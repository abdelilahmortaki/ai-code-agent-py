from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agent.codebase import CodebaseService
from agent.config import ProjectConfig
from agent.module_resolver import resolve_module
from agent.paths import resolve_within
from agent.versioning import VersionIndexer, _database_url, _redact_secrets, compute_file_hash

if TYPE_CHECKING:
    from agent.db.store import PgStore


class SymbolParseError(Exception):
    """Raised when Java source is malformed or structurally unbalanced."""


@dataclass(frozen=True)
class Symbol:
    """A logical Java declaration with exact source location metadata."""

    file: str
    module: str
    package_name: str
    owner: str
    symbol_type: str
    name: str
    qualified_name: str
    signature: str
    start_line: int
    end_line: int
    source: str


@dataclass(frozen=True)
class _Token:
    """A scanned token with its exact source offsets and line numbers."""

    value: str
    kind: str  # "ident" | "number" | "string" | "char" | "punct"
    start: int
    end: int
    start_line: int
    end_line: int


# Java type declaration keywords (sorted).
_TYPE_KEYWORDS = frozenset({"class", "interface", "enum", "record"})

# Java member modifiers (sorted).
_MODIFIERS = frozenset(
    {
        "abstract",
        "default",
        "final",
        "native",
        "private",
        "protected",
        "public",
        "sealed",
        "static",
        "strictfp",
        "synchronized",
        "transient",
        "volatile",
    }
)


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch in "_$"


def _is_ident_part(ch: str) -> bool:
    return ch.isalnum() or ch in "_$"


class _JavaScanner:
    """Hand-written Java tokenizer (pure stdlib, no regex).

    Skips whitespace, line comments (``//``), block comments (``/* ... */``)
    including javadoc, and collapses string/char literals (with backslash
    escapes) into single tokens, so braces inside them never affect nesting
    depth. Tracks 1-based line numbers.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self._buffer: list[_Token | None] = []

    # ------------------------------------------------------------- low level

    def _advance(self, n: int) -> None:
        self.line += self.text[self.pos : self.pos + n].count("\n")
        self.pos += n

    def _skip_trivia(self) -> None:
        text, n = self.text, len(self.text)
        while self.pos < n:
            ch = text[self.pos]
            if ch in " \t\r\n\f\v":
                self._advance(1)
                continue
            if ch == "/" and self.pos + 1 < n:
                nxt = text[self.pos + 1]
                if nxt == "/":
                    end = text.find("\n", self.pos)
                    self._advance(n - self.pos if end == -1 else end - self.pos)
                    continue
                if nxt == "*":
                    end = text.find("*/", self.pos + 2)
                    if end == -1:
                        raise SymbolParseError(
                            f"unterminated block comment starting at line {self.line}"
                        )
                    self._advance(end + 2 - self.pos)
                    continue
            break

    def _read_text_block(self) -> _Token:
        """Consume a Java 15+ text block (``\"\"\" ... \"\"\"``) as one token.

        Text blocks may span lines and contain quotes, braces and other
        structure that must never affect nesting depth; only the closing
        ``\"\"\"`` (or the escaped ``\\\"\"\"``, skipped by the backslash
        rule) terminates the token.
        """
        text, n, start, start_line = self.text, len(self.text), self.pos, self.line
        self._advance(3)  # opening """
        while self.pos < n:
            if text.startswith('"""', self.pos):
                self._advance(3)
                return _Token(
                    text[start : self.pos],
                    "string",
                    start,
                    self.pos,
                    start_line,
                    self.line,
                )
            ch = text[self.pos]
            if ch == "\\":
                self._advance(2)
                continue
            self._advance(1)
        raise SymbolParseError(f"unterminated text block starting at line {start_line}")

    def _read_quoted(self, quote: str) -> _Token:
        text, n, start, start_line = self.text, len(self.text), self.pos, self.line
        self._advance(1)  # opening quote
        while self.pos < n:
            ch = text[self.pos]
            if ch == "\\":
                self._advance(2)
                continue
            if ch == "\n":
                raise SymbolParseError(
                    f"unterminated {quote}-literal at line {start_line}"
                )
            if ch == quote:
                self._advance(1)
                return _Token(
                    text[start : self.pos],
                    "string" if quote == '"' else "char",
                    start,
                    self.pos,
                    start_line,
                    self.line,
                )
            self._advance(1)
        raise SymbolParseError(f"unterminated {quote}-literal at line {start_line}")

    def _read_ident(self) -> _Token:
        text, n, start, start_line = self.text, len(self.text), self.pos, self.line
        self._advance(1)
        while self.pos < n and _is_ident_part(text[self.pos]):
            self._advance(1)
        return _Token(text[start : self.pos], "ident", start, self.pos, start_line, self.line)

    def _read_number(self) -> _Token:
        text, n, start, start_line = self.text, len(self.text), self.pos, self.line
        while self.pos < n and (text[self.pos].isalnum() or text[self.pos] in "_."):
            self._advance(1)
        return _Token(text[start : self.pos], "number", start, self.pos, start_line, self.line)

    def _scan(self) -> _Token | None:
        self._skip_trivia()
        if self.pos >= len(self.text):
            return None
        ch = self.text[self.pos]
        if ch == '"' and self.text.startswith('"""', self.pos):
            return self._read_text_block()
        if ch == '"':
            return self._read_quoted('"')
        if ch == "'":
            return self._read_quoted("'")
        if _is_ident_start(ch):
            return self._read_ident()
        if ch.isdigit() or (
            ch == "." and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()
        ):
            return self._read_number()
        token = _Token(ch, "punct", self.pos, self.pos + 1, self.line, self.line)
        self._advance(1)
        return token

    # ----------------------------------------------------------- token API

    def _ensure(self, n: int) -> None:
        while len(self._buffer) < n:
            self._buffer.append(self._scan())

    def peek(self) -> _Token | None:
        self._ensure(1)
        return self._buffer[0]

    def peek2(self) -> _Token | None:
        self._ensure(2)
        return self._buffer[1]

    def consume(self) -> _Token | None:
        self._ensure(1)
        return self._buffer.pop(0)

    def consume_balanced(
        self, open_c: str, close_c: str, sig: list[_Token] | None = None
    ) -> _Token:
        """Consume from the next token through its matching close token.

        Depth is tracked on tokens only, so braces/parens inside strings,
        chars and comments (collapsed by the tokenizer) cannot affect it.
        Returns the closing token.
        """
        depth = 0
        while True:
            token = self.peek()
            if token is None:
                raise SymbolParseError(
                    f"unbalanced '{open_c}'...'{close_c}' group (expected closing "
                    f"'{close_c}', reached end of file)"
                )
            if sig is not None:
                sig.append(token)
            if token.value == open_c:
                depth += 1
                self.consume()
            elif token.value == close_c:
                depth -= 1
                self.consume()
                if depth == 0:
                    return token
            else:
                self.consume()

    def consume_annotation(self) -> None:
        """Consume one annotation: @name(.name)* with optional balanced args."""
        at = self.consume()
        if at is None or at.value != "@":
            raise SymbolParseError(
                f"expected '@' at line {at.start_line if at else self.line}"
            )
        name = self.consume()
        if name is None or name.kind != "ident":
            raise SymbolParseError(f"malformed annotation at line {at.start_line}")
        while self.peek() is not None and self.peek().value == ".":
            self.consume()
            part = self.consume()
            if part is None or part.kind != "ident":
                raise SymbolParseError(f"malformed annotation at line {at.start_line}")
        if self.peek() is not None and self.peek().value == "(":
            self.consume_balanced("(", ")")


def _normalize_signature(tokens: list[_Token]) -> str:
    """Whitespace-normalize a header token stream into a single line.

    Deterministic: the same token stream always yields the same string.
    Annotations are excluded from the signature by the scanner, so any '@'
    here is part of the '@interface' keyword.
    """
    parts: list[str] = []
    prev = ""
    for token in tokens:
        value = token.value
        if parts:
            if value in "(,.;:)]><-" or prev in "([.<@-":
                parts.append(value)
            else:
                parts.append(" " + value)
        else:
            parts.append(value)
        prev = value
    return "".join(parts)


class JavaSymbolParser:
    """Pure-stdlib Java declaration parser producing logical symbols.

    Detects top-level and nested types (class/interface/enum/record/
    annotation) plus methods and constructors, each with an exact source
    slice. Methods are never split: the full balanced body is consumed.
    Malformed or unbalanced source raises SymbolParseError.
    """

    def __init__(self, file: str = "", root: str | Path | None = None) -> None:
        self.file = file
        self.root = Path(root) if root is not None else None

    # ------------------------------------------------------------------ API

    def parse(self, text: str) -> list[Symbol]:
        self._scanner = _JavaScanner(text)
        self._module = ""
        self._package_name = ""
        if self.root is not None:
            self._module = resolve_module(self.root, self.file)
        self._symbols: list[Symbol] = []
        self._parse_block(owner="", enclosing_name="", is_enum=False, top_level=True)
        return self._symbols

    # ------------------------------------------------------------ structure

    def _parse_block(
        self,
        owner: str,
        enclosing_name: str,
        is_enum: bool,
        top_level: bool = False,
    ) -> _Token | None:
        """Parse a type body (or the compilation unit when top_level).

        Returns the closing '}' token of a type body, or None for the
        (EOF-terminated) top level.
        """
        while True:
            token = self._scanner.peek()
            if token is None:
                if top_level:
                    return None
                raise SymbolParseError(
                    f"unexpected end of file inside type body (missing '}}' for "
                    f"'{enclosing_name}' near line {self._scanner.line})"
                )
            if token.value == "}":
                self._scanner.consume()
                if top_level:
                    raise SymbolParseError(
                        f"unbalanced closing '}}' at line {token.start_line}"
                    )
                return token
            if token.value in (";", ","):
                self._scanner.consume()
                continue
            if token.value in ("package", "import", "module"):
                if not top_level:
                    pass  # restricted keyword used as an identifier; fall through
                elif token.value == "package" and not self._package_name:
                    self._parse_package()
                    continue
                elif token.value == "import":
                    self._parse_import()
                    continue
                elif token.value == "module":
                    self._parse_module_decl()
                    continue
            symbol = self._parse_member(
                owner=owner,
                enclosing_name=enclosing_name,
                is_enum=is_enum,
                top_level=top_level,
            )
            if symbol is not None:
                self._symbols.append(symbol)

    def _parse_package(self) -> None:
        self._scanner.consume()  # 'package'
        parts: list[str] = []
        while True:
            token = self._scanner.consume()
            if token is None:
                raise SymbolParseError("unexpected end of file in package declaration")
            if token.value == ";":
                break
            if token.kind == "ident":
                parts.append(token.value)
        self._package_name = ".".join(parts)

    def _parse_import(self) -> None:
        self._scanner.consume()  # 'import'
        while True:
            token = self._scanner.consume()
            if token is None:
                raise SymbolParseError("unexpected end of file in import declaration")
            if token.value == ";":
                return

    def _parse_module_decl(self) -> None:
        """Skip a module-info.java body: module a.b { ... } (not a symbol)."""
        self._scanner.consume()  # 'module'
        while True:
            token = self._scanner.consume()
            if token is None:
                raise SymbolParseError("unexpected end of file in module declaration")
            if token.value == "{":
                self._scanner.consume_balanced("{", "}")
                return

    def _parse_member(
        self,
        owner: str,
        enclosing_name: str,
        is_enum: bool,
        top_level: bool,
    ) -> Symbol | None:
        """Parse one declaration; returns a symbol or None for non-symbols."""
        start_token = self._scanner.peek()
        assert start_token is not None
        saw_modifier = False
        sig_prefix: list[_Token] = []

        # Annotations (with balanced args) and modifiers precede the
        # declaration; "@interface" is the annotation TYPE keyword.
        while True:
            token = self._scanner.peek()
            if token is None:
                raise SymbolParseError("unexpected end of file inside type body")
            if token.value == "@":
                nxt = self._scanner.peek2()
                if nxt is not None and nxt.value == "interface":
                    break  # @interface declaration
                self._scanner.consume_annotation()
                continue
            if token.value in _MODIFIERS:
                sig_prefix.append(token)
                self._scanner.consume()
                saw_modifier = True
                continue
            if token.value == "non":
                nxt = self._scanner.peek2()
                if nxt is not None and nxt.value == "-":
                    self._scanner.consume()
                    self._scanner.consume()
                    sealed = self._scanner.consume()
                    if sealed is None or sealed.value != "sealed":
                        raise SymbolParseError(
                            f"malformed 'non-sealed' modifier at line {token.start_line}"
                        )
                    sig_prefix.extend([token, nxt, sealed])
                    saw_modifier = True
                    continue
            break

        token = self._scanner.peek()
        assert token is not None
        if token.value == "@":  # @interface
            return self._parse_type(owner, start_token, sig_prefix)
        if token.value in _TYPE_KEYWORDS:
            return self._parse_type(owner, start_token, sig_prefix)

        # Enum constants: bare identifier at member level inside an enum body.
        if is_enum and not saw_modifier and token.kind == "ident":
            nxt = self._scanner.peek2()
            is_constant = nxt is None or nxt.value in ("(", "{", ",", ";", "}")
            if is_constant and token.value != enclosing_name:
                self._scanner.consume()  # constant name
                if self._scanner.peek() is not None and self._scanner.peek().value == "(":
                    self._scanner.consume_balanced("(", ")")
                if self._scanner.peek() is not None and self._scanner.peek().value == "{":
                    self._scanner.consume_balanced("{", "}")
                return None

        return self._parse_method_or_field(
            owner=owner,
            enclosing_name=enclosing_name,
            start_token=start_token,
            sig_prefix=sig_prefix,
            top_level=top_level,
        )

    def _parse_method_or_field(
        self,
        owner: str,
        enclosing_name: str,
        start_token: _Token,
        sig_prefix: list[_Token],
        top_level: bool,
    ) -> Symbol | None:
        """Scan a member: a method/constructor, a field, or an initializer."""
        sig: list[_Token] = list(sig_prefix)
        last_ident: _Token | None = None
        prev_was_ident = False
        saw_equals = False

        while True:
            token = self._scanner.peek()
            if token is None:
                raise SymbolParseError(
                    f"unexpected end of file in member declaration starting at "
                    f"line {start_token.start_line}"
                )
            value = token.value
            if value == "(":
                if last_ident is not None and not saw_equals and prev_was_ident:
                    name_token = last_ident
                    self._scanner.consume_balanced("(", ")", sig)
                    end_token = self._scan_method_tail(sig)
                    sym_type = (
                        "constructor"
                        if name_token.value == enclosing_name
                        else "method"
                    )
                    return self._build_symbol(
                        start_token, owner, sym_type, name_token.value, sig, end_token
                    )
                self._scanner.consume_balanced("(", ")", sig)
                prev_was_ident = False
                continue
            if value == "=":
                saw_equals = True
                prev_was_ident = False
                self._scanner.consume()
                continue
            if value == "{":
                if (
                    last_ident is not None
                    and not saw_equals
                    and prev_was_ident
                    and last_ident.value == enclosing_name
                ):
                    # record compact constructor:  Name { ... }
                    end_token = self._scanner.consume_balanced("{", "}")
                    return self._build_symbol(
                        start_token, owner, "constructor", last_ident.value, sig, end_token
                    )
                # Initializer block or anonymous body at member level.
                self._scanner.consume_balanced("{", "}")
                prev_was_ident = False
                continue
            if value == ";":
                self._scanner.consume()
                if top_level:
                    raise SymbolParseError(
                        f"unexpected top-level declaration at line {token.start_line}"
                    )
                return None  # field or stray terminator: not a symbol
            if value == ",":
                sig.append(token)
                self._scanner.consume()
                prev_was_ident = False
                continue
            if value == "@":
                self._scanner.consume_annotation()
                continue
            if token.kind == "ident":
                last_ident = token
                prev_was_ident = True
            else:
                prev_was_ident = False
            sig.append(token)
            self._scanner.consume()

    def _scan_method_tail(self, sig: list[_Token]) -> _Token:
        """After a method's parameter list, find '{' (body) or ';' (bodyless)."""
        while True:
            token = self._scanner.peek()
            if token is None:
                raise SymbolParseError(
                    "unexpected end of file in method declaration (missing body or ';')"
                )
            if token.value == "{":
                return self._scanner.consume_balanced("{", "}")
            if token.value == ";":
                self._scanner.consume()
                return token
            sig.append(token)
            self._scanner.consume()

    def _parse_type(
        self, owner: str, start_token: _Token, sig_prefix: list[_Token] | None = None
    ) -> Symbol | None:
        """Parse a type declaration and its body, then its members."""
        sig: list[_Token] = list(sig_prefix or [])
        kind = ""
        while True:
            token = self._scanner.peek()
            if token is None:
                raise SymbolParseError(
                    f"unexpected end of file in type declaration at line "
                    f"{start_token.start_line}"
                )
            if token.value == "@":
                nxt = self._scanner.peek2()
                if nxt is not None and nxt.value == "interface":
                    kind = "annotation"
                    sig.append(token)
                    self._scanner.consume()
                    kw = self._scanner.consume()
                    if kw is None or kw.value != "interface":
                        raise SymbolParseError(
                            f"malformed @interface at line {token.start_line}"
                        )
                    sig.append(kw)
                    break
                self._scanner.consume_annotation()
                continue
            if token.value in _MODIFIERS:
                sig.append(token)
                self._scanner.consume()
                continue
            if token.value == "non":
                nxt = self._scanner.peek2()
                if nxt is not None and nxt.value == "-":
                    self._scanner.consume()
                    self._scanner.consume()
                    sealed = self._scanner.consume()
                    if sealed is None or sealed.value != "sealed":
                        raise SymbolParseError(
                            f"malformed 'non-sealed' at line {token.start_line}"
                        )
                    sig.extend([token, nxt, sealed])
                    continue
            if token.value in _TYPE_KEYWORDS:
                kind = token.value
                sig.append(token)
                self._scanner.consume()
                break
            raise SymbolParseError(
                f"expected a type declaration keyword at line {token.start_line}"
            )

        name_token = self._scanner.consume()
        if name_token is None or name_token.kind != "ident":
            raise SymbolParseError(
                f"expected a type name at line "
                f"{name_token.start_line if name_token else self._scanner.line}"
            )
        name = name_token.value
        sig.append(name_token)

        # Header tail: generics, record components, extends/implements/permits.
        while True:
            token = self._scanner.peek()
            if token is None:
                raise SymbolParseError(
                    f"unexpected end of file in '{kind} {name}' header"
                )
            if token.value == "{":
                self._scanner.consume()
                break
            if token.value in ("(", "<"):
                self._scanner.consume_balanced(token.value, ")" if token.value == "(" else ">", sig)
                continue
            sig.append(token)
            self._scanner.consume()

        child_owner = f"{owner}.{name}" if owner else name
        index = len(self._symbols)
        closing = self._parse_block(
            owner=child_owner,
            enclosing_name=name,
            is_enum=(kind == "enum"),
            top_level=False,
        )
        if closing is None:  # unreachable for type bodies; defensive
            raise SymbolParseError(
                f"unbalanced type body for '{name}' (missing '}}')"
            )
        # Insert the parent symbol at its declaration position so symbols
        # appear in declaration order (parent before its members).
        self._symbols.insert(
            index,
            self._build_symbol(start_token, owner, kind, name, sig, closing),
        )
        return None

    def _build_symbol(
        self,
        start_token: _Token,
        owner: str,
        symbol_type: str,
        name: str,
        sig: list[_Token],
        end_token: _Token,
    ) -> Symbol:
        source = self._scanner.text[start_token.start : self._scanner.pos]
        return Symbol(
            file=self.file,
            module=self._module,
            package_name=self._package_name,
            owner=owner,
            symbol_type=symbol_type,
            name=name,
            qualified_name=".".join(part for part in (self._package_name, owner, name) if part),
            signature=_normalize_signature(sig),
            start_line=start_token.start_line,
            end_line=end_token.end_line,
            source=source,
        )


class SymbolIndexer:
    """Indexes Java symbols into PostgreSQL via the store, per file."""

    def __init__(self, codebase: CodebaseService, store: PgStore | None) -> None:
        self.codebase = codebase
        self.store = store

    def index_symbols(self, project: ProjectConfig) -> dict:
        if self.store is None:
            raise RuntimeError("database is not configured")
        version_result = VersionIndexer(self.codebase, self.store).index_version(project)
        version = version_result["version"]
        root = Path(project.repo_root).resolve()

        files_indexed = 0
        symbols_indexed = 0
        failed_files: list[str] = []

        for rel in self.codebase.scan_paths(project):
            if not rel.endswith(".java"):
                continue
            try:
                target = resolve_within(root, rel)
                content = target.read_text(encoding="utf-8", errors="replace")
                file_row = self.store.upsert_file(
                    project_version_id=version["id"],
                    path=rel,
                    language="java",
                    file_hash=compute_file_hash(target),
                )
                symbols = JavaSymbolParser(file=rel, root=root).parse(content)
                for symbol in symbols:
                    self.store.insert_symbol(
                        project_version_id=version["id"],
                        file_id=file_row["id"],
                        module=symbol.module,
                        package_name=symbol.package_name,
                        owner=symbol.owner,
                        symbol_type=symbol.symbol_type,
                        name=symbol.name,
                        qualified_name=symbol.qualified_name,
                        signature=symbol.signature,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        source=symbol.source,
                    )
                files_indexed += 1
                symbols_indexed += len(symbols)
            except (SymbolParseError, OSError):
                failed_files.append(rel)
        return {
            "version": version,
            "files_indexed": files_indexed,
            "symbols_indexed": symbols_indexed,
            "failed_files": failed_files,
        }


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.symbols",
        description="Index Java symbols for a project snapshot.",
    )
    parser.add_argument("--repo-root", required=True, help="Absolute path of the project to index")
    parser.add_argument("--name", default="", help="Project name (defaults to the directory name)")
    parser.add_argument(
        "--external-id",
        default="",
        help="Unique external id (defaults to the resolved repo root path)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides DATABASE_URL)",
    )
    args = parser.parse_args(argv)

    dsn = _database_url(args.url)
    if not dsn:
        print(
            "No database URL configured: pass --url or set DATABASE_URL",
            file=sys.stderr,
        )
        return 1

    try:
        from agent.db.store import PgStore

        store = PgStore(dsn)
        root = Path(args.repo_root).resolve()
        name = args.name or root.name
        external_id = args.external_id or str(root)
        project = ProjectConfig(
            id=external_id,
            name=name,
            repo_root=str(root),
            stories_file="",
            index_file="",
        )
        result = SymbolIndexer(CodebaseService(), store).index_symbols(project)
    except Exception as exc:
        print(f"Indexing failed: {_redact_secrets(str(exc), dsn)}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
