"""Code graph relationships for an indexed project version (F2.2).

Extracts deterministic, project-internal edges from the stored
``code_symbols`` rows of one ``project_version`` (plus the file texts
for import statements) and persists them into ``code_edges``:

- IMPORTS:  importing type symbol -> imported project type symbol
- CALLS:    calling method/constructor symbol -> called method symbol
            (or constructor symbol for ``new Type(...)``)
- INHERITS: subclass -> superclass
- IMPLEMENTS: class -> interface
- TESTED_BY: production type symbol -> test type symbol
            (tests detected by path ``/src/test/`` or ``Test``/``Tests``
            suffix; direction is production -> test)

Only resolvable project-internal targets produce edges: external or
ambiguous references are counted in ``unresolved_skipped`` and never
fail the run, and no symbol nodes are ever fabricated for them.

Everything is a deterministic function of (stored rows, file texts):
the same version and files produce the same edge set, which is
deduplicated and sorted before persistence. Builds are idempotent
(delete-then-insert for the version).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from agent.codebase import CodebaseService
from agent.config import ProjectConfig
from agent.paths import resolve_within
from agent.symbols import SymbolParseError, _JavaScanner, _Token

if TYPE_CHECKING:
    from agent.db.store import PgStore

IMPORTS = "IMPORTS"
CALLS = "CALLS"
INHERITS = "INHERITS"
IMPLEMENTS = "IMPLEMENTS"
TESTED_BY = "TESTED_BY"

RELATIONS = frozenset({IMPORTS, CALLS, INHERITS, IMPLEMENTS, TESTED_BY})

_TYPE_KINDS = frozenset({"class", "interface", "enum", "record", "annotation"})

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

_CALL_KEYWORDS = frozenset(
    {
        "assert",
        "case",
        "catch",
        "default",
        "else",
        "for",
        "if",
        "instanceof",
        "return",
        "super",
        "switch",
        "synchronized",
        "throw",
        "while",
    }
)

_RESERVED_WORDS = frozenset(
    {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "package",
        "private",
        "protected",
        "public",
        "record",
        "return",
        "sealed",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "var",
        "void",
        "volatile",
        "while",
    }
)


def _tokens(text: str) -> list[_Token]:
    try:
        scanner = _JavaScanner(text)
        tokens: list[_Token] = []
        while True:
            token = scanner.consume()
            if token is None:
                break
            tokens.append(token)
        return tokens
    except SymbolParseError:
        return []


def _single_type_imports(text: str) -> list[tuple[str, str]]:
    """Return [(simple_name, full_dotted_path)] for single-type imports.

    ``import static`` and wildcard imports (``import a.b.*;``) are skipped
    and never create a graph target; only the final simple segment of each
    single-type import is kept.
    """
    tokens = _tokens(text)
    result: list[tuple[str, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].value == "import":
            j = i + 1
            if j < n and tokens[j].value == "static":
                while j < n and tokens[j].value != ";":
                    j += 1
                i = j + 1
                continue
            parts: list[str] = []
            wildcard = False
            while j < n and tokens[j].value != ";":
                if tokens[j].kind == "ident":
                    parts.append(tokens[j].value)
                elif tokens[j].value == "*":
                    wildcard = True
                elif tokens[j].value != ".":
                    break
                j += 1
            if not wildcard and parts:
                result.append((parts[-1], ".".join(parts)))
            i = j + 1
        else:
            i += 1
    return result


def _header_types(signature: str, keyword: str) -> list[str]:
    """Collect the dotted type names following ``extends``/``implements``."""
    tokens = _tokens(signature)
    result: list[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i].value == keyword:
            j = i + 1
            while j < n:
                value = tokens[j].value
                if value in ("extends", "implements", "permits", "{"):
                    break
                if value == ",":
                    j += 1
                    continue
                if tokens[j].kind != "ident":
                    break
                name_parts: list[str] = []
                k = j
                while (
                    k < n
                    and (tokens[k].kind == "ident" or tokens[k].value == ".")
                    and tokens[k].value not in ("extends", "implements", "permits")
                ):
                    if tokens[k].kind == "ident":
                        name_parts.append(tokens[k].value)
                    k += 1
                if name_parts:
                    result.append(".".join(name_parts))
                if k < n and tokens[k].value == "<":
                    depth = 1
                    k += 1
                    while k < n and depth:
                        if tokens[k].value == "<":
                            depth += 1
                        elif tokens[k].value == ">":
                            depth -= 1
                        k += 1
                j = k
            break
        i += 1
    return result


def _field_var_types(source: str) -> dict[str, str]:
    """Map field name -> declared (outermost) type for a type body source."""
    tokens = _tokens(source)
    fields: dict[str, str] = {}
    brace = 0
    paren = 0
    angle = 0
    seg_start = 0
    n = len(tokens)
    i = 0
    while i < n:
        value = tokens[i].value
        if value == "<":
            angle += 1
        elif value == ">":
            angle = max(0, angle - 1)
        elif angle == 0 and value == "{":
            if brace == 0:
                seg_start = i + 1
            brace += 1
        elif angle == 0 and value == "}":
            brace = max(0, brace - 1)
        elif angle == 0 and value == "(":
            paren += 1
        elif angle == 0 and value == ")":
            paren = max(0, paren - 1)
        elif angle == 0 and paren == 0 and brace == 1 and value in (";", ",", "="):
            seg = tokens[seg_start:i]
            if seg and seg[-1].kind == "ident":
                name = seg[-1].value
                type_idents = [
                    t.value
                    for t in seg[:-1]
                    if t.kind == "ident"
                    and t.value[0].isupper()
                    and t.value not in _MODIFIERS
                ]
                if type_idents and name[0].islower():
                    fields[name] = type_idents[0]
            seg_start = i + 1
        i += 1
    return fields


def _param_var_types(signature: str) -> dict[str, str]:
    """Map parameter name -> declared (outermost) type for a method signature."""
    tokens = _tokens(signature)
    depth = 0
    start = -1
    end = -1
    for i, token in enumerate(tokens):
        if token.value == "(":
            depth += 1
            if start == -1:
                start = i
        elif token.value == ")":
            depth -= 1
            if start != -1 and depth == 0:
                end = i
                break
    if start == -1 or end == -1:
        return {}
    result: dict[str, str] = {}
    seg: list[_Token] = []
    angle = 0
    for token in tokens[start + 1 : end]:
        if token.value == "<":
            angle += 1
            seg.append(token)
        elif token.value == ">":
            angle = max(0, angle - 1)
            seg.append(token)
        elif token.value == "," and angle == 0:
            _param_from_seg(seg, result)
            seg = []
        else:
            seg.append(token)
    _param_from_seg(seg, result)
    return result


def _param_from_seg(seg: list[_Token], result: dict[str, str]) -> None:
    if not seg or seg[-1].kind != "ident":
        return
    name = seg[-1].value
    type_idents = [
        t.value
        for t in seg[:-1]
        if t.kind == "ident"
        and t.value[0].isupper()
        and t.value not in _MODIFIERS
        and t.value != "final"
    ]
    if type_idents and name[0].islower():
        result[name] = type_idents[0]


def _local_var_types(source: str) -> dict[str, str]:
    """Map local variable name -> declared type for a method body source."""
    tokens = _tokens(source)
    result: dict[str, str] = {}
    n = len(tokens)
    for i in range(n - 1):
        token = tokens[i]
        if token.kind != "ident" or not token.value[0].isupper():
            continue
        nxt = tokens[i + 1]
        if nxt.kind != "ident" or nxt.value in _RESERVED_WORDS:
            continue
        after = tokens[i + 2] if i + 2 < n else None
        if after is not None and after.value == "(":
            continue
        result[nxt.value] = token.value
    return result


def _method_param_count(signature: str) -> int:
    """Number of parameters in the first paren group of a signature."""
    tokens = _tokens(signature)
    for i, token in enumerate(tokens):
        if token.value != "(":
            continue
        inner = 0
        commas = 0
        j = i + 1
        while j < len(tokens):
            value = tokens[j].value
            if value == "(":
                inner += 1
            elif value == ")":
                if inner == 0:
                    return 0 if j == i + 1 else commas + 1
                inner -= 1
            elif value == "," and inner == 0:
                commas += 1
            j += 1
        return 0
    return 0


def _call_sites(source: str) -> list[dict]:
    """Extract deterministic call-site evidence from a method body source."""
    tokens = _tokens(source)
    sites: list[dict] = []
    n = len(tokens)
    start = 0
    depth = 0
    for idx, token in enumerate(tokens):
        if token.value == "(":
            depth += 1
            if depth == 1:
                start = idx + 1
        elif token.value == ")":
            depth -= 1
            if depth == 0 and start:
                break
    i = start
    while i < n:
        if tokens[i].value != "(":
            i += 1
            continue
        prev = tokens[i - 1] if i >= 1 else None
        if prev is None or prev.kind != "ident":
            i += 1
            continue
        method = prev.value
        if method in _CALL_KEYWORDS:
            i += 1
            continue

        chain: list[str] = [method]
        j = i - 2
        while j >= 0 and tokens[j].value == ".":
            if j >= 1 and tokens[j - 1].kind == "ident":
                chain.insert(0, tokens[j - 1].value)
                j -= 2
            else:
                break

        is_new = False
        type_hint: str | None = None
        if j >= 0 and tokens[j].value == "new":
            is_new = True
            type_hint = chain[0]
        elif j >= 0 and tokens[j].value == ".":
            k = j - 1
            if k >= 0 and tokens[k].value == ")":
                depth = 0
                while k >= 0:
                    value = tokens[k].value
                    if value == ")":
                        depth += 1
                    elif value == "(":
                        depth -= 1
                        if depth == 0:
                            break
                    k -= 1
                if (
                    k >= 2
                    and tokens[k].value == "("
                    and tokens[k - 1].kind == "ident"
                    and tokens[k - 2].value == "new"
                ):
                    type_hint = tokens[k - 1].value

        depth = 0
        commas = 0
        k = i
        closed = False
        while k < n:
            value = tokens[k].value
            if value == "(":
                depth += 1
            elif value == ")":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
            elif value == "," and depth == 1:
                commas += 1
            k += 1
        if not closed:
            i += 1
            continue
        arg_count = 0 if (i + 1 < n and tokens[i + 1].value == ")") else commas + 1

        if is_new or type_hint:
            sites.append(
                {
                    "method": method,
                    "receiver": None,
                    "is_new": is_new,
                    "type_hint": type_hint,
                    "arg_count": arg_count,
                }
            )
        elif chain:
            receiver = chain[-2] if len(chain) >= 2 else None
            if receiver in ("this", "super") or receiver is None:
                sites.append(
                    {
                        "method": method,
                        "receiver": None,
                        "is_new": False,
                        "type_hint": None,
                        "arg_count": arg_count,
                    }
                )
            else:
                sites.append(
                    {
                        "method": method,
                        "receiver": receiver,
                        "is_new": False,
                        "type_hint": receiver if receiver[0].isupper() else None,
                        "arg_count": arg_count,
                    }
                )
        else:
            sites.append(
                {
                    "method": method,
                    "receiver": None,
                    "is_new": False,
                    "type_hint": None,
                    "arg_count": arg_count,
                }
            )
        i += 1
    return sites


def _type_qual(symbol: dict) -> str:
    owner = symbol.get("owner", "") or ""
    name = symbol.get("name", "") or ""
    return f"{owner}.{name}" if owner else name


class CodeGraphBuilder:
    """Build and persist the deterministic code graph of one project version."""

    def __init__(self, codebase: CodebaseService, store: PgStore | None) -> None:
        self.codebase = codebase
        self.store = store

    def build(self, project_version_id: str, repo_root: str) -> dict:
        if self.store is None:
            raise RuntimeError("database is not configured")
        symbols = self.store.list_symbols(project_version_id)

        by_path: dict[str, list[dict]] = {}
        for symbol in symbols:
            by_path.setdefault(symbol["path"], []).append(symbol)

        full_path_index: dict[str, list[dict]] = {}
        owner_index: dict[str, list[dict]] = {}
        simple_index: dict[str, list[dict]] = {}
        for symbol in symbols:
            simple_index.setdefault(symbol["name"], []).append(symbol)
            key = _type_qual(symbol)
            if symbol.get("qualified_name"):
                full_path_index.setdefault(symbol["qualified_name"], []).append(symbol)
            owner_index.setdefault(key, []).append(symbol)

        def resolve(name: str, module: str, package_name: str, path: str) -> dict | None:
            """Resolve a type reference to a symbol.

            The Java namespace (qualified_name / owner.name / package_name) is
            authoritative; the Maven module is only a disambiguation signal.
            Duplicate qualified names across modules are resolved to the
            symbol in the current module when unambiguous, otherwise they are
            ambiguous and resolve to None (never an arbitrary pick).
            """
            if "." in name:
                candidates = [
                    c
                    for c in full_path_index.get(name, [])
                    if c["symbol_type"] in _TYPE_KINDS
                ]
                if len(candidates) == 1:
                    return candidates[0]
                if len(candidates) > 1:
                    in_module = [c for c in candidates if c.get("module") == module]
                    if len(in_module) == 1:
                        return in_module[0]
                    return None
                candidates = [
                    c
                    for c in owner_index.get(name, [])
                    if c["symbol_type"] in _TYPE_KINDS
                ]
                if len(candidates) == 1:
                    return candidates[0]
                return None
            candidates = [
                c for c in simple_index.get(name, []) if c["symbol_type"] in _TYPE_KINDS
            ]
            tier_same_file = [c for c in candidates if c["path"] == path]
            if tier_same_file:
                if len(tier_same_file) == 1:
                    return tier_same_file[0]
                return None
            tier_same_package = [
                c for c in candidates if c.get("package_name") == package_name
            ]
            if tier_same_package:
                if len(tier_same_package) == 1:
                    return tier_same_package[0]
                return None
            tier_same_module = [c for c in candidates if c.get("module") == module]
            if tier_same_module:
                if len(tier_same_module) == 1:
                    return tier_same_module[0]
                return None
            if len(candidates) == 1:
                return candidates[0]
            return None

        root = Path(repo_root).resolve()
        file_texts: dict[str, str] = {}
        files_skipped = 0
        for path in sorted(by_path):
            try:
                target = resolve_within(root, path)
                file_texts[path] = target.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                files_skipped += 1
                file_texts[path] = ""

        edges: set[tuple[str, str, str]] = set()
        unresolved = 0

        for path in sorted(by_path):
            text = file_texts[path]
            file_symbols = by_path[path]
            module = next(
                (s["module"] for s in file_symbols if s["symbol_type"] in _TYPE_KINDS and s["module"]),
                "",
            ) or (file_symbols[0]["module"] if file_symbols else "")
            package_name = next(
                (
                    s["package_name"]
                    for s in file_symbols
                    if s["symbol_type"] in _TYPE_KINDS and s.get("package_name")
                ),
                "",
            ) or (file_symbols[0].get("package_name", "") if file_symbols else "")

            imports = _single_type_imports(text)
            type_symbols = [s for s in file_symbols if s["symbol_type"] in _TYPE_KINDS]

            for type_symbol in type_symbols:
                for _, full in imports:
                    # Explicit single-type imports are authoritative FQNs:
                    # no fallback to the simple name, so an import of an
                    # external type can never fabricate an edge to a project
                    # type that merely shares the simple name.
                    target = resolve(full, module, package_name, path)
                    if target is None:
                        unresolved += 1
                        continue
                    edges.add((type_symbol["id"], target["id"], IMPORTS))

            for type_symbol in type_symbols:
                for name in _header_types(type_symbol["signature"], "extends"):
                    target = resolve(name, type_symbol["module"], type_symbol["package_name"], path)
                    if target is None:
                        unresolved += 1
                        continue
                    edges.add((type_symbol["id"], target["id"], INHERITS))
                for name in _header_types(type_symbol["signature"], "implements"):
                    target = resolve(name, type_symbol["module"], type_symbol["package_name"], path)
                    if target is None:
                        unresolved += 1
                        continue
                    edges.add((type_symbol["id"], target["id"], IMPLEMENTS))

            method_symbols = [
                s for s in file_symbols if s["symbol_type"] in ("method", "constructor")
            ]
            owner_types: dict[str, list[dict]] = {}
            for method in method_symbols:
                owner_types.setdefault(method["owner"], [])
                owner_types[method["owner"]].extend(
                    t for t in type_symbols if _type_qual(t) == method["owner"]
                )

            for method in method_symbols:
                fields: dict[str, str] = {}
                for owner_type in owner_types.get(method["owner"], []):
                    fields.update(_field_var_types(owner_type["source"]))
                params = _param_var_types(method["signature"])
                locals_ = _local_var_types(method["source"])
                for call in _call_sites(method["source"]):
                    if call["is_new"] or call["type_hint"]:
                        type_name = call["type_hint"]
                    elif call["receiver"]:
                        type_name = (
                            locals_.get(call["receiver"])
                            or params.get(call["receiver"])
                            or fields.get(call["receiver"])
                            or (call["receiver"] if call["receiver"][0].isupper() else None)
                        )
                    else:
                        unresolved += 1
                        continue
                    if not type_name:
                        unresolved += 1
                        continue
                    target_type = resolve(type_name, method["module"], package_name, path)
                    if target_type is None:
                        unresolved += 1
                        continue
                    if call["is_new"]:
                        candidates = [
                            s
                            for s in symbols
                            if s["symbol_type"] == "constructor"
                            and s["name"] == target_type["name"]
                            and s["package_name"] == target_type.get("package_name")
                            and s["owner"] == _type_qual(target_type)
                        ]
                    else:
                        candidates = [
                            s
                            for s in symbols
                            if s["symbol_type"] == "method"
                            and s["name"] == call["method"]
                            and s["package_name"] == target_type.get("package_name")
                            and s["owner"] == _type_qual(target_type)
                        ]
                    exact_arity = [
                        c
                        for c in candidates
                        if _method_param_count(c["signature"]) == call["arg_count"]
                    ]
                    pool = exact_arity or candidates
                    if not pool:
                        unresolved += 1
                        continue
                    target = sorted(pool, key=lambda c: (c["start_line"], c["id"]))[0]
                    edges.add((method["id"], target["id"], CALLS))

            for type_symbol in type_symbols:
                is_test = (
                    "/src/test/" in path
                    or type_symbol["name"].endswith("Test")
                    or type_symbol["name"].endswith("Tests")
                )
                if not is_test:
                    continue
                referenced: set[str] = set()
                for simple, full in imports:
                    referenced.add(simple)
                referenced.update(
                    token.value
                    for token in _tokens(type_symbol["source"])
                    if token.kind == "ident" and token.value[0].isupper()
                )
                referenced.discard(type_symbol["name"])
                for name in sorted(referenced):
                    target = resolve(name, module, package_name, path)
                    if target is None:
                        continue
                    target_path = target.get("path", "")
                    target_name = target.get("name", "")
                    if "/src/test/" in target_path or target_name.endswith("Test") or target_name.endswith("Tests"):
                        continue
                    if target["symbol_type"] not in _TYPE_KINDS:
                        continue
                    edges.add((target["id"], type_symbol["id"], TESTED_BY))

        sorted_edges = sorted(edges)
        inserted = self.store.replace_edges(project_version_id, sorted_edges)

        edges_by_type = {
            relation: sum(1 for _, _, rel in sorted_edges if rel == relation)
            for relation in sorted(RELATIONS)
        }
        return {
            "version_id": project_version_id,
            "files_processed": len(by_path),
            "files_skipped": files_skipped,
            "symbols_loaded": len(symbols),
            "edges_inserted": inserted,
            "edges_by_type": edges_by_type,
            "unresolved_skipped": unresolved,
        }


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    from agent.db.store import PgStore
    from agent.versioning import _database_url, _redact_secrets

    parser = argparse.ArgumentParser(
        prog="python -m agent.graph",
        description="Build the deterministic code graph (code_edges) of a project version.",
    )
    parser.add_argument(
        "--repo-root",
        default="",
        help="Absolute path of the project (its resolved path is the external "
        "id); not needed when --project-id is used",
    )
    parser.add_argument(
        "--project-id",
        default="",
        help="Runtime project id (external_id, e.g. upload-123); takes "
        "precedence over the repo-root lookup",
    )
    parser.add_argument("--name", default="", help="Project name fallback for lookup")
    parser.add_argument("--version", type=int, default=None, help="Version number (default: latest)")
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides DATABASE_URL)",
    )
    args = parser.parse_args(argv)

    if not args.project_id and not args.repo_root:
        parser.error("either --project-id or --repo-root is required")

    dsn = _database_url(args.url)
    if not dsn:
        print(
            "No database URL configured: pass --url or set DATABASE_URL",
            file=sys.stderr,
        )
        return 1

    try:
        store = PgStore(dsn)
        project_row = None
        if args.project_id:
            project_row = store.find_project_by_external_id(args.project_id)
        if project_row is None and args.repo_root:
            project_row = store.find_project_by_external_id(
                str(Path(args.repo_root).resolve())
            )
        if project_row is None and args.name:
            project_row = store.find_project_by_name(args.name)
        if project_row is None:
            print(
                f"Project not found (project-id: {args.project_id or '-'}, "
                f"repo root: {args.repo_root or '-'})",
                file=sys.stderr,
            )
            return 1
        # File texts for import scanning are read from disk; without an
        # explicit --repo-root, read them from the stored index-time root
        # instead of the CLI's current working directory.
        root = (
            Path(args.repo_root).resolve()
            if args.repo_root
            else Path(project_row.get("repo_root") or ".")
        )
        if args.version is not None:
            version = store.find_version_by_number(project_row["id"], args.version)
        else:
            version = store.latest_version(project_row["id"])
        if version is None:
            print("No versions found for this project", file=sys.stderr)
            return 1
        builder = CodeGraphBuilder(CodebaseService(), store)
        result = builder.build(version["id"], str(root))
        result["project_id"] = project_row["id"]
        result["version_number"] = version["version_number"]
    except Exception as exc:
        print(f"Graph build failed: {_redact_secrets(str(exc), dsn)}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
