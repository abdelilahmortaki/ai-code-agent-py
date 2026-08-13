"""Contextual embedding payloads for Java symbols (F1.4).

Builds deterministic, dependency-aware text payloads per symbol that
describe where the symbol lives (module/file/class), what it is (symbol
type, signature) and which project types it depends on. Dependencies
come from simple parser/symbol evidence only -- imports, signature
types, field/member types and direct type references in the source --
so no F2 graph extraction is required.

Every build/extract entry point is defensive: malformed input never
raises; a source that cannot be scanned simply contributes no
dependencies and the payload is still produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.symbols import JavaSymbolParser, Symbol, SymbolParseError, _JavaScanner, _Token

# Curated set of java.lang (and core JDK) types that add no embedding
# signal as dependencies: references to them resolve through the JVM
# itself, so they never tell the payload which project types a symbol
# actually relates to. Curated (not exhaustive) by design.
JAVA_LANG_TYPES = frozenset(
    {
        "Appendable",
        "ArithmeticException",
        "ArrayIndexOutOfBoundsException",
        "ArrayStoreException",
        "AssertionError",
        "AutoCloseable",
        "Boolean",
        "Byte",
        "CharSequence",
        "Character",
        "Class",
        "ClassCastException",
        "ClassNotFoundException",
        "CloneNotSupportedException",
        "Cloneable",
        "Comparable",
        "ConcurrentModificationException",
        "Double",
        "Enum",
        "EnumConstantNotPresentException",
        "Error",
        "Exception",
        "ExceptionInInitializerError",
        "Float",
        "IllegalAccessException",
        "IllegalArgumentException",
        "IllegalStateException",
        "IncompatibleClassChangeError",
        "IndexOutOfBoundsException",
        "InstantiationException",
        "Integer",
        "InterruptedException",
        "Iterable",
        "LinkageError",
        "Long",
        "Math",
        "Module",
        "NegativeArraySizeException",
        "NoSuchElementException",
        "NoSuchFieldException",
        "NoSuchMethodException",
        "NullPointerException",
        "Number",
        "NumberFormatException",
        "Object",
        "OutOfMemoryError",
        "Package",
        "Process",
        "ReflectiveOperationException",
        "Runtime",
        "RuntimeException",
        "RuntimePermission",
        "Runnable",
        "SecurityException",
        "SecurityManager",
        "Short",
        "StackOverflowError",
        "StrictMath",
        "String",
        "StringBuffer",
        "StringBuilder",
        "System",
        "Thread",
        "ThreadDeath",
        "ThreadGroup",
        "ThreadLocal",
        "Throwable",
        "TypeNotPresentException",
        "UnsupportedClassVersionError",
        "UnsupportedOperationException",
        "VirtualMachineError",
        "Void",
    }
)


def _dotted_path(package_name: str, owner: str, name: str) -> str:
    """Join non-empty Java namespace segments with dots."""
    return ".".join(part for part in (package_name, owner, name) if part)


def _extract_imports(text: str) -> list[str]:
    """Simple type names from single-type import declarations.

    Wildcard (``import a.b.*;``) and static (``import static ...;``)
    imports are skipped; only the final simple segment of each import is
    kept. May raise SymbolParseError for structurally malformed text.
    """
    names: set[str] = set()
    scanner = _JavaScanner(text)
    while True:
        token = scanner.consume()
        if token is None:
            break
        if token.kind == "ident" and token.value == "import":
            parts: list[str] = []
            skip = False
            while True:
                nxt = scanner.consume()
                if nxt is None:
                    skip = True
                    break
                if nxt.value == ";":
                    break
                if nxt.kind != "ident":
                    if nxt.value == "*":
                        skip = True
                    continue
                if nxt.value == "static":
                    skip = True
                    continue
                parts.append(nxt.value)
            if not skip and parts:
                names.add(parts[-1])
    return sorted(names)


def _owner_type_map(text: str) -> dict[str, Symbol]:
    """Map dotted path -> type symbol for one compilation unit.

    The dotted path is the Java namespace package_name.owner.name (empty
    segments omitted), so ``com.shoppoc.domain.CustomerService.Inner``
    addresses the nested type and ``com.shoppoc.domain.CustomerService``
    its enclosing type. Raises SymbolParseError for malformed input;
    callers treat that as "no owner evidence available".
    """
    symbols = JavaSymbolParser().parse(text)
    owners: dict[str, Symbol] = {}
    for symbol in symbols:
        if symbol.symbol_type not in ("class", "interface", "enum", "record", "annotation"):
            continue
        owners[_dotted_path(symbol.package_name, symbol.owner, symbol.name)] = symbol
    return owners


def _field_types(source: str) -> list[str]:
    """Capitalized identifiers in field-type position of a type body.

    The type source is scanned with _JavaScanner at brace depth 1 /
    paren depth 0: declaration segments terminating in ';', ',' or '='
    with no '(' before the terminator are field candidates; the final
    (name) identifier of a candidate is excluded, all other capitalized
    identifiers in the segment are treated as types. May raise
    SymbolParseError for structurally malformed source.
    """
    scanner = _JavaScanner(source)
    brace_depth = 0
    paren_depth = 0
    segment: list[_Token] = []
    saw_paren = False
    found: list[str] = []
    while True:
        token = scanner.consume()
        if token is None:
            break
        value = token.value
        if value in ("{", "}"):
            brace_depth += 1 if value == "{" else -1
            if brace_depth < 0:
                brace_depth = 0
            segment = []
            saw_paren = False
            continue
        if value in ("(", ")"):
            paren_depth += 1 if value == "(" else -1
            if paren_depth < 0:
                paren_depth = 0
            if value == "(":
                saw_paren = True
            segment.append(token)
            continue
        if brace_depth != 1 or paren_depth != 0:
            continue
        if value in (";", ",", "="):
            if not saw_paren and segment:
                idents = [t for t in segment if t.kind == "ident"]
                if idents:
                    found.extend(t.value for t in idents[:-1] if t.value[:1].isupper())
            segment = []
            saw_paren = False
            continue
        segment.append(token)
    return sorted(set(found))


def _dependencies(
    symbol: Symbol, imports: list[str], owner_map: dict[str, Symbol]
) -> list[str]:
    """Union of dependency evidence for one symbol, filtered and sorted.

    Sources: (a) single-type imports, (b) capitalized identifiers in the
    signature, (c) field/member types of the owner type's source, (d)
    capitalized identifiers in the symbol's own source. Excludes
    non-uppercase-starting tokens (primitives/keywords), the curated
    java.lang set, the symbol's own name and every segment of its owner
    path.
    """
    deps: set[str] = set()
    excluded = {symbol.name} | set(part for part in symbol.owner.split(".") if part)

    deps.update(imports)

    try:
        scanner = _JavaScanner(symbol.signature)
        while True:
            token = scanner.consume()
            if token is None:
                break
            if token.kind == "ident" and token.value[:1].isupper():
                deps.add(token.value)
    except SymbolParseError:
        pass

    owner = owner_map.get(_dotted_path(symbol.module, symbol.owner or symbol.name, ""))
    if owner is not None:
        try:
            deps.update(_field_types(owner.source))
        except SymbolParseError:
            pass

    try:
        scanner = _JavaScanner(symbol.source)
        while True:
            token = scanner.consume()
            if token is None:
                break
            if token.kind == "ident" and token.value[:1].isupper():
                deps.add(token.value)
    except SymbolParseError:
        pass

    return sorted(
        dep
        for dep in deps
        if dep not in excluded
        and dep not in JAVA_LANG_TYPES
        and dep[:1].isupper()
    )


def _render_payload(symbol: Symbol, deps: list[str]) -> str:
    symbol_type = symbol.symbol_type.upper()
    if symbol.symbol_type == "method":
        kind = "Method"
    elif symbol.symbol_type == "constructor":
        kind = "Constructor"
    else:
        kind = "Symbol"
    lines = [
        f"Module: {symbol.module}",
        f"File: {Path(symbol.file).name}",
        f"Package: {symbol.package_name}",
        f"Class: {symbol.owner or symbol.name}",
        f"Symbol Type: {symbol_type}",
        f"{kind}: {symbol.name}",
        f"Qualified Name: {symbol.qualified_name}",
        f"Signature: {symbol.signature}",
        "",
        "Useful dependencies:",
    ]
    lines.extend(deps)
    lines.append("")
    lines.append(symbol.source)
    return "\n".join(lines)


def extract_dependencies(symbol: Symbol, file_text: str) -> list[str]:
    """Dependency type names of a symbol, from evidence in the file text.

    Deterministic (sorted, deduplicated) and never raises: malformed
    file text or a scanner failure contributes nothing.
    """
    try:
        imports = _extract_imports(file_text)
    except SymbolParseError:
        imports = []
    try:
        owner_map = _owner_type_map(file_text)
    except SymbolParseError:
        owner_map = {}
    return _dependencies(symbol, imports, owner_map)


def build_contextual_payload(symbol: Symbol, file_text: str) -> str:
    """One contextual embedding payload for a symbol.

    Never raises: a symbol whose evidence cannot be scanned still gets a
    payload (header + source, empty dependency list).
    """
    return _render_payload(symbol, extract_dependencies(symbol, file_text))


def build_file_payloads(symbols: list[Symbol], file_text: str) -> list[str]:
    """Payloads for every symbol; the file is parsed and imports are
    extracted once, and the owner map is built once."""
    try:
        imports = _extract_imports(file_text)
    except SymbolParseError:
        imports = []
    try:
        owner_map = _owner_type_map(file_text)
    except SymbolParseError:
        owner_map = {}
    return [
        _render_payload(symbol, _dependencies(symbol, imports, owner_map))
        for symbol in symbols
    ]


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.payloads",
        description="Print contextual embedding payloads for Java symbols.",
    )
    parser.add_argument("--file", required=True, help="Path of the Java source file")
    parser.add_argument(
        "--symbol", default="", help="Only emit payloads for this symbol name"
    )
    args = parser.parse_args(argv)

    try:
        content = Path(args.file).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"payloads: cannot read {args.file}: {exc}", file=sys.stderr)
        return 1

    try:
        symbols = JavaSymbolParser(file=args.file).parse(content)
    except SymbolParseError as exc:
        print(f"payloads: cannot parse {args.file}: {exc}", file=sys.stderr)
        return 1

    if args.symbol:
        matched = [symbol for symbol in symbols if symbol.name == args.symbol]
        if not matched:
            print(
                f"payloads: no symbol named '{args.symbol}' in {args.file}",
                file=sys.stderr,
            )
            return 1
        symbols = matched

    payloads = build_file_payloads(symbols, content)
    if payloads:
        marker = "=" * 72
        print(f"\n{marker}\n".join(payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
