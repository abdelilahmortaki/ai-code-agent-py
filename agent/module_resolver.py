"""Deterministic Maven module resolution for source files.

The Maven module of a source file is the repo-relative directory of the
nearest ancestor (walking from the file's directory toward the repo root)
that contains a ``pom.xml``. A ``pom.xml`` at the repo root marks the root
project module. Projects without any ``pom.xml`` fall back to the root
project identifier (the root directory name).

No Maven reactor parsing or dependency resolution is performed: the rule
above is purely structural and deterministic.
"""

from __future__ import annotations

from pathlib import Path


def resolve_module(root: str | Path, file_path: str | Path) -> str:
    """Return the Maven module identifier for a source file under ``root``.

    ``file_path`` may be absolute or repo-relative; it does not need to
    exist. The result is a POSIX-style repo-relative directory path (or the
    root project identifier for the root module and non-Maven projects).
    """
    root_path = Path(root).resolve()
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    current = candidate.resolve().parent
    while current != root_path and root_path in current.parents:
        if (current / "pom.xml").is_file():
            return current.relative_to(root_path).as_posix()
        current = current.parent
    return root_path.name
