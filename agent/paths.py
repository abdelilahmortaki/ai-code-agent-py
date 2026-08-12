from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_within(root: str | os.PathLike[str], relative_path: str | os.PathLike[str]) -> Path:
    """Resolve a relative path and ensure it remains inside ``root``."""
    raw_path = os.fspath(relative_path)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Path must be a non-empty relative path")

    # Reject syntax for either platform even when running on the other one.
    windows_path = PureWindowsPath(raw_path)
    if ".." in PurePosixPath(raw_path).parts or ".." in windows_path.parts:
        raise ValueError(f"Path traversal is not allowed: {relative_path}")
    if (
        Path(raw_path).is_absolute()
        or PurePosixPath(raw_path).is_absolute()
        or windows_path.root
        or windows_path.drive
    ):
        raise ValueError(f"Path must be relative: {relative_path}")

    root_path = Path(root).resolve()
    target = (root_path / raw_path).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Path escapes root: {relative_path}") from exc
    return target
