from __future__ import annotations
from pathlib import Path

from agent.config import SUPPORTED_EXTENSIONS, ProjectConfig
from agent.paths import resolve_within

class CodebaseService:
    def __init__(self, max_file_chars: int = 12000):
        self.max_file_chars = max_file_chars

    def scan(self, project: ProjectConfig) -> list[tuple[str, str]]:
        """Return (relative_path, content) for every supported file in the repo."""
        root = Path(project.repo_root).resolve()
        if not root.exists():
            raise ValueError(f"Repo root not found: {root}")

        excluded = set(project.normalized_excluded_directories())
        docs: list[tuple[str, str]] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            if self._is_excluded(rel, excluded):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if content and not content.isspace():
                    docs.append((rel, self._limit(content, self.max_file_chars)))
            except Exception:
                pass

        return docs

    def read_file(self, project: ProjectConfig, relative_path: str) -> str | None:
        """Read the full content of a single file (capped at max_file_chars)."""
        root = Path(project.repo_root).resolve()
        try:
            target = resolve_within(root, relative_path)
        except ValueError:
            return None
        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
            return self._limit(content, self.max_file_chars)
        except Exception:
            return None

    # ------------------------------------------------------------------ helpers

    def _is_excluded(self, rel: str, excluded: set[str]) -> bool:
        for part in rel.split("/")[:-1]:
            if part in excluded:
                return True
        return False

    def _limit(self, text: str, max_chars: int) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + "\n...[TRUNCATED]"
