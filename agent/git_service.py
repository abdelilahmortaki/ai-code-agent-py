from __future__ import annotations
import subprocess
from pathlib import Path

from agent.config import ProjectConfig


class GitDiffService:
    def diff(self, project: ProjectConfig, touched_paths: list[str]) -> str:
        if not touched_paths:
            return ""
        try:
            root = Path(project.repo_root).resolve()
            cmd = ["git", "diff", "--no-ext-diff", "--unified=3", "--"] + touched_paths
            result = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"git diff failed (exit {result.returncode}): {result.stderr}"
                )
            return result.stdout
        except Exception as e:
            raise RuntimeError(f"Could not produce git diff: {e}") from e
