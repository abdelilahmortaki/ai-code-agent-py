from __future__ import annotations
from pathlib import Path

from agent.config import ProjectConfig
from agent.guard import PatchGuardService
from agent.models import PatchFile


class PatchApplierService:
    def __init__(self, guard: PatchGuardService) -> None:
        self.guard = guard

    def apply(self, project: ProjectConfig, files: list[PatchFile]) -> list[str]:
        self.guard.validate(project, files)
        root = Path(project.repo_root).resolve()
        touched: list[str] = []

        for pf in files:
            target = (root / pf.path).resolve()
            if pf.operation.lower() == "delete":
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(pf.content or "", encoding="utf-8")
            touched.append(pf.path)

        return touched
