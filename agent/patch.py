from __future__ import annotations
from pathlib import Path
from agent.config import ProjectConfig
from agent.guard import PatchGuardService
from agent.materializer import PatchMaterializer
from agent.models import PatchProposal
from agent.paths import resolve_within

class PatchApplierService:
    def __init__(self, guard: PatchGuardService, materializer: PatchMaterializer | None = None) -> None:
        self.guard = guard
        self.materializer = materializer or PatchMaterializer()

    def apply(self, project: ProjectConfig, files: list[PatchProposal]) -> list[str]:
        self.guard.validate(project, files)
        root = Path(project.repo_root).resolve()
        touched = []
        for patch in self.materializer.materialize(project, files):
            target = resolve_within(root, patch.path)
            if patch.operation == "delete":
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(patch.content or "", encoding="utf-8")
            touched.append(patch.path)
        return touched