from __future__ import annotations
import hashlib
from pathlib import Path
from agent.config import ProjectConfig
from agent.guard import PatchGuardService
from agent.materializer import PatchMaterializer
from agent.models import PatchPlan
from agent.paths import resolve_within

STALE_PATCH = "STALE_PATCH"

class StalePatchError(ValueError):
    pass

class PatchApplierService:
    def __init__(self, guard: PatchGuardService, materializer: PatchMaterializer | None = None) -> None:
        self.guard = guard
        self.materializer = materializer or PatchMaterializer()

    def apply(self, project: ProjectConfig, plan: PatchPlan) -> list[str]:
        self.guard.validate(project, plan.files)
        root = Path(project.repo_root).resolve()
        for patch in plan.files:
            if patch.operation.lower() in {"modify", "delete"} and patch.base_file_hash:
                target = resolve_within(root, patch.path)
                if hashlib.sha256(target.read_bytes()).hexdigest() != patch.base_file_hash:
                    raise StalePatchError(STALE_PATCH)
        touched = []
        for patch in plan.files:
            target = resolve_within(root, patch.path)
            if patch.operation.lower() == "delete":
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(patch.content or "", encoding="utf-8")
            touched.append(patch.path)
        return touched
