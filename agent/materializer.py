from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from agent.config import ProjectConfig
from agent.models import PatchProposal
from agent.paths import resolve_within

@dataclass(frozen=True)
class MaterializedPatch:
    path: str
    operation: str
    content: str | None

class PatchMaterializer:
    def materialize(self, project: ProjectConfig, files: list[PatchProposal]) -> list[MaterializedPatch]:
        root = Path(project.repo_root).resolve()
        result = []
        for proposal in files:
            target = resolve_within(root, proposal.path)
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            operation = proposal.operation.lower()
            content = proposal.content if operation == "create" else None
            if operation == "modify":
                content = self.materialize_content(current, proposal)
            result.append(MaterializedPatch(proposal.path, operation, content))
        return result

    @staticmethod
    def materialize_content(current: str, proposal: PatchProposal) -> str:
        if not proposal.edits:
            raise ValueError(f"modify requires exact edits: {proposal.path}")
        matches = []
        for edit in proposal.edits:
            if not edit.before:
                raise ValueError(f"exact edit anchor cannot be empty: {proposal.path}")
            start = current.find(edit.before)
            if start < 0 or start != current.rfind(edit.before):
                raise ValueError(f"exact edit anchor is not unique: {proposal.path}")
            matches.append((start, start + len(edit.before), edit.after))
        matches.sort()
        if any(end > next_start for (_, end, _), (next_start, _, _) in zip(matches, matches[1:])):
            raise ValueError(f"exact edits overlap: {proposal.path}")
        output = current
        for start, end, after in reversed(matches):
            output = output[:start] + after + output[end:]
        return output