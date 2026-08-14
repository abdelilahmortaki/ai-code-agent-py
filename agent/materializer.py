from __future__ import annotations
import hashlib
from pathlib import Path
from agent.config import ProjectConfig
from agent.models import PatchFile, PatchPlan, PatchProposalPlan, ProposalFile
from agent.paths import resolve_within

EDIT_ANCHOR_NOT_FOUND = "EDIT_ANCHOR_NOT_FOUND"
EDIT_ANCHOR_AMBIGUOUS = "EDIT_ANCHOR_AMBIGUOUS"
EDIT_ANCHOR_OVERLAP = "EDIT_ANCHOR_OVERLAP"
PROTECTED_CONTENT_CHANGED = "PROTECTED_CONTENT_CHANGED"

class PatchMaterializer:
    def materialize(self, project: ProjectConfig, plan: PatchProposalPlan) -> PatchPlan:
        root = Path(project.repo_root).resolve()
        result: list[PatchFile] = []
        for proposal in plan.files:
            operation = proposal.operation.lower()
            target = resolve_within(root, proposal.path)
            raw = target.read_bytes() if target.exists() else b""
            current = raw.decode("utf-8")
            base_file_hash = None
            if operation in {"modify", "delete"} and target.exists():
                base_file_hash = hashlib.sha256(raw).hexdigest()
            content = proposal.content if operation == "create" else None
            if operation == "modify":
                content = self.materialize_content(current, proposal)
            result.append(PatchFile(path=proposal.path, operation=operation, content=content, base_file_hash=base_file_hash))
        return PatchPlan(storyId=plan.storyId, summary=plan.summary, files=result, tests=plan.tests, notes=plan.notes)

    @staticmethod
    def materialize_content(current: str, proposal: ProposalFile) -> str:
        if not proposal.edits:
            raise ValueError(f"modify requires exact edits: {proposal.path}")
        matches = []
        for edit in proposal.edits:
            if not edit.before:
                raise ValueError(EDIT_ANCHOR_NOT_FOUND)
            start = current.find(edit.before)
            if start < 0:
                raise ValueError(EDIT_ANCHOR_NOT_FOUND)
            if start != current.rfind(edit.before):
                raise ValueError(EDIT_ANCHOR_AMBIGUOUS)
            matches.append((start, start + len(edit.before), edit.after))
        matches.sort()
        if any(end > next_start for (_, end, _), (next_start, _, _) in zip(matches, matches[1:])):
            raise ValueError(EDIT_ANCHOR_OVERLAP)
        output = current
        for start, end, after in reversed(matches):
            output = output[:start] + after + output[end:]
        return output
