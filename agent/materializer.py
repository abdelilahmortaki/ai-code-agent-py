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
            result.append(PatchFile(
                path=proposal.path,
                operation=operation,
                content=content,
                base_file_hash=base_file_hash,
                reason=proposal.reason,
            ))
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
            replacement = edit.after
            if start < 0:
                start, end, replacement = PatchMaterializer._find_eol_equivalent(
                    current, edit.before, edit.after
                )
            else:
                if start != current.rfind(edit.before):
                    raise ValueError(EDIT_ANCHOR_AMBIGUOUS)
                end = start + len(edit.before)
            matches.append((start, end, replacement))
        matches.sort()
        if any(end > next_start for (_, end, _), (next_start, _, _) in zip(matches, matches[1:])):
            raise ValueError(EDIT_ANCHOR_OVERLAP)
        output = current
        for start, end, after in reversed(matches):
            output = output[:start] + after + output[end:]
        return output

    @staticmethod
    def _find_eol_equivalent(current: str, before: str, after: str) -> tuple[int, int, str]:
        """Resolve an anchor differing from the source only by CRLF versus LF."""
        normalized_current, boundaries = PatchMaterializer._normalize_newlines(current)
        normalized_before = before.replace("\r\n", "\n")
        positions = []
        offset = 0
        while True:
            found = normalized_current.find(normalized_before, offset)
            if found < 0:
                break
            positions.append(found)
            offset = found + 1
        if not positions:
            raise ValueError(EDIT_ANCHOR_NOT_FOUND)
        if len(positions) > 1:
            raise ValueError(EDIT_ANCHOR_AMBIGUOUS)

        start = boundaries[positions[0]]
        end = boundaries[positions[0] + len(normalized_before)]
        source_span = current[start:end]
        source_eol = "\r\n" if "\r\n" in source_span else "\n"
        replacement = after.replace("\r\n", "\n").replace("\n", source_eol)
        return start, end, replacement

    @staticmethod
    def _normalize_newlines(text: str) -> tuple[str, list[int]]:
        normalized: list[str] = []
        boundaries = [0]
        index = 0
        while index < len(text):
            if text.startswith("\r\n", index):
                normalized.append("\n")
                index += 2
            else:
                normalized.append(text[index])
                index += 1
            boundaries.append(index)
        return "".join(normalized), boundaries
