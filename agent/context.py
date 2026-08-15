"""Explainable Context Builder (F2.5).

Turns a hybrid retrieval ranking into an ordered, explainable generation
context. Every context item records the selected symbol, why it was
selected (reason), which retrieval source produced it, a priority, an
estimated token size and its included/excluded state. Optional simulated
budget support (``estimated_token_budget``) excludes low-value items
first while preserving main/high-confidence evidence and useful
dependencies/tests.

Token estimates are F2 heuristics only: ``estimated_tokens =
ceil(chars / 4)``. Exact token accounting is F3 scope.

CRITICAL contract: the context selects SYMBOLS, but generation must
receive COMPLETE authoritative files. The ContextBuilder therefore
derives the unique selected file paths and loads the FULL current file
content for each (respecting ``max_context_files``) into
``ContextBundle.file_contents``; method snippets are never passed as
complete MODIFY source.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from agent.codebase import CodebaseService

if TYPE_CHECKING:
    from agent.config import ProjectConfig

_PRIORITY_LEVEL = {"high": 0, "medium": 1, "low": 2}

# Order of item sources; used only as a stable sort key.
_SOURCE_RANK = {
    "main_candidate": 0,
    "ticket_evidence": 1,
    "associated_test": 2,
    "dependency": 3,
    "related_class": 4,
    "signature": 5,
}

_GRAPH_TYPE_PRIORITY = {
    "TESTED_BY": "associated_test",
    "INHERITS": "related_class",
    "IMPLEMENTS": "related_class",
    "IMPORTS": "dependency",
    "CALLS": "dependency",
}


def estimate_tokens(chars: int) -> int:
    """F2 heuristic token estimate: ceil(chars / 4)."""
    return max(0, math.ceil(int(chars) / 4))


class ContextItem(BaseModel):
    symbol_id: str | None = None
    qualified_name: str = ""
    symbol_type: str = ""
    file_path: str = ""
    signature: str = ""
    reasons: list[str] = []
    retrieval_sources: list[str] = []
    priority: str = "low"
    estimated_tokens: int = 0
    included: bool = True
    final_score: float = 0.0
    source: str = "ticket_evidence"
    graph_relations: list[dict] = []


class ContextBundle(BaseModel):
    retrieval_mode: str = "legacy"
    project_version_id: str | None = None
    version_number: int | None = None
    graph_depth: int = 0
    items: list[ContextItem] = []
    selected_files: list[str] = []
    file_tokens: list[dict] = []
    estimated_token_budget: int | None = None
    estimated_tokens: int = 0
    included_count: int = 0
    excluded_count: int = 0
    # Full authoritative file contents for generation. Excluded from API
    # serialization: the UI only needs items/selected files, and the runtime
    # passes these contents to the generation provider instead.
    file_contents: dict[str, str] = Field(default_factory=dict, exclude=True)

    @field_validator("project_version_id", mode="before")
    @classmethod
    def _coerce_version_id(cls, v: object) -> object:
        if v is None:
            return None
        return str(v)


def build_legacy_bundle(
    relevant_files: list[tuple[str, str]],
    max_context_files: int = 6,
    project_version_id: str | None = None,
    version_number: int | None = None,
) -> ContextBundle:
    """Explainable bundle for the legacy SemanticIndexService path.

    Each legacy file becomes one context item with a single retrieval
    reason ("legacy semantic file match"); full file content is still
    loaded so generation receives authoritative files.
    """
    items: list[ContextItem] = []
    file_contents: dict[str, str] = {}
    file_tokens: list[dict] = []
    for path, content in relevant_files[:max_context_files]:
        file_contents[path] = content
        file_tokens.append(
            {
                "path": path,
                "chars": len(content),
                "estimated_tokens": estimate_tokens(len(content)),
            }
        )
        items.append(
            ContextItem(
                symbol_id=None,
                qualified_name=path,
                symbol_type="file",
                file_path=path,
                signature="",
                reasons=["legacy semantic file match"],
                retrieval_sources=["legacy"],
                priority="medium",
                estimated_tokens=estimate_tokens(len(content)),
                included=True,
                final_score=0.0,
                source="ticket_evidence",
                graph_relations=[],
            )
        )
    return ContextBundle(
        retrieval_mode="legacy",
        project_version_id=project_version_id,
        version_number=version_number,
        graph_depth=0,
        items=items,
        selected_files=list(file_contents.keys()),
        file_tokens=file_tokens,
        estimated_token_budget=None,
        estimated_tokens=sum(item.estimated_tokens for item in items),
        included_count=len(items),
        excluded_count=0,
        file_contents=file_contents,
    )


class ContextBuilder:
    """Build an explainable, budget-aware generation context (F2.5)."""

    def __init__(self, codebase: CodebaseService) -> None:
        self.codebase = codebase

    def build(
        self,
        project: ProjectConfig,
        query: str,
        retrieval: dict,
        estimated_token_budget: int | None = None,
        max_context_files: int = 6,
    ) -> ContextBundle:
        """Select context items from a hybrid retrieval result.

        Items are categorized (main candidate / ticket evidence / related
        class / dependency / associated test), prioritized, optionally
        reduced to a simulated token budget, and their unique file paths
        are resolved to full authoritative file contents.
        """
        items: list[ContextItem] = []
        for index, raw in enumerate(retrieval.get("items", [])):
            source, priority = self._categorize(raw, index)
            items.append(self._to_item(raw, source, priority))

        self._apply_budget(items, estimated_token_budget)

        included = [item for item in items if item.included]
        selected_files, file_tokens, file_contents = self._load_files(
            project, included, max_context_files
        )

        included_count = sum(1 for item in items if item.included)
        return ContextBundle(
            retrieval_mode="hybrid",
            project_version_id=retrieval.get("project_version_id"),
            version_number=retrieval.get("version_number"),
            graph_depth=int(retrieval.get("graph_depth", 0)),
            items=items,
            selected_files=selected_files,
            file_tokens=file_tokens,
            estimated_token_budget=estimated_token_budget,
            estimated_tokens=sum(item.estimated_tokens for item in included),
            included_count=included_count,
            excluded_count=len(items) - included_count,
            file_contents=file_contents,
        )

    # ------------------------------------------------------------------ private

    def _categorize(self, raw: dict, index: int) -> tuple[str, str]:
        """Assign a (source, priority) label to a retrieved item.

        The first ranked item is the main candidate. Test/dependency/related
        class labels come from graph relations; lexical/vector matches on the
        ticket are ticket evidence. Deterministic and precedence based.
        """
        graph_relations = [
            relation.get("relation_type", "")
            for relation in (raw.get("graph") or {}).get("relations", [])
        ]
        reasons = " ".join(raw.get("reasons", []))
        sources = set(raw.get("retrieval_sources", []))

        if index == 0:
            return "main_candidate", "high"

        # Graph relation precedence: a test is more informative than a plain
        # import, so TESTED_BY beats IMPORTS/CALLS for the same symbol.
        for relation_type in ("TESTED_BY", "INHERITS", "IMPLEMENTS", "IMPORTS", "CALLS"):
            if relation_type not in graph_relations:
                continue
            source = _GRAPH_TYPE_PRIORITY[relation_type]
            return source, "medium"

        if sources & {"lexical", "vector"}:
            is_exact = (
                "exact identifier match" in reasons
                or "qualified name match" in reasons
                or "case-insensitive" in reasons
            )
            return "ticket_evidence", ("high" if is_exact else "medium")

        if sources == {"graph"}:
            return "dependency", "medium"

        return "ticket_evidence", "low"

    def _to_item(self, raw: dict, source: str, priority: str) -> ContextItem:
        symbol = raw.get("symbol") or {}
        graph = raw.get("graph")
        relations = (graph or {}).get("relations", []) or []
        symbol_id = raw.get("symbol_id") or symbol.get("id")
        return ContextItem(
            symbol_id=str(symbol_id) if symbol_id else None,
            qualified_name=raw.get("qualified_name") or symbol.get("qualified_name", ""),
            symbol_type=raw.get("symbol_type") or symbol.get("symbol_type", ""),
            file_path=raw.get("file_path") or symbol.get("path", ""),
            signature=raw.get("signature") or symbol.get("signature", ""),
            reasons=list(raw.get("reasons", [])),
            retrieval_sources=list(raw.get("retrieval_sources", [])),
            priority=priority,
            estimated_tokens=estimate_tokens(
                len(raw.get("source") or symbol.get("source") or "")
            ),
            included=True,
            final_score=float(raw.get("final_score", 0.0)),
            source=source,
            graph_relations=[
                {
                    "relation_type": relation.get("relation_type", ""),
                    "direction": relation.get("direction", ""),
                }
                for relation in relations
            ],
        )

    def _apply_budget(
        self, items: list[ContextItem], budget: int | None
    ) -> None:
        """Exclude low-value items when the simulated budget is exceeded.

        Items are considered in deterministic priority order (priority level,
        then final score DESC, then qualified name). When an item does not
        fit the remaining budget it is excluded and the next item is tried,
        so main/high-confidence evidence and useful dependencies survive.
        """
        if budget is None:
            for item in items:
                item.included = True
            return
        ordered = sorted(
            items,
            key=lambda it: (
                _PRIORITY_LEVEL.get(it.priority, 2),
                -it.final_score,
                it.qualified_name,
            ),
        )
        used = 0
        for item in ordered:
            if item.estimated_tokens <= budget - used:
                item.included = True
                used += item.estimated_tokens
            else:
                item.included = False

    def _load_files(
        self,
        project: ProjectConfig,
        items: list[ContextItem],
        max_context_files: int,
    ) -> tuple[list[str], list[dict], dict[str, str]]:
        """Load FULL current file content for the unique selected paths.

        Files are picked in item priority order so higher-value evidence
        wins the ``max_context_files`` cap. The loaded content is exactly
        what the generation provider receives.
        """
        ordered = sorted(
            items,
            key=lambda it: (
                _PRIORITY_LEVEL.get(it.priority, 2),
                -it.final_score,
                it.qualified_name,
            ),
        )
        selected: list[str] = []
        for item in ordered:
            if item.file_path and item.file_path not in selected:
                selected.append(item.file_path)
            if len(selected) >= max_context_files:
                break

        file_contents: dict[str, str] = {}
        file_tokens: list[dict] = []
        for path in selected:
            content = self.codebase.read_file(project, path) or ""
            file_contents[path] = content
            file_tokens.append(
                {
                    "path": path,
                    "chars": len(content),
                    "estimated_tokens": estimate_tokens(len(content)),
                }
            )
        return selected, file_tokens, file_contents
