from __future__ import annotations

from typing import Callable, Protocol, TypedDict

from agent.config import ProjectConfig
from agent.models import PatchProposalPlan, TestFailureContext, UserStory


class NormalizedUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GenerationProvider(Protocol):
    def generate_patch_plan(
        self,
        project: ProjectConfig,
        story: UserStory,
        relevant_files: list[tuple[str, str]],
        static_analysis: str,
        log_callback: Callable[[str], None],
        validation_feedback: str = "",
    ) -> PatchProposalPlan: ...

    def generate_fix_plan(
        self,
        ctx: TestFailureContext,
        log_callback: Callable[[str], None],
    ) -> PatchProposalPlan: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model_identity(self) -> str: ...


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model_identity(self) -> str: ...

    @property
    def dimensions(self) -> int | None: ...
