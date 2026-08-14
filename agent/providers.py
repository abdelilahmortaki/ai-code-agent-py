from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypedDict

from agent.config import ProjectConfig
from agent.models import PatchProposalPlan, TestFailureContext, UserStory


class NormalizedUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ProviderCapabilities:
    """Deterministic FinOps capability report for a generation provider/model.

    Static per provider/model — never derived from a single invocation's
    runtime state. `exact_token_counting` is False when the provider cannot
    count tokens exactly without a dependency we do not bundle; F3 must then
    apply its documented fallback rather than silently guessing a count.
    """

    exact_token_counting: bool = False
    usage: bool = False
    notes: tuple[str, ...] = ()


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

    def count_tokens(self, text: str) -> int | None:
        """Exact token count when the provider supports it; None = unsupported/unknown."""
        ...

    def normalize_usage(self, usage: object | None) -> NormalizedUsage | None:
        """Map a provider-native usage object into NormalizedUsage; None when no usage."""
        ...

    def capabilities(self) -> ProviderCapabilities:
        """Static capability report consumed by F3 guardrail/fallback logic."""
        ...


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model_identity(self) -> str: ...

    @property
    def dimensions(self) -> int | None: ...
