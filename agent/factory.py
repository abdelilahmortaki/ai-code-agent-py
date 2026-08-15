from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.config import BedrockConfig, ProjectConfig, Settings
from agent.guard import PromptGuardService
from agent.models import PatchProposalPlan, TestFailureContext, UserStory
from agent.providers import EmbeddingProvider, GenerationProvider, NormalizedUsage, ProviderCapabilities


def _noop(msg: str) -> None:
    pass


@dataclass(frozen=True)
class ProviderRuntime:
    generation: GenerationProvider
    embedding: EmbeddingProvider


class BedrockGenerationProvider:
    """Provider-neutral adapter that delegates all generation to BedrockService."""

    provider_name = "bedrock"

    def __init__(self, service: object, config: BedrockConfig) -> None:
        self._service = service
        self.model_identity = config.model_id

    def generate_patch_plan(
        self,
        project: ProjectConfig,
        story: UserStory,
        relevant_files: list[tuple[str, str]],
        static_analysis: str,
        log_callback: Callable[[str], None] = _noop,
        validation_feedback: str = "",
    ) -> PatchProposalPlan:
        return self._service.generate_patch_plan(
            project,
            story,
            relevant_files,
            static_analysis,
            log_callback,
            validation_feedback=validation_feedback,
        )

    def generate_fix_plan(
        self,
        ctx: TestFailureContext,
        log_callback: Callable[[str], None] = _noop,
    ) -> PatchProposalPlan:
        return self._service.generate_fix_plan(ctx, log_callback)

    @property
    def last_usage(self) -> NormalizedUsage | None:
        return self.normalize_usage(getattr(self._service, "last_usage", None))

    @staticmethod
    def normalize_usage(usage: object | None) -> NormalizedUsage | None:
        if usage is None:
            return None

        def _pick(*names: str) -> object | None:
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if value is not None:
                    return value
            return None

        normalized: NormalizedUsage = {}
        input_tokens = _pick("inputTokens", "input_tokens")
        output_tokens = _pick("outputTokens", "output_tokens")
        total_tokens = _pick("totalTokens", "total_tokens")
        if input_tokens is not None:
            normalized["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            normalized["output_tokens"] = int(output_tokens)
        if total_tokens is not None:
            normalized["total_tokens"] = int(total_tokens)
        return normalized

    @staticmethod
    def count_tokens(text: str) -> None:
        return None

    @staticmethod
    def capabilities() -> ProviderCapabilities:
        return ProviderCapabilities(
            exact_token_counting=False,
            usage=True,
            notes=(
                "Bedrock ConverseStream reports usage (inputTokens/outputTokens/totalTokens), "
                "captured into the service's last_usage after each generation. Exact token "
                "counting is unsupported without a bundled tokenizer for the chosen model.",
            ),
        )


def create_provider_runtime(settings: Settings, prompt_guard: PromptGuardService) -> ProviderRuntime:
    if settings.ai_provider == "bedrock":
        from agent.bedrock_service import BedrockService
        from agent.index import EmbeddingsService

        embedding = EmbeddingsService(settings.bedrock)
        generation = BedrockGenerationProvider(
            BedrockService(settings.bedrock, prompt_guard),
            settings.bedrock,
        )
        return ProviderRuntime(generation=generation, embedding=embedding)

    if settings.ai_provider == "azure":
        from agent.azure_embeddings import AzureOpenAIEmbeddingProvider
        from agent.azure_generation import AzureOpenAIGenerationProvider

        azure = settings.azure
        api_key = azure.api_key.get_secret_value()
        return ProviderRuntime(
            generation=AzureOpenAIGenerationProvider(
                azure.endpoint,
                api_key,
                azure.deployment,
                prompt_guard,
            ),
            embedding=AzureOpenAIEmbeddingProvider(
                azure.endpoint,
                api_key,
                azure.embedding_deployment,
                dimensions=azure.embedding_dimensions,
            ),
        )

    raise ValueError("AI_PROVIDER must be one of: azure, bedrock")


def create_database_store(settings: Settings):
    """Build a PgStore when a database URL is configured, else return None.

    Lazy-imports psycopg so the rest of the app runs without it installed.
    Fails fast (without echoing the DSN) when the database is unreachable.
    """
    url = settings.database.url.get_secret_value()
    if not url:
        return None

    from agent.db.store import PgStore, PgStoreError

    store = PgStore(url)
    try:
        store.ping()
    except PgStoreError as exc:
        raise RuntimeError("database configured but unreachable") from exc
    return store
