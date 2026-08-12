from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.config import BedrockConfig, ProjectConfig, Settings
from agent.guard import PromptGuardService
from agent.models import PatchPlan, TestFailureContext, UserStory
from agent.providers import EmbeddingProvider, GenerationProvider


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
    ) -> PatchPlan:
        return self._service.generate_patch_plan(
            project, story, relevant_files, static_analysis, log_callback
        )

    def generate_fix_plan(
        self,
        ctx: TestFailureContext,
        log_callback: Callable[[str], None] = _noop,
    ) -> PatchPlan:
        return self._service.generate_fix_plan(ctx, log_callback)


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
            ),
        )

    raise ValueError("AI_PROVIDER must be one of: azure, bedrock")
