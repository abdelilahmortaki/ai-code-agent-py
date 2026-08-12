from __future__ import annotations

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from agent.providers import NormalizedUsage


_BATCH_SIZE = 32
_MAX_INPUT_CHARS = 32_000


class AzureOpenAIEmbeddingProvider:
    provider_name = "azure"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        timeout: float = 180.0,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")
        if not api_key.strip():
            raise ValueError("AZURE_OPENAI_API_KEY is required")
        if not deployment.strip():
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required")

        base_url = endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url += "/openai/v1"
        self.model_identity = deployment
        self.dimensions: int | None = None
        self.last_usage: NormalizedUsage | None = None
        self._client = OpenAI(
            api_key=api_key,
            base_url=f"{base_url}/",
            timeout=timeout,
            max_retries=0,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Azure embeddings require non-empty text inputs")
            if len(text) > _MAX_INPUT_CHARS:
                raise ValueError("Azure embedding input exceeds the supported size")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start:start + _BATCH_SIZE]))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self.model_identity,
                input=texts,
            )
        except APITimeoutError as exc:
            raise RuntimeError("Azure OpenAI embedding request timed out") from exc
        except APIConnectionError as exc:
            raise RuntimeError("Azure OpenAI embedding connection failed") from exc
        except RateLimitError as exc:
            raise RuntimeError("Azure OpenAI embedding rate limit reached") from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                message = "Azure OpenAI embedding authentication or authorization failed"
            elif status == 429:
                message = "Azure OpenAI embedding rate limit reached"
            elif isinstance(status, int) and status >= 500:
                message = "Azure OpenAI embedding service error"
            else:
                message = "Azure OpenAI embedding request failed"
            raise RuntimeError(message) from exc
        except Exception as exc:
            raise RuntimeError("Azure OpenAI embedding request failed") from exc

        self.last_usage = self._normalize_usage(getattr(response, "usage", None))
        items = sorted(getattr(response, "data", []), key=lambda item: getattr(item, "index", 0))
        if len(items) != len(texts):
            raise RuntimeError("Azure OpenAI returned an incomplete embedding response")

        vectors: list[list[float]] = []
        for item in items:
            vector = [float(value) for value in getattr(item, "embedding", [])]
            if not vector:
                raise RuntimeError("Azure OpenAI returned an empty embedding vector")
            if self.dimensions is None:
                self.dimensions = len(vector)
            if len(vector) != self.dimensions:
                raise RuntimeError("REBUILD_REQUIRED: embedding dimensions changed")
            vectors.append(vector)
        return vectors

    @staticmethod
    def _normalize_usage(usage: object | None) -> NormalizedUsage | None:
        if usage is None:
            return None
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        values = {
            "input_tokens": input_tokens,
            "total_tokens": total_tokens,
        }
        return {key: int(value) for key, value in values.items() if value is not None}
