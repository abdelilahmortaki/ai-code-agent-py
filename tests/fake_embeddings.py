"""Deterministic fake embedding provider for F2 acceptance fixtures.

No external credentials are required. Vectors are bag-of-words hashes of
the input tokens normalized to unit length, so cosine similarity
reflects shared tokens: a query embedding a technical identifier is
similar to the symbol payload that mentions it. Deterministic per text.
"""

from __future__ import annotations

import hashlib

_DEFAULT_DIMENSIONS = 16


def hash_embedding(text: str, dimensions: int = _DEFAULT_DIMENSIONS) -> list[float]:
    """Unit vector from the hashed token bag of ``text``."""
    vector = [0.0] * dimensions
    for token in text.split():
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vector[digest % dimensions] += 1.0
    norm = sum(x * x for x in vector) ** 0.5
    if norm:
        vector = [x / norm for x in vector]
    return vector


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_identity = "fake-model-v1"
    dimensions = _DEFAULT_DIMENSIONS

    def __init__(
        self,
        provider_name: str = "fake",
        model_identity: str = "fake-model-v1",
        dimensions: int = _DEFAULT_DIMENSIONS,
    ) -> None:
        self.provider_name = provider_name
        self.model_identity = model_identity
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [hash_embedding(text, self.dimensions) for text in texts]
