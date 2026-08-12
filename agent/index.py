from __future__ import annotations
import json
import math
from pathlib import Path

import boto3

from agent.config import BedrockConfig, ProjectConfig
from agent.codebase import CodebaseService
from agent.models import IndexedChunk, RepoChunk
from agent.providers import EmbeddingProvider

# Amazon Titan Embed Text v2 truncation limit (safe character ceiling)
_TITAN_MAX_CHARS = 8000


class EmbeddingsService:
    """
    Embeds text using Amazon Titan Embed Text v2 via AWS Bedrock.
    boto3 picks up credentials from the environment (AWS_ACCESS_KEY_ID /
    AWS_SECRET_ACCESS_KEY / AWS_REGION loaded by python-dotenv).
    """

    def __init__(self, bedrock_cfg: BedrockConfig) -> None:
        self.model_id = bedrock_cfg.embeddings_model_id
        self.provider_name = "bedrock"
        self.model_identity = self.model_id
        self.dimensions = 1024
        self._client = boto3.client("bedrock-runtime", region_name=bedrock_cfg.region)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({
            "inputText": text[:_TITAN_MAX_CHARS],
            "dimensions": 1024,
            "normalize": True,
        })
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]


class SemanticIndexService:
    """
    Builds a chunk-level embedding index for fast retrieval,
    but search() returns *unique file paths* so the orchestrator
    can pass full file contents to OpenAI instead of partial chunks.
    """

    def __init__(self, codebase: CodebaseService, embeddings: EmbeddingProvider) -> None:
        self.codebase = codebase
        self.embeddings = embeddings

    # ------------------------------------------------------------------ public

    def rebuild(self, project: ProjectConfig) -> list[str]:
        """Scan repo, embed all chunks, persist the index. Returns chunk ids."""
        docs = self.codebase.scan(project)
        chunks = self._chunk_docs(docs, max_chars=1800)
        texts = [f"FILE: {c.path}\nCHUNK: {c.chunk_index}\n{c.text}" for c in chunks]
        vectors = self.embeddings.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding provider returned an incomplete result")
        dimensions = len(vectors[0]) if vectors else self.embeddings.dimensions
        if dimensions is None:
            raise ValueError("Embedding provider did not report vector dimensions")
        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("Embedding provider returned inconsistent vector dimensions")

        indexed = [
            IndexedChunk(
                path=c.path,
                chunk_index=c.chunk_index,
                text=c.text,
                embedding=vectors[i] if i < len(vectors) else [],
            )
            for i, c in enumerate(chunks)
        ]
        self._persist(project, indexed, dimensions)
        return [f"{ic.path}#{ic.chunk_index}" for ic in indexed]

    def search(self, project: ProjectConfig, query: str, top_k: int) -> list[str]:
        """
        Returns the unique file paths of the top-K most relevant chunks.
        Callers should then read the *full file content* for each path.
        """
        metadata, indexed = self._load(project)
        if not indexed:
            return []

        q_vecs = self.embeddings.embed_texts([query])
        if not q_vecs:
            return []
        q = q_vecs[0]
        if len(q) != metadata["dimensions"]:
            raise ValueError("REBUILD_REQUIRED: embedding dimensions differ")

        ranked = sorted(indexed, key=lambda c: self._cosine(q, c.embedding), reverse=True)

        seen: list[str] = []
        for chunk in ranked:
            if chunk.path not in seen:
                seen.append(chunk.path)
            if len(seen) >= top_k:
                break
        return seen

    def exists(self, project: ProjectConfig) -> bool:
        return self._index_path(project).exists()

    # ------------------------------------------------------------------ private

    def _chunk_docs(self, docs: list[tuple[str, str]], max_chars: int) -> list[RepoChunk]:
        chunks: list[RepoChunk] = []
        for path, content in docs:
            if not content or content.isspace():
                continue
            idx = 0
            chunk_index = 0
            while idx < len(content):
                end = min(len(content), idx + max_chars)
                if end < len(content):
                    nl = content.rfind("\n", idx + 200, end)
                    if nl > idx + 200:
                        end = nl
                part = content[idx:end].strip()
                if part:
                    chunks.append(RepoChunk(path=path, chunk_index=chunk_index, text=part))
                idx = end
                chunk_index += 1
        return chunks

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def _persist(self, project: ProjectConfig, indexed: list[IndexedChunk], dimensions: int) -> None:
        path = self._index_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "embedding": {
                        "provider": self.embeddings.provider_name,
                        "deployment_or_model": self.embeddings.model_identity,
                        "dimensions": dimensions,
                    },
                    "chunks": [ic.model_dump() for ic in indexed],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load(self, project: ProjectConfig) -> tuple[dict, list[IndexedChunk]]:
        path = self._index_path(project)
        if not path.exists():
            return {"dimensions": 0}, []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("REBUILD_REQUIRED: index metadata missing") from exc
        if not isinstance(data, dict) or not isinstance(data.get("embedding"), dict):
            raise ValueError("REBUILD_REQUIRED: index metadata missing")
        metadata = data["embedding"]
        if (
            metadata.get("provider") != self.embeddings.provider_name
            or metadata.get("deployment_or_model") != self.embeddings.model_identity
        ):
            raise ValueError("REBUILD_REQUIRED: embedding provider or deployment differs")
        dimensions = metadata.get("dimensions")
        chunks = data.get("chunks")
        if not isinstance(dimensions, int) or dimensions <= 0 or not isinstance(chunks, list):
            raise ValueError("REBUILD_REQUIRED: index metadata invalid")
        indexed = [IndexedChunk(**item) for item in chunks]
        if any(len(chunk.embedding) != dimensions for chunk in indexed):
            raise ValueError("REBUILD_REQUIRED: indexed vector dimensions differ")
        return {"dimensions": dimensions}, indexed

    @staticmethod
    def _index_path(project: ProjectConfig) -> Path:
        return Path(project.index_file).resolve()
