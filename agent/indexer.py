"""Persist code symbols and contextual embeddings into PostgreSQL/pgvector (F1.5).

Builds a version snapshot, then per Java file: parses symbols, renders the
deterministic contextual payloads (F1.4), embeds them via the configured
embedding provider and persists symbol rows plus pgvector embeddings through
the PgStore. The legacy JSON index (SemanticIndexService) is left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from agent.codebase import CodebaseService
from agent.config import ProjectConfig
from agent.paths import resolve_within
from agent.payloads import build_file_payloads
from agent.providers import EmbeddingProvider
from agent.symbols import JavaSymbolParser, SymbolParseError
from agent.versioning import (
    VersionIndexer,
    _database_url,
    _redact_secrets,
    compute_file_hash,
)

if TYPE_CHECKING:
    from agent.db.store import PgStore

# Marker appended to payloads that are truncated at max_payload_chars.
_TRUNCATION_MARKER = "\n...[TRUNCATED]"


def truncate_payload(payload: str, max_chars: int) -> str:
    """Deterministically truncate a payload to ``max_chars`` characters.

    The result never exceeds ``max_chars``: the marker is accounted for
    inside the budget so a truncated payload still ends with it.
    """
    if len(payload) <= max_chars:
        return payload
    if max_chars < len(_TRUNCATION_MARKER):
        raise ValueError(
            f"max_payload_chars must be at least {len(_TRUNCATION_MARKER)} "
            "to fit the truncation marker"
        )
    return payload[: max_chars - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


class SymbolEmbeddingIndexer:
    """Persist code symbols and their contextual embeddings into PostgreSQL.

    Provider failures abort the whole run (fail-fast); per-file parse or I/O
    failures are recorded in ``failed_files`` and skipped.
    """

    def __init__(
        self,
        codebase: CodebaseService,
        store: PgStore | None,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.codebase = codebase
        self.store = store
        self.embedding_provider = embedding_provider

    def index(
        self,
        project: ProjectConfig,
        max_payload_chars: int = 16000,
    ) -> dict:
        """Create a version snapshot and persist symbols + embeddings."""
        if self.store is None:
            raise RuntimeError("database is not configured")
        version_result = VersionIndexer(self.codebase, self.store).index_version(project)
        version = version_result["version"]
        root = Path(project.repo_root).resolve()

        files_indexed = 0
        symbols_indexed = 0
        embeddings_indexed = 0
        failed_files: list[str] = []
        last_batch_dimensions: int | None = None

        for rel in self.codebase.scan_paths(project):
            if not rel.endswith(".java"):
                continue
            try:
                target = resolve_within(root, rel)
                content = target.read_text(encoding="utf-8", errors="replace")
                file_row = self.store.upsert_file(
                    project_version_id=version["id"],
                    path=rel,
                    language="java",
                    file_hash=compute_file_hash(target),
                )
                symbols = JavaSymbolParser(file=rel, root=root).parse(content)
                payloads = [
                    truncate_payload(payload, max_payload_chars)
                    for payload in build_file_payloads(symbols, content)
                ]
                if payloads:
                    vectors = self.embedding_provider.embed_texts(payloads)
                    if len(vectors) != len(payloads):
                        raise ValueError(
                            "Embedding provider returned an incomplete result: "
                            f"expected {len(payloads)} vectors, got {len(vectors)}"
                        )
                    if not vectors or not vectors[0]:
                        raise ValueError("Embedding provider returned an empty vector")
                    batch_dimensions = len(vectors[0])
                    last_batch_dimensions = batch_dimensions
                    if any(len(vector) != batch_dimensions for vector in vectors):
                        raise ValueError(
                            "Embedding provider returned mixed vector dimensions "
                            f"in one batch: expected {batch_dimensions}"
                        )
                    for symbol, vector in zip(symbols, vectors):
                        symbol_row = self.store.insert_symbol(
                            project_version_id=version["id"],
                            file_id=file_row["id"],
                            module=symbol.module,
                            package_name=symbol.package_name,
                            owner=symbol.owner,
                            symbol_type=symbol.symbol_type,
                            name=symbol.name,
                            qualified_name=symbol.qualified_name,
                            signature=symbol.signature,
                            start_line=symbol.start_line,
                            end_line=symbol.end_line,
                            source=symbol.source,
                        )
                        self.store.insert_embedding(
                            project_version_id=version["id"],
                            symbol_id=symbol_row["id"],
                            embedding=vector,
                            provider=self.embedding_provider.provider_name,
                            deployment_or_model=self.embedding_provider.model_identity,
                        )
                    symbols_indexed += len(symbols)
                    embeddings_indexed += len(vectors)
                files_indexed += 1
            except (SymbolParseError, OSError):
                failed_files.append(rel)

        return {
            "version": version,
            "files_indexed": files_indexed,
            "symbols_indexed": symbols_indexed,
            "embeddings_indexed": embeddings_indexed,
            "failed_files": failed_files,
            "embedding": {
                "provider": self.embedding_provider.provider_name,
                "model": self.embedding_provider.model_identity,
                "dimensions": last_batch_dimensions,
            },
        }


# --------------------------------------------------------------------------- CLI


def _build_embedding_provider() -> EmbeddingProvider:
    from agent.config import Settings
    from agent.factory import create_provider_runtime
    from agent.guard import PromptGuardService

    settings = Settings.load("config.yml")
    return create_provider_runtime(settings, PromptGuardService()).embedding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.indexer",
        description="Index Java symbols and contextual embeddings into PostgreSQL (pgvector).",
    )
    parser.add_argument("--repo-root", required=True, help="Absolute path of the project to index")
    parser.add_argument("--name", default="", help="Project name (defaults to the directory name)")
    parser.add_argument(
        "--external-id",
        default="",
        help="Unique external id (defaults to the resolved repo root path)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides DATABASE_URL)",
    )
    parser.add_argument(
        "--max-payload-chars",
        type=int,
        default=16000,
        help="Maximum characters per embedding payload (default: 16000)",
    )
    args = parser.parse_args(argv)

    dsn = _database_url(args.url)
    if not dsn:
        print(
            "No database URL configured: pass --url or set DATABASE_URL"
            "or set database.url in config.yml",
            file=sys.stderr,
        )
        return 1

    try:
        from agent.db.store import PgStore

        store = PgStore(dsn)
        root = Path(args.repo_root).resolve()
        name = args.name or root.name
        external_id = args.external_id or str(root)
        project = ProjectConfig(
            id=external_id,
            name=name,
            repo_root=str(root),
            stories_file="",
            index_file="",
        )
        result = SymbolEmbeddingIndexer(
            CodebaseService(),
            store,
            _build_embedding_provider(),
        ).index(project, max_payload_chars=args.max_payload_chars)
        if result["symbols_indexed"] != result["embeddings_indexed"]:
            print(
                "warning: symbols_indexed "
                f"({result['symbols_indexed']}) != embeddings_indexed "
                f"({result['embeddings_indexed']})",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Indexing failed: {_redact_secrets(str(exc), dsn)}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
