"""Persist code symbols and contextual embeddings into PostgreSQL/pgvector (F1.5/F1.6).

Builds a version snapshot, then per Java file: parses symbols, renders the
deterministic contextual payloads (F1.4), embeds them via the configured
embedding provider and persists symbol rows plus pgvector embeddings through
the PgStore. The legacy JSON index (SemanticIndexService) is left untouched.

Two entry points on SymbolEmbeddingIndexer:
- index(): full re-index of every Java file (F1.5);
- index_incremental(): hash-based incremental re-index (F1.6) that reuses
  symbols/embeddings of unchanged files, re-indexes changed/new files and
  leaves deleted files implicit (each run creates its own version rows).
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
        """Create a version snapshot and persist symbols + embeddings (full re-index)."""
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
                files, symbols, embeddings = self._index_java_file(
                    version["id"], root, rel, max_payload_chars
                )
                files_indexed += files
                symbols_indexed += symbols
                embeddings_indexed += embeddings
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
                "dimensions": self.embedding_provider.dimensions,
            },
        }

    def index_incremental(
        self,
        project: ProjectConfig,
        max_payload_chars: int = 16000,
    ) -> dict:
        """Incrementally re-index a project by comparing file hashes (F1.6).

        Flow: ensure the project row exists, snapshot the previous latest
        version, create the new version (VersionIndexer.index_version), then
        classify every current Java file against the previous version's stored
        hashes:

        - unchanged: symbols + embeddings are copied from the previous version
          (store.copy_unchanged_file_symbols, which re-verifies the hash); a
          file with zero copied symbols falls back to a full re-parse;
        - changed/new: re-parsed and re-embedded via the same per-file path as
          the full index;
        - deleted: nothing to do -- every version owns its rows, so files that
          disappeared from the scan simply have no rows in the new version.

        Returned counters: files_total = current Java files in the scan;
        files_reused = unchanged files whose symbols were actually copied;
        files_changed/files_added = hash-changed/new files; files_deleted =
        previous Java files absent from the current scan; files_indexed =
        files re-parsed in this run (changed + new + reuse fallbacks);
        symbols_reused/embeddings_reused = rows copied from the previous
        version; symbols_indexed/embeddings_indexed = rows inserted by
        re-parsing.
        """
        if self.store is None:
            raise RuntimeError("database is not configured")
        root = Path(project.repo_root).resolve()
        project_row = self.store.upsert_project(
            external_id=project.id,
            name=project.name,
            repo_root=str(root),
        )
        previous_version = self.store.latest_version(project_row["id"])
        version = VersionIndexer(self.codebase, self.store).index_version(project)["version"]

        current_hashes: dict[str, str] = {}
        current_java_paths: list[str] = []
        for rel in self.codebase.scan_paths(project):
            if not rel.endswith(".java"):
                continue
            current_java_paths.append(rel)
            try:
                target = resolve_within(root, rel)
                current_hashes[rel] = compute_file_hash(target)
            except (OSError, ValueError):
                continue

        prev_files: list[dict] = []
        if previous_version is not None:
            prev_files = self.store.list_files(previous_version["id"])
        prev_java = {
            row["path"]: row
            for row in prev_files
            if row["path"].endswith(".java")
        }
        current_set = set(current_java_paths)

        deleted = sorted(path for path in prev_java if path not in current_set)
        unchanged: list[str] = []
        changed: list[str] = []
        added: list[str] = []
        for rel in sorted(current_set):
            prev_row = prev_java.get(rel)
            if prev_row is None:
                added.append(rel)
            elif rel in current_hashes and prev_row["file_hash"] == current_hashes[rel]:
                unchanged.append(rel)
            else:
                changed.append(rel)

        files_indexed = 0
        symbols_indexed = 0
        embeddings_indexed = 0
        symbols_reused = 0
        embeddings_reused = 0
        failed_files: list[str] = []
        fallback: list[str] = []

        if unchanged and previous_version is not None:
            copy_result = self.store.copy_unchanged_file_symbols(
                new_version_id=version["id"],
                old_version_id=previous_version["id"],
                paths=unchanged,
                provider=self.embedding_provider.provider_name,
                deployment_or_model=self.embedding_provider.model_identity,
                dimensions=self.embedding_provider.dimensions,
            )
            symbols_reused += copy_result["symbols_copied"]
            embeddings_reused += copy_result["embeddings_copied"]
            fallback = [
                rel
                for rel in unchanged
                if copy_result["symbols_by_path"].get(rel, 0) == 0
            ]
            unchanged = [rel for rel in unchanged if rel not in fallback]

        for rel in sorted(changed + added + fallback):
            try:
                files, symbols, embeddings = self._index_java_file(
                    version["id"], root, rel, max_payload_chars
                )
                files_indexed += files
                symbols_indexed += symbols
                embeddings_indexed += embeddings
            except (SymbolParseError, OSError):
                failed_files.append(rel)

        return {
            "version": version,
            "previous_version": previous_version,
            "files_total": len(current_set),
            "files_indexed": files_indexed,
            "files_reused": len(unchanged),
            "files_changed": len(changed),
            "files_added": len(added),
            "files_deleted": len(deleted),
            "symbols_indexed": symbols_indexed,
            "symbols_reused": symbols_reused,
            "embeddings_indexed": embeddings_indexed,
            "embeddings_reused": embeddings_reused,
            "failed_files": failed_files,
            "embedding": {
                "provider": self.embedding_provider.provider_name,
                "model": self.embedding_provider.model_identity,
                "dimensions": self.embedding_provider.dimensions,
            },
        }

    # ------------------------------------------------------------------ shared

    def _index_java_file(
        self,
        project_version_id: str,
        root: Path,
        rel: str,
        max_payload_chars: int,
    ) -> tuple[int, int, int]:
        """Parse, embed and persist one Java file; returns (files, symbols,
        embeddings) counts. Raises SymbolParseError/OSError on per-file
        failure and ValueError on embedding contract violations (fail-fast)."""
        target = resolve_within(root, rel)
        content = target.read_text(encoding="utf-8", errors="replace")
        file_row = self.store.upsert_file(
            project_version_id=project_version_id,
            path=rel,
            language="java",
            file_hash=compute_file_hash(target),
        )
        symbols = JavaSymbolParser(file=rel, root=root).parse(content)
        payloads = [
            truncate_payload(payload, max_payload_chars)
            for payload in build_file_payloads(symbols, content)
        ]
        if not payloads:
            return 1, 0, 0
        vectors = self.embedding_provider.embed_texts(payloads)
        if len(vectors) != len(payloads):
            raise ValueError(
                "Embedding provider returned an incomplete result: "
                f"expected {len(payloads)} vectors, got {len(vectors)}"
            )
        if not vectors or not vectors[0]:
            raise ValueError("Embedding provider returned an empty vector")
        batch_dimensions = len(vectors[0])
        if any(len(vector) != batch_dimensions for vector in vectors):
            raise ValueError(
                "Embedding provider returned mixed vector dimensions "
                f"in one batch: expected {batch_dimensions}"
            )
        for symbol, vector in zip(symbols, vectors):
            symbol_row = self.store.insert_symbol(
                project_version_id=project_version_id,
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
                project_version_id=project_version_id,
                symbol_id=symbol_row["id"],
                embedding=vector,
                provider=self.embedding_provider.provider_name,
                deployment_or_model=self.embedding_provider.model_identity,
            )
        return 1, len(symbols), len(vectors)


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
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help="Reuse unchanged files (hash-based) instead of a full re-index",
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
        indexer = SymbolEmbeddingIndexer(
            CodebaseService(),
            store,
            _build_embedding_provider(),
        )
        if args.incremental:
            result = indexer.index_incremental(
                project, max_payload_chars=args.max_payload_chars
            )
        else:
            result = indexer.index(project, max_payload_chars=args.max_payload_chars)
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
