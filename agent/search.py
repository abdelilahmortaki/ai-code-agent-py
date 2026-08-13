"""Exact/lexical symbol retrieval over PostgreSQL (F2.1).

Provides LexicalSearchService: deterministic lexical search over indexed
symbols (exact name fast path plus case-insensitive, qualified, prefix,
fuzzy trigram and substring matches). The CLI searches the latest indexed
version of a project and prints JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from agent.versioning import _database_url, _redact_secrets

if TYPE_CHECKING:
    from agent.db.store import PgStore


class LexicalSearchService:
    """Exact/lexical search over symbols persisted in PostgreSQL."""

    _MAX_QUERY_CHARS = 500
    _FILTER_KEYS = frozenset({"symbol_type", "file_id"})

    def __init__(self, store: PgStore | None = None) -> None:
        self.store = store

    def search(
        self,
        project_version_id: str,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search symbols within a single project version.

        Validation: the query is stripped and must be non-empty and at most
        500 characters; top_k must be between 1 and 50; filters may only
        contain the keys ``symbol_type`` and ``file_id``. Any violation
        raises ValueError.
        """
        if self.store is None:
            raise RuntimeError("database is not configured")
        query, top_k, filters = self._validate(query, top_k, filters)
        return self.store.search_lexical(
            project_version_id=project_version_id,
            query=query,
            top_k=top_k,
            symbol_type=filters.get("symbol_type") or None,
            file_id=filters.get("file_id") or None,
        )

    def search_latest(
        self,
        project_id: str,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> dict:
        """Search the highest-numbered version of a project by external id.

        Returns ``{"project_version_id", "version_number", "results"}``.
        Raises ValueError when the project has no indexed version.
        """
        if self.store is None:
            raise RuntimeError("database is not configured")
        project = self.store.find_project_by_external_id(project_id)
        if project is None:
            raise ValueError(f"unknown project: {project_id}")
        version = self.store.latest_version(project["id"])
        if version is None:
            raise ValueError("no versions for project")
        return {
            "project_version_id": version["id"],
            "version_number": version["version_number"],
            "results": self.search(
                version["id"], query, top_k=top_k, filters=filters
            ),
        }

    def _validate(self, query: str, top_k: int, filters: dict | None) -> tuple:
        """Normalize and validate search inputs; returns (query, top_k, filters)."""
        stripped = query.strip() if isinstance(query, str) else ""
        if not stripped:
            raise ValueError("query must be non-empty")
        if len(stripped) > self._MAX_QUERY_CHARS:
            raise ValueError(
                f"query must be at most {self._MAX_QUERY_CHARS} characters"
            )
        if not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        normalized = dict(filters or {})
        unknown = sorted(set(normalized) - self._FILTER_KEYS)
        if unknown:
            raise ValueError(f"unsupported filter keys: {', '.join(unknown)}")
        file_id = normalized.get("file_id")
        if file_id is not None:
            if isinstance(file_id, uuid.UUID):
                normalized["file_id"] = str(file_id)
            else:
                try:
                    uuid.UUID(str(file_id))
                except (ValueError, AttributeError, TypeError) as exc:
                    raise ValueError("file_id must be a valid UUID") from exc
        return stripped, top_k, normalized


# --------------------------------------------------------------------------- CLI


def _find_project(store: PgStore, repo_root: str, name: str) -> dict | None:
    """Resolve the project row for the CLI: by external id (repo root path),
    falling back to a name lookup when --name is given."""
    project = store.find_project_by_external_id(repo_root)
    if project is None and name:
        project = store.find_project_by_name(name)
    return project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.search",
        description="Exact/lexical search over indexed symbols (F2.1).",
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Absolute path of the indexed project (its resolved path is the "
        "project's external id)",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Project name, used as a fallback when the repo root has no "
        "project row",
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of results (1-50, default 10)",
    )
    parser.add_argument(
        "--symbol-type",
        default="",
        help="Restrict results to a symbol type (e.g. class, method)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides DATABASE_URL)"
        "database.url)",
    )
    args = parser.parse_args(argv)

    dsn = _database_url(args.url)
    if not dsn:
        print(
            "No database URL configured: pass --url or set DATABASE_URL",
            file=sys.stderr,
        )
        return 1

    try:
        from agent.db.store import PgStore

        store = PgStore(dsn)
        external_id = str(Path(args.repo_root).resolve())
        project = _find_project(store, external_id, args.name)
        if project is None:
            print(f"Project not found for repo root: {external_id}", file=sys.stderr)
            return 1
        filters = {"symbol_type": args.symbol_type} if args.symbol_type else None
        result = LexicalSearchService(store).search_latest(
            project["external_id"], args.query, top_k=args.top_k, filters=filters
        )
    except Exception as exc:
        print(f"Search failed: {_redact_secrets(str(exc), dsn)}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
