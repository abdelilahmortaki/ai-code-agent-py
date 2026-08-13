"""PostgreSQL + pgvector persistence layer.

psycopg and pgvector are imported lazily inside functions so that module
import and compileall succeed without the dependencies installed.
"""

import os
from pathlib import Path
from typing import Any, Sequence

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"
_MIGRATION_FILES = ("001_code_index_foundation.sql",)


class PersistenceError(RuntimeError):
    """Raised for persistence failures; messages never contain the DSN."""


def database_url() -> str | None:
    raw = os.getenv("DATABASE_URL")
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def run_migrations(url: str | None = None) -> int:
    target = url if url is not None else database_url()
    if target is None:
        raise PersistenceError("DATABASE_URL is not set")
    import psycopg  # noqa: PLC0415

    try:
        with psycopg.connect(target) as conn:
            for file_name in _MIGRATION_FILES:
                sql = (_MIGRATIONS_DIR / file_name).read_text()
                with conn.cursor() as cur:
                    cur.execute(sql)
            conn.commit()
    except PersistenceError:
        raise
    except Exception as exc:
        raise PersistenceError("migration run failed") from exc
    return len(_MIGRATION_FILES)


class PgStore:
    """PostgreSQL/pgvector store. Opens a connection per operation."""

    def __init__(self, url: str):
        self._url = url

    @classmethod
    def from_env(cls) -> "PgStore":
        url = database_url()
        if url is None:
            raise PersistenceError("DATABASE_URL is not set")
        return cls(url)

    def _connect(self):
        import psycopg  # noqa: PLC0415
        from psycopg.rows import dict_row

        try:
            conn = psycopg.connect(self._url, row_factory=dict_row)
        except Exception as exc:
            raise PersistenceError("could not connect to database") from exc
        from pgvector.psycopg import register_vector  # noqa: PLC0415

        try:
            register_vector(conn)
        except Exception as exc:
            conn.close()
            raise PersistenceError("could not initialize vector support") from exc
        return conn

    def _require_nonempty(self, value: Any, name: str) -> str:
        if value is None or not str(value).strip():
            raise ValueError(f"{name} must not be empty")
        return value

    def upsert_project(self, project_id: str, name: str, repo_root: str) -> dict:
        self._require_nonempty(project_id, "project_id")
        self._require_nonempty(name, "name")
        self._require_nonempty(repo_root, "repo_root")
        sql = """
            INSERT INTO projects (id, name, repo_root)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, repo_root = EXCLUDED.repo_root
            RETURNING id, name, repo_root, created_at
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (project_id, name, repo_root))
                return _normalize_row(cur.fetchone())

    def insert_project_version(
        self, project_id: str, source_hash: str, branch: str | None = None, commit_sha: str | None = None
    ) -> dict:
        self._require_nonempty(project_id, "project_id")
        self._require_nonempty(source_hash, "source_hash")
        sql = """
            INSERT INTO project_versions (project_id, source_hash, branch, commit_sha)
            VALUES (%s, %s, %s, %s)
            RETURNING id, project_id, source_hash, branch, commit_sha, created_at
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (project_id, source_hash, branch, commit_sha))
                return _normalize_row(cur.fetchone())

    def insert_code_file(
        self, project_version_id: str, path: str, file_hash: str, language: str, module: str | None = None
    ) -> dict:
        self._require_nonempty(project_version_id, "project_version_id")
        self._require_nonempty(path, "path")
        self._require_nonempty(file_hash, "file_hash")
        self._require_nonempty(language, "language")
        sql = """
            INSERT INTO code_files (project_version_id, path, module, file_hash, language)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, project_version_id, path, module, file_hash, language, created_at
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (project_version_id, path, module, file_hash, language))
                return _normalize_row(cur.fetchone())

    def insert_code_symbol(
        self,
        file_id: str,
        symbol_type: str,
        name: str,
        owner: str | None = None,
        qualified_name: str | None = None,
        signature: str | None = None,
        start_line: int = 0,
        end_line: int = 0,
        source: str = "",
    ) -> dict:
        self._require_nonempty(file_id, "file_id")
        self._require_nonempty(symbol_type, "symbol_type")
        self._require_nonempty(name, "name")
        if start_line < 0 or end_line < 0:
            raise ValueError("start_line and end_line must be >= 0")
        sql = """
            INSERT INTO code_symbols
                (file_id, owner, symbol_type, name, qualified_name, signature,
                 start_line, end_line, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, file_id, owner, symbol_type, name, qualified_name,
                      signature, start_line, end_line, source
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (file_id, owner, symbol_type, name, qualified_name, signature, start_line, end_line, source),
                )
                return _normalize_row(cur.fetchone())

    def insert_code_edge(
        self, project_version_id: str, source_symbol_id: str, target_symbol_id: str, edge_type: str
    ) -> dict:
        self._require_nonempty(project_version_id, "project_version_id")
        self._require_nonempty(source_symbol_id, "source_symbol_id")
        self._require_nonempty(target_symbol_id, "target_symbol_id")
        self._require_nonempty(edge_type, "edge_type")
        sql = """
            INSERT INTO code_edges (project_version_id, source_symbol_id, target_symbol_id, edge_type)
            VALUES (%s, %s, %s, %s)
            RETURNING id, project_version_id, source_symbol_id, target_symbol_id, edge_type
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (project_version_id, source_symbol_id, target_symbol_id, edge_type))
                return _normalize_row(cur.fetchone())

    def insert_symbol_embedding(
        self, symbol_id: str, embedding: Sequence[float], provider: str, deployment_or_model: str
    ) -> dict:
        self._require_nonempty(symbol_id, "symbol_id")
        self._require_nonempty(provider, "provider")
        self._require_nonempty(deployment_or_model, "deployment_or_model")
        values = _validate_embedding(embedding)
        sql = """
            INSERT INTO symbol_embeddings (symbol_id, embedding, provider, deployment_or_model, dimensions)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symbol_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                provider = EXCLUDED.provider,
                deployment_or_model = EXCLUDED.deployment_or_model,
                dimensions = EXCLUDED.dimensions
            RETURNING symbol_id, provider, deployment_or_model, dimensions, created_at
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol_id, values, provider, deployment_or_model, len(values)))
                return _normalize_row(cur.fetchone())

    def find_nearest(
        self, embedding: Sequence[float], provider: str, deployment_or_model: str, limit: int = 5
    ) -> list[dict]:
        self._require_nonempty(provider, "provider")
        self._require_nonempty(deployment_or_model, "deployment_or_model")
        values = _validate_embedding(embedding)
        if limit <= 0:
            raise ValueError("limit must be > 0")
        sql = """
            SELECT se.symbol_id, se.provider, se.deployment_or_model, se.dimensions,
                   se.embedding <=> %s AS distance
            FROM symbol_embeddings se
            WHERE se.provider = %s AND se.deployment_or_model = %s AND se.dimensions = %s
            ORDER BY distance
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (values, provider, deployment_or_model, len(values), limit))
                rows = cur.fetchall()
        return [_normalize_row(row) for row in rows]


def _validate_embedding(embedding: Sequence[float]) -> list[float]:
    values = list(embedding)
    if not values:
        raise ValueError("embedding must not be empty")
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding values must be numeric") from exc
        if fv != fv or fv in (float("inf"), float("-inf")):
            raise ValueError("embedding values must be finite")
    return [float(v) for v in values]


def _normalize_row(row: dict | None) -> dict:
    if row is None:
        return {}
    normalized = dict(row)
    for key in ("embedding",):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = [float(v) for v in value]
    return normalized


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="postgres", description="Run PostgreSQL migrations.")
    parser.add_argument("--url", default=None, help="Database URL (defaults to DATABASE_URL env var).")
    args = parser.parse_args(argv)
    if args.url is None and database_url() is None:
        print("DATABASE_URL is not set (pass --url or set DATABASE_URL)", file=sys.stderr)
        return 2
    try:
        run_migrations(args.url)
    except PersistenceError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for file_name in _MIGRATION_FILES:
        print(f"Applied migration: {file_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
