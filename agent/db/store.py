from __future__ import annotations

import uuid
from typing import Any, Sequence

import psycopg
from psycopg.rows import dict_row


class PgStoreError(RuntimeError):
    """Raised when a PostgreSQL operation fails."""


def _vector_to_list(value: Any) -> list[float]:
    """Normalize a pgvector value returned by PostgreSQL into a list of floats.

    psycopg has no adapter for the ``vector`` type, so values come back as
    their text form (e.g. ``"[0.1,0.2,...]"``) unless a third-party adapter is
    registered; both forms are handled here.
    """
    if value is None:
        return []
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [float(x) for x in to_list()]
    if isinstance(value, str):
        stripped = value.strip("[]")
        if not stripped:
            return []
        return [float(x) for x in stripped.split(",")]
    return [float(x) for x in value]


class PgStore:
    """Minimal PostgreSQL persistence layer for the indexed code base."""

    def __init__(self, url: str) -> None:
        self._url = url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._url, row_factory=dict_row)

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def ping(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except psycopg.Error as exc:
            raise PgStoreError("database ping failed") from exc

    def close(self) -> None:
        """No-op: connections are short-lived per call and already closed."""

    def upsert_project(self, external_id: str, name: str, repo_root: str = "") -> dict:
        if not external_id or not name:
            raise ValueError("external_id and name must be non-empty")
        sql = (
            "INSERT INTO projects (id, external_id, name, repo_root) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (external_id) DO UPDATE SET name = EXCLUDED.name, "
            "repo_root = EXCLUDED.repo_root "
            "RETURNING id, external_id, name, repo_root, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (self._new_id(), external_id, name, repo_root))
                    return dict(cur.fetchone())
        except psycopg.Error as exc:
            raise PgStoreError("failed to upsert project") from exc

    def create_version(self, project_id: str, version_number: int, label: str = "") -> dict:
        if version_number <= 0:
            raise ValueError("version_number must be positive")
        sql = (
            "INSERT INTO project_versions (id, project_id, version_number, label) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (project_id, version_number) DO UPDATE SET label = EXCLUDED.label "
            "RETURNING id, project_id, version_number, label, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (self._new_id(), project_id, version_number, label))
                    return dict(cur.fetchone())
        except psycopg.Error as exc:
            raise PgStoreError("failed to create project version") from exc

    def upsert_file(
        self,
        project_version_id: str,
        path: str,
        language: str = "",
        file_hash: str = "",
    ) -> dict:
        if not path:
            raise ValueError("path must be non-empty")
        sql = (
            "INSERT INTO code_files (id, project_version_id, path, language, file_hash) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (project_version_id, path) DO UPDATE SET language = EXCLUDED.language, "
            "file_hash = EXCLUDED.file_hash "
            "RETURNING id, project_version_id, path, language, file_hash, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (self._new_id(), project_version_id, path, language, file_hash))
                    return dict(cur.fetchone())
        except psycopg.Error as exc:
            raise PgStoreError("failed to upsert code file") from exc

    def insert_symbol(
        self,
        project_version_id: str,
        name: str,
        symbol_type: str,
        file_id: str | None = None,
        module: str = "",
        owner: str = "",
        signature: str = "",
        start_line: int = 0,
        end_line: int = 0,
        source: str = "",
    ) -> dict:
        if not name:
            raise ValueError("name must be non-empty")
        sql = (
            "INSERT INTO code_symbols "
            "(id, project_version_id, file_id, module, owner, symbol_type, name, "
            "signature, start_line, end_line, source) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id, project_version_id, file_id, module, owner, symbol_type, "
            "name, signature, start_line, end_line, source, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            self._new_id(),
                            project_version_id,
                            file_id,
                            module,
                            owner,
                            symbol_type,
                            name,
                            signature,
                            start_line,
                            end_line,
                            source,
                        ),
                    )
                    return dict(cur.fetchone())
        except psycopg.Error as exc:
            raise PgStoreError("failed to insert code symbol") from exc

    def insert_edge(
        self,
        project_version_id: str,
        source_symbol_id: str,
        target_symbol_id: str,
        relation_type: str,
    ) -> dict:
        if not relation_type:
            raise ValueError("relation_type must be non-empty")
        sql = (
            "INSERT INTO code_edges (id, project_version_id, source_symbol_id, "
            "target_symbol_id, relation_type) "
            "VALUES (%s, %s, %s, %s, %s) "
            "RETURNING id, project_version_id, source_symbol_id, target_symbol_id, "
            "relation_type, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (self._new_id(), project_version_id, source_symbol_id, target_symbol_id, relation_type),
                    )
                    return dict(cur.fetchone())
        except psycopg.Error as exc:
            raise PgStoreError("failed to insert code edge") from exc

    def insert_embedding(
        self,
        project_version_id: str,
        symbol_id: str,
        embedding: Sequence[float],
        provider: str,
        deployment_or_model: str,
    ) -> dict:
        if not provider or not deployment_or_model:
            raise ValueError("provider and deployment_or_model must be non-empty")
        vector = [float(x) for x in embedding]
        if not vector:
            raise ValueError("embedding must not be empty")
        if not all(x == x and x not in (float("inf"), float("-inf")) for x in vector):
            raise ValueError("embedding must contain only finite numbers")
        dimensions = len(vector)
        sql = (
            "INSERT INTO symbol_embeddings (id, project_version_id, symbol_id, embedding, "
            "provider, deployment_or_model, dimensions) "
            "VALUES (%s, %s, %s, %s::vector, %s, %s, %s) "
            "ON CONFLICT (symbol_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
            "provider = EXCLUDED.provider, deployment_or_model = EXCLUDED.deployment_or_model, "
            "dimensions = EXCLUDED.dimensions "
            "RETURNING id, project_version_id, symbol_id, embedding, provider, "
            "deployment_or_model, dimensions, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            self._new_id(),
                            project_version_id,
                            symbol_id,
                            str(vector),
                            provider,
                            deployment_or_model,
                            dimensions,
                        ),
                    )
                    row = dict(cur.fetchone())
                    row["embedding"] = _vector_to_list(row["embedding"])
                    return row
        except psycopg.Error as exc:
            raise PgStoreError("failed to insert symbol embedding") from exc

    def search_similar(
        self,
        embedding: Sequence[float],
        provider: str,
        deployment_or_model: str,
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not provider or not deployment_or_model:
            raise ValueError("provider and deployment_or_model must be non-empty")
        vector = [float(x) for x in embedding]
        if not vector:
            raise ValueError("embedding must not be empty")
        dimensions = len(vector)
        sql = (
            "SELECT s.id, s.project_version_id, s.file_id, s.module, s.owner, "
            "s.symbol_type, s.name, s.signature, s.start_line, s.end_line, "
            "se.embedding, se.provider, se.deployment_or_model, se.dimensions, "
            "se.embedding <=> %s::vector AS distance "
            "FROM symbol_embeddings se "
            "JOIN code_symbols s ON s.id = se.symbol_id "
            "WHERE se.provider = %s "
            "AND se.deployment_or_model = %s "
            "AND se.dimensions = %s "
            "AND (%s::uuid IS NULL OR s.file_id = %s::uuid) "
            "ORDER BY se.embedding <=> %s::vector "
            "LIMIT %s"
        )
        params: tuple = (
            str(vector),
            provider,
            deployment_or_model,
            dimensions,
            file_id,
            file_id,
            str(vector),
            top_k,
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = [dict(row) for row in cur.fetchall()]
        except psycopg.Error as exc:
            raise PgStoreError("failed to search similar symbols") from exc
        for row in rows:
            row["embedding"] = _vector_to_list(row["embedding"])
        return rows

    def count_symbols(self, project_version_id: str | None = None) -> int:
        sql = (
            "SELECT count(*) AS n FROM code_symbols "
            "WHERE (%s::uuid IS NULL OR project_version_id = %s::uuid)"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (project_version_id, project_version_id))
                    return int(cur.fetchone()["n"])
        except psycopg.Error as exc:
            raise PgStoreError("failed to count symbols") from exc
