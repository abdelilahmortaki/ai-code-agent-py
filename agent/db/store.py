from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
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

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        """Run a block inside one database transaction.

        Commits on success, rolls back and re-raises on any exception
        (psycopg.Error is wrapped in PgStoreError), and always closes the
        connection in ``finally``.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        except psycopg.Error as exc:
            raise PgStoreError("database transaction failed") from exc
        finally:
            conn.close()

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

    def find_project_by_external_id(self, external_id: str) -> dict | None:
        """Return a project row by its external id, or None if it does not exist."""
        if not external_id:
            return None
        sql = (
            "SELECT id, external_id, name, repo_root, created_at FROM projects "
            "WHERE external_id = %s"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (external_id,))
                    row = cur.fetchone()
                    return dict(row) if row is not None else None
        except psycopg.Error as exc:
            raise PgStoreError("failed to find project by external id") from exc

    def find_project_by_name(self, name: str) -> dict | None:
        """Return the most recently created project with the given name, or None."""
        if not name:
            return None
        sql = (
            "SELECT id, external_id, name, repo_root, created_at FROM projects "
            "WHERE lower(name) = lower(%s) "
            "ORDER BY created_at DESC, id ASC LIMIT 1"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (name,))
                    row = cur.fetchone()
                    return dict(row) if row is not None else None
        except psycopg.Error as exc:
            raise PgStoreError("failed to find project by name") from exc

    def create_version(
        self,
        project_id: str,
        version_number: int,
        label: str = "",
        source_hash: str = "",
        branch: str = "",
        commit_sha: str = "",
    ) -> dict:
        if version_number <= 0:
            raise ValueError("version_number must be positive")
        sql = (
            "INSERT INTO project_versions "
            "(id, project_id, version_number, label, source_hash, branch, commit_sha) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (project_id, version_number) DO UPDATE SET label = EXCLUDED.label, "
            "source_hash = EXCLUDED.source_hash, branch = EXCLUDED.branch, "
            "commit_sha = EXCLUDED.commit_sha "
            "RETURNING id, project_id, version_number, label, source_hash, branch, "
            "commit_sha, created_at"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (self._new_id(), project_id, version_number, label, source_hash, branch, commit_sha),
                    )
                    return dict(cur.fetchone())
        except psycopg.Error as exc:
            raise PgStoreError("failed to create project version") from exc

    def next_version_number(self, project_id: str) -> int:
        """Return the next version_number for a project (max + 1, starting at 1)."""
        sql = "SELECT COALESCE(MAX(version_number), 0) + 1 AS next FROM project_versions WHERE project_id = %s"
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (project_id,))
                    return int(cur.fetchone()["next"])
        except psycopg.Error as exc:
            raise PgStoreError("failed to compute next version number") from exc

    def latest_version(self, project_id: str) -> dict | None:
        """Return the highest-numbered version of a project, or None if none exist."""
        sql = (
            "SELECT id, project_id, version_number, label, source_hash, branch, "
            "commit_sha, created_at FROM project_versions "
            "WHERE project_id = %s ORDER BY version_number DESC LIMIT 1"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (project_id,))
                    row = cur.fetchone()
                    return dict(row) if row is not None else None
        except psycopg.Error as exc:
            raise PgStoreError("failed to load latest project version") from exc

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

    def list_files(self, project_version_id: str) -> list[dict]:
        """Return every code_files row of a version: id, path, language, file_hash."""
        sql = (
            "SELECT id, path, language, file_hash FROM code_files "
            "WHERE project_version_id = %s ORDER BY path"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (project_version_id,))
                    return [dict(row) for row in cur.fetchall()]
        except psycopg.Error as exc:
            raise PgStoreError("failed to list code files") from exc

    def copy_unchanged_file_symbols(
        self,
        new_version_id: str,
        old_version_id: str,
        paths: Sequence[str],
        provider: str,
        deployment_or_model: str,
        dimensions: int | None = None,
    ) -> dict:
        """Copy symbols + embeddings of unchanged files into a new version.

        Runs in a single transaction. A file is only copied when ALL of the
        following hold:

        - its path exists in both versions AND the stored file_hash matches;
        - every old symbol of the file has an embedding row (complete set);
        - every old embedding of the file carries the same provider,
          deployment_or_model and (when known) dimensions as the current run.

        If any provenance requirement fails, the file is not copied and the
        caller re-parses/re-embeds it. Symbol rows are recreated with fresh
        ids; embeddings are copied with an exact-content symbol remap
        (module, package_name, owner, symbol_type, name, qualified_name,
        signature, start_line, end_line, source).

        Returns {"paths", "symbols_copied", "embeddings_copied",
        "symbols_by_path"} where symbols_by_path maps each copied path to the
        number of symbols copied for it. An empty ``paths`` list is a no-op.
        """
        if not paths:
            return {"paths": [], "symbols_copied": 0, "embeddings_copied": 0, "symbols_by_path": {}}
        symbol_sql = (
            "INSERT INTO code_symbols "
            "(id, project_version_id, file_id, module, package_name, owner, symbol_type, "
            "name, qualified_name, signature, start_line, end_line, source) "
            "SELECT gen_random_uuid(), %s, new_files.id, old_s.module, old_s.package_name, "
            "old_s.owner, old_s.symbol_type, old_s.name, old_s.qualified_name, "
            "old_s.signature, old_s.start_line, old_s.end_line, old_s.source "
            "FROM code_symbols old_s "
            "JOIN code_files old_files ON old_files.id = old_s.file_id "
            "AND old_files.project_version_id = %s "
            "JOIN code_files new_files ON new_files.project_version_id = %s "
            "AND new_files.path = old_files.path "
            "AND new_files.file_hash = old_files.file_hash "
            "AND new_files.path = ANY(%s) "
            "WHERE old_s.project_version_id = %s "
            "AND old_s.id IN ("
            "  SELECT se3.symbol_id FROM symbol_embeddings se3 "
            "  WHERE se3.provider = %s AND se3.deployment_or_model = %s "
            "  AND (%s::int IS NULL OR se3.dimensions = %s)"
            ") "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM code_symbols missing "
            "  WHERE missing.project_version_id = %s AND missing.file_id = old_files.id "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM symbol_embeddings e WHERE e.symbol_id = missing.id"
            "  )"
            ") "
            "RETURNING id AS new_symbol_id, file_id AS new_file_id"
        )
        embedding_sql = (
            "INSERT INTO symbol_embeddings "
            "(id, project_version_id, symbol_id, embedding, provider, "
            "deployment_or_model, dimensions) "
            "SELECT gen_random_uuid(), %s, new_s.id, old_se.embedding, "
            "old_se.provider, old_se.deployment_or_model, old_se.dimensions "
            "FROM symbol_embeddings old_se "
            "JOIN code_symbols old_s ON old_s.id = old_se.symbol_id "
            "AND old_s.project_version_id = %s "
            "JOIN code_files old_files ON old_files.id = old_s.file_id "
            "JOIN code_files new_files ON new_files.project_version_id = %s "
            "AND new_files.path = old_files.path "
            "AND new_files.path = ANY(%s) "
            "JOIN code_symbols new_s ON new_s.project_version_id = %s "
            "AND new_s.file_id = new_files.id "
            "AND new_s.module = old_s.module "
            "AND new_s.package_name = old_s.package_name "
            "AND new_s.owner = old_s.owner "
            "AND new_s.symbol_type = old_s.symbol_type "
            "AND new_s.name = old_s.name "
            "AND new_s.qualified_name = old_s.qualified_name "
            "AND new_s.signature = old_s.signature "
            "AND new_s.start_line = old_s.start_line "
            "AND new_s.end_line = old_s.end_line "
            "AND new_s.source = old_s.source "
            "WHERE old_se.project_version_id = %s"
        )
        symbol_params = (
            new_version_id,
            old_version_id,
            new_version_id,
            list(paths),
            old_version_id,
            provider,
            deployment_or_model,
            dimensions,
            dimensions,
            old_version_id,
        )
        embedding_params = (
            new_version_id,
            old_version_id,
            new_version_id,
            list(paths),
            new_version_id,
            old_version_id,
        )
        try:
            with self.transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(symbol_sql, symbol_params)
                    copied_rows = [dict(row) for row in cur.fetchall()]
                    cur.execute(embedding_sql, embedding_params)
                    embeddings_copied = cur.rowcount
                    file_ids = [row["new_file_id"] for row in copied_rows]
                    paths_by_file: dict = {}
                    if file_ids:
                        cur.execute(
                            "SELECT id, path FROM code_files "
                            "WHERE project_version_id = %s AND id = ANY(%s)",
                            (new_version_id, file_ids),
                        )
                        for row in cur.fetchall():
                            paths_by_file[row["id"]] = row["path"]
        except PgStoreError:
            raise
        except psycopg.Error as exc:
            raise PgStoreError("failed to copy unchanged file symbols") from exc
        symbols_by_path: dict[str, int] = {}
        for row in copied_rows:
            path = paths_by_file.get(row["new_file_id"])
            if path is not None:
                symbols_by_path[path] = symbols_by_path.get(path, 0) + 1
        return {
            "paths": list(paths),
            "symbols_copied": len(copied_rows),
            "embeddings_copied": embeddings_copied,
            "symbols_by_path": symbols_by_path,
        }

    def insert_symbol(
        self,
        project_version_id: str,
        name: str,
        symbol_type: str,
        file_id: str | None = None,
        module: str = "",
        package_name: str = "",
        owner: str = "",
        qualified_name: str = "",
        signature: str = "",
        start_line: int = 0,
        end_line: int = 0,
        source: str = "",
    ) -> dict:
        if not name:
            raise ValueError("name must be non-empty")
        sql = (
            "INSERT INTO code_symbols "
            "(id, project_version_id, file_id, module, package_name, owner, "
            "symbol_type, name, qualified_name, signature, start_line, end_line, source) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id, project_version_id, file_id, module, package_name, owner, "
            "symbol_type, name, qualified_name, signature, start_line, end_line, "
            "source, created_at"
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
                            package_name,
                            owner,
                            symbol_type,
                            name,
                            qualified_name,
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
        project_version_id: str | None = None,
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
            "AND (%s::uuid IS NULL OR s.project_version_id = %s::uuid) "
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
            project_version_id,
            project_version_id,
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

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escape ILIKE wildcards so the query is matched literally.

        Escapes backslash, percent and underscore; the SQL uses
        ``ESCAPE '\\'`` for the pattern operators.
        """
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def search_lexical(
        self,
        project_version_id: str,
        query: str,
        top_k: int,
        min_similarity: float = 0.3,
        symbol_type: str | None = None,
        file_id: str | None = None,
    ) -> list[dict]:
        """Two-stage exact/lexical symbol search over PostgreSQL (F2.1).

        Stage 1 probes the exact-name btree index (``name = query``, score
        100.0). Stage 2 scans the same version with precedence-band scoring:

        - exact_ci 95.0: ``lower(name) = lower(query)``;
        - qualified 90.0: the module.owner.name or owner.name qualified name
          equals the query (case-insensitive);
        - prefix 80.0: ``name ILIKE 'query%'``;
        - fuzzy 40.0 + 40.0 * similarity: pg_trgm similarity >= min_similarity
          with a query of at least 3 characters and at most the symbol name
          length + 3 (so multi-word fragments cannot be mistaken for a typo of
          an identifier);
        - contains 20.0: the query appears in signature, module, path, source
          or name.

        Stage 1 rows (score 100.0) outrank every stage-2 band; both stages
        run inside one transaction, duplicates are removed by symbol id and
        the merged set is ordered by ``lexical_score DESC, name ASC,
        start_line ASC, id ASC`` (fully deterministic) and capped at top_k.
        Only parameterized SQL is used; ILIKE patterns escape ``\\ % _`` via
        ``_escape_like`` with ``ESCAPE '\\'``.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query:
            raise ValueError("query must be non-empty")
        escaped = self._escape_like(query)
        prefix_pattern = escaped + "%"
        contains_pattern = "%" + escaped + "%"
        version_filter = "s.project_version_id = %s"
        symbol_type_filter = "(%s::text IS NULL OR s.symbol_type = %s)"
        file_filter = "(%s::uuid IS NULL OR s.file_id = %s::uuid)"

        exact_sql = (
            "SELECT s.id, s.project_version_id, s.file_id, f.path, s.module, "
            "s.owner, s.symbol_type, s.name, s.signature, s.start_line, "
            "s.end_line, s.source "
            "FROM code_symbols s "
            "JOIN code_files f ON f.id = s.file_id "
            f"WHERE {version_filter} AND {symbol_type_filter} AND {file_filter} "
            "AND s.name = %s"
        )
        exact_params = (project_version_id, symbol_type, symbol_type, file_id, file_id, query)

        # A candidate is any row whose name, qualified name, signature, module,
        # path or source mentions the query; the CASE bands assign the score.
        stage2_sql = (
            "SELECT s.id, s.project_version_id, s.file_id, f.path, s.module, "
            "s.owner, s.symbol_type, s.name, s.signature, s.start_line, "
            "s.end_line, s.source, "
            "CASE "
            "WHEN lower(s.name) = lower(%s) THEN 'exact_ci' "
            "WHEN lower(concat_ws('.', s.module, NULLIF(s.owner, ''), s.name)) = lower(%s) "
            "OR lower(concat_ws('.', NULLIF(s.owner, ''), s.name)) = lower(%s) THEN 'qualified' "
            "WHEN s.name ILIKE %s ESCAPE '\\' THEN 'prefix' "
            "WHEN similarity(s.name, %s) >= %s AND char_length(%s) >= 3 "
            "AND char_length(%s) <= char_length(s.name) + 3 THEN 'fuzzy' "
            "ELSE 'contains' "
            "END AS match_kind, "
            "CASE "
            "WHEN lower(s.name) = lower(%s) THEN 95.0 "
            "WHEN lower(concat_ws('.', s.module, NULLIF(s.owner, ''), s.name)) = lower(%s) "
            "OR lower(concat_ws('.', NULLIF(s.owner, ''), s.name)) = lower(%s) THEN 90.0 "
            "WHEN s.name ILIKE %s ESCAPE '\\' THEN 80.0 "
            "WHEN similarity(s.name, %s) >= %s AND char_length(%s) >= 3 "
            "AND char_length(%s) <= char_length(s.name) + 3 "
            "THEN 40.0 + 40.0 * similarity(s.name, %s) "
            "ELSE 20.0 "
            "END AS lexical_score, "
            "similarity(s.name, %s) AS name_sim "
            "FROM code_symbols s "
            "JOIN code_files f ON f.id = s.file_id "
            f"WHERE {version_filter} AND {symbol_type_filter} AND {file_filter} "
            "AND ( "
            "lower(s.name) = lower(%s) "
            "OR lower(concat_ws('.', s.module, NULLIF(s.owner, ''), s.name)) = lower(%s) "
            "OR lower(concat_ws('.', NULLIF(s.owner, ''), s.name)) = lower(%s) "
            "OR s.name ILIKE %s ESCAPE '\\' "
            "OR (similarity(s.name, %s) >= %s AND char_length(%s) >= 3 "
            "AND char_length(%s) <= char_length(s.name) + 3) "
            "OR s.signature ILIKE %s ESCAPE '\\' "
            "OR s.module ILIKE %s ESCAPE '\\' "
            "OR f.path ILIKE %s ESCAPE '\\' "
            "OR s.source ILIKE %s ESCAPE '\\' "
            ") "
            "ORDER BY lexical_score DESC, s.name ASC, s.start_line ASC, s.id ASC "
            "LIMIT %s"
        )
        stage2_params = (
            query,  # 1  CASE exact_ci
            query,  # 2  CASE qualified full
            query,  # 3  CASE qualified short
            prefix_pattern,  # 4  CASE prefix
            query,  # 5  CASE fuzzy similarity
            min_similarity,  # 6  CASE fuzzy threshold
            query,  # 7  CASE fuzzy length
            query,  # 8  CASE fuzzy length guard
            query,  # 9  CASE score exact_ci
            query,  # 10 CASE score qualified full
            query,  # 11 CASE score qualified short
            prefix_pattern,  # 12 CASE score prefix
            query,  # 13 CASE score fuzzy similarity
            min_similarity,  # 14 CASE score fuzzy threshold
            query,  # 15 CASE score fuzzy length
            query,  # 16 CASE score fuzzy length guard
            query,  # 17 CASE score fuzzy score
            query,  # 18 name_sim
            project_version_id,  # 19 version
            symbol_type,  # 20 symbol_type null check
            symbol_type,  # 21 symbol_type equality
            file_id,  # 22 file_id null check
            file_id,  # 23 file_id equality
            query,  # 24 WHERE exact_ci
            query,  # 25 WHERE qualified full
            query,  # 26 WHERE qualified short
            prefix_pattern,  # 27 WHERE prefix
            query,  # 28 WHERE fuzzy similarity
            min_similarity,  # 29 WHERE fuzzy threshold
            query,  # 30 WHERE fuzzy length
            query,  # 31 WHERE fuzzy length guard
            contains_pattern,  # 32 WHERE signature contains
            contains_pattern,  # 33 WHERE module contains
            contains_pattern,  # 34 WHERE path contains
            contains_pattern,  # 35 WHERE source contains
        )
        try:
            with self.transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(exact_sql, exact_params)
                    stage1 = [dict(row) for row in cur.fetchall()]
                    for row in stage1:
                        row["lexical_score"] = 100.0
                        row["match_kind"] = "exact"
                        row["name_sim"] = 1.0
                        row["matched_fields"] = ["name"]
                    cur.execute(stage2_sql, stage2_params + (top_k + len(stage1),))
                    stage2 = [dict(row) for row in cur.fetchall()]
        except PgStoreError as exc:
            cause = exc.__cause__
            if cause is not None and (
                "similarity" in str(cause) or "pg_trgm" in str(cause)
            ):
                raise PgStoreError(
                    "failed to search symbols lexically: pg_trgm is not enabled; "
                    "apply agent/db/migrations/0003_lexical_search.sql"
                ) from exc
            raise
        except psycopg.Error as exc:
            raise PgStoreError("failed to search symbols lexically") from exc
        return self._merge_lexical(stage1, stage2, top_k, query)

    @staticmethod
    def _merge_lexical(
        stage1: list[dict], stage2: list[dict], top_k: int, query: str
    ) -> list[dict]:
        """Merge exact (stage 1) and scored (stage 2) rows deterministically.

        Stage-1 rows always win on duplicate ids (they outrank stage 2).
        Final ordering: lexical_score DESC, name ASC, start_line ASC, id ASC.
        """
        query_lower = query.lower()
        merged: dict[str, dict] = {}
        for row in stage1 + stage2:
            row["qualified_name"] = ".".join(
                part for part in (row["module"], row["owner"], row["name"]) if part
            )
            if row["match_kind"] == "contains":
                row["matched_fields"] = PgStore._lexical_matched_fields(row, query_lower)
            elif row["match_kind"] == "qualified":
                row["matched_fields"] = ["qualified_name"]
            else:
                row["matched_fields"] = ["name"]
            row["lexical_score"] = float(row["lexical_score"])
            row["name_sim"] = float(row["name_sim"])
            source = row["source"] or ""
            if len(source) > 2000:
                row["source"] = source[:2000] + "...[TRUNCATED]"
            merged.setdefault(row["id"], row)
        return sorted(
            merged.values(),
            key=lambda r: (-r["lexical_score"], r["name"], r["start_line"], r["id"]),
        )[:top_k]

    @staticmethod
    def _lexical_matched_fields(row: dict, query_lower: str) -> list[str]:
        """Sorted list of fields containing the query for a 'contains' row."""
        fields: list[str] = []
        for field in ("signature", "module", "path", "source", "name"):
            value = row.get(field)
            if value and query_lower in str(value).lower():
                fields.append(field)
        return sorted(fields)

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
