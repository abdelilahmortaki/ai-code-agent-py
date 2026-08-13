-- F2.1 Exact/lexical symbol retrieval over PostgreSQL.

-- Trigram similarity for fuzzy identifier matching (near-exact typos).
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Fast path for exact technical identifiers (name lookups scoped per version).
CREATE INDEX IF NOT EXISTS idx_code_symbols_name_exact
    ON code_symbols (project_version_id, name);

-- Trigram indexes for fuzzy name/signature matching.
CREATE INDEX IF NOT EXISTS idx_code_symbols_name_trgm
    ON code_symbols USING GIN (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_code_symbols_signature_trgm
    ON code_symbols USING GIN (signature gin_trgm_ops);

-- Module-scoped lookups (module contains-matches and qualified names).
CREATE INDEX IF NOT EXISTS idx_code_symbols_module
    ON code_symbols (project_version_id, module);

-- Java namespace lookups: persisted qualified_name and package_name.
CREATE INDEX IF NOT EXISTS idx_code_symbols_qualified_name_trgm
    ON code_symbols USING GIN (qualified_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_code_symbols_package_name_trgm
    ON code_symbols USING GIN (package_name gin_trgm_ops);

-- File path contains-matches.
CREATE INDEX IF NOT EXISTS idx_code_files_path_trgm
    ON code_files USING GIN (path gin_trgm_ops);
