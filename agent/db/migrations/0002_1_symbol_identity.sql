-- F1.3 follow-up: symbol identity (package_name + qualified_name).
--
-- code_symbols previously carried only `module` (now the Maven module) with
-- no Java package or qualified-name columns. This migration adds the Java
-- namespace identity used by payloads, lexical search and graph resolution.
-- Applies after 0002_version_identity.sql and before 0003_lexical_search.sql.

ALTER TABLE code_symbols ADD COLUMN IF NOT EXISTS package_name TEXT;
ALTER TABLE code_symbols ADD COLUMN IF NOT EXISTS qualified_name TEXT;

-- Fast exact lookups of qualified names scoped per project version.
CREATE INDEX IF NOT EXISTS idx_code_symbols_qualified_name
    ON code_symbols (project_version_id, qualified_name);
