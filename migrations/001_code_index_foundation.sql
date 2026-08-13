CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    repo_root TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_hash TEXT NOT NULL,
    branch TEXT,
    commit_sha TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_version_id UUID NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    module TEXT,
    file_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_version_id, path)
);

CREATE TABLE IF NOT EXISTS code_symbols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES code_files(id) ON DELETE CASCADE,
    owner TEXT,
    symbol_type TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT,
    signature TEXT,
    start_line INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS code_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_version_id UUID NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
    source_symbol_id UUID NOT NULL REFERENCES code_symbols(id) ON DELETE CASCADE,
    target_symbol_id UUID NOT NULL REFERENCES code_symbols(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbol_embeddings (
    symbol_id UUID PRIMARY KEY REFERENCES code_symbols(id) ON DELETE CASCADE,
    embedding vector NOT NULL,
    provider TEXT NOT NULL,
    deployment_or_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_project_versions_project_id ON project_versions (project_id);
CREATE INDEX IF NOT EXISTS idx_code_files_project_version_id ON code_files (project_version_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_file_id ON code_symbols (file_id);
CREATE INDEX IF NOT EXISTS idx_code_edges_project_version_id ON code_edges (project_version_id);
CREATE INDEX IF NOT EXISTS idx_code_edges_source_symbol_id ON code_edges (source_symbol_id);
CREATE INDEX IF NOT EXISTS idx_code_edges_target_symbol_id ON code_edges (target_symbol_id);
CREATE INDEX IF NOT EXISTS idx_symbol_embeddings_scope ON symbol_embeddings (provider, deployment_or_model, dimensions);
