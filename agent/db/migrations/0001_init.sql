-- F1.1 PostgreSQL persistence foundation.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    repo_root TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_versions (
    id UUID PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_id, version_number)
);

CREATE TABLE IF NOT EXISTS code_files (
    id UUID PRIMARY KEY,
    project_version_id UUID NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    file_hash TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_version_id, path)
);

CREATE TABLE IF NOT EXISTS code_symbols (
    id UUID PRIMARY KEY,
    project_version_id UUID NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
    file_id UUID REFERENCES code_files(id) ON DELETE CASCADE,
    module TEXT NOT NULL DEFAULT '',
    owner TEXT NOT NULL DEFAULT '',
    symbol_type TEXT NOT NULL,
    name TEXT NOT NULL,
    signature TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_edges (
    id UUID PRIMARY KEY,
    project_version_id UUID NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
    source_symbol_id UUID NOT NULL REFERENCES code_symbols(id) ON DELETE CASCADE,
    target_symbol_id UUID NOT NULL REFERENCES code_symbols(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS symbol_embeddings (
    id UUID PRIMARY KEY,
    project_version_id UUID NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
    symbol_id UUID NOT NULL UNIQUE REFERENCES code_symbols(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
