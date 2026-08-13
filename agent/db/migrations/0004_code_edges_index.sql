-- F2.2 Code graph relationship queries over code_edges.

-- Fast lookups of outgoing edges (source-scoped) per relation type.
CREATE INDEX IF NOT EXISTS idx_code_edges_source ON code_edges (project_version_id, source_symbol_id, relation_type);

-- Fast lookups of incoming edges (target-scoped) per relation type.
CREATE INDEX IF NOT EXISTS idx_code_edges_target ON code_edges (project_version_id, target_symbol_id, relation_type);
