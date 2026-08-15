# F1 Acceptance Evidence (#10-#15) — real VM + real Azure + real ShopPoc

Date: 2026-08-15
Project: upload-1786756681 (fresh API upload of egawilldoit/shoppoc-app @ 8f18e3b)
Backend: dev @ 4340f3dd + overnight/f1-acceptance fix
DB: agentdb_local (PostgreSQL 16 + pgvector 0.8.6)

## #10 PostgreSQL + pgvector foundation
- vector 0.8.6 + pg_trgm 1.6 extensions installed.
- Tables present: projects, project_versions, code_files, code_symbols, code_edges, symbol_embeddings, schema_migrations.
- 7 migrations applied in order; re-run idempotent (verified by F1 audit agent).
- Real Azure 3072-dim embeddings persist (see #14).

## #11 Project version identity + hashes
- v1 source_hash=d7ac74e5... v2 source_hash=3c0957e5... (64 hex each) — version-specific, changes when content changes.
- branch=dev, commit_sha=4340f3dd captured for direct Git-backed index.
- file_hash 64-hex per code_files row.
- project_version_id per indexed state (v1 7f9a691c, v2 99c6cbb5).

## #12 Java symbol parser (real ShopPoc)
915 symbols across 5 types (v2):
- method 675, class 136, constructor 73, interface 23, enum 8
Qualified names/signatures/start-end/full source persisted as code_symbols rows.

## #13 Contextual embedding payloads
Real context-preview payload for the NOT_FOUND ticket selected:
- com.shoppoc.shared.error.DomainError.notFound (method) with reasons
  ['direct lexical match', 'semantic match', 'related dependency (CALLS incoming)'],
  retrieval_sources [lexical, vector, graph], priority high, signature preserved.

## #14 pgvector symbol embeddings
- 915 embeddings, provider=azure, deployment=text-embedding-3-large, dimensions=3072.
- Exact nearest-neighbor (<=>) proof: querying DomainError.notFound returns
  DomainError.forbidden (0.2435), DomainError.validation (0.2633), ... (top-5 all DomainError methods).

## #15 Incremental indexing (REAL, live Azure, after dimensions fix)
- v1 -> v2: files_total=159 (Java), files_reused=158, files_changed=1 (the accepted DomainError.java),
  symbols_reused=905, embeddings_reused=905, failed=[].
- New project version created per run; unchanged files reused WITHOUT re-embedding
  (byte-identical copy w/ provider/model/dimension provenance check);
  changed file re-parsed/re-embedded; historical v1 preserved.
- Fix applied: AzureOpenAIEmbeddingProvider now accepts configured dimensions
  (AZURE_EMBEDDING_DIMENSIONS=3072) so unchanged-file reuse works instead of
  conservative re-embed fallback. REBUILD_REQUIRED guard retained.

## F1 runtime acceptance
Full runtime-e2e.py destructive run: 61 checks, 0 fails, RESULT PASS.
Evidence: .agent/e2e/runtime-e2e-20260815-012449132.json
- first generation -> reject (human_decision=rejected)
- second generation -> accept (human_decision=accepted)
- exact one-line NOT_FOUND -> RESOURCE_NOT_FOUND mutation; old=0 new=1; CRLF=0
- source SHA immutable through pending/reject/second-pending/before-accept
- both runs persisted (azure/gpt-5-mini), status completed
