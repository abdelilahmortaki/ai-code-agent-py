-- F1.2 Project version identity and file hashing.

ALTER TABLE project_versions
    ADD COLUMN IF NOT EXISTS source_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS commit_sha TEXT NOT NULL DEFAULT '';

-- source_hash is always supplied explicitly by the indexer; keep the ''
-- defaults on branch/commit_sha so inserts without git metadata still work.
ALTER TABLE project_versions ALTER COLUMN source_hash DROP DEFAULT;

ALTER TABLE code_files ALTER COLUMN file_hash SET NOT NULL;
