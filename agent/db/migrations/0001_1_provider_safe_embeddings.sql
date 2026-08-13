-- F1.1 follow-up: provider-safe symbol_embeddings.
--
-- The original schema fixed embedding vector(1024), which rejects any other
-- provider dimension (e.g. Azure 3072) at insert time. This migration drops
-- the fixed typmod, renames model -> deployment_or_model, and records the
-- actual dimension of each stored vector. Safe to apply on both a fresh
-- database (after 0001_init.sql) and the overnight database where
-- 0001_init.sql is already recorded.

-- Drop the fixed dimension typmod so any provider dimension is accepted.
ALTER TABLE symbol_embeddings ALTER COLUMN embedding TYPE vector USING embedding::vector;

-- Explicit model/deployment identity.
ALTER TABLE symbol_embeddings RENAME COLUMN model TO deployment_or_model;

-- Backfill dimensions from the actual stored vectors, then enforce.
ALTER TABLE symbol_embeddings ADD COLUMN IF NOT EXISTS dimensions INTEGER;
UPDATE symbol_embeddings
   SET dimensions = vector_dims(embedding)
 WHERE dimensions IS NULL;
ALTER TABLE symbol_embeddings ALTER COLUMN dimensions SET NOT NULL;
ALTER TABLE symbol_embeddings
    ADD CONSTRAINT symbol_embeddings_dimensions_check CHECK (dimensions > 0);

ALTER TABLE symbol_embeddings ALTER COLUMN provider SET NOT NULL;
ALTER TABLE symbol_embeddings ALTER COLUMN deployment_or_model SET NOT NULL;
