-- 023_logged_embeddings.sql
-- Convert paper_embeddings from UNLOGGED to LOGGED for crash safety,
-- and widen the vector column to support multiple embedding dimensions.
--
-- Context: paper_embeddings was UNLOGGED for faster bulk writes during
-- initial embedding. A PostgreSQL crash wiped all 32M embeddings because
-- UNLOGGED tables are truncated on recovery. This migration prevents that.
--
-- Also changes embedding column from vector(768) to vector (untyped),
-- allowing different models to store different-dimensional embeddings
-- in the same table (e.g. 768 for SPECTER2/INDUS, 1024 for GTE).

BEGIN;

-- 1. Convert to LOGGED so WAL protects the data.
--    This is a metadata-only change when the table is empty (current state).
--    On a populated table it rewrites all pages to WAL — plan for downtime.
ALTER TABLE paper_embeddings SET LOGGED;

-- 2. Widen embedding column: vector(768) -> vector (any dimension).
--    The model_name column already partitions by model, so mixed dimensions
--    are safe. Per-model partial HNSW indexes (migration 004) enforce
--    dimensional consistency at the index level.
ALTER TABLE paper_embeddings ALTER COLUMN embedding TYPE vector;

-- 3. Drop the old global HNSW index from migration 001 if it exists.
--    We use per-model partial indexes (migration 004) instead.
DROP INDEX IF EXISTS idx_embed_hnsw;
DROP INDEX IF EXISTS idx_embeddings_hnsw;

-- 4. Also convert _to_embed if it exists (scratch table for embedding pipeline).
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_class WHERE relname = '_to_embed' AND relpersistence = 'u') THEN
        ALTER TABLE _to_embed SET LOGGED;
        RAISE NOTICE 'Converted _to_embed to LOGGED';
    END IF;
END $$;

COMMIT;
