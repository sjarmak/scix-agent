-- 027_per_model_hnsw_rebuild.sql
-- Create per-model partial HNSW indexes for all active embedding models.
--
-- pgvector 0.8.2 supports parallel HNSW builds (max_parallel_maintenance_workers=7)
-- and iterative scan (relaxed_order) for filtered queries.
--
-- Index parameters:
--   m=16           — graph degree, good balance of recall vs build time
--   ef_construction=64 — lower than 200 for 32M rows (acceptable recall,
--                         significantly faster build). Matches CLAUDE.md guidance.
--
-- The indus model (32M rows, 768d) is the critical index. specter2 and nomic
-- (20K rows each) are small but included for completeness.
--
-- NOTE: The indus index build on 32M rows will take significant time even with
-- parallel workers. Run during low-traffic window.

BEGIN;

-- Drop stale indexes from earlier migrations if they somehow survived
DROP INDEX IF EXISTS idx_embed_hnsw_specter2;
DROP INDEX IF EXISTS idx_embed_hnsw_openai;

-- Per-model HNSW indexes with cosine distance.
-- The embedding column is untyped vector (migration 023), so we must cast
-- to vector(768) in the index expression to pin the dimensionality.

-- indus: 32M rows, 768d — primary retrieval model
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_indus
    ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'indus';

-- specter2: 20K rows, 768d — citation-proximity similarity
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_specter2
    ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'specter2';

-- nomic: 20K rows, 768d
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_nomic
    ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'nomic';

COMMIT;
