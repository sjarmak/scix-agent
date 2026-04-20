-- migration: 054_paper_embeddings_halfvec_index — HNSW on embedding_hv
--
-- Bead: scix_experiments-0vy
-- Prereq: migration 053 applied; scripts/backfill_halfvec.py --model indus
--         has completed for at least the rows you want indexed.
--
-- This migration creates the new partial HNSW index on the halfvec(768)
-- shadow column. The old index (idx_embed_hnsw_indus on vector_cosine_ops)
-- is NOT dropped here — the code deploy (src/scix/search.py) cuts queries
-- over to the new column, then a follow-up migration drops the old index
-- once the cutover is verified on the 50-query eval.
--
-- CREATE INDEX CONCURRENTLY cannot run inside a transaction. Psycopg-based
-- migration runners in this repo (scripts/migrate.py) accept files without
-- BEGIN/COMMIT and execute them with autocommit; verify that before running.
--
-- Runtime / resources (estimate on this host, per scix_test dry-run):
--   - Build time: ~45-90 minutes wall clock for 32M halfvec(768) rows.
--   - Peak RAM: ~10-15 GB (parallel workers × maintenance_work_mem).
--   - Disk: ~60 GB new index (~half of the 120 GB vector_cosine_ops index).
--   - MUST be invoked via scix-batch to keep user@1000.service OOM isolation.
--     Recommended: scix-batch --mem-high 20G --mem-max 30G psql "$SCIX_DSN" \
--         -v ON_ERROR_STOP=1 -f migrations/054_paper_embeddings_halfvec_index.sql

SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 7;

-- NOTE on the partial-index predicate: we intentionally match the shape of
-- the legacy idx_embed_hnsw_indus (WHERE model_name='indus', no NOT NULL
-- clause) so the planner can match index-to-query by lexical predicate
-- equality. pgvector's HNSW silently skips NULL rows at build time, so
-- indexing partially-backfilled state is safe — the index just grows
-- incrementally as the backfill progresses.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embed_hnsw_indus_hv
    ON paper_embeddings
    USING hnsw (embedding_hv halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'indus';

-- If CREATE INDEX CONCURRENTLY fails midway, the index is left INVALID.
-- Drop it and retry:
--     DROP INDEX CONCURRENTLY IF EXISTS idx_embed_hnsw_indus_hv;
--
-- Post-verification (also encoded in tests/test_halfvec_migration.py):
--     SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
--       FROM pg_indexes
--      WHERE tablename='paper_embeddings'
--        AND indexname LIKE 'idx_embed_hnsw_indus%';
