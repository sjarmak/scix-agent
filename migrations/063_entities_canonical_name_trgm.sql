-- migration: 063_entities_canonical_name_trgm — pg_trgm GIN on entities.canonical_name
--
-- Bead: scix_experiments-dbl.8
--
-- Problem: EntityResolver._match_fuzzy() does
--     similarity(lower(canonical_name), lower($1)) > $2
-- which forces a Seq Scan over the 19M-row entities table (~40s p50 measured
-- on prod 2026-04-29). That made fuzzy resolve unusable as a default
-- fallback in the entity tool.
--
-- This migration installs the pg_trgm extension (no-op if already present)
-- and creates a GIN trigram index on lower(canonical_name) so the resolver
-- can use the index-aware `%` operator and pull only matching rows. The
-- companion code change (src/scix/entity_resolver.py) switches the WHERE
-- clause from a similarity()-only filter to a `%`-then-similarity() pair;
-- the planner uses gin_trgm_ops via the index, then similarity() is
-- computed only on the candidate set for ranking.
--
-- CREATE INDEX CONCURRENTLY cannot run inside a transaction, so this file
-- has no BEGIN/COMMIT — the psql runner must execute statements in
-- autocommit. Apply with:
--
--     scix-batch --mem-high 4G --mem-max 8G \
--         psql "$SCIX_DSN" -v ON_ERROR_STOP=1 \
--             -f migrations/063_entities_canonical_name_trgm.sql
--
-- Runtime / resources (estimate, prod 2026-04-29):
--   - Source: 19,347,591 rows × avg ~22 chars canonical_name → ~2.7 GB heap.
--   - Build time: ~10-25 minutes wall clock with 4 parallel workers.
--   - Peak RAM: ~3-6 GB (parallel workers × maintenance_work_mem).
--   - Disk: ~1.5-3 GB GIN trigram index.
--
-- If CREATE INDEX CONCURRENTLY fails midway, the index is left INVALID.
-- Drop it and retry:
--     DROP INDEX CONCURRENTLY IF EXISTS idx_entities_canonical_trgm;

CREATE EXTENSION IF NOT EXISTS pg_trgm;

SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entities_canonical_trgm
    ON entities USING gin (lower(canonical_name) gin_trgm_ops);

-- The schema_migrations row insert lives in its own transaction so the file
-- can run with statements in autocommit. The CONCURRENTLY index above and
-- this INSERT are independent — re-running this migration after the index
-- already exists is a no-op for both.
INSERT INTO schema_migrations (version, filename)
    VALUES (63, '063_entities_canonical_name_trgm.sql')
    ON CONFLICT (version) DO NOTHING;

-- Post-verification:
--     SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
--       FROM pg_indexes
--      WHERE tablename='entities'
--        AND indexname='idx_entities_canonical_trgm';
--
--     EXPLAIN (ANALYZE) SELECT id, canonical_name,
--         similarity(lower(canonical_name), 'scibert') AS sim
--       FROM entities
--      WHERE lower(canonical_name) % 'scibert'
--        AND similarity(lower(canonical_name), 'scibert') > 0.3
--      ORDER BY sim DESC LIMIT 20;
--
-- Expected: Bitmap Index Scan on idx_entities_canonical_trgm, sub-second.
