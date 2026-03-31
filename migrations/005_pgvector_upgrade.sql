-- 005_pgvector_upgrade.sql
-- Upgrade pgvector extension and enable iterative index scans.
--
-- pgvector 0.8.0+ adds iterative index scans which solve the post-filtering
-- problem: with selective filters (e.g., "astrophysics papers from 2023"),
-- the old approach could return fewer than k results because it scans a fixed
-- number of index entries then post-filters. Iterative scans automatically
-- expand the search until enough results pass the filter.
--
-- This migration is safe to run on any pgvector version:
-- - ALTER EXTENSION ... UPDATE upgrades to the latest installed version
-- - The SET commands are session-level and only take effect in 0.8.0+
--
-- Also enables in 0.8.0+:
-- - halfvec (float16) quantization — halves embedding storage
-- - Binary quantization for further compression
-- - Improved HNSW build performance

BEGIN;

-- Upgrade pgvector to the latest version available on the server.
-- If already at the latest version, this is a no-op.
ALTER EXTENSION vector UPDATE;

-- Note: iterative scan settings are session-level (SET/SET LOCAL), not
-- persisted in the schema. Application code in search.py configures
-- hnsw.iterative_scan per-transaction when pgvector >= 0.8.0 is detected.
--
-- Available modes:
--   off           — default pre-0.8.0 behavior (post-filtering only)
--   relaxed_order — iterative scan, may return results slightly out of order
--                   (best for filtered search where recall > exact ordering)
--   strict_order  — iterative scan with exact distance ordering
--                   (slower but guarantees order; use for unfiltered top-k)

COMMIT;
