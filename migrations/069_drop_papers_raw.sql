-- Migration 069: Drop papers.raw JSONB column (ADR-011, Option B)
--
-- ADR: docs/ADR/011_drop_papers_raw_column.md
--
-- papers.raw was the initial-ingest catch-all for ADS JSONL fields without
-- dedicated columns. Every valuable field has had a dedicated column since
-- migration 012; the ordered reference[] list it carried is superseded by
-- citation_edges (Option B: positional [N] marker resolution is disabled in
-- citation_context.py until ref_position lands via Option A re-ingest);
-- the citation[] field was never consumed.
--
-- PRECONDITION (operator-verified before applying to prod)
-- ---------------------------------------------------------
-- A full archive of (bibcode, raw) for every row WHERE raw IS NOT NULL
-- exists on the NAS:
--   /mnt/scix_offload/papers_raw_archive/papers_raw_2026-06-09.jsonl.zst
-- Row count of the archive must equal
--   SELECT count(*) FROM papers WHERE raw IS NOT NULL
-- at archive time. The canonical upstream record additionally lives in
-- /mnt/scix_offload/ads_metadata_by_year_picard/ (re-ingest is the defined
-- recovery path).
--
-- LOCKING / SPACE
-- ---------------
-- DROP COLUMN is a catalog-only change: brief AccessExclusiveLock, no table
-- rewrite. TOAST pages are returned to the free-space map by routine vacuum
-- (reused by future writes); the relation file does NOT shrink until a
-- pg_repack/VACUUM FULL maintenance window with ~450 GB free (deferred —
-- see ADR-011 Step 6).
--
-- IDEMPOTENCY: DROP COLUMN IF EXISTS.

ALTER TABLE papers DROP COLUMN IF EXISTS raw;

ANALYZE papers;
