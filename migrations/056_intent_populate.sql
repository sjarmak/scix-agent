-- 056_intent_populate.sql
-- Backfill marker for the citation_contexts.intent population job (PRD MH-1
-- of docs/prd/scix_deep_search_v1.md).
--
-- This migration does NOT change schema. It records an idempotent marker row
-- in ingest_log so that operator runs of scripts/backfill_citation_intent.py
-- have a stable, queryable handle for status, restart, and audit.
--
-- The PRD calls this table scix_ingest_log; the actual table in this codebase
-- (introduced in migrations/002_ingest_log.sql) is named ingest_log. We use
-- the existing table — no rename, no new schema object — to keep this
-- migration a pure marker.
--
-- Idempotency:
--   * Re-running this migration leaves a single 'pending' row in ingest_log
--     with filename = 'intent_backfill:citation_contexts'.
--   * If the backfill has already advanced the row's status (e.g. to
--     'in_progress' or 'complete'), the ON CONFLICT clause leaves it alone.
--
-- Rollback: DELETE FROM ingest_log WHERE filename = 'intent_backfill:citation_contexts';

BEGIN;

INSERT INTO ingest_log (filename, status, started_at)
VALUES ('intent_backfill:citation_contexts', 'pending', NOW())
ON CONFLICT (filename) DO NOTHING;

COMMIT;
