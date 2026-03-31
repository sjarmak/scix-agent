-- 002_ingest_log.sql
-- Tracks ingestion progress per file for resumability.

BEGIN;

CREATE TABLE IF NOT EXISTS ingest_log (
    filename TEXT PRIMARY KEY,
    records_loaded INTEGER NOT NULL DEFAULT 0,
    errors_skipped INTEGER NOT NULL DEFAULT 0,
    edges_loaded INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'in_progress',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

COMMIT;
