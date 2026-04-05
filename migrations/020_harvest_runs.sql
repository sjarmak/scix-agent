-- 020_harvest_runs.sql
-- Harvest run tracking for external data source ingestion

BEGIN;

CREATE TABLE IF NOT EXISTS harvest_runs (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    records_fetched INT NOT NULL DEFAULT 0,
    records_upserted INT NOT NULL DEFAULT 0,
    cursor_state JSONB,
    error_message TEXT,
    config JSONB,
    counts JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_harvest_runs_source ON harvest_runs(source);

COMMIT;
