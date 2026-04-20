-- 036_link_runs_watermark.sql
-- M10 / u13: Watermark table for the incremental entity-linking pipeline,
-- plus a tiny alerts sink used by the circuit-breaker and
-- watermark-staleness checks.
--
-- link_runs is append-only: every incremental run inserts one row with
-- the max entry_date it processed, the row count, and a status so that
-- a tripped circuit breaker (graceful degradation) can still advance the
-- watermark while recording that it did so with zero link writes.
--
-- alerts is a generic sink the breaker and other M10 health checks write
-- into. Operational paging reads rows with severity='page' and marks
-- acked_at when handled.
--
-- Idempotent: safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS link_runs (
    run_id         BIGSERIAL   PRIMARY KEY,
    max_entry_date TIMESTAMPTZ,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT now(),
    rows_linked    INTEGER     NOT NULL DEFAULT 0,
    status         TEXT        NOT NULL DEFAULT 'ok'
        CHECK (status IN ('ok', 'tripped', 'failed')),
    trip_count     INTEGER     NOT NULL DEFAULT 0,
    note           TEXT
);

CREATE INDEX IF NOT EXISTS idx_link_runs_timestamp_desc
    ON link_runs (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_link_runs_max_entry_date_desc
    ON link_runs (max_entry_date DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS alerts (
    id         BIGSERIAL   PRIMARY KEY,
    severity   TEXT        NOT NULL
        CHECK (severity IN ('info', 'warn', 'page')),
    source     TEXT        NOT NULL,
    message    TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    acked_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerts_unacked_severity
    ON alerts (severity, created_at DESC)
    WHERE acked_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_alerts_source
    ON alerts (source);

COMMIT;
