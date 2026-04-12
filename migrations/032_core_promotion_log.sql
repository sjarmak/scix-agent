-- 032_core_promotion_log.sql
-- M3.5.2: Curated entity core lifecycle tables.
--
-- Creates:
--   * curated_entity_core — the actual core membership list (10K cap)
--   * core_promotion_log  — append-only event log of promote/demote events
--
-- Idempotent: safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS curated_entity_core (
    entity_id      INT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    query_hits_14d INT NOT NULL DEFAULT 0,
    promoted_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curated_entity_core_hits
    ON curated_entity_core(query_hits_14d);

CREATE TABLE IF NOT EXISTS core_promotion_log (
    id             SERIAL PRIMARY KEY,
    entity_id      INT NOT NULL,
    action         TEXT NOT NULL CHECK (action IN ('promote', 'demote')),
    query_hits_14d INT,
    reason         TEXT,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_core_promotion_log_entity
    ON core_promotion_log(entity_id);

CREATE INDEX IF NOT EXISTS idx_core_promotion_log_ts
    ON core_promotion_log(ts);

COMMIT;
