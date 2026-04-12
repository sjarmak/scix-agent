-- 034_jit_cache.sql
-- M11b: Persisted JIT cache (partitioned by expires_at).
--
-- Creates:
--   * document_entities_jit_cache — partitioned cache table with tier=5,
--     candidate_set_hash / model_version / expires_at columns.
--   * document_entities_jit_cache_default — catch-all DEFAULT partition so
--     INSERTs succeed before the daily partition-creation cron runs.
--   * Supporting index on (bibcode, candidate_set_hash, model_version).
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Partitioned parent table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_entities_jit_cache (
    bibcode             TEXT        NOT NULL,
    entity_id           INT         NOT NULL,
    link_type           TEXT        NOT NULL,
    confidence          REAL,
    match_method        TEXT,
    evidence            JSONB,
    harvest_run_id      INT,
    tier                SMALLINT    NOT NULL DEFAULT 5,
    tier_version        INT         NOT NULL DEFAULT 1,
    candidate_set_hash  TEXT        NOT NULL,
    model_version       TEXT        NOT NULL,
    expires_at          TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT document_entities_jit_cache_tier_check CHECK (tier = 5),
    PRIMARY KEY (
        bibcode,
        entity_id,
        link_type,
        candidate_set_hash,
        model_version,
        expires_at
    )
) PARTITION BY RANGE (expires_at);

-- ---------------------------------------------------------------------------
-- 2. DEFAULT partition — catches any row whose expires_at does not fall into
--    a dated partition. The daily cron (scripts/jit_cache_cleanup.py) will
--    create forward-dated range partitions and eventually detach/drop the
--    default — but for tests and bootstrap the DEFAULT is sufficient so
--    INSERTs always succeed.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_entities_jit_cache_default
    PARTITION OF document_entities_jit_cache DEFAULT;

-- ---------------------------------------------------------------------------
-- 3. Lookup index — the hot path is
--    (bibcode, candidate_set_hash, model_version) equality.
--    Defined on the parent so all partitions inherit.
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_document_entities_jit_cache_lookup
    ON document_entities_jit_cache (bibcode, candidate_set_hash, model_version);

CREATE INDEX IF NOT EXISTS idx_document_entities_jit_cache_expires
    ON document_entities_jit_cache (expires_at);

COMMIT;
