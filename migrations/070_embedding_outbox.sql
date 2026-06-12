-- Migration 070: PG→Qdrant transactional outbox for forward embedding writes
--
-- PRD: docs/prd/qdrant_nas_migration.md (MH-9, "Transactional outbox for
-- forward writes"). Bead: scix_experiments-8m0a.
--
-- Why an outbox at all
-- --------------------
-- The new-paper flow is PG-first: daily_sync.sh step 5 (scripts/embed.py)
-- writes INDUS vectors into paper_embeddings. The serving dense lane reads
-- from Qdrant (scix_indus_v2_papers_s1, ADR-013). Without a sync mechanism
-- the Qdrant collection silently goes stale every day — and because the
-- env-flag flip to qdrant makes Qdrant authoritative, "stale" reads as
-- "rollback to yesterday's corpus" with no error. The outbox closes that gap.
--
-- Why a trigger instead of the PRD's app-level dual write
-- ------------------------------------------------------
-- MH-9 sketched a dual write inside embed.py (BEGIN; INSERT paper_embeddings;
-- INSERT vector_outbox; COMMIT). A trigger is strictly stronger: it captures
-- EVERY writer of paper_embeddings (embed.py COPY-upsert, ad-hoc backfills,
-- manual fixes) with zero chance of a code path forgetting the second write.
-- The cost is one extra row-insert per embedding write, which is negligible
-- next to the embedding compute itself.
--
-- Schema is a true queue, not a watermark log
-- -------------------------------------------
-- embedding_outbox(bibcode, model_name, op, enqueued_at) carries no ack
-- column: the worker (scripts/qdrant_outbox_sync.py) claims a batch with
-- FOR UPDATE SKIP LOCKED, upserts to Qdrant, and DELETEs the claimed rows in
-- the same transaction only after the upsert succeeds. At-least-once delivery
-- + idempotent uuid5(bibcode) upsert = safe. Drained rows leave the table, so
-- the lag metric is simply "rows remaining + age of the oldest row".
--
-- OPERATIONAL ORDER (per bead 8m0a — read before applying to prod)
-- ----------------------------------------------------------------
-- SAFE NOW: the table + trigger function + trigger DDL. The trigger only adds
-- work when paper_embeddings is written. Apply the trigger to PROD *after* the
-- full-corpus Qdrant load (scripts/qdrant_full_load.py) completes — installing
-- it mid-load would have every bulk upsert also fire the trigger and contend
-- on paper_embeddings writes. The one-time gap (papers embedded before the
-- trigger existed, e.g. the 1,266-paper 2026-06-11 batch) is reconciled by
-- `qdrant_outbox_sync.py --backfill-since <date>`, NOT by this migration.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Outbox queue table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_outbox (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bibcode     TEXT        NOT NULL,
    model_name  TEXT        NOT NULL,
    op          TEXT        NOT NULL CHECK (op IN ('INSERT', 'UPDATE', 'DELETE', 'BACKFILL')),
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE embedding_outbox IS
    'PG→Qdrant forward-write queue (migration 070, PRD MH-9). One row per '
    'paper_embeddings write, inserted by trigger trg_embedding_outbox. Drained '
    'and deleted by scripts/qdrant_outbox_sync.py. No ack column — drained rows '
    'are removed, so row count + oldest enqueued_at IS the sync-lag metric.';
COMMENT ON COLUMN embedding_outbox.op IS
    'INSERT/UPDATE → upsert current vector to Qdrant; DELETE → remove the '
    'point; BACKFILL → enqueued by the worker''s --backfill-since reconcile, '
    'treated as an upsert.';

-- Drain order is (model_name, enqueued_at, id): the worker routes per
-- model_name and claims oldest-first within each. `id` is the tiebreaker for
-- rows sharing an enqueued_at (a bulk backfill INSERT..SELECT stamps one
-- now() across all rows), and including it lets the LIMIT-batched claim be a
-- pure index scan with no sort node even on a million-row backfill. The
-- leading (model_name, enqueued_at) prefix still serves report_lag's
-- min(enqueued_at) GROUP BY model_name.
CREATE INDEX IF NOT EXISTS idx_embedding_outbox_drain
    ON embedding_outbox (model_name, enqueued_at, id);

-- ---------------------------------------------------------------------------
-- 2. Trigger function — enqueue one row per paper_embeddings write
-- ---------------------------------------------------------------------------
-- AFTER trigger so it only fires once the row change is durable in the same
-- transaction. On DELETE the NEW tuple is NULL, so we read OLD.
CREATE OR REPLACE FUNCTION embedding_outbox_enqueue() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO embedding_outbox (bibcode, model_name, op)
            VALUES (OLD.bibcode, OLD.model_name, 'DELETE');
        RETURN OLD;
    ELSE
        INSERT INTO embedding_outbox (bibcode, model_name, op)
            VALUES (NEW.bibcode, NEW.model_name, TG_OP);
        RETURN NEW;
    END IF;
END
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION embedding_outbox_enqueue() IS
    'Row-level enqueue for embedding_outbox. Captures every paper_embeddings '
    'INSERT/UPDATE/DELETE so the Qdrant sync worker never misses a write '
    '(migration 070).';

-- ---------------------------------------------------------------------------
-- 3. Trigger on paper_embeddings
-- ---------------------------------------------------------------------------
-- DROP+CREATE (not CREATE OR REPLACE — Postgres has no CREATE OR REPLACE
-- TRIGGER on this version) so the migration is idempotent on re-run.
DROP TRIGGER IF EXISTS trg_embedding_outbox ON paper_embeddings;
CREATE TRIGGER trg_embedding_outbox
    AFTER INSERT OR UPDATE OR DELETE ON paper_embeddings
    FOR EACH ROW EXECUTE FUNCTION embedding_outbox_enqueue();

-- ---------------------------------------------------------------------------
-- 4. Record the migration
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version, filename)
    VALUES (70, '070_embedding_outbox.sql')
    ON CONFLICT (version) DO NOTHING;

COMMIT;
