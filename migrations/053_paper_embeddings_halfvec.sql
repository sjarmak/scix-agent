-- migration: 053_paper_embeddings_halfvec — halfvec storage for INDUS
--
-- Bead: scix_experiments-0vy
-- PRD/notes: docs/runbooks/halfvec_migration.md
--
-- This migration is the ONLINE (shadow-column) leg of the halfvec cutover.
-- It deliberately does NOT rewrite the existing `embedding` column, because
-- ALTER COLUMN ... TYPE halfvec(768) USING ... holds ACCESS EXCLUSIVE on
-- paper_embeddings for the entire rewrite (~multi-hour on 32M rows / 125 GB
-- TOAST) and would block the daily embed.py cron. Instead we add a nullable
-- halfvec(768) shadow column and backfill it out-of-band in batches via
-- scripts/backfill_halfvec.py. The new HNSW index is built separately
-- (CREATE INDEX CONCURRENTLY cannot run inside a transaction) — see
-- migration 054 and the runbook for the full sequence.
--
-- Idempotent: every DDL uses IF NOT EXISTS. Safe to re-run.
--
-- Dependencies:
--   - paper_embeddings exists (migrations 001 / 004 / 005).
--   - pgvector >= 0.8.0 (migration 005 upgraded to 0.8.2).
--   - halfvec type + halfvec_cosine_ops opclass (shipped with pgvector 0.7+).

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Shadow column: embedding_hv halfvec(768)
-- ---------------------------------------------------------------------------
-- Nullable so the ALTER is metadata-only (no rewrite, no lock scan). The
-- backfill script populates it for model_name='indus' rows. The pilot
-- embeddings (nomic, specter2) stay on the legacy vector column — per bead
-- 0vy non-goals they are not being migrated.

ALTER TABLE paper_embeddings
    ADD COLUMN IF NOT EXISTS embedding_hv halfvec(768);

COMMENT ON COLUMN paper_embeddings.embedding_hv IS
    'halfvec(768) shadow column for INDUS. Populated by '
    'scripts/backfill_halfvec.py and by scripts/embed.py on new writes. '
    'Replaces (embedding::vector(768)) as the canonical retrieval column '
    'for model_name=''indus''. Pilot models keep using embedding.';

-- ---------------------------------------------------------------------------
-- 2. Progress-tracking table for the backfill
-- ---------------------------------------------------------------------------
-- The backfill is batched by bibcode range so restarts are cheap. We persist
-- cursor + counts here so the script is idempotent across OOM kills and
-- systemd scope restarts.

CREATE TABLE IF NOT EXISTS halfvec_backfill_progress (
    id             SERIAL PRIMARY KEY,
    model_name     TEXT NOT NULL,
    last_bibcode   TEXT,
    rows_updated   BIGINT NOT NULL DEFAULT 0,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    note           TEXT
);

CREATE INDEX IF NOT EXISTS idx_halfvec_backfill_progress_model
    ON halfvec_backfill_progress (model_name, started_at DESC);

-- ---------------------------------------------------------------------------
-- 3. Assertion: column is present with the expected type
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    col_type TEXT;
BEGIN
    SELECT format_type(atttypid, atttypmod)
      INTO col_type
      FROM pg_attribute
     WHERE attrelid = 'paper_embeddings'::regclass
       AND attname  = 'embedding_hv';

    IF col_type IS NULL THEN
        RAISE EXCEPTION 'migration 053: embedding_hv column not added';
    END IF;
    IF col_type <> 'halfvec(768)' THEN
        RAISE EXCEPTION 'migration 053: embedding_hv has unexpected type %', col_type;
    END IF;
END $$;

COMMIT;

-- ---------------------------------------------------------------------------
-- NEXT STEPS (executed outside this file — see docs/runbooks/halfvec_migration.md)
-- ---------------------------------------------------------------------------
--   1. scix-batch python scripts/backfill_halfvec.py --model indus --dsn "$SCIX_DSN"
--        → populates embedding_hv for all existing indus rows
--   2. migration 054_paper_embeddings_halfvec_index.sql  (CREATE INDEX CONCURRENTLY)
--        → builds idx_embed_hnsw_indus_hv on embedding_hv halfvec_cosine_ops
--   3. Deploy src/scix/search.py + scripts/embed.py changes
--        → queries and new writes cut over to embedding_hv
--   4. Smoke-test 50-query eval (results/halfvec_migration/post.json)
--   5. Later migration to DROP INDEX idx_embed_hnsw_indus + DROP COLUMN embedding
--        (paper_embeddings_nomic / _specter2 also read embedding — gated on
--         retiring those pilots).
