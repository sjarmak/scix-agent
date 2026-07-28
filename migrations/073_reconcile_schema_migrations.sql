-- Migration 073: Reconcile schema_migrations with the actual state of 069-072
--
-- Bead: scix_experiments-crz (GOAL.md W8, acceptance criterion A12).
-- Audit: docs/ops/migration_ledger_reconciliation_2026-07-27.md — read it before
-- running this. It carries the evidence for every verdict encoded below.
--
-- WHY THIS EXISTS
-- ---------------
-- `select max(version) from schema_migrations` returns 68 in prod, but 072 is
-- applied (indus_qdrant_synced holds 35,463,731 rows). The ledger has been
-- unable to answer "what is actually deployed?" since 2026-06-09, and that is the
-- root cause of the s7cy incident: a destructive DDL landed without its companion
-- code cutover because nothing tracked what had been applied.
--
-- WHAT IS AND IS NOT RECORDED HERE
-- --------------------------------
--   069  NOT recorded — papers.raw still exists in prod, ~91.5% of rows non-NULL.
--        The migration was never applied. Its *code* half shipped on 2026-06-10
--        (commit 7d6a131 dropped "raw" from field_mapping.COLUMN_ORDER), so the
--        column is now a frozen zombie with no writer. Applying 069 is an
--        operator decision gated on ADR-011 (still Status: Proposed), not a
--        ledger fix. See audit §F1.
--   070  NOT recorded — embedding_outbox never existed in prod. daily_sync Step 7
--        logged `relation "embedding_outbox" does not exist` on 2026-06-21/22/23
--        and 06-29, and no successful drain appears anywhere in the log history.
--        The file is also now UNRUNNABLE: its CREATE TRIGGER targets
--        paper_embeddings, which no longer exists. Migration 074 marks it
--        superseded. See audit §F2.
--   071  RECORDED — its two named indexes are absent, so the post-condition holds
--        and re-running is a no-op. But they are absent because paper_embeddings
--        itself was dropped, not because this file ran. The note column carries
--        that distinction; without it, this row would be a quiet falsehood.
--   072  RECORDED — verified object-for-object: live DDL, primary key and table
--        comment all match the file exactly.
--   073  RECORDED — self-recording. Only 056, 059, 063, 064 and 070 in this tree
--        ever recorded themselves; from here on, every migration should.
--
-- THE note COLUMN
-- ---------------
-- schema_migrations(version, applied_at, filename) cannot express "post-condition
-- satisfied, but not by this file" — the exact state migration 071 is in. Without
-- a note, recording 071 asserts it ran (false) and omitting it asserts the indexes
-- still exist (also false). ADD COLUMN of a nullable TEXT with no default is a
-- catalog-only change: brief lock, no table rewrite, trivial on a 58-row table.
--
-- LOCKING / SAFETY: one ALTER (catalog-only) + four INSERTs into a 58-row table,
-- inside one transaction. No row data is touched. Idempotent: ADD COLUMN IF NOT
-- EXISTS + ON CONFLICT DO NOTHING, so a re-run is a no-op.
--
-- APPLY WITH (a human must run this; it has NOT been executed):
--   psql -d scix -v ON_ERROR_STOP=1 -f migrations/073_reconcile_schema_migrations.sql
--   python scripts/verify_migration_ledger.py --dsn "dbname=scix"

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Give the ledger somewhere to record provenance
-- ---------------------------------------------------------------------------
ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS note TEXT;

COMMENT ON COLUMN schema_migrations.note IS
    'Why this row is here when the plain (version, filename) pair would be '
    'ambiguous: retroactive reconciliation, supersession, or a post-condition '
    'satisfied by something other than this file. NULL for migrations that were '
    'simply executed in order. Added by migration 073.';

-- ---------------------------------------------------------------------------
-- 2. Migration 071 — post-condition satisfied by supersession
-- ---------------------------------------------------------------------------
-- Recorded so the ledger stops implying idx_embed_hnsw_indus{,_hv} still exist.
-- applied_at is the date the superseding drop landed (bead s7cy), not today,
-- because that is when the post-condition actually became true.
INSERT INTO schema_migrations (version, applied_at, filename, note)
VALUES (
    71,
    TIMESTAMPTZ '2026-07-14 00:00:00-04',
    '071_drop_paper_embeddings_indus_indexes.sql',
    'Post-condition satisfied by supersession, not by execution. Both target '
    'indexes (idx_embed_hnsw_indus, idx_embed_hnsw_indus_hv) are absent because '
    'paper_embeddings was dropped wholesale on 2026-07-14 (bead s7cy, recorded '
    'retroactively by migration 074) — there is no evidence this file was ever '
    'run. Re-running it is a safe no-op (DROP INDEX ... IF EXISTS). Recorded '
    '2026-07-27 by migration 073; evidence in '
    'docs/ops/migration_ledger_reconciliation_2026-07-27.md.'
)
ON CONFLICT (version) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3. Migration 072 — genuinely applied, verified object-for-object
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version, applied_at, filename, note)
VALUES (
    72,
    TIMESTAMPTZ '2026-07-14 00:00:00-04',
    '072_indus_qdrant_synced.sql',
    'Applied ~2026-07-14 (bead s7cy); recorded retroactively 2026-07-27 by '
    'migration 073. Verified: indus_qdrant_synced exists with bibcode TEXT '
    'PRIMARY KEY + synced_at TIMESTAMPTZ NOT NULL DEFAULT now(), table comment '
    'matches the file exactly, 35,463,731 rows seeded from Qdrant '
    'scix_indus_v2_papers_s1. applied_at is approximate — the true timestamp was '
    'never recorded, which is what migration 073 exists to stop happening.'
)
ON CONFLICT (version) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 4. Self-record
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version, filename, note)
VALUES (
    73,
    '073_reconcile_schema_migrations.sql',
    'Ledger reconciliation for 069-072 (bead crz). Deliberately records nothing '
    'for 069 (papers.raw still present, ~91.5% of rows non-NULL) or 070 '
    '(embedding_outbox never existed in prod; file is unrunnable since '
    'paper_embeddings was dropped). Recording either as applied would recreate '
    'the failure mode this migration removes.'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
