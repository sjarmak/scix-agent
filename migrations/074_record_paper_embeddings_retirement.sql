-- Migration 074: Retroactively record the paper_embeddings retirement (bead s7cy)
--
-- Bead: scix_experiments-crz (GOAL.md W8). Audit:
-- docs/ops/migration_ledger_reconciliation_2026-07-27.md §F3.
-- Depends on migration 073 (adds schema_migrations.note). Run 073 first.
--
-- ============================================================================
-- READ THIS BEFORE RUNNING. THIS FILE CONTAINS DROP STATEMENTS.
-- ============================================================================
-- Against production `scix` as audited on 2026-07-27 every statement below is a
-- NO-OP: paper_embeddings, embedding_outbox, embedding_outbox_enqueue() and
-- trg_embedding_outbox are all already absent. The file exists to make that
-- absence *recorded and replayable*, not to perform it.
--
-- VERIFY THE NO-OP PRECONDITION FIRST. If either name resolves, STOP — this file
-- would then be genuinely destructive and the audit is out of date:
--
--   psql -d scix -tAc "select coalesce(to_regclass('public.paper_embeddings')::text,'ABSENT'),
--                             coalesce(to_regclass('public.embedding_outbox')::text,'ABSENT')"
--   -- must print:  ABSENT|ABSENT
--
-- WHY THIS FILE EXISTS
-- --------------------
-- On ~2026-07-14 the ~195 GB multi-model `paper_embeddings` table was dropped
-- from production. There is no migration file for it, no schema_migrations row,
-- no entry in logs/, and nothing in psql history. The only record is prose in
-- commit de0e006 and in docs/ADR/015 — which is still marked
-- "Status: Proposed (artifacts authored 2026-06-22; NOT executed)".
--
-- That untracked destructive DDL *is* the s7cy incident. Leaving it unrecorded
-- means a replay of migrations 001..073 against a fresh database produces a
-- schema with paper_embeddings in it — a schema production has not had since
-- 2026-07-14 — and the ledger keeps lying in the same way that caused the
-- original failure.
--
-- SCOPE NOTE (deliberate divergence from ADR-015)
-- ----------------------------------------------
-- ADR-015 scoped the offload to the INDUS footprint and explicitly kept the
-- pilot-model rows (nomic, specter2, specter3) and their indexes. What actually
-- executed dropped the whole table, taking the pilots with it. This file records
-- what happened, not what ADR-015 planned. The divergence is called out in the
-- audit; ADR-015 should be updated to Superseded/Executed-as-modified by whoever
-- owns it.
--
-- REPLACES: migration 070, which can no longer run (its CREATE TRIGGER targets
-- paper_embeddings). 070 was applied and recorded in scix_test; it is marked
-- superseded in the ledger below rather than deleted, because editing or removing
-- an already-applied migration is its own integrity break.
--
-- LOCKING: DROP TABLE takes AccessExclusiveLock, but only against objects that no
-- longer exist in prod, so nothing is locked. Idempotent throughout (IF EXISTS).
--
-- ORDER MATTERS, and not in the obvious direction. paper_embeddings must go
-- FIRST. On a database where migration 070 *was* applied (i.e. scix_test),
-- trg_embedding_outbox sits on paper_embeddings and depends on
-- embedding_outbox_enqueue(); dropping the function first fails with
-- "cannot drop function ... because other objects depend on it". Dropping the
-- table first removes the trigger with it — along with every index on the table
-- (idx_embed_hnsw_indus, idx_embed_hnsw_indus_hv, idx_embed_hnsw_nomic,
-- idx_embed_hnsw_specter2) — after which the function drops cleanly. No CASCADE
-- is needed or wanted: CASCADE would hide exactly this kind of dependency.
-- (Verified by replaying this file against a throwaway database carrying the
-- migration-070 objects; the wrong order was caught there, not in prod.)
--
-- APPLY WITH (a human must run this; it has NOT been executed):
--   psql -d scix -v ON_ERROR_STOP=1 -f migrations/074_record_paper_embeddings_retirement.sql
--   python scripts/verify_migration_ledger.py --dsn "dbname=scix"

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. The staging table — first, so its trigger goes with it
-- ---------------------------------------------------------------------------
-- The dense lane serves from Qdrant scix_indus_v2_papers_s1 (ADR-013) and
-- unembedded-detection anti-joins indus_qdrant_synced (migration 072), so
-- nothing reads this table. scix.embed writes straight to Qdrant since s7cy.
-- This also removes trg_embedding_outbox, unblocking the function drop below.
DROP TABLE IF EXISTS paper_embeddings;

-- ---------------------------------------------------------------------------
-- 2. Migration 070's outbox apparatus
-- ---------------------------------------------------------------------------
-- Safe now that no trigger references the function. The queue table has no
-- dependants of its own.
DROP FUNCTION IF EXISTS embedding_outbox_enqueue();
DROP TABLE IF EXISTS embedding_outbox;

-- ---------------------------------------------------------------------------
-- 3. Record
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version, applied_at, filename, note)
VALUES (
    74,
    TIMESTAMPTZ '2026-07-14 00:00:00-04',
    '074_record_paper_embeddings_retirement.sql',
    'Retroactive record of the untracked paper_embeddings drop executed ~'
    '2026-07-14 (bead s7cy) with no migration file, no ledger row and no log '
    'entry. applied_at is the approximate date of the original DDL, not of this '
    'file. Supersedes migration 070, which is unrunnable now that its trigger '
    'target is gone. Diverges from ADR-015 scope: the pilot-model rows and '
    'indexes were dropped too. Authored 2026-07-27, bead crz; evidence in '
    'docs/ops/migration_ledger_reconciliation_2026-07-27.md.'
)
ON CONFLICT (version) DO NOTHING;

-- The version-70 tombstone. Two databases reach this statement in different
-- states and the row has to be truthful in both:
--
--   * `scix`      — no version-70 row at all; 070 never ran. The INSERT fires.
--   * `scix_test` — 070 genuinely ran and self-recorded a plain, note-less row.
--     Section 2 above has just dropped the objects that row vouches for, so
--     leaving it untouched would leave behind a bare "applied" claim whose
--     post-condition no longer holds — verify_migration_ledger.py reports that
--     as DIVERGENT, correctly. DO UPDATE replaces it with this tombstone.
--
-- The update is guarded on `note IS NULL` so it can only overwrite an
-- unannotated row: if a human has already written provenance here, theirs wins.
-- applied_at is deliberately left alone by the update — on a database where 070
-- really ran, the original apply time is a fact worth keeping.
INSERT INTO schema_migrations (version, applied_at, filename, note)
VALUES (
    70,
    TIMESTAMPTZ '2026-07-14 00:00:00-04',
    '070_embedding_outbox.sql',
    'SUPERSEDED, NOT IN FORCE. Do NOT read this row as "applied": it records '
    'that migration 070''s objects (embedding_outbox, idx_embedding_outbox_drain, '
    'embedding_outbox_enqueue(), trg_embedding_outbox) are intentionally absent. '
    'In production the migration never ran at all — daily_sync Step 7 logged '
    '`relation "embedding_outbox" does not exist` on 2026-06-21/22/23 and 06-29, '
    'and no successful outbox drain appears anywhere in the log history. On a '
    'database where it did run (scix_test), migration 074 dropped those objects '
    'above. Either way the PG->Qdrant forward-write guarantee it promised is not '
    'in force, and the file is unrunnable: its CREATE TRIGGER targets '
    'paper_embeddings, dropped 2026-07-14. Recorded by migration 074 so a future '
    'runner does not attempt it.'
)
ON CONFLICT (version) DO UPDATE
    SET filename = EXCLUDED.filename,
        note = EXCLUDED.note
  WHERE schema_migrations.note IS NULL;

COMMIT;
