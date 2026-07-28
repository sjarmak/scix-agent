# Migration ledger reconciliation, 069–072

**Date:** 2026-07-27
**Bead:** `scix_experiments-crz` (GOAL.md W8, acceptance criterion A12)
**Databases audited:** `scix` (production, local), `scix_test` (schema-only test DB)
**Method:** read-only — `to_regclass`, `information_schema.columns`, `pg_class`,
`pg_trigger`, `pg_proc`, `pg_views`, `TABLESAMPLE`, plus `logs/daily_sync.log*`
and `git log`. No DDL, no INSERT, no DELETE was executed against either database.

---

## 1. Headline findings

**Three of the four migrations 069–072 are unrecorded, and only one of them is
actually applied to production.** `select max(version) from schema_migrations`
returns 68 in `scix`; the true state is that 072 is applied, 071's post-condition
holds by supersession, and 069 and 070 were never applied at all.

Two findings are more serious than the missing ledger rows and are stated first.

### F1 (CRITICAL) — Migration 069 is *partially applied*: the code half shipped, the schema half did not

`papers.raw` still exists in production. The ingest code that populated it does not.

| Half | State | Evidence |
|---|---|---|
| Code cutover | **shipped 2026-06-10** | commit `7d6a131` removed `"raw"` from `field_mapping.COLUMN_ORDER` and deleted the `row["raw"] = json.dumps(raw_fields)` assignment, replacing it with the comment *"the raw catch-all column is gone"* |
| Migration 069 | **never applied** | `information_schema.columns` still reports `papers.raw jsonb` (73 columns present, code writes 67) |

The consequence is a half-populated zombie column:

```
-- TABLESAMPLE SYSTEM (0.05) on papers
sampled = 17,602    raw IS NOT NULL = 16,114    → ~91.5% of rows still carry raw
-- rows ingested since 2026-06-10 all have raw = NULL:
select entry_date, raw is null from papers where entry_date >= '2026-07-24' → all true
```

So `papers.raw` is neither present-and-maintained nor absent. It is a silently
frozen column: 91.5% legacy JSONB, 8.5% NULL, with no writer. Any consumer that
treats `raw IS NULL` as "this field was absent upstream" is now wrong for every
paper ingested in the last seven weeks.

This is **the same failure shape as the s7cy incident, inverted**. In s7cy the
schema change landed without the code cutover. Here the code cutover landed
without the schema change — and it landed inside a commit titled
*"chore: PEP 8 lint cleanup + ruff/black enforcement infra"*, which is why nobody
saw it. The commit that authored migration 069 (`a87d56b`, 2026-06-14) came
**four days after** the code that assumed it had already run.

Disk stake: `papers` is 414 GB total — 59 GB heap, **297 GB TOAST**, 59 GB
indexes. The TOAST bulk is `raw`. ADR-011's stated precondition is satisfied
(`/mnt/scix_offload/papers_raw_archive/papers_raw_2026-06-09.jsonl.zst`, 5.4 GB
compressed, exists), and no live code path reads `papers.raw` — the only two
grep hits are a docstring in `scripts/metadata_coverage.py` and an unrelated
local variable in `scripts/canary_ner.py`. ADR-011 is still **Status: Proposed**,
so applying 069 remains an operator decision, not a cleanup task.

### F2 (CRITICAL) — Migration 070 was never applied to production, but its consumer was deployed and ran against the missing table for five weeks

`embedding_outbox`, `idx_embedding_outbox_drain`, `embedding_outbox_enqueue()`
and `trg_embedding_outbox` are all absent from `scix`, and no `schema_migrations`
row for version 70 exists. That is not the s7cy retirement removing them — they
were never there:

```
logs/daily_sync.log
[2026-06-21T10:18:20Z] Step 7/7: Syncing new embeddings to Qdrant (outbox drain)...
2026-06-21 06:18:21,313 ERROR qdrant_outbox_sync: outbox sync failed:
    relation "embedding_outbox" does not exist
```

The same error on 2026-06-22, 2026-06-23 and 2026-06-29. Across both
`daily_sync.log` and `daily_sync.log.1` there is **not one successful outbox
drain, ever**. Commit `4245300` shipped `scripts/qdrant_outbox_sync.py` and
wired it in as daily_sync Step 7; the table it drains was only ever created in
`scix_test`.

Two implications:

1. The PG→Qdrant forward-write guarantee that migration 070 and PRD MH-9 promised
   **never existed in production**. Between the ADR-013 cutover and the s7cy
   direct-to-Qdrant rewrite, nothing propagated `paper_embeddings` writes to
   Qdrant. (The s7cy watermark seed from the live Qdrant collection fixed this
   forward; it is noted here because the ADR/PRD record still describes a
   guarantee that was not in force.)
2. Migration 070 **can no longer be run as written**. Its `CREATE TRIGGER
   trg_embedding_outbox ON paper_embeddings` will fail — `paper_embeddings` no
   longer exists — and because the file is wrapped in `BEGIN`/`COMMIT`, the whole
   migration rolls back. It is dead as a forward migration.

Note also that `git log` records the s7cy commit message asserting *"the
migration-070 PG-to-Qdrant outbox are gone (ADR-015)"*. In production it was
never present to be gone.

### F3 (HIGH) — `paper_embeddings` was dropped in production with no migration file, no ledger row, and no log entry

`paper_embeddings` is absent from `scix`. Nothing in `migrations/` drops it.
Nothing in `schema_migrations` records it. Nothing in `logs/` mentions it.
`~/.psql_history` has no matching statement. The only record of the drop is prose
in the commit message of `de0e006` and in `docs/ADR/015`, which is still marked
**Status: Proposed (artifacts authored 2026-06-22; NOT executed)**.

This is the untracked destructive DDL at the centre of the s7cy incident. It is
what §4 of this document proposes to record retroactively.

### F4 (HIGH) — `scix` and `scix_test` have diverged in *opposite directions* on all four migrations

`scix_test` is documented in CLAUDE.md as "the full schema (all migrations
applied), no data". It is not.

| Object | `scix` (prod) | `scix_test` |
|---|---|---|
| `max(version)` in `schema_migrations` | 68 (58 rows) | 70 (**42 rows**) |
| `papers.raw` | **present** | absent |
| `paper_embeddings` | **absent** | present |
| `embedding_outbox` + index + fn + trigger | **absent** | present |
| `idx_embed_hnsw_indus`, `idx_embed_hnsw_indus_hv` | absent | **present** |
| `indus_qdrant_synced` | present (35,463,731 rows) | **absent** |

Every row differs. `scix_test`'s ledger holds 42 rows — versions 1–39, 44, 56,
59 and 70 — written between 2026-04-13 and 2026-06-11, so it is a *stale* record
rather than an empty one. Its top row, version 70, comes from migration 070's own
self-recording `INSERT`. Self-recording is rare but not unique to 070: `056`,
`059`, `063`, `064` and `070` each insert their own ledger row; every other file
in `migrations/` relies on the runner to do it, which is why the ledger drifts.

This directly threatens acceptance criterion **A13** ("suite green"): the tests
added by `de0e006` (`tests/test_embed_qdrant_store.py`,
`tests/test_qdrant_dense.py`) target a schema in which `indus_qdrant_synced`
exists and `paper_embeddings` does not. `scix_test` is the exact inverse.
Convergence steps are in §5.

---

## 2. Per-migration reconciliation table (production `scix`)

| Mig | What the file does | What `scix` shows | Verdict |
|---|---|---|---|
| **069**<br>`069_drop_papers_raw.sql` | `ALTER TABLE papers DROP COLUMN IF EXISTS raw;` then `ANALYZE papers;` | `papers.raw jsonb` **present and nullable**; ~91.5% of rows non-NULL (TABLESAMPLE 0.05%: 16,114 / 17,602); every row ingested since 2026-06-10 is NULL; 297 GB TOAST attached | **NOT APPLIED** — and *partially applied* in the cross-layer sense: its companion code cutover shipped on 2026-06-10 (F1) |
| **070**<br>`070_embedding_outbox.sql` | Creates `embedding_outbox` table, `idx_embedding_outbox_drain`, `embedding_outbox_enqueue()`, `trg_embedding_outbox` on `paper_embeddings`; self-inserts version 70 | All four objects **absent**; version 70 **absent** from `schema_migrations`; `daily_sync` Step 7 logged `relation "embedding_outbox" does not exist` on 4 dates; zero successful drains in log history | **NOT APPLIED — and now UNRUNNABLE** (F2). Its `CREATE TRIGGER ... ON paper_embeddings` cannot succeed; the file must be superseded, not replayed |
| **071**<br>`071_drop_paper_embeddings_indus_indexes.sql` | `DROP INDEX CONCURRENTLY IF EXISTS idx_embed_hnsw_indus_hv;` and `... idx_embed_hnsw_indus;` | Both indexes **absent** — but so is their parent table `paper_embeddings`. No log, history, or ledger evidence that this file was ever executed. Both indexes are still **present in `scix_test`** | **POST-CONDITION SATISFIED BY SUPERSESSION.** The named objects are gone, but because of the untracked `paper_embeddings` drop (F3), not because this migration ran. Re-running it is a proven-safe no-op (`IF EXISTS` on a nonexistent index is a `NOTICE`, not an error) |
| **072**<br>`072_indus_qdrant_synced.sql` | `CREATE TABLE IF NOT EXISTS indus_qdrant_synced (bibcode TEXT PRIMARY KEY, synced_at TIMESTAMPTZ NOT NULL DEFAULT now())` + `COMMENT ON TABLE` | Table present. Live DDL byte-matches the file: `bibcode text NOT NULL`, `synced_at timestamptz NOT NULL DEFAULT now()`, PK `indus_qdrant_synced_pkey` on `bibcode`. Table comment matches the file's `COMMENT ON TABLE` exactly. 35,463,731 rows | **APPLIED** (2026-07-14 per bead s7cy; seed confirmed by row count) |

### Objects verified absent-and-orphan-free

No view, materialised view, or function in `scix` references `paper_embeddings`
or `embedding_outbox`. The only surviving `%embed%` relations are
`section_embeddings`, `section_embeddings_pkey` and
`idx_section_embeddings_hnsw` — all migration 061, unrelated. The pilot indexes
`idx_embed_hnsw_nomic` and `idx_embed_hnsw_specter2` are also gone (they went
with the table), which means the ADR-015 "pilots are out of scope" constraint was
**not** honoured by whatever executed the drop. `halfvec_backfill_progress`
(migration 053) survives as a now-orphaned artifact.

---

## 3. Pre-existing ledger gaps (context, not in scope)

`schema_migrations` in `scix` holds 58 rows with `max(version) = 68`. Versions
**9, 10, 12, 14, 17, 18, 23, 27, 53, 54** are also missing from the range 1–68.
This document does not reconcile those: they predate the audit window, several
correspond to files that no longer exist under their original names, and
resolving them needs its own evidence pass. They are recorded here so that
`max(version)` is understood to be an unreliable watermark **even below 69**, and
so that a future auto-runner is not built on the assumption that the ledger is
contiguous.

---

## 4. Reconciling writes — authored, NOT executed

Two SQL files carry the reconciliation. **Neither has been run. A human must run
them.** Both are read-safe to inspect and neither touches row data.

### `migrations/073_reconcile_schema_migrations.sql`

* Adds a nullable `note TEXT` column to `schema_migrations`. This is the change
  that lets the ledger stop lying: without it, migration 071 has no truthful
  representation — recording it plainly asserts the file ran (false), and
  omitting it asserts the indexes still exist (also false). Adding a nullable
  column with no default is a catalog-only change; no table rewrite.
* Records **071** with a note stating the post-condition holds by supersession
  and the file itself was never executed.
* Records **072** with its verification evidence.
* Records **073** itself, generalising the self-recording convention that only
  `056`, `059`, `063`, `064` and `070` previously followed.
* Records **nothing** for 069 or 070. Both are genuinely not applied to `scix`;
  inserting them as applied would be exactly the falsehood this bead exists to
  remove. (070 does get a row later, from 074 — but as an explicit `SUPERSEDED,
  NOT IN FORCE` tombstone, not an applied-claim. See below.)

### `migrations/074_record_paper_embeddings_retirement.sql`

Retroactively records the untracked s7cy drop (F3) so that a replay of the
migration chain reproduces production rather than a schema production has not had
since 2026-07-14. Every statement is `IF EXISTS`, and **all of them are no-ops
against `scix` today** — the file's purpose is to make the history replayable and
to close the "destructive DDL with no migration file" hole.

It also writes a **tombstone row for version 70**, whose note begins
`SUPERSEDED, NOT IN FORCE` and states that migration 070's objects are
intentionally absent. The row exists only so a future auto-runner does not
attempt a migration that cannot succeed. `verify_migration_ledger.py` treats a
note carrying that marker as an inverted claim and will fail if the objects ever
*do* appear — so the tombstone cannot quietly decay into a false applied-claim.
The alternative, recording 070 as plainly applied, would have been the same
category of lie this document is fixing.

The marker says "not in force" rather than "not applied" because the row must be
truthful on both databases. In `scix` the migration never ran. In `scix_test` it
*did* run and self-recorded a plain, note-less row — and section 2 of 074 then
drops the objects that row vouches for, which would leave a bare applied-claim
whose post-condition no longer holds. So the version-70 insert uses
`ON CONFLICT (version) DO UPDATE … WHERE schema_migrations.note IS NULL`: it
replaces an *unannotated* row with the tombstone, leaves any row a human has
already annotated alone, and never touches `applied_at` (on a database where 070
really ran, the original apply time is worth keeping). What is true in both
databases, and what the marker asserts, is that the objects are intentionally
gone and the PG→Qdrant guarantee is not in force.

Run 073 before 074. Verify the no-op precondition first:

```bash
psql -d scix -tAc "select coalesce(to_regclass('public.paper_embeddings')::text,'ABSENT'),
                          coalesce(to_regclass('public.embedding_outbox')::text,'ABSENT')"
# must print: ABSENT|ABSENT   — if not, STOP and re-audit; 074 would be destructive.

psql -d scix -v ON_ERROR_STOP=1 -f migrations/073_reconcile_schema_migrations.sql
psql -d scix -v ON_ERROR_STOP=1 -f migrations/074_record_paper_embeddings_retirement.sql
python scripts/verify_migration_ledger.py --dsn "dbname=scix"
```

### How 073 and 074 were validated without touching `scix`

Both files were replayed against **two** throwaway databases (created and dropped
inside this audit; no other database was written), because the two shapes fail in
different ways and validating only one hides the other:

| Throwaway DB | Seeded to look like | Result after 073 + 074 |
|---|---|---|
| `crz_fix_prodshape` | `scix`: ledger 1–68 (with the §3 gaps), `papers.raw` present, `indus_qdrant_synced` present, no `paper_embeddings` / `embedding_outbox` | `verify_migration_ledger.py` → **exit 0**, "Ledger agrees with the catalog for migrations 069-074" |
| `crz_fix_testshape` | `scix_test`: ledger rows 1–39, 44, 56, 59 **and 70** (note-less, as 070's self-record leaves it), `papers.raw` absent, `paper_embeddings` + both indus indexes + the whole 070 outbox apparatus present, `indus_qdrant_synced` absent | after the §5 order (072, 073, 074) → exit 1 with **069 as the only divergence**, which §5 caveat 2 explains |

The second shape is the one that matters and it was got wrong the first time: an
earlier validation seeded the 070 *objects* but not 070's own `schema_migrations`
row. With that row present, `ON CONFLICT (version) DO NOTHING` preserved a plain,
note-less version-70 claim while 074 dropped the objects underneath it, and the
verifier reported version 70 `DIVERGENT` — on exactly the database the file was
described as validated against. That is what the `DO UPDATE … WHERE note IS NULL`
above fixes, and it was found by seeding the harder shape faithfully rather than
approximately.

Replaying also caught a real bug in the first draft of 074: it dropped
`embedding_outbox_enqueue()` before `paper_embeddings`, and Postgres refused —
`cannot drop function embedding_outbox_enqueue() because other objects depend on
it (trigger trg_embedding_outbox on table paper_embeddings)`. On production, where
both objects are already gone, that ordering error would have been invisible; on
`scix_test` it would have aborted the migration. The file now drops the table
first so the trigger goes with it. **This is the argument for the file existing
at all**: the original untracked drop was never replayable, so this defect had
nowhere to surface.

Post-fix the replay is clean and idempotent on both shapes: every file re-runs
with no error and no duplicate rows.

### Deliberately not proposed here

* **Applying 069.** ADR-011 is Status: Proposed. Dropping a 297 GB TOAST payload
  is an operator decision with its own sign-off, not a ledger fix. What this
  audit adds is that the code half already shipped, so the decision is now
  "finish it or revert the code", not "should we start".
* **Deleting migration 070's file.** It is applied and recorded in `scix_test`;
  deleting or editing an applied migration is its own integrity break. 074's
  ledger note marks it superseded instead.

---

## 5. `scix_test` convergence (needed for A13)

`scix_test` already satisfies 069 (`papers.raw` is absent there) and has 070
applied and recorded. What it lacks is 072 — the table the new tests need — plus
the 074 retirement that makes it stop carrying `paper_embeddings`. Suggested
order, for a human to run:

```bash
psql -d scix_test -v ON_ERROR_STOP=1 -f migrations/072_indus_qdrant_synced.sql
psql -d scix_test -v ON_ERROR_STOP=1 -f migrations/073_reconcile_schema_migrations.sql
psql -d scix_test -v ON_ERROR_STOP=1 -f migrations/074_record_paper_embeddings_retirement.sql
python scripts/verify_migration_ledger.py --dsn "dbname=scix_test"
```

Replayed against a faithful copy of the `scix_test` shape, that sequence leaves
**069 as the only divergence** (verified; see the validation table in §4).

Two caveats, both requiring a human decision rather than an assumption:

1. **073 and 074 encode production's provenance.** Their note text describes what
   happened to `scix`, and the 071 and 072 notes will carry prod's story into
   `scix_test` verbatim. The one place this would have asserted something false
   *there* — the version-70 row — is handled: the tombstone is worded for both
   databases and only overwrites an unannotated row (see §4). Acceptable for a
   disposable schema-only database; not something to do silently.
2. **069 will still report divergent on `scix_test` afterwards** — its
   post-condition holds (`papers.raw` is absent there) but no ledger row exists,
   and 073 deliberately does not write one because 069 is *not* applied in prod.
   Backfilling it is a one-line `INSERT` a human can make once they have decided
   whether `scix_test`'s ledger is worth maintaining at all:

   ```sql
   INSERT INTO schema_migrations (version, filename, note)
   VALUES (69, '069_drop_papers_raw.sql',
           'Applied to scix_test only; papers.raw is still present in production.')
   ON CONFLICT (version) DO NOTHING;
   ```

`scix_test`'s ledger will remain sparse below version 69 regardless: it holds 42
rows for versions 1–39, 44, 56, 59 and 70, last written 2026-06-11. This audit
does not backfill it; see §3.

---

## 6. Verification

`scripts/verify_migration_ledger.py` re-runs every check in §2 read-only and
exits non-zero on any divergence between what `schema_migrations` claims and what
the catalog shows. It is the command that proves acceptance criterion A12.

```bash
python scripts/verify_migration_ledger.py --dsn "dbname=scix"
```

Actual output against `scix` on 2026-07-27, before 073/074 are run:

```
  [ok  ] 69  069_drop_papers_raw.sql                      ok (unrecorded, not applied)
  [ok  ] 70  070_embedding_outbox.sql                     ok (unrecorded, not applied)
  [FAIL] 71  071_drop_paper_embeddings_indus_indexes.sql  DIVERGENT: applied in DB but absent from ledger
  [FAIL] 72  072_indus_qdrant_synced.sql                  DIVERGENT: applied in DB but absent from ledger
  [ok  ] 73  073_reconcile_schema_migrations.sql          ok (unrecorded, not applied)
  [FAIL] 74  074_record_paper_embeddings_retirement.sql   DIVERGENT: applied in DB but absent from ledger
3 divergence(s)   → exit 1
```

069 and 070 are reported `ok` because the ledger and the catalog *agree*: both
say "not applied". That agreement is the point — A12 asks whether the ledger is
truthful, not whether every migration has been run. F1 and F2 remain open
questions for an operator; they are not ledger defects.

After a human runs 073 and 074 against `scix` the script exits 0. That was
confirmed by replay against a throwaway database seeded to production's shape,
not asserted from reading the SQL — and separately against a `scix_test`-shaped
database, where the residual divergence is 069 alone (§4, §5).
