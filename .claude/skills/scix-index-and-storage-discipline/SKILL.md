---
name: scix-index-and-storage-discipline
description: >
  How index and storage changes are made safely in SciX. Load BEFORE building,
  rebuilding, dropping, or benchmarking any Postgres index (HNSW, GIN, DiskANN,
  expression indexes); before any halfvec/quantization change; before
  reclaiming disk (DROP INDEX vs DELETE vs DROP TABLE, VACUUM FULL, pg_repack);
  before placing new data on DS (local NVMe) vs NAS (/mnt, NFS); and when
  working the ADR-015/016 reclamation line or the dqfe quantization spike.
  Covers the four don't-trust-a-new-index rules (bead 12rp, the 56-hour DiskANN
  loss), the halfvec shadow-column cutover and its planner traps, the
  disk-at-99% crisis, and never-live-write-on-NAS. NOT for Qdrant serving
  (scix-vector-serving-qdrant), the embed pipeline (scix-embedding-pipeline),
  DSN guards (scix-db-safety-and-telemetry), or change approval
  (scix-change-control).
---

# SciX index and storage discipline

Date-stamped facts below are as of **2026-07-07**. This skill teaches the
committed reality at HEAD; anything only in the uncommitted working tree is
labeled in-flight.

## Vocabulary (defined once)

| Term                        | Meaning here                                                                                                                                               |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DS**                      | The local serving disk: `/dev/nvme1n1p2`, 1.9 TB NVMe, mounted at `/`. Everything latency-critical lives here.                                             |
| **NAS**                     | The NFS mount at `/mnt` (50 TB, `192.168.1.86:/var/nfs/shared/...`). Backup/archive/cold tier only.                                                        |
| **pgvector**                | Postgres extension providing the `vector`/`halfvec` types and HNSW indexes. Version 0.8.2 in prod.                                                         |
| **halfvec**                 | pgvector's float16 vector type: 2 bytes/dim instead of 4. The project's only approved lossy storage quantization.                                          |
| **HNSW**                    | Graph-based approximate-nearest-neighbor index type. Build cost scales with graph memory: ~2.07 KB/tuple at `m=16`.                                        |
| **pgvectorscale / DiskANN** | The extension and index type that failed catastrophically here in June 2026 (see rule history). Settled-closed, not a live option.                         |
| **expression index**        | An index on a computed expression like `((embedding)::vector(768))` rather than a bare column. pgvectorscale cannot scan these.                            |
| **opclass**                 | The operator class an index is built with (`vector_cosine_ops` vs `halfvec_cosine_ops`). Part of the DDL identity; changing it changes what was validated. |
| **soak**                    | A mandated waiting period (30 days post-cutover) during which the old serving path is kept intact as rollback.                                             |
| **seal**                    | ADR-016's operation: move a closed year's full-text to read-only NAS shards and remove it from Postgres.                                                   |
| **TOAST**                   | Postgres's out-of-line storage for large column values. Co-TOASTed columns share one relation, which constrains partial reclaim (see ADR-016 A1).          |

Sibling map: dense-lane serving internals → `scix-vector-serving-qdrant`;
embed/outbox/daily_sync → `scix-embedding-pipeline`; DSN and prod-write guards
→ `scix-db-safety-and-telemetry`; approval gates and who signs off →
`scix-change-control`; running anything heavy → `scix-memory-and-batch-discipline`.

---

## 1. The four "don't trust a new index" rules

Source of authority: `docs/ADR/013_dense_lane_qdrant.md` lines 89–100
("Validation rules this failure bought — binding for future index/lane work"),
mirrored in the repo `CLAUDE.md` Don't-list. These are binding, not advisory.

1. **No index is trusted until one query has returned from it.** Any new
   index type or config gets a ≤50k-row scratch build plus a
   forced-index-scan smoke test _before_ any multi-hour build.
2. **Benchmark DDL must be byte-identical to production DDL** (opclass,
   column type, storage params), or the benchmark is treated as fiction.
3. **Never drop a serving index before its replacement is validated
   scannable.** Disk pressure is solved some other way first.
4. **Architecture pins are re-examined when they start selecting immature
   components.** "The only option that preserves the pin" is a trigger to
   question the pin, not to accept the option.

### The failure that bought them (bead 12rp + ADR-013, June 2026)

Every link verified in-session at the time; chronicle preserved so nobody
re-fights it:

- `paper_embeddings.embedding` was a _dimensionless_ `vector` column
  (multi-model table design). DiskANN requires a fixed dimension, so the
  prod rebuild script cast `((embedding)::vector(768))`, producing a true
  **expression index that pgvectorscale cannot scan**
  (`assertion failed: attnum > 0`, reproducible at 50k rows in seconds).
- Meanwhile the _validated_ benchmark variant used
  `USING diskann (embedding halfvec_cosine_ops)` on a properly typed pilot
  table. Prod built `vector_cosine_ops` on float32. The go/no-go numbers
  described a configuration production never ran. That DDL divergence is
  bead `scix_experiments-12rp` (P1 bug, CLOSED).
- No scratch-index query was run before committing to the 56-hour build.
  The index finished `valid=t` and **crashed every scan**. Worse than
  useless: the planner auto-selected it, so it would have broken all dense
  queries, not merely failed to help.
- The serving HNSW index had been dropped **before** the replacement was
  validated → **~2 weeks of lexical-only serving**. The measured cost is in
  `results/litreview_rerun_2026-06-11.md` (two literature reviews materially
  changed by the missing dense lane).
- Outcome: the dense lane left pgvector entirely and now serves from Qdrant
  (ADR-013). The pgvectorscale line is **settled-closed** (12rp CLOSED); do
  not reopen it without a new ADR.

### The scratch-build smoke test (rule 1, executable recipe)

Run this pattern before ANY multi-hour index build. It is a write to a
scratch table, so it needs a test DSN or explicit prod sign-off per
`scix-db-safety-and-telemetry` and `scix-change-control`. **Do not run
casually against prod**; shown here with its guards:

```bash
# Heavy-ish (builds a 50k index): wrap in scix-batch on this host.
scix-batch psql "$SCIX_TEST_DSN" -v ON_ERROR_STOP=1 <<'SQL'
-- 1. Scratch table with the SAME column type as prod (not a convenient one).
CREATE TABLE scratch_idx_smoke AS
  SELECT * FROM the_real_table LIMIT 50000;   -- ≤50k rows, rule 1

-- 2. Index DDL BYTE-IDENTICAL to the intended prod DDL (rule 2):
--    same index type, same opclass, same expression/bare-column shape,
--    same WITH params, same WHERE predicate.
CREATE INDEX scratch_idx ON scratch_idx_smoke USING hnsw (embedding_col halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);

-- 3. Force the planner onto the index and require a row back.
SET enable_seqscan = off;
EXPLAIN ANALYZE
  SELECT id FROM scratch_idx_smoke
   ORDER BY embedding_col <=> (SELECT embedding_col FROM scratch_idx_smoke LIMIT 1)
   LIMIT 5;
-- PASS only if the plan shows an Index Scan on scratch_idx AND rows returned.
-- "Index created without error" is NOT a pass. valid=t is NOT a pass.

DROP TABLE scratch_idx_smoke;
SQL
```

Checklist before any multi-hour index build:

- [ ] Scratch build at ≤50k rows completed AND a forced-index-scan query
      returned rows from it.
- [ ] The scratch DDL is byte-identical to the prod DDL you will run
      (diff the two statements; do not eyeball).
- [ ] The column type in the scratch table matches prod (a dimensionless
      `vector` column and a `vector(768)` column are different animals;
      that difference caused the 56-hour loss).
- [ ] The benchmark that justified this build used this exact DDL. If not,
      the benchmark is fiction (rule 2); redo it.
- [ ] The old serving index still exists and will not be dropped until the
      new one has served a validated query (rule 3).
- [ ] Build command is wrapped in `scix-batch` with explicit memory bounds
      (this installation's host also runs the Gas City supervisor; see
      `scix-memory-and-batch-discipline`).
- [ ] `CREATE INDEX CONCURRENTLY` is run with autocommit (it cannot run in
      a transaction block); a midway failure leaves an INVALID index that
      must be `DROP INDEX CONCURRENTLY`-ed before retrying (migration 054
      header documents this).

---

## 2. Halfvec cutover mechanics (the shadow-column pattern)

The halfvec migration (migrations 053/054, bead `scix_experiments-0vy`,
runbook `docs/runbooks/halfvec_migration.md`, outcome
`docs/runbooks/halfvec_migration_outcome.md`) is this project's canonical
online column-type cutover. The pattern generalizes to any "change the
storage type of a huge hot column" task.

**Status note (2026-07-07):** the `paper_embeddings` table this migration
targeted was dropped in prod ~2026-06-29/30 (bead `s7cy`, P1 OPEN — see
`scix-embedding-pipeline`), so the concrete objects below are historical.
The _pattern_ and its traps are the durable knowledge. The dense lane now
serves from Qdrant (ADR-013).

### The pattern, in order

1. **Shadow column, not ALTER TYPE.**
   `ALTER TABLE ... ADD COLUMN IF NOT EXISTS embedding_hv halfvec(768)` —
   nullable, so the ALTER is metadata-only (no rewrite, no lock scan).
   The rejected alternative, `ALTER COLUMN ... TYPE halfvec(768) USING ...`,
   holds ACCESS EXCLUSIVE for the entire rewrite (multi-hour on 32M rows /
   125 GB TOAST) and blocks every writer. Migration 053's header states
   this rationale.
2. **Out-of-band batched backfill with a persisted cursor.**
   `scripts/backfill_halfvec.py` batches by bibcode range and persists
   cursor + counts in `halfvec_backfill_progress` (created by 053), so the
   job is idempotent across OOM kills and scope restarts. Any backfill you
   write here gets the same restartable-cursor shape.
3. **`CREATE INDEX CONCURRENTLY` on the shadow column** (migration 054),
   autocommit, never inside BEGIN/COMMIT. The partial predicate
   intentionally matches the legacy index's shape (`WHERE
model_name='indus'`, no NOT NULL clause) so the planner can match
   index-to-query by lexical predicate equality. pgvector HNSW silently
   skips NULL rows at build time, so indexing a partially backfilled
   column is safe; the index grows as the backfill progresses.
4. **Query cutover behind an env flag.** `SCIX_USE_HALFVEC=1` routes INDUS
   dense SQL to `embedding_hv` (`src/scix/search.py:44`,
   `_HALFVEC_ENABLED`; default `0`). Flag flip only after the 50-query
   eval acceptance gate.
5. **Drop the legacy index/column in a later, separately gated migration**
   (rule 3 again). In real history the legacy drop happened out of order,
   which is part of why dense search went dark; see the outcome doc.

### The traps this migration paid for

- **The LHS-cast planner trap.** The new index is on the _bare_ column
  `embedding_hv`. Query code that wraps the column —
  `(pe.embedding_hv)::halfvec(768)` — defeats planner index matching:
  ~12 min parallel seq scan instead of sub-100 ms HNSW, verified via
  EXPLAIN. `src/scix/search.py:683–692` gates the cast on `use_halfvec`
  for exactly this reason, and
  `tests/test_halfvec_migration.py::test_indus_query_uses_embedding_hv_no_lhs_cast`
  pins it. General rule: the query's index-argument expression must match
  the index definition lexically; casts are part of that identity.
- **The HNSW-build ceiling at ~32M rows on this host.** Four independent
  build attempts (2026-04-29/30, table in the outcome doc) all stalled in
  pgvector's "graph no longer fits into maintenance_work_mem" spill cycle:
  ~2.07 KB/tuple at `m=16`, ~1.04 KB/tuple at `m=8`, versus 62 GB physical
  RAM. Attempt #2 (`mwm=16GB × 4 workers` = 64 GB) pushed 16 GB into swap.
  The constraint is structural (per-tuple graph memory × tuples > RAM),
  not tunable. Do not attempt a full-corpus pgvector HNSW build on this
  host expecting different results.
- **Size payoff when it works:** `results/halfvec_migration/sizes_after.txt`
  records the halfvec HNSW at 24 GB vs the float32 index's 120 GB (80%
  reduction, acceptance was ≥40%).

### Quantization policy (what is allowed to be lossy)

| Technique                                 | Status                        | Evidence / authority                                                                                                                                                                             |
| ----------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `halfvec` (float16) storage               | **Safe, approved**            | Migrations 053/054; CLAUDE.md "halfvec is the safe quantization"                                                                                                                                 |
| Scalar quantization INT8 (Qdrant serving) | **Approved, validated**       | `docs/prd/qdrant_nas_migration.md` MH-11; recall within 1% of f32 on all canary slices                                                                                                           |
| Binary quantization                       | **Banned for storage**        | CLAUDE.md: >40% nDCG@10 loss on scientific text; INDUS 768d is below Qdrant's 1024d BQ guidance (MH-11: "BQ NOT enabled"). Allowed only as a first-pass filter, never the stored representation. |
| 3–4 bit (TurboQuant)                      | **Open spike, unproven here** | Bead `scix_experiments-dqfe`, below                                                                                                                                                              |

### The dqfe quantization spike (open, candidate only)

Bead `scix_experiments-dqfe` (P2, OPEN, created 2026-07-03, no results yet):
evaluate the TurboQuant claim that 3–4 bit embeddings give ~5× memory
savings with no ranking loss ("search only cares which vector is closest"),
measured as ranking parity on this project's hybrid INDUS+BM25 RRF stack.
It extends the ADR-016 storage line and complements the ~493 GB offload.

Discipline for whoever picks it up:

- The claim is **unproven on this corpus**. The prior here is adverse:
  binary (1-bit) quantization lost >40% nDCG@10 on scientific text, which
  is why it is banned. 3–4 bit must be _measured_, not assumed, against
  the gold sets (`eval/` — see `scix-eval-and-evidence`).
- A parity result is evidence for a proposal, not a green light: any change
  to the stored representation of a serving lane is an ADR-pinned axis and
  goes through `scix-change-control`. PROVISIONAL pending Stephanie (Q5):
  treat quantization changes as HALT-branch-ready requiring her sign-off.
- Any candidate index built for the spike obeys the four rules in §1,
  including the 50k scratch + forced-scan test.

---

## 3. Disk crisis and reclamation (ADR-015 / ADR-016)

### Timeline you need to know (all date-stamped)

| Date          | Event                                                                                                                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-29/30 | Halfvec cutover; HNSW rebuild attempts stall; DS already at 94% during attempt #4                                                                                                        |
| 2026-06-11    | DiskANN 56 h build lost; Qdrant cutover (ADR-013); 30-day soak clock starts                                                                                                              |
| 2026-06-22    | ADR-015 authored (Proposed): staged INDUS offload, Stage 1 soak-gated to **2026-07-11**; DS ~98%, ≈42 GB free                                                                            |
| 2026-06-29    | ADR-016 accepted: DS at 99%, 25 GB free; scix DB 1198 GB; `papers_fulltext` alone 493 GB                                                                                                 |
| 2026-06-29/30 | **Out-of-process `DROP TABLE paper_embeddings`** in prod (bead `s7cy`, P1 OPEN) — NOT the ADR-015 path; see below                                                                        |
| 2026-06-30    | ADR-016 Phase 1 executed: `papers_fulltext` sealed to NAS, ~493 GB reclaimed; A3 records DS at 72% (532 GB free); body phases 2&3 cancelled                                              |
| 2026-07-07    | `df` today: DS **89% used, 212 GB free**. The ~320 GB regrowth since 06-30 is unexplained by any doc in the repo; treat free space as a monitored, scarce resource, not a solved problem |

### The reclamation physics (get this right before proposing anything)

- **`DROP INDEX` returns space to the OS immediately at commit.** No
  `VACUUM FULL`, no `pg_repack`, no free-space precondition. This is why
  ADR-015 Stage 1 (drop the two INDUS HNSW indexes, ~120 GB + the halfvec
  index) was the designated relief valve. Migration
  `071_drop_paper_embeddings_indus_indexes.sql` implements it with
  `DROP INDEX CONCURRENTLY IF EXISTS` (autocommit only, idempotent), with
  the operator preconditions in its header.
- **`DELETE`/`DROP COLUMN` do NOT return space to the OS.** They return
  pages to the free-space map. Shrinking the relation file requires
  `pg_repack` (online) or `VACUUM FULL` (exclusive), and either transiently
  needs scratch disk ≈ the live table size. That is the ADR-011 caveat,
  and it applies to row/column reclaim (ADR-015 Stage 2), never to index
  drops. Bead descriptions have confused these before; ADR-015 §Context
  point 3 is the corrective.
- **`DROP TABLE` frees table + TOAST + all indexes instantly**, but
  forfeits everything in it. ADR-015 reserved this (Stage 2b) behind a
  verified NAS archive, a quiesced outbox, and its own sign-off.
- **Co-TOASTed columns cannot be reclaimed independently.** ADR-016 A1:
  `papers_fulltext` was 493 GB = 4.5 GB heap + 29 GB indexes + 460 GB
  TOAST, and the TOAST held `sections` (~214 GB) AND `sections_tsv`
  (~217 GB) together. Keeping one while reclaiming the other means
  rewriting the whole 460 GB TOAST with ~220 GB scratch — impossible at
  25 GB free. Column-level reclaim plans must check TOAST co-location
  first.

### What ADR-016 teaches (the storage-decision patterns)

1. **Measure `idx_scan` before believing an index is load-bearing.** The
   `sections_tsv` GIN (27 GB) had `idx_scan=0` with `stats_reset` NULL and
   peer indexes at millions of scans, so the zero was real: a dead index
   backing an unexposed lane. It was dropped with its column. The body
   BM25 expression GIN had `idx_scan=1258`: alive, kept. Read-only check:

   ```sql
   -- Read-only. Requires a DB connection; on this installation that means
   -- prod — go through scix-db-safety-and-telemetry first.
   SELECT indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid))
     FROM pg_stat_user_indexes ORDER BY pg_relation_size(indexrelid) DESC LIMIT 20;
   -- Also check: SELECT stats_reset FROM pg_stat_database WHERE datname = current_database();
   -- idx_scan=0 is only meaningful if stats_reset is old/NULL and peers show traffic.
   ```

2. **When hot data is a small fraction, rebuild small instead of rewriting
   big.** Hot rows (≥2025) were 2.3% of `papers_fulltext`. So: seal cold
   years to NAS, `CREATE TABLE papers_fulltext_hot AS SELECT ... WHERE
year >= 2025` (~10 GB, fits in 25 GB free), one-transaction rename swap,
   verify reads on both tiers, then `DROP TABLE papers_fulltext_old`
   (~470 GB reclaimed instantly, zero scratch). The rejected alternative
   (keep 14.6M slim sealed stubs) was a 14.6M-row rewrite with heavy WAL
   on a near-full disk.
3. **A derived structure can be bigger than its source; check before
   materializing.** A3's measurement: raw `body` ~208 GB, but a stored
   `body_tsv` _with positions_ would be ~236 GB. So "materialize the
   tsvector, drop the prose" was net-NEGATIVE (+28 GB) plus a 32M-row
   rewrite, and phases 2&3 were **cancelled**. Body stays in PG; body-BM25
   keeps its expression GIN (39 GB). Do not resurrect phases 2/3 without
   new math.
4. **Validation gates mirror ADR-013 discipline** for any seal/reclaim:
   no PG deletion until the NAS shard has returned one read for a sampled
   bibcode; shard verified by row-count + per-bibcode checksum parity;
   content parity byte-equals on a sample; heavy steps under `scix-batch`
   with `max_parallel_workers_per_gather=0` and bounded `work_mem`.
   Tooling exists: `scripts/seal_fulltext_to_nas.py` (build/verify
   subcommands), `scripts/coldtext_swap_papers_fulltext.py`,
   `src/scix/coldtext/` (`HOT_WINDOW_START_YEAR = 2025` in
   `src/scix/coldtext/route.py:31`).

### The negative example: the out-of-process DROP TABLE (live fire)

ADR-015 authorized **artifacts only** (Stage 1 = index drops, soak-gated to
2026-07-11; Stage 2 = row reclaim behind a NAS archive, "NOT pre-authored").
Instead, a full `DROP TABLE paper_embeddings` was executed in prod
~2026-06-29/30, out of process, **without** the Stage-2 NAS archive and
without the companion code cutover. Consequences (bead `s7cy`, P1 OPEN):
`daily_sync.sh` aborts at Step 5 every run since 2026-06-30, new papers get
no dense vectors (gap grows ~1–3k/day), and the in-PG rollback source
ADR-013 relied on is gone. Committed HEAD `src/scix/embed.py` and
`scripts/embed_fast.py` still target the dropped table; a direct-to-Qdrant
remediation (`src/scix/qdrant_dense.py`, migration 072, a
`indus_qdrant_synced` watermark) exists but is **uncommitted in the working
tree — in-flight, not canon. PROVISIONAL pending Stephanie (Q2):** this
skill teaches committed reality; do not treat the working-tree fix as the
standard path. Full incident and remediation state:
`scix-embedding-pipeline`.

The lesson for THIS skill: the reclamation ladder (index drop → gated row
reclaim → gated table drop) exists precisely so that disk pressure never
justifies skipping a rung. It was skipped once; the cost is an open P1 and
a lost rollback source.

### Reclamation pre-flight (before dropping anything)

```bash
# Read-only pre-flight for the ADR-015 line. Prints PG↔Qdrant parity,
# per-index reclaimable bytes, NAS archive/snapshot presence, and the soak
# clock, ending in a READY / NOT READY verdict. Read-only but touches prod
# and loads a DB connection: run under scix-batch, per its own header.
scix-batch python scripts/audit_paper_embeddings.py --allow-prod
```

- [ ] Audit verdict READY (parity: derived cache fully covers the source).
- [ ] Soak clock satisfied, or a documented evidence-based early-retirement
      sign-off per `feedback_no_destructive_cleanup_without_evidence.md`
      (referenced by ADR-015 and `docs/prd/qdrant_nas_migration.md`).
- [ ] You know which reclamation physics applies (index drop = instant;
      row/column = repack + scratch; table = everything gone).
- [ ] Rollback path stated in writing, including how long it takes after
      the drop (ADR-015: Stage 1 degrades rollback from "instant env-flag
      flip" to "rebuild index ~45–90 min, then flip").
- [ ] The change is going through `scix-change-control`, not around it.

---

## 4. NAS-vs-DS placement rules

Authority: repo `CLAUDE.md` (Do-list, Don't-list), ADR-016 §Context,
`docs/prd/qdrant_nas_migration.md` "Headline correction". The load-bearing
rule, stated plainly:

> **Never run a live-write workload on NAS.** No Postgres data dir, no
> Qdrant `storage_path`/`wal_dir`, no SQLite, nothing that mmaps or holds
> locks, ever on `/mnt`.

Why (evidence, not vibes — PRD "Headline correction" section): Qdrant's own
docs state it won't work on NFS; v1.15+ ships a runtime FS-compatibility
check; GitHub issues #6135/#5065/#4145/#4862 document RocksDB lock failures
and snapshot-metadata corruption on NFS; NFS+mmap cache-coherency and
lock-recovery bugs are long documented (Red Hat #151284, SQLite "How To
Corrupt"). Failure mode is refuse-to-start at best, **silent corruption
later** at worst. The only block-safe way to put live data on the NAS would
be an iSCSI LUN, which this project explicitly declined (ADR-016 non-goals).

**Write-once/read-only files on NAS are fine.** That asymmetry is the whole
design of the cold tier.

### Placement decision table

| Data                                                                                                            | Lives on                                      | Why                                                                                     |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------- |
| Postgres cluster (the 1.2 TB `scix` DB)                                                                         | DS                                            | Live-write; NFS unsafe                                                                  |
| Qdrant runtime (`storage_path`, WAL) — collection `scix_indus_v2_papers_s1`                                     | DS (docker bind, local NVMe)                  | Live-write; ADR-013 says "never NAS"; ~83 GB per ADR-016 (dated 2026-06-29)             |
| Everything that powers _search_ (tsvectors, GIN indexes, metadata, citation graph, entity graph, dense vectors) | DS, all years                                 | ADR-016: search never touches NAS; no cross-tier RRF fanout                             |
| Qdrant snapshots                                                                                                | NAS (`/mnt/qdrant_snapshots`, container bind) | One-shot sequential write then read-only; `scripts/qdrant_snapshot_to_nas.sh`           |
| Sealed cold text (year ≤2024 `papers_fulltext` shards)                                                          | NAS `/mnt/scix_coldtext/v1/{year}/`           | Write-once, checksummed, read-only; ADR-016                                             |
| Raw ADS JSONL corpus mirror                                                                                     | NAS `/mnt/scix_offload/`                      | Backup duplicate; upstream ADS is the real DR tier                                      |
| Any _new_ dataset/artifact you create                                                                           | **DS by default**                             | CLAUDE.md: NAS only for (a) doesn't fit, (b) backup duplicate, (c) archival raw content |

Corollaries:

- A "move the 1.2 TB PG to NAS" idea resurfaces periodically (beads
  `7ysy`/`ymnv` territory). The answer is the rule above: not as a live
  data dir, ever. Tiering happens by _content_ (seal read-only years to
  NAS), not by relocating live storage.
- `read_paper`/`read_paper_section` on sealed years pay an NFS read,
  mitigated by a DS read-through cache; that is an accepted, human-paced
  cost (ADR-016 consequences). Search latency is unaffected because search
  structures never left DS.
- Check placement before writing, not after:

  ```bash
  df -h / /mnt                      # DS vs NAS free space, read-only
  findmnt -T /path/you/plan/to/use  # which filesystem a path actually is
  ```

---

## 5. When NOT to use this skill

- Configuring or debugging the Qdrant collection, payload schema, SQ-INT8
  serving params, or REST-vs-gRPC → `scix-vector-serving-qdrant`.
- The embed/ingest/outbox path, `daily_sync.sh`, or the s7cy incident's
  remediation state → `scix-embedding-pipeline`.
- DSN selection, `is_production_dsn`, `--allow-prod`, test-skip semantics →
  `scix-db-safety-and-telemetry`.
- Whether a change needs an ADR, a bead, or Stephanie's sign-off →
  `scix-change-control` (short answer for everything in this skill:
  ADR-pinned axis, yes it does).
- Sizing/wrapping a heavy job so it doesn't OOM the co-hosted supervisor →
  `scix-memory-and-batch-discipline`.
- Retrieval quality, RRF, gold-set evaluation → `scix-retrieval-architecture`
  and `scix-eval-and-evidence`.

---

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`
at HEAD `452ab86` (note: **not `main`**; working tree also carried the
uncommitted s7cy remediation, which this skill does not canonize).
Verification was source-reading only: no DB connections, no data scripts,
no docker. Live-state claims (whether migration 071 was ever applied as
written, current index inventory, Qdrant `points_count`, why DS regrew to
89% between 06-30 and 07-07) are unverified here and must be re-checked
against the live systems through the proper guards before being relied on.

One-line re-verification commands (all read-only):

```bash
git -C . branch --show-current && git rev-parse --short HEAD   # provenance pin
sed -n '89,101p' docs/ADR/013_dense_lane_qdrant.md             # the four rules, verbatim
grep -n "SCIX_USE_HALFVEC" src/scix/search.py | head -3        # halfvec gate still at search.py:44
sed -n '1,25p' migrations/071_drop_paper_embeddings_indus_indexes.sql  # Stage-1 drop + preconditions
grep -n "Amendment A3" docs/ADR/016_time_partitioned_cold_text_tier.md # body phases 2&3 still cancelled
bd show dqfe | head -5                                         # quantization spike still open?
bd show s7cy | head -5                                         # ingest fire still open?
git show HEAD:src/scix/embed.py | grep -c paper_embeddings     # >0 ⇒ committed embed path still targets the dropped table
df -h / /mnt                                                   # DS/NAS pressure, date-stamp any number you quote
ls /mnt/scix_coldtext/v1/ /mnt/qdrant_snapshots/ 2>/dev/null   # cold tier + snapshot dirs present
```

Drift watchlist: the 2026-07-11 soak date (stale after it passes), the DS
`df` numbers, the s7cy/dqfe bead states, and every "PROVISIONAL pending
Stephanie" marker (Q2 in §3, Q5 in §2) — resolve those when her real
answers land.
