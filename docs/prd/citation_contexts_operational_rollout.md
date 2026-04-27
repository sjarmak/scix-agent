# PRD: Citation-Contexts Operational Rollout

> Bead: `scix_experiments-79n.3` (parent: `scix_experiments-79n`,
> sibling: `scix_experiments-79n.1` — extraction throughput
> backfill, see `docs/prd/citation_contexts_throughput.md`).
>
> Cross-ref discipline: 79n.1 owns the extraction pipeline, pilot
> measurement, throughput tuning, and ingest_log integration. **This
> PRD owns** storage headroom resolution (the 79n.1 hard-blocker),
> partition strategy, index rebuild, `v_claim_edges` materialization
> evolution, MCP downtime planning, latency benchmarking, and
> backup-tier integration. Do not duplicate scope.

## Status (2026-04-27)

Build PRD. The 79n.1 storage hard-blocker (DS has 274 GB free,
projected 79n.1 growth needs ~450 GB) is **resolved here** via
declarative range partitioning by `source_year` plus a
detach-and-archive scheme that keeps the live table on DS and ships
cold partitions to NAS-only `pg_dump` without dropping data
permanently. Live data stays on DS per CLAUDE.md storage-tiering;
NFSv3 is never used as a live PG data dir.

## Summary

Post-79n.1, `citation_contexts` grows from **825,115** to a
projected **210M–300M** rows (79n.1 §Target state). At today's
**1,844 B/row**, that is **~390–560 GB** of additional heap
(estimate, see [Storage probe](#current-storage-probe-2026-04-27)).
The DS partition has **274 GB free** at the time of this PRD —
**hard overshoot** under the 450 GB midpoint estimate. This PRD
resolves the overshoot by partitioning `citation_contexts` by
`source_year` and detaching cold partitions to NAS-only `pg_dump`
storage, keeps the live MCP query path (`v_claim_edges` materialized
view) intact, schedules index rebuilds inside operator-defined
maintenance windows, and adds a runnable latency benchmark on a
10% sample loaded into `scix_test`.

## Pre-conditions

The plan in this PRD does not execute until **all** of the following
are true:

1. **79n.1 pilot measurement complete.** A representative 10K-paper
   pilot has run end-to-end and the contexts/paper ratio is
   recorded in `docs/citation_contexts_coverage.md`. The 27.1
   baseline used here is from existing 30,443 sources; if the pilot
   measures >35 contexts/paper (or <21), the storage projections in
   §Storage plan resolution must be re-run with the new ratio
   *before* the partition migration is applied.
2. **Operator-scheduled maintenance window** of ≥4 hours,
   off-business-hours UTC, agreed up front. Used for the partition
   migration and the per-shard index rebuild. Local MCP only at
   that point per CLAUDE.local.md (mcp.sjarmak.ai is intentionally
   down), so user-facing impact is limited to a single operator's
   tooling — but the window must still be scheduled, not improvised.
3. **`scix_test` 10% sample loaded and benchmarked.** Acceptance gate
   A4 ([Latency benchmark plan](#latency-benchmark-plan)) must pass
   on the test fixture before the prod partition migration is
   committed.
4. **A NAS pg_dump destination exists.** `/mnt/postgres/scix_dumps/`
   created, mode 0700 (owner-only), with at least the projected 79n.1 dump size
   free on the NAS volume. NAS currently has 49 TB free; this is
   non-blocking but the directory itself must exist before the
   first archive partition is detached.

## Current storage probe (2026-04-27)

| Surface                                    | Value           |
| ------------------------------------------ | --------------- |
| DS partition (`/dev/nvme1n1p2`)            | 1.9 TB total    |
| DS used                                    | 1.6 TB (86%)    |
| DS free                                    | **274 GB**      |
| NAS (`/mnt`, NFS)                          | 50 TB total     |
| NAS free                                   | **49 TB**       |
| `scix` database total                      | 1,162 GB        |
| `papers` (heap+indexes+TOAST)              | 411 GB          |
| `paper_embeddings`                         | 253 GB          |
| `papers_fulltext`                          | 242 GB          |
| `citation_edges`                           | 45 GB           |
| `agent_document_context`                   | 35 GB           |
| `idx_embed_hnsw_indus` (single index)      | **120 GB**      |
| `citation_contexts` total relation         | 1,553 MB        |
| `citation_contexts` heap                   | 1,451 MB        |
| `citation_contexts` indexes (3 btrees)     | 101 MB          |
| `v_claim_edges` (matview, total)           | 1,916 MB        |
| `v_claim_edges` (matview, indexes)         | 183 MB          |

Reproduce:

```bash
df -h /
df -h /mnt
psql -d scix -c "SELECT pg_size_pretty(pg_database_size('scix'));"
psql -d scix -c "
  SELECT relname, pg_size_pretty(pg_total_relation_size(c.oid)) AS sz
  FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
  WHERE n.nspname='public' AND c.relkind IN ('r','m')
  ORDER BY pg_total_relation_size(c.oid) DESC LIMIT 10;
"
psql -d scix -c "
  SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid))
  FROM pg_stat_user_indexes WHERE indexrelname LIKE '%hnsw%'
  ORDER BY pg_relation_size(indexrelid) DESC;
"
```

Headroom math (79n.1 midpoint):

| Component                          | Estimate       |
| ---------------------------------- | -------------- |
| Heap @ 270M × 1,844 B              | ~452 GB        |
| Three btrees @ ~12% of heap (probe) | ~54 GB         |
| **Total citation_contexts post-run** | **~506 GB**    |
| `v_claim_edges` matview growth     | +30–50 GB est. |
| **DS deficit vs 274 GB free**      | **~−260 GB**   |

**This PRD's plan must remove ~260 GB of DS pressure or move ~260
GB of cold partitions off DS — the partition+archive path below is
the recommended resolution.**

## Storage plan resolution

### Option analysis

| Option                                               | Pro                                | Con                                                                                                                              |
| ---------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **(A)** Drop obsolete data on DS to free space       | No new schema, fast                | Specific candidates not identified by this PRD; would need a separate cleanup audit per `feedback_no_destructive_cleanup_without_evidence.md`. |
| **(B)** Add storage to DS (NVMe upgrade)             | Cleanest                           | Hardware project, not a PRD-deliverable. Out of scope.                                                                            |
| **(C)** Move `citation_contexts` to a NAS tablespace | Frees ~500 GB of DS                | **Forbidden by CLAUDE.md** — NFSv3 not safe for live-write workloads; explicitly called out for vectors and applies equally here. |
| **(D)** Partition by `source_year`, detach + dump cold partitions to NAS, keep recent partitions hot | DS keeps only hot data; recoverable via `pg_restore` from NAS dump | Requires migration; partitions add ongoing operational complexity; queries spanning years cost a partition-pruning planner pass. |
| **(E)** Cap the 79n.1 backfill at recent years only  | Trivial to implement               | Throws away 79n.1's coverage goal — defeats the purpose.                                                                          |

**Recommendation: Option D — partition by `source_year`, detach
cold partitions, archive via `pg_dump` to `/mnt/postgres/scix_dumps/`,
drop archived partitions from DS.**

Rationale:

- Aligns with CLAUDE.md storage-tiering: live writes on DS,
  duplicate/archive on NAS via `pg_dump`. Identical pattern to the
  one already documented for irreplaceable derived tables in
  CLAUDE.md §Storage tiering.
- Reversible: a detached + dumped partition can be re-attached after
  `pg_restore` if a future query needs cold data. We document the
  re-attach procedure below so it is not folklore.
- Partition pruning by `source_year` matches the dominant query
  shape: `claim_blame` walks `WHERE source_bibcode = %s` (single
  year prunes via the partition key once we add a join condition);
  `find_replications` walks `WHERE target_bibcode = %s` (does NOT
  prune by `source_year` — see [Partition strategy](#partition-strategy)
  for how this is handled).

### Recommended path

1. Apply migration `063_partition_citation_contexts.sql` — see DDL
   below.
2. Hot retention: keep all partitions ≥ `(current_year − 5)` on DS.
   Initial implementation: keep `2021..2026` (six partitions plus
   the catch-all "pre-2021" partition for legacy data) on DS;
   dump+detach the catch-all once partition is full and stable.
3. Annual cron (Jan 15 each year): identify the partition that
   crossed the 5-year retention boundary, `pg_dump` it to
   `/mnt/postgres/scix_dumps/`, verify the dump, then `ALTER TABLE
   ... DETACH PARTITION` and `DROP TABLE` the detached partition.
4. The dump filename convention is
   `citation_contexts_y<YEAR>_<dumpdate>.dump` (custom format,
   compressed) so re-attach can target the right archive.

This plan **frees** an estimated 200–280 GB of DS post-run if the
catch-all "pre-2021" partition (containing the long historical
tail) is detached after the initial backfill stabilizes.

## Partition strategy

### Choice: range partition by `source_year`

| Candidate scheme           | Why not                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hash on `source_bibcode`   | Even distribution, but no archive boundary — every partition has a uniform mix of decades, so we can't ship "old" data off DS.                     |
| Range on `target_year`     | Aligned with `find_replications` query (target lookup) but not with `cited_by_intent` and `claim_blame` (both filter on `source_bibcode`).         |
| Range on `source_bibcode` (alphanumeric) | Bibcode prefix encodes year, but year extraction at planner time requires expression-based partitioning which precludes constraint exclusion in 16. |
| **Range on `source_year` (NEW column derived at insert time)** | Matches archive boundary (cold years detach and dump cleanly). Partition-prune for `cited_by_intent` and writer-side INSERTs. `find_replications` is the trade-off — see below. |

### Required schema change

`citation_contexts` does not currently have `source_year`. Migration
must add it as a generated/insert-side column. Recommended approach:
add as a non-nullable column populated at insert time by the
extraction pipeline (`src/scix/citation_context.py:613` — the bulk
INSERT statement), with a one-time backfill of the existing 825K
rows from `papers.year`.

### DDL (recommended migration `063_partition_citation_contexts.sql`)

```sql
BEGIN;

-- 1. Backfill source_year on existing rows (one-shot).
ALTER TABLE citation_contexts ADD COLUMN source_year SMALLINT;
UPDATE citation_contexts cc
SET source_year = p.year
FROM papers p
WHERE p.bibcode = cc.source_bibcode
  AND cc.source_year IS NULL;

-- 2. Defensive: any rows where the source paper is not in `papers`
--    (should be zero given the 79n.1 SELECT joins on body IS NOT
--    NULL) get bucketed into the catch-all "year 0" partition.
UPDATE citation_contexts SET source_year = 0 WHERE source_year IS NULL;

ALTER TABLE citation_contexts ALTER COLUMN source_year SET NOT NULL;

-- 3. Rename the existing table out of the way and recreate as
--    partitioned. NOTE: we cannot ALTER an unpartitioned table
--    into a partitioned table in Postgres 16 — must rename + copy.
ALTER TABLE citation_contexts RENAME TO citation_contexts_legacy;
ALTER INDEX citation_contexts_pkey RENAME TO citation_contexts_legacy_pkey;
ALTER INDEX idx_citctx_source_target RENAME TO idx_citctx_legacy_source_target;
ALTER INDEX idx_citctx_target RENAME TO idx_citctx_legacy_target;

CREATE TABLE citation_contexts (
    id BIGINT GENERATED ALWAYS AS IDENTITY,  -- BIGINT, see §id-overflow. ALWAYS prevents any future INSERT from supplying an explicit id and colliding with the sequence.
    source_bibcode TEXT NOT NULL,
    target_bibcode TEXT NOT NULL,
    context_text TEXT NOT NULL,
    char_offset INTEGER,
    section_name TEXT,
    intent TEXT,
    source_year SMALLINT NOT NULL,
    PRIMARY KEY (id, source_year)  -- partition key MUST be in PK
) PARTITION BY RANGE (source_year);

-- 4. Create partitions. Convention:
--    - one partition per year for years 2010..2026 (inclusive)
--    - one catch-all "legacy" partition for source_year < 2010
--    - one default partition for source_year > 2026 (defensive)
CREATE TABLE citation_contexts_legacy_pre2010
    PARTITION OF citation_contexts
    FOR VALUES FROM (0) TO (2010);

DO $$
DECLARE y INT;
BEGIN
    FOR y IN 2010..2026 LOOP
        EXECUTE format(
            'CREATE TABLE citation_contexts_y%s PARTITION OF citation_contexts FOR VALUES FROM (%s) TO (%s);',
            y, y, y + 1
        );
    END LOOP;
END$$;

CREATE TABLE citation_contexts_default
    PARTITION OF citation_contexts DEFAULT;

-- 5. Indexes on the partitioned parent — Postgres 16 propagates them
--    to all partitions automatically.
CREATE INDEX idx_citctx_source_target
    ON citation_contexts (source_bibcode, target_bibcode);
CREATE INDEX idx_citctx_target
    ON citation_contexts (target_bibcode);

-- 6. Move data from legacy table into partitions in batches. This
--    is the slow step; size the batch to fit in maintenance_work_mem
--    (currently 8GB).
INSERT INTO citation_contexts
    (source_bibcode, target_bibcode, context_text, char_offset,
     section_name, intent, source_year)
SELECT
    cc.source_bibcode, cc.target_bibcode, cc.context_text,
    cc.char_offset, cc.section_name, cc.intent, cc.source_year
FROM citation_contexts_legacy cc;

-- 7. v_claim_edges depends on citation_contexts. Drop and recreate
--    it pointing at the new partitioned table (it picks up the new
--    table by name automatically — same name).
DROP MATERIALIZED VIEW IF EXISTS v_claim_edges CASCADE;
-- Re-run the body of migrations/057_v_claim_edges.sql here.

-- 8. Drop the legacy table once data is verified (manual step,
--    not in this transaction).

COMMIT;
```

The DROP+rebuild of `v_claim_edges` in step 7 takes the materialized
view's existing build time. At 825K-row baseline it is ~1 minute;
at 270M-row scale, expect 30–90 minutes (estimate, see
[`v_claim_edges` evolution](#v_claim_edges-evolution)).

### `id` overflow note

The current `id` column is `INTEGER` (int4, max 2,147,483,647) with a
`SERIAL` sequence. The 79n.1 projected upper bound (300M) is well
inside int4 range, but a follow-on extraction or a re-run that
double-inserts would push past 1B and start eroding headroom. The
new partitioned table uses `BIGINT GENERATED ALWAYS AS IDENTITY`
to remove this as a future concern. Sequence values from the legacy
table are not preserved (the partitioned table assigns fresh IDs);
if any external system references `citation_contexts.id` as a
foreign key (none found in `git grep "citation_contexts.id"` on
2026-04-27), this assumption must be re-validated before the
migration.

## Index rebuild plan

### Current indexes (probed 2026-04-27)

| Index                         | Size    | Purpose                                                                                  |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `citation_contexts_pkey`      | 35 MB   | Surrogate primary key on `id`.                                                           |
| `idx_citctx_source_target`    | 43 MB   | `claim_blame`-style queries; covers `cited_by_intent` walks.                              |
| `idx_citctx_target`           | 23 MB   | `find_replications`-style queries.                                                       |

At 270M rows, btree-index size scales roughly linearly — expect
each btree to grow to ~10–15 GB per partition (0.04–0.05× heap, per
PG btree fillfactor 90 + key width). For 17 hot partitions that is
~170–250 GB across all three indexes — already accounted for in
the storage projection.

### Build order during migration

1. **Drop the legacy indexes** at the moment we drop the legacy
   table (step 8 of the DDL). No fast-path here — the legacy table
   is gone in one statement.
2. **Build partition-level btrees in parallel.** Each partition
   gets its three indexes built in a separate `psql` session.
   Postgres 16 honors `max_parallel_maintenance_workers=7` per
   build — at 8 GB `maintenance_work_mem` and 7 parallel workers
   per build, four concurrent partition builds saturate the box.
3. **Rebuild `v_claim_edges`.** This is the single longest blocking
   step. See timing in [`v_claim_edges` evolution](#v_claim_edges-evolution).

### Estimated wall-clock per phase

These are **estimates** (not measured at scale). Methodology:
extrapolate from current btree sizes scaled linearly to 270M rows
under PG 16 fillfactor-90 defaults, with parallel index workers
saturating at `max_parallel_maintenance_workers=7` (probed
2026-04-27).

| Phase                                                             | Est. wall-clock |
| ----------------------------------------------------------------- | --------------- |
| Schema migration (rename, create partitions, INSERT...SELECT)     | 2–4 h (data move bound by sequential scan of 270M-row legacy)|
| Per-partition btree builds (17 partitions × 3 indexes, 4-way concurrent) | 1–3 h     |
| `v_claim_edges` rebuild (DROP + CREATE MATERIALIZED VIEW)         | 30 min–2 h     |
| ANALYZE on all partitions                                         | 15–30 min      |
| **Total maintenance window**                                      | **4–10 h**     |

The 4-hour pre-condition window above is a hard floor; if the
INSERT...SELECT exceeds that, the operator pauses, finishes
mid-window, and resumes the next window — `INSERT INTO ...
PARTITION (...) SELECT ... WHERE source_year = X` is partition-
local and resumable per partition.

## `v_claim_edges` evolution

### Current state

- `v_claim_edges` is **already a `MATERIALIZED VIEW`** (probed at
  `\d+ v_claim_edges` and `migrations/057_v_claim_edges.sql`). The
  79n.3 acceptance criterion phrasing ("continue as VIEW, may need
  MATERIALIZED VIEW post-scale") was based on the parent bead
  description; the actual state is already materialized.
- Refresh: `scripts/refresh_v_claim_edges.py` runs `REFRESH
  MATERIALIZED VIEW CONCURRENTLY v_claim_edges` against the prod
  DSN, daily, called from `scripts/daily_sync.sh:119`.
- Refresh duration today: ~1–3 minutes (823K rows).

### Evolution path

1. **No structural change required.** The MATERIALIZED VIEW
   correctly absorbs the partitioned table: it is a separate
   physical relation built from the partitioned `citation_contexts`,
   so partitioning the underlying table doesn't break the view's
   indexes.
2. **Refresh duration grows.** At 270M underlying rows, the
   `DISTINCT ON` + JOIN + ORDER BY in
   `migrations/057_v_claim_edges.sql:47-68` becomes the dominant
   cost. Estimate:
   - existing build: 823K rows / ~60 s ≈ 14K rows/s of view-build
     throughput.
   - 270M rows at 14K rows/s ≈ **5.4 hours**.
   - `REFRESH ... CONCURRENTLY` adds ~1.5–2× overhead (PG must
     compute the diff against the existing materialized rows), so
     concurrent refresh at full scale: **8–11 hours**. *Estimate.*
3. **Mitigations** if the daily concurrent refresh becomes
   intolerable:
   - Move the daily refresh from `daily_sync.sh` (currently runs
     06:15 UTC then a chain of harvest+ingest+embed+refresh) into
     a dedicated weekly cron (Sunday 02:00 UTC), and let the
     daily refresh be best-effort skipped if the prior run is
     still in flight.
   - Add a `last_refreshed_at` column to a sidecar metadata table
     so MCP responses can carry the freshness timestamp — agents
     reading `v_claim_edges` will know the data lag without out-
     of-band documentation.
   - **Do not** drop CONCURRENTLY in favor of a fast non-
     concurrent refresh: that would lock readers for the duration
     of the rebuild and make `claim_blame`/`find_replications`
     unavailable for hours.
4. **Post-79n.1 first refresh is one-time bigger.** Plan for the
   first post-backfill REFRESH to take **8–12 hours**. Schedule it
   into the same maintenance window as the partition migration
   (or the night after).

## MCP downtime plan

The CLAUDE.local.md note that `mcp.sjarmak.ai` is intentionally
down means user-facing downtime is local-MCP-only. The maintenance
operations below still degrade the local MCP for the operator
running it; the durations and mitigations apply to that.

### Operations that block MCP queries

| Op                                                       | Affected tools                                       | Downtime type    | Est. duration | Mitigation                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------- | ---------------- | ------------- | --------------------------------------------------------------------------- |
| Partition migration (DDL, INSERT...SELECT, DROP legacy)  | `claim_blame`, `find_replications`, `cited_by_intent`, `citation_traverse` (intent annotations) | **Hard** — table renamed mid-transaction | 4–10 h        | Maintenance window. Local MCP returns 5xx during this. No fallback responses — agents see errors. |
| `v_claim_edges` REFRESH (CONCURRENTLY, post-migration)   | `claim_blame`, `find_replications` (slight latency increase)        | **Soft** — readers OK, writers slow | 8–12 h        | Concurrent refresh keeps reads online; latency increase is noticeable but tools remain answering.  |
| Per-partition btree REINDEX (annual maintenance)         | Queries hitting that partition only                  | **Soft** — REINDEX CONCURRENTLY | 30–60 min/partition | REINDEX CONCURRENTLY — reads stay online.                                                          |
| Annual archive partition detach + drop                   | None (cold partition, no live readers)               | **None**         | 1–5 min        | DETACH is a metadata-only op; DROP after dump completes.                                          |

### Cron jobs that depend on `citation_contexts`

Found by `grep -rn 'citation_contexts\|v_claim_edges'` across
`scripts/` (2026-04-27):

- `scripts/daily_sync.sh:119` — daily REFRESH at 06:15 UTC.
- `scripts/refresh_v_claim_edges.py` — refresh script invoked above.

Operator action: **before the partition migration window**, comment
out line 119 of `daily_sync.sh` (or set `SCIX_BATCH=true` to no-op
the refresh) until the migration completes. Otherwise the daily
cron will collide with the migration's `v_claim_edges` rebuild and
either fail or double-rebuild.

The 79n.1 backfill itself does not write to `v_claim_edges` (it
writes only to `citation_contexts`); the daily refresh picks up
new rows on its next run. There is no cross-script lock today.

### Local MCP fallback during downtime

For the 4–10 h hard-downtime window, the four tools above will
return `psycopg.errors.UndefinedTable` (during the rename) or hang
(during the INSERT...SELECT batch). A clean fallback is out of
scope for this PRD — the operator scheduling the window is the
only consumer per CLAUDE.local.md, and the mitigation is "don't
run the local MCP during the window." If a future deploy re-opens
the public MCP, this PRD's downtime budget must be revisited.

## Latency benchmark plan

Acceptance gate A4: post-scale `claim_blame` and `find_replications`
respond <5s p95. Methodology below produces a defensible answer
**before** the prod cutover.

### Step 1 — Provision `scix_test` with a 10% sample

> **Pre-requisite (added in wave-2 review).** `scix_test` has the full
> schema but **no data** by design (CLAUDE.md "Testing — Database
> Safety"). The original Step 1 SQL below was self-referential — it
> truncated `scix_test.citation_contexts` then `INSERT…SELECT` from
> the same (now-empty) table, producing 0 rows and a false-positive
> A4 gate pass. Operator must seed `scix_test.papers` /
> `scix_test.citation_contexts` / `scix_test.citation_edges` from
> prod **first**, via one of:
>
> 1. **`pg_dump | pg_restore` of the prod tables** (preferred — fast,
>    self-contained):
>    ```bash
>    pg_dump -d scix --data-only \
>      --table=papers --table=citation_contexts --table=citation_edges \
>      --format=custom --compress=6 \
>      --file=/tmp/scix_seed.dump
>    pg_restore -d scix_test --data-only --disable-triggers \
>      --jobs=4 /tmp/scix_seed.dump
>    rm /tmp/scix_seed.dump
>    ```
>    This is **slow** at full prod scale; for a benchmark fixture, prefer
>    the `dblink` form below.
>
> 2. **`dblink` cross-DB read into `scix_test`** (faster — sources from
>    prod via TABLESAMPLE without needing a full restore):
>    ```sql
>    CREATE EXTENSION IF NOT EXISTS dblink;
>    INSERT INTO papers
>      SELECT * FROM dblink(
>        'dbname=scix',
>        'SELECT * FROM papers TABLESAMPLE SYSTEM (10)'
>      ) AS t(<full papers row type>);
>    -- then INSERT INTO citation_contexts / citation_edges using
>    -- joins as in the next block, but reading via dblink from prod.
>    ```
>
> Once `scix_test.papers` is populated, the original SQL below is
> correct and produces a representative sample. Verify
> `SELECT count(*) FROM papers` is non-zero in `scix_test` before
> running the block.

```bash
export SCIX_TEST_DSN="dbname=scix_test"
psql "$SCIX_TEST_DSN" -c "
  -- A. drop any prior fixture data
  TRUNCATE citation_contexts;
  TRUNCATE citation_edges;

  -- B. sample ~10% of source papers, stratified by year so the
  --    distribution matches prod. We sample papers (the seeds) and
  --    pull all of their citation_contexts plus the citation_edges
  --    those rows reference, so the JOIN graph used by v_claim_edges
  --    is dense enough to exercise the indexes.
  WITH sampled_papers AS (
    SELECT bibcode FROM papers TABLESAMPLE SYSTEM (10)
  )
  INSERT INTO citation_contexts
    SELECT cc.* FROM citation_contexts cc
    JOIN sampled_papers sp ON sp.bibcode = cc.source_bibcode;

  INSERT INTO citation_edges
    SELECT ce.* FROM citation_edges ce
    JOIN sampled_papers sp ON sp.bibcode = ce.source_bibcode;
"
```

The sample's row count is the validation lever: if it lands within
±20% of one tenth of the prod-projected row count (i.e.
**21M–30M rows** at the 270M projection), the sample is
representative and the benchmark is fair. Operator records the
actual sample size in the bead notes.

### Step 2 — Refresh `v_claim_edges` against the sample

```bash
psql "$SCIX_TEST_DSN" -c "REFRESH MATERIALIZED VIEW v_claim_edges;"
psql "$SCIX_TEST_DSN" -c "ANALYZE citation_contexts; ANALYZE v_claim_edges;"
```

### Step 3 — Run the representative query mix

Each query is a representative shape from `claim_blame.py:412-432`
(`_walk_reverse_references`) and `find_replications.py:270-295`
(`_query_citations`). We run 100 invocations of each against random
bibcodes and capture timing.

```sql
-- claim_blame mainline: walk reverse refs from a source
EXPLAIN (ANALYZE, BUFFERS)
SELECT source_bibcode, target_bibcode, context_snippet, intent,
       section_name, source_year, target_year, char_offset
FROM v_claim_edges
WHERE source_bibcode = $1
ORDER BY target_year ASC NULLS LAST, char_offset ASC NULLS LAST
LIMIT 1000;

-- find_replications mainline: forward citations to target
EXPLAIN (ANALYZE, BUFFERS)
SELECT vce.source_bibcode AS citing_bibcode,
       vce.source_year AS year,
       vce.intent, vce.context_snippet, vce.section_name
FROM v_claim_edges vce
WHERE vce.target_bibcode = $1
LIMIT 200;

-- cited_by_intent (queries citation_contexts directly, not
-- v_claim_edges — exercises partition pruning if source_year is
-- pushed down by the planner)
EXPLAIN (ANALYZE, BUFFERS)
SELECT cc.source_bibcode, p.title, p.authors[1], p.citation_count
FROM citation_contexts cc
LEFT JOIN papers p ON cc.source_bibcode = p.bibcode
WHERE cc.target_bibcode = $1
  AND cc.intent = 'method'
ORDER BY p.citation_count DESC NULLS LAST, cc.id ASC
LIMIT 50;
```

### Step 4 — Drive the workload with `pgbench`

```bash
# Save the queries above into ~/scix_bench_qmix.sql with \set
# directives binding $1 to a random bibcode from a pre-generated
# bibcode list.
pgbench -d scix_test \
  -f ~/scix_bench_qmix.sql \
  -c 4 -j 4 -T 300 \
  -P 30 \
  --report-per-command \
  > /home/ds/projects/scix_experiments/logs/scix_bench_post_partition.log
```

`pgbench -P 30 --report-per-command` emits per-statement latency
percentiles every 30s. Acceptance gate compares the p95 of each
statement against a 5,000 ms budget.

### Step 5 — Compare against pre-partition baseline

Run the same `pgbench` workload against the **current** prod
schema (before any migration) using `dbname=scix` read-only. The
comparison establishes that the partition migration does not
*regress* latency on small bibcode lookups (the partitioning by
`source_year` should be neutral for `find_replications` and
positive for `cited_by_intent`).

Pre-baseline command (read-only is safe against prod since
`pgbench -f` runs only the SELECTs in the file):

```bash
pgbench -d scix \
  -f ~/scix_bench_qmix.sql \
  -c 4 -j 4 -T 300 \
  -P 30 \
  --report-per-command \
  > /home/ds/projects/scix_experiments/logs/scix_bench_pre_partition.log
```

### Acceptance

PASS when post-partition p95 ≤ pre-partition p95 × 1.5, AND p95 <
5000 ms for each of the three statements. Record both logs in the
bead.

## Backup tier integration

### Live backup of `citation_contexts` heads

Per CLAUDE.md storage-tiering: weekly `pg_dump` of irreplaceable
derived tables to `/mnt/postgres/`. `citation_contexts` is
irreplaceable in the sense that re-deriving it costs the full
79n.1 throughput run.

Add migration-time directory:

```bash
sudo mkdir -p /mnt/postgres/scix_dumps/
sudo chown postgres:postgres /mnt/postgres/scix_dumps/
sudo chmod 0700 /mnt/postgres/scix_dumps/   # owner-only — dumps contain
                                            # full citation_contexts
                                            # corpus including any
                                            # sensitive context snippets
```

### Weekly cron (new)

Add to crontab on the DS host (alongside the existing
`daily_sync.sh` cron):

```cron
# Sunday 03:00 UTC: dump live citation_contexts partitions to NAS.
# Only the hot-tier partitions are dumped here; archived partitions
# are dumped once at detach time and never re-dumped.
0 3 * * 0 /home/ds/projects/scix_experiments/scripts/weekly_pg_dump_citation_contexts.sh \
    >> /home/ds/projects/scix_experiments/logs/weekly_pg_dump.log 2>&1
```

The script (out of scope to write here, but specified):

1. Set `umask 0077` at the top so dump files land at mode `0600`
   instead of inheriting the shell default `0644` (world-readable on
   the NAS mount).
2. `pg_dump -d scix --schema=public --table='citation_contexts*' --table=v_claim_edges --format=custom --compress=6 --file /mnt/postgres/scix_dumps/citation_contexts_$(date +%Y%m%d).dump`
3. Verify dump with `pg_restore --list` (catalog read-only check —
   does NOT verify row data; for a stronger check, do a periodic
   `pg_restore --schema-only` against a scratch DB).
4. Retain last 4 weekly dumps; rotate older to `/mnt/postgres/scix_dumps/archive/`.
5. Wrap in `$SCIX_BATCH` per CLAUDE.md memory-isolation rule —
   `pg_dump` of a 500 GB table allocates significantly.

### Annual archive partition dump (new)

When a partition crosses the 5-year retention boundary
(see [Storage plan resolution](#storage-plan-resolution) step 3):

```bash
umask 0077                             # dump files land at mode 0600.
YEAR=2018                              # whichever partition is being archived
[[ "$YEAR" =~ ^[0-9]{4}$ ]] || { echo "Invalid YEAR" >&2; exit 1; }

pg_dump -d scix \
  --table=citation_contexts_y${YEAR} \
  --format=custom --compress=6 \
  --file=/mnt/postgres/scix_dumps/archive/citation_contexts_y${YEAR}_$(date +%Y%m%d).dump

# pg_restore --list only validates the catalog; for a stronger check,
# add: pg_restore --schema-only --no-owner -d scix_test <dump-file>
pg_restore --list /mnt/postgres/scix_dumps/archive/citation_contexts_y${YEAR}_*.dump >/dev/null

psql -d scix <<SQL
ALTER TABLE citation_contexts DETACH PARTITION citation_contexts_y${YEAR};
DROP TABLE citation_contexts_y${YEAR};
SQL
```

Re-attach procedure (folklore-prevention):

```bash
# Restore the dumped partition table into a staging schema first.
pg_restore -d scix \
  --schema=public \
  /mnt/postgres/scix_dumps/archive/citation_contexts_y${YEAR}_<date>.dump

psql -d scix <<SQL
ALTER TABLE citation_contexts ATTACH PARTITION citation_contexts_y${YEAR}
    FOR VALUES FROM (${YEAR}) TO ($((YEAR + 1)));
SQL
```

## Acceptance gates

A1 through A5 are runnable checks. All five must PASS before the
operational rollout is considered complete.

### A1. Storage headroom check

```bash
df -h / | awk 'NR==2 {print $4}'
```

PASS if free space ≥ 100 GB after the partition migration. If the
catch-all `_pre2010` partition was archived as part of the
migration, this should land closer to 200 GB free.

### A2. Partition queryability

```sql
SELECT count(*) FROM citation_contexts;
SELECT count(DISTINCT source_year) FROM citation_contexts;
\d+ citation_contexts
```

PASS if the row count matches the legacy table's pre-migration
count (within the dropped-rows budget, see partition migration
step 6), AND `\d+` shows the partitioned-table layout (`Partitions:
... `).

### A3. `v_claim_edges` queryable post-rebuild

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM v_claim_edges
WHERE source_bibcode = '2020ApJ...893...85B';
```

PASS if planner picks `idx_v_claim_edges_source_intent` and runtime
< 5 s. (Same probe as 79n.1 §A3, deliberately — same gate, owned
here.)

### A4. p95 latency on `claim_blame` / `find_replications`

Run the workload from §[Latency benchmark plan](#latency-benchmark-plan)
against the live prod instance after migration:

```bash
pgbench -d scix -f ~/scix_bench_qmix.sql -c 4 -j 4 -T 300 -P 30 \
  --report-per-command > /home/ds/projects/scix_experiments/logs/scix_bench_post_prod.log
```

PASS if p95 < 5000 ms for each of the three statements in the
mix.

### A5. Backup tier dump exists and verifies

```bash
LATEST=$(find /mnt/postgres/scix_dumps/ -maxdepth 1 -name 'citation_contexts_*.dump' \
  -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
test -n "$LATEST" && pg_restore --list "$LATEST" >/dev/null
```

PASS if exit 0.

## Open questions / TBD

1. **`source_year` derivation cost on the 79n.1 hot path.** The
   recommended migration adds `source_year` as a NOT NULL column.
   The 79n.1 extraction pipeline (`src/scix/citation_context.py:613`)
   currently does not write `source_year`. Decision needed before
   migration goes live: either (a) extend the bulk INSERT to
   compute `source_year` from the source paper's `year` field
   (already on the row at SELECT time, no JOIN cost), or (b) make
   `source_year` a generated column with `GENERATED ALWAYS AS
   ((SELECT year FROM papers WHERE bibcode = source_bibcode))` —
   but PG generated columns can't reference other tables, so (b)
   is not actually available; (a) is the only viable path.
   **Owner: 79n.1 sibling, or a follow-up bead.**
2. **Catch-all partition strategy.** The `_pre2010` catch-all is
   convenient but can grow uncontrolled if 79n.1 turns out to
   process many sub-2010 source papers. Pilot measurement (79n.1
   pre-condition 1) should report the count of `source_year < 2010`
   sources; if it exceeds ~5M, this PRD's DDL should be updated to
   split the legacy bucket by decade instead.
3. **Refresh windowing under partitioned storage.** The 8–12 hour
   first-refresh estimate is extrapolated, not measured. The
   benchmark plan in §Latency benchmark plan tests the *steady-
   state* `v_claim_edges`, not the rebuild itself. A separate
   timed refresh against the 10% sample should be run to validate
   the linear extrapolation.
4. **Hot-tier retention years.** "≥ current_year − 5" is operator
   intuition. A future audit (out of scope) of how often
   `claim_blame` and `find_replications` requests target
   pre-2021 source papers should establish whether the 5-year
   window is too aggressive (resulting in misses) or too generous
   (wastes DS).
5. **Public MCP re-deploy.** If `mcp.sjarmak.ai` is brought back
   up before the partition migration, the §MCP downtime plan
   needs a public status-page strategy and a graceful-degradation
   path for the four affected tools. Not addressed here.

## Cross-references

- Sibling: `scix_experiments-79n.1` —
  `docs/prd/citation_contexts_throughput.md`. Owns extraction
  throughput, pilot measurement, ingest_log integration. Storage
  hard-blocker called out there is **resolved here** in
  §Storage plan resolution.
- Parent: `scix_experiments-79n` — coverage gap motivating both
  79n.1 and 79n.3.
- Related: `migrations/057_v_claim_edges.sql` — current
  materialized view definition, will be re-run after partitioning.
- Related: `migrations/011_citation_contexts.sql` — current table
  definition (unpartitioned).
- Related: `scripts/refresh_v_claim_edges.py` — daily REFRESH
  script invoked by `scripts/daily_sync.sh:119`. Must be paused
  during the migration window.
- Related: `src/scix/claim_blame.py:412-432` and
  `src/scix/find_replications.py:270-295` — the SQL shapes the
  benchmark plan exercises.
- Related: `CLAUDE.md` §Storage tiering — DS-primary,
  NAS-duplicate-only policy this PRD operates within.
- Related: `CLAUDE.local.md` §Public service — confirms MCP is
  intentionally not exposed publicly today, scoping the downtime
  blast radius.
