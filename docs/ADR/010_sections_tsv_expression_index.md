# ADR-010: Drop `sections_tsv` Stored Column, Replace with Expression GIN Index

- **Status**: Proposed
- **Deciders**: SciX maintainers
- **Scope**: `papers_fulltext.sections_tsv` column + `idx_papers_fulltext_sections_tsv` GIN index
- **Related beads**: `scix_experiments-zsbd` (section embeddings), `scix_experiments-6hr7` (citation_contexts backfill)
- **Related ADRs**: ADR-009 (body-AI OA gate, which chose expression index over GENERATED STORED for the same reason)

## Context

NVMe (`/dev/nvme1n1p2`, 1.9 TB) is 96% full with ~93 GB free after a Tier-1
cleanup pass. Two multi-day pipelines are storage-gated:

- **`scix_experiments-zsbd`**: section_pipeline encode of 14.9M papers into
  `section_embeddings` (150–450M halfvec(1024) rows, projected 300+ GB).
- **`scix_experiments-6hr7`**: citation_contexts shard backfill (~232 GB at
  full population).

Neither can start without first recovering a substantial block of headroom.

`papers_fulltext.sections_tsv` is a regular `tsvector` column (not GENERATED —
migration 063 deliberately avoided GENERATED ALWAYS to sidestep the full heap
rewrite and the 1 MB tsvector ceiling on outlier rows). It was populated via
`scripts/backfill_sections_tsv.py` using the `safe_sections_tsv(jsonb)`
helper. Its current footprint:

| Object | Size |
|---|---|
| `sections_tsv` column (TOAST) | ~214 GB |
| `idx_papers_fulltext_sections_tsv` GIN index | ~27 GB |
| **Total** | **~241 GB** |

The analogous pattern for `papers.body` already runs in production without a
stored column:

```sql
-- ix_papers_body_tsv (from migration, no stored column)
CREATE INDEX ix_papers_body_tsv ON public.papers
    USING gin (to_tsvector('english'::regconfig, body))
    WHERE body IS NOT NULL AND length(body) <= 1048575;
```

This confirms expression GIN indexes work for the same BM25 workload on this
corpus.

## Decision

Drop `papers_fulltext.sections_tsv` (the stored column) and replace
`idx_papers_fulltext_sections_tsv` with an expression GIN index over
`safe_sections_tsv(sections)`.

### Proposed DDL (migration 069)

```sql
-- Step 1: build the replacement expression index CONCURRENTLY
-- (no table lock; reads and writes proceed normally)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_fulltext_sections_tsv_expr
    ON papers_fulltext
    USING gin (safe_sections_tsv(sections));

-- Step 2: drop the old column-backed index (instant)
DROP INDEX CONCURRENTLY IF EXISTS idx_papers_fulltext_sections_tsv;

-- Step 3: drop the stored column (AccessExclusiveLock, brief — column
-- drop itself is a catalog operation; actual TOAST reclaim happens at VACUUM)
ALTER TABLE papers_fulltext DROP COLUMN IF EXISTS sections_tsv;

-- Step 4: reclaim TOAST space; run after-hours, may take several hours
-- on 214 GB of TOAST data
VACUUM (VERBOSE, ANALYZE) papers_fulltext;

-- Step 5: rename the new index to the canonical name so downstream
-- references in pg_stat_user_indexes / monitoring remain stable
ALTER INDEX idx_papers_fulltext_sections_tsv_expr
    RENAME TO idx_papers_fulltext_sections_tsv;
```

### Code changes required

Two call sites use `sections_tsv` as a column reference rather than purely
as an implicit GIN target:

1. **`src/scix/mcp_server.py` line 5503** — `ts_rank(pf.sections_tsv, ...)`.
   Must change to `ts_rank(safe_sections_tsv(pf.sections), ...)`. This is a
   per-row function call on the already-fetched `sections` JSONB and is
   evaluated only over the GIN-filtered candidate set (typically ≤ fanout rows,
   default 100), so overhead is negligible.

2. **`migrations/064_section_bm25_index.sql` `search_sections_bm25` SQL
   function** — line 58 uses `ts_rank_cd(pf.sections_tsv, ...)`. If this
   function remains in use it must be updated to
   `ts_rank_cd(safe_sections_tsv(pf.sections), ...)` via a `CREATE OR REPLACE`.

Both call sites use `sections_tsv` only for `ts_rank`/`ts_rank_cd` scoring on
the GIN-filtered result set, never for `SELECT sections_tsv FROM ...` projection.
No call site reads the stored tsvector bytes directly; transition is safe.

### Why `safe_sections_tsv` is valid as an expression index key

PostgreSQL requires index expressions to be IMMUTABLE. `safe_sections_tsv` is
declared `IMMUTABLE` (migration 063, line 124). Its internals call
`to_tsvector('english', ...)` which is itself IMMUTABLE. The `PARALLEL UNSAFE`
annotation is a separate axis (it restricts use inside parallel workers due to
the `BEGIN/EXCEPTION` subtransaction, but does not affect index usability).

The planner uses an expression index when the query `WHERE` clause contains
the same expression. The existing queries use:

```sql
WHERE pf.sections_tsv @@ plainto_tsquery('english', %s)
```

After migration, that becomes:

```sql
WHERE safe_sections_tsv(pf.sections) @@ plainto_tsquery('english', %s)
```

The planner will match the expression to the index key and use GIN index scan
for the `@@` test.

### PARALLEL UNSAFE and expression index builds

`CREATE INDEX CONCURRENTLY` spawns a parallel index build by default.
`safe_sections_tsv` is `PARALLEL UNSAFE`. PostgreSQL will fall back to a
single-worker build automatically when the index expression is PARALLEL UNSAFE
— no manual override needed, but build time will be longer (see estimate
below). To avoid surprises, the migration can set
`SET max_parallel_maintenance_workers = 0;` explicitly before the
`CREATE INDEX CONCURRENTLY`.

## Consequences

### Positive

- Recovers ~214 GB of NVMe headroom (the stored column TOAST), enabling zsbd
  and 6hr7 to proceed.
- After the expression index is built, net disk impact is ~27 GB (the GIN
  index stays; the 214 GB TOAST disappears). Peak during build is ~+40 GB for
  the new index before the old one is dropped.
- No semantic change to BM25 retrieval results: `safe_sections_tsv` is the
  same function that produced the stored values.
- Consistent with the production pattern used by `ix_papers_body_tsv` on
  `papers.body`.
- Eliminates the maintenance burden of keeping the stored column in sync with
  the `sections` JSONB on future updates.

### Negative / acceptable trade-offs

- `ts_rank` / `ts_rank_cd` scoring calls in application code now invoke
  `safe_sections_tsv(pf.sections)` per result row at query time instead of
  reading a pre-computed column. This is evaluated only on the GIN-filtered
  set (≤ fanout rows), so wall-clock impact per query is small (< 1 ms added
  for 100 rows). See Open Questions for the p95 concern.
- `safe_sections_tsv` is `PARALLEL UNSAFE`. The expression index build runs
  single-threaded. Estimated build time: 6–12 hours (vs 3–6 hours for the
  column-backed GIN build in migration 064). Must be wrapped in `scix-batch`.
- `VACUUM (VERBOSE, ANALYZE) papers_fulltext` is required after the column
  drop to actually return the TOAST pages to free space. This is a long-running
  read-heavy operation (several hours on a 214 GB TOAST store). Disk headroom
  is not recovered until VACUUM completes.
- `ALTER TABLE ... DROP COLUMN` takes `AccessExclusiveLock` briefly (catalog
  operation only, not a heap rewrite), but any concurrent session holding an
  open transaction on `papers_fulltext` will be blocked until it commits.

## Migration cost estimate

| Phase | Duration | Peak extra disk | Lock type |
|---|---|---|---|
| `CREATE INDEX CONCURRENTLY` (new expr index) | 6–12 h | +40 GB | none (CONCURRENTLY) |
| `DROP INDEX CONCURRENTLY` (old column index) | seconds | -27 GB | none |
| `ALTER TABLE DROP COLUMN` | seconds | catalog only | AccessExclusiveLock (brief) |
| `VACUUM (VERBOSE) papers_fulltext` | 4–8 h | none | ShareUpdateExclusiveLock |
| **Net headroom recovered** | | **~214 GB** | |

All steps must be run under `scix-batch` to avoid OOM-killing the gascity
supervisor (see CLAUDE.md §Memory isolation).

## Rollback plan

If the expression index build fails or the planner stops using the index after
migration:

1. Re-add the column: `ALTER TABLE papers_fulltext ADD COLUMN sections_tsv tsvector;`
2. Backfill via `scripts/backfill_sections_tsv.py` (resumable, idempotent).
3. Rebuild `CREATE INDEX idx_papers_fulltext_sections_tsv ON papers_fulltext USING gin (sections_tsv);`
4. Revert `src/scix/mcp_server.py` and `search_sections_bm25` to the original
   `pf.sections_tsv` column references.

## Open questions

1. **`ts_rank` latency regression on high-fanout queries.** The two call sites
   invoke `ts_rank(safe_sections_tsv(pf.sections), ...)` over at most `fanout`
   rows (default 100, max observed ~500). `safe_sections_tsv` on a typical
   paper (p50: ~5 KB/section × 30 sections = 150 KB text) takes ~0.3–1 ms per
   row. At 100 rows that is 30–100 ms added latency per query. This is within
   the 500 ms p95 budget from migration 064, but it should be measured in
   staging before cutting over. If latency is unacceptable, a covering
   expression `ts_rank` can be rewritten as an ORDER BY on the GIN score
   (PG16 supports `@@ ... ORDER BY ts_rank(...)` with an index scan if
   `gin_fuzzy_search_limit = 0`), or a partial pre-computed column limited to
   commonly-retrieved papers can be maintained.

2. **Is `safe_sections_tsv` truly equivalent to the backfilled column values?**
   Migration 063 populated the column via `safe_sections_tsv(sections)` in
   batch UPDATE statements. The expression index re-evaluates the same function
   at index-build time. Any row where the JSONB was updated between the
   original backfill and the new index build would show the new value in the
   index (correct behaviour). Rows where `sections IS NULL` return
   `''::tsvector` from the function — GIN does not index empty tsvectors, so
   those rows are silently absent from the index (same as current behaviour
   since NULLs are not indexed). No discrepancy is expected, but a spot-check
   of 1 000 random rows (`safe_sections_tsv(sections) = sections_tsv`) before
   dropping the column would provide confidence.

3. **Does `search_sections_bm25` (migration 064 SQL function) need a
   `CREATE OR REPLACE` update?** That function is defined in SQL and directly
   references `pf.sections_tsv`. After the column drop it will fail with
   `column "sections_tsv" does not exist`. Migration 069 must include a
   `CREATE OR REPLACE FUNCTION search_sections_bm25(...)` that substitutes
   `safe_sections_tsv(pf.sections)` for `pf.sections_tsv`. Confirm whether
   this function is still actively called via MCP (the `mcp_server.py` BM25
   leg appears to use its own inline SQL rather than calling
   `search_sections_bm25` directly — verify before migration).

4. **PARALLEL UNSAFE and future index maintenance.** Auto-ANALYZE triggers
   an expression index re-evaluation on sampled rows. PARALLEL UNSAFE is fine
   for this (auto-analyze is single-process). However, if `safe_sections_tsv`
   is ever rewritten to be `PARALLEL SAFE`, the index build would benefit from
   parallel workers. That refactor (removing the `BEGIN/EXCEPTION` subtransaction
   by using a `DO INSTEAD NOTHING` trigger or a C-language try/catch) is out
   of scope here but worth tracking.
