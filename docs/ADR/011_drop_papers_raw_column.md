# ADR-011: Drop `papers.raw` JSONB Column

- **Status**: Proposed
- **Deciders**: SciX maintainers
- **Scope**: `papers` table, ingest pipeline (`src/scix/field_mapping.py`), citation-context pipeline (`src/scix/citation_context.py`), test suite
- **Supersedes**: none
- **Related**: storage tiering policy (CLAUDE.md), bead `scix_experiments-zsbd` (section_pipeline encode), bead `scix_experiments-6hr7` (citation_contexts shard backfill)

## Context

NVMe storage (`/dev/nvme1n1p2`, 1.9 TB) is 96% full with ~93 GB free. Two active
workstreams require staging capacity — section_pipeline INDUS encode and citation_contexts
shard backfill — and both are gated on available disk.

`papers.raw` is a JSONB column introduced during initial ingest as a catch-all for ADS
JSONL fields not yet mapped to dedicated SQL columns. The column now has three problems:

1. **Redundant content.** `field_mapping.py` explicitly notes that `reference` is
   "preserved in raw for provenance" and that `citation` is "kept only in raw (incomplete
   per ADS API)". All other valuable ADS fields received dedicated columns in migrations
   001–012. The provenance copy of `reference` is fully superseded by `citation_edges`
   (299M rows, primary key `(source_bibcode, target_bibcode)`). The `citation` field
   was never used for retrieval or analysis. The `body` field that was temporarily stored
   in `raw` was stripped out per-row by `scripts/backfill_body.py` during the body-ingest
   phase; that backfill is complete.

2. **Significant disk cost.** `pg_column_size` reports ~11 GB of compressed JSONB column
   data. The `papers` TOAST table is 292 GB total; `body` accounts for ~204 GB and the
   `tsvector tsv` column for ~29 GB, leaving ~59 GB unaccounted that spans `raw` plus
   array columns. The true reclaim after `DROP COLUMN` + `VACUUM FULL` is estimated at
   15–30 GB (see Consequences for the caveat on when `VACUUM FULL` can run).

3. **The source of truth already exists on NAS.** Every ADS record's raw JSONL is on
   `/mnt/scix_offload/ads_metadata_by_year_picard/` — the same files documented in
   CLAUDE.md as the canonical upstream archive, covering years 1800–2026 as `.jsonl.gz`
   or `.jsonl[.xz]`. Re-ingest is the defined recovery path; no off-database copy of
   `papers.raw` is needed for that.

## Callsite Inventory

Full grep: `grep -rn "papers\.raw\|p\.raw\|\.raw::jsonb\|FROM papers WHERE raw\|SELECT raw FROM papers\|raw->" src/ scripts/ tests/ migrations/`

### `src/scix/citation_context.py` lines 649–654 and 796–808

The pipeline SELECT reads `p.raw` to obtain the `reference` array — an ordered list of
bibcodes used to resolve positional citation markers (`[N]`) in body text. The WHERE
clause filters with `p.raw::jsonb ? 'reference'` before streaming rows.

**Fields actually accessed**: `raw_dict.get("reference")` — the reference bibcode list
only. No other key from `raw` is read.

**Refactor path**: `citation_edges` holds all outgoing edges as
`(source_bibcode, target_bibcode)` pairs with 299M rows and a covering primary key.
However, `citation_context.py` requires the references in **ordered array form**
(the position of a bibcode in the array is the index used by `[N]` markers). The
`citation_edges` table does not store this ordering. Two options:

- **Option A (recommended)**: Add a nullable `INTEGER` column `ref_position` to
  `citation_edges` during re-ingest (a zero-cost schema change on a new column). During
  re-ingest, `field_mapping.py` already has the ordered `reference[]` array; emit
  `(source, target, position)` triples. Then replace the `raw`-based pipeline SELECT
  with a query that assembles the ordered list via `array_agg(target_bibcode ORDER BY
  ref_position)` grouped by `source_bibcode`, joined against papers for the body. This
  fully eliminates the `raw` dependency without losing the ordering signal.

- **Option B (simpler, lower-fidelity)**: Accept that ordered-position resolution
  (`[N]` markers) is used for ~15–20% of citations in the body; the majority of
  citation markers are author-year format resolved independently of position. Drop the
  `? 'reference'` filter and replace it with `EXISTS (SELECT 1 FROM citation_edges ce
  WHERE ce.source_bibcode = p.bibcode)`. The position-based resolver degrades to
  full-list scan (already implemented as the fallback path in `_resolve_marker_by_scan`).
  Noop for papers with no `citation_edges` rows. No schema change needed.

Option B enables the `DROP COLUMN` immediately. Option A requires a re-ingest pass over
the 2021–2026 JSONL files before the column can be dropped, but produces higher-quality
citation context resolution.

### `scripts/backfill_body.py` lines 23–55

Dead code. Uses `raw ? 'body'` and `raw->>'body'` to find rows where body text was
stored in `raw` but not yet promoted to `papers.body`. The backfill is complete: memory
notes confirm `papers.body` has 14.9M rows, matching the full-text population. Running
`backfill_body.py` today returns "Nothing to backfill." The script can be deleted or
retained as historical reference; it will no longer compile meaningfully against the
schema after `DROP COLUMN`.

### `tests/test_schema.py` lines 110–121

The `test_jsonb_raw_field` test directly writes to and reads from `papers.raw`. It
tests the generic JSONB property, not any application logic. Delete this test after
`DROP COLUMN`.

### `tests/test_field_mapping.py` (multiple lines)

Tests that validate `transform_record` produces the expected `raw` dict for unmapped
fields, that `COLUMN_ORDER[-1] == "raw"`, and that `raw IS NULL` when all fields are
mapped. These tests must be updated to remove the `raw` assertion cases and update
`COLUMN_ORDER` expectations.

### `tests/test_metadata_coverage.py` lines 339–368

Integration tests asserting that `facility` and similar unmapped fields appear in
`papers[0]["raw"]`. After the column drop and the ingest change (see Migration Plan),
these must be updated to reflect that `facility` is already a dedicated column and
`raw` will not exist.

### `src/scix/field_mapping.py` lines 93 and 287–292

`COLUMN_ORDER` includes `"raw"` as the last element. `transform_record` collects
unmapped residual JSONL fields into `raw_fields` and serializes them as JSONB. After
the column drop, both must be removed: drop `"raw"` from `COLUMN_ORDER`, remove the
`raw_fields` collection loop, and remove the `row["raw"] = ...` assignment.

## Decision

Drop `papers.raw`. Immediate steps:

1. Refactor `citation_context.py` using Option B (unblock the column drop with no
   re-ingest dependency). Track Option A (add `ref_position` to `citation_edges`) as a
   separate follow-up bead to improve citation resolution quality.
2. Execute `ALTER TABLE papers DROP COLUMN raw;` — DDL-only, fast, no table rewrite.
3. Remove `raw` from `field_mapping.py` `COLUMN_ORDER` and `transform_record`.
4. Delete `scripts/backfill_body.py`.
5. Update affected tests.
6. Defer `VACUUM FULL` / `pg_repack` to a maintenance window when disk allows it
   (see Consequences).

## Consequences

### Positive

- Eliminates the JSONB column from all new writes immediately after the `ALTER TABLE`.
- Frees ~11 GB of column-data pages immediately to the free-space map (available for
  new writes without `VACUUM FULL`).
- Estimated ~15–30 GB reclaimed after a future `VACUUM FULL` or `pg_repack` on `papers`.
- Removes the `raw::jsonb ? 'reference'` predicate from the citation pipeline; the
  replacement `EXISTS` predicate hits `idx_cite_target` (or a new partial index on
  `citation_edges.source_bibcode`, which already forms the primary key).
- Removes the ingest overhead of serializing residual fields to JSONB on every record.

### Negative / Caveats

- **VACUUM FULL is not immediately viable.** `VACUUM FULL papers` rewrites the entire
  table and requires ~411 GB of free disk during the rewrite (the full current table
  size). With 93 GB free, this would abort partway. `pg_repack` has the same footprint
  requirement. Disk reclaim must wait until disk is freed by other means (e.g., after
  section_pipeline and citation_contexts backfills complete and their temp staging is
  cleaned up), or until the Qdrant migration moves `paper_embeddings` (253 GB) off NVMe.
  Until then, the 11 GB column-data pages are returned to the free-space map for
  PostgreSQL reuse — not returned to the OS.

- **`reference[]` ordering lost.** Citation context resolution degrades slightly for
  papers using positional `[N]` citation markers until Option A (re-ingest with
  `ref_position`) is implemented. The fallback scan-based resolver still fires.

- **Unknown consumers.** gascity workers or ad-hoc scripts that query `papers.raw`
  directly will break. Mitigation: the grep above found no additional callsites in
  `src/`, `scripts/`, `tests/`, or `migrations/`. The MCP server tools do not expose
  `raw` in any tool response (confirmed in `mcp_server.py`). The eval harness and viz
  module do not reference `raw`. Risk is low but operators should audit any external
  notebook or one-off scripts before executing the migration.

- **Schema evolution / new ADS fields.** If ADS adds fields not in the current column
  set, they will be silently dropped on ingest until a new column is added. Previously,
  `raw` would have caught them. Mitigation: the ADS API schema is stable on multi-year
  timescales; the field list has not changed since migration 012 was finalized. Monitor
  ADS API changelog and add columns via migration when new fields appear.

- **Compliance.** No audit or compliance requirement to retain raw upstream payload in
  the database was identified. CLAUDE.md does not mention one. The NAS archive is the
  upstream record.

## Migration Plan

### Step 1 — Refactor `citation_context.py` (no downtime)

Replace `_SELECT_PAPERS_BASE`:

```python
_SELECT_PAPERS_BASE = """
    SELECT p.bibcode, p.body
    FROM papers p
    WHERE p.body IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM citation_edges ce
          WHERE ce.source_bibcode = p.bibcode
      )
      AND NOT EXISTS (
          SELECT 1 FROM citation_contexts cc
          WHERE cc.source_bibcode = p.bibcode
      )
"""
```

Replace the `for bibcode, body, raw_val in cur:` loop body. Fetch the ordered reference
list from `citation_edges` inside the loop (or via a batched JOIN query to avoid N+1):

```sql
-- Per-paper ordered list, fallback to insertion order (no ref_position yet)
SELECT target_bibcode
FROM citation_edges
WHERE source_bibcode = %s
ORDER BY (target_bibcode)  -- arbitrary stable order until ref_position added
```

Call `process_paper(bibcode, body, refs)` as before.

### Step 2 — Drop column (fast DDL, ~1 second)

```sql
ALTER TABLE papers DROP COLUMN IF EXISTS raw;
```

This acquires `AccessExclusiveLock` briefly for the catalog update only. No table
rewrite. Disk pages are not freed to the OS yet — they are marked dead in the free-space
map and available for future inserts.

### Step 3 — Update `field_mapping.py`

- Remove `"raw"` from `COLUMN_ORDER`.
- Remove the `raw_fields` dict accumulation loop and `row["raw"] = ...` assignment.
- Remove `"raw"` from `_MAPPED_JSONL_FIELDS` (or confirm it is already excluded as a
  special case — it is not in any of the `DIRECT_*` frozensets; the sentinel comment at
  line 194 explains why `reference` and `citation` are excluded from `_MAPPED_JSONL_FIELDS`
  but note they will still be silently ignored after the loop is removed).

### Step 4 — Remove dead code

Delete `scripts/backfill_body.py`. The script is not referenced by `cron`, `Makefile`,
or any other orchestration file (confirmed by grep).

### Step 5 — Update tests

- Delete `test_schema.py::TestPapersTable::test_jsonb_raw_field`.
- In `test_field_mapping.py`: remove assertions on `row[COL["raw"]]`, update
  `COLUMN_ORDER` length/last-element checks.
- In `test_metadata_coverage.py`: remove `papers[0]["raw"]` assertions; `facility` is
  already asserted via its dedicated column elsewhere.

### Step 6 — VACUUM FULL (deferred maintenance window)

When NVMe has ≥ 450 GB free (after Qdrant migration of `paper_embeddings` or equivalent
space event):

```sql
-- Online alternative preferred if available:
-- pg_repack -t papers  (requires pg_repack extension)

-- Blocking alternative (requires maintenance window):
VACUUM FULL ANALYZE papers;
```

This reclaims the dead pages from the OS and shrinks the table relation file. Estimated
duration: 2–6 hours depending on I/O. Requires `AccessExclusiveLock` for the full
duration if using `VACUUM FULL`; `pg_repack` operates online with a brief final lock.

## Rollback Plan

If `papers.raw` needs to be restored:

1. Add the column back: `ALTER TABLE papers ADD COLUMN raw jsonb;`
2. Re-ingest from NAS: `python -m scix.ingest ads_metadata_by_year_picard/<year>/` for
   each affected year (2021–2026 are present as `.jsonl.xz`, `.jsonl`, `.jsonl.gz` files
   on `/mnt/scix_offload/ads_metadata_by_year_picard/`; pre-2021 years are also available
   as `.jsonl.gz`). The ingest pipeline already handles all three formats.
3. Re-ingest cost estimate: ~5.5M records/year × 6 years = ~7M records in the recent
   cohort. At observed ingest throughput (~20K records/s on NVMe with COPY), bulk
   re-ingest of the 2021–2026 cohort takes ~6 minutes of wall time, plus additional time
   for the HNSW index to catch up on any new `paper_embeddings` inserts. The JSONB
   serialization step in `transform_record` adds minimal overhead. Full corpus
   re-ingest (all years, ~7M+ records) would take ~30–60 minutes.

## Open Questions

1. **Option A for `citation_edges.ref_position`**: Should the re-ingest to populate
   `ref_position` be scheduled immediately as a follow-up bead, or deferred until
   citation-context recall quality degrades measurably? The fallback scan resolver fires
   today for papers where marker parsing fails, so quality regression may be minor. A
   sampling comparison of context quality before/after would inform priority.

2. **`citation` field in `raw`**: The `field_mapping.py` comment at line 195 notes
   `citation` is "kept only in raw (incomplete per ADS API; derived from reference[]
   inverse)". After the column drop, this field is permanently lost. Is there any
   downstream use case — e.g., forward-citation counts or citation provenance — for
   which `citation[]` from ADS provides signal not already in `citation_edges`?
   Confirm explicitly before executing Step 2.
