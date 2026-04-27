# PRD: Citation-Contexts Extraction Throughput Backfill

> Bead: `scix_experiments-79n.1` (parent: `scix_experiments-79n`,
> sibling: `scix_experiments-79n.3` — operational rollout, hot-table
> growth, index rebuild, MCP downtime).
>
> This PRD covers the **extraction throughput** plan only: how to drive
> the existing pipeline in `src/scix/citation_context.py` from its
> current 30,443-source-paper coverage to the full
> 9,885,995-source-paper extractable population. Storage planning,
> index rebuild, partition strategy, `v_claim_edges` materialization
> strategy, and MCP-side downtime planning are **out of scope** —
> see `scix_experiments-79n.3`.

## Status (2026-04-27)

Build PRD — ready to execute once cross-referenced concerns in
`79n.3` (storage headroom in particular) are resolved. Pipeline code
already lands the heavy lifting. The remaining work is operator
choreography, plus three small deltas in
`scripts/extract_citation_contexts.py` (production-safety flag,
ingest_log integration, optional sharding flag) needed before a
multi-day run can be left unattended.

> **Reading note (added in wave-1 review).** Line numbers in this PRD
> reference `src/scix/citation_context.py` as it existed at
> `fcd450f` (pre-wave-1). Bead `scix_experiments-79n.2` (merged in
> wave-1) added ~250 lines of author-year extraction at the top of
> the file; line numbers below are no longer accurate. Use **function
> names** for navigation (`extract_citation_contexts`,
> `_enrich_with_sections`, `_SELECT_PAPERS`, etc.) — these are
> stable. Additionally, the 79n.2 author-year extractor produces
> rows from papers whose citation style does not use `[N]` markers,
> so the **~267M row estimate is now conservative**. Pilot
> measurement should run with both extractors enabled before the
> full run is kicked off.

## Goal

Run the existing citation-context extraction pipeline over the full
extractable source-paper population. "Done" means:

1. `citation_contexts` row count grows from **825,115** to a
   measured count consistent with **80% or more** of the
   9,885,995-paper eligible set having been processed (estimated
   landing band: **~210M to ~270M rows**, see
   [Target state](#target-state) for the math).
2. `v_claim_edges` remains queryable while the run is in flight
   (no extended downtime; refresh cadence handled in `79n.3`).
3. `ingest_log` records a single canonical end-state row under
   filename `'citctx_full_backfill_2026'` so subsequent runs can
   prove the backfill ran.
4. Coverage of 1000 random `target_bibcode`s sampled from
   `citation_edges` rises from the baseline **11.4%** (measured
   2026-04-27) to **>50%**.

## Current state (real DB probes, 2026-04-27)

All numbers below were probed live against `dbname=scix` on
2026-04-27. Reproduce by re-running the SQL in the lines beneath
each row.

| Surface                                                                 | Value                |
| ----------------------------------------------------------------------- | -------------------- |
| `citation_contexts` rows                                                | **825,115**          |
| Distinct `source_bibcode` in `citation_contexts`                        | **30,443**           |
| Distinct `target_bibcode` in `citation_contexts`                        | ~250,000 (est)       |
| Avg context rows per source paper (rough)                               | **27.10**            |
| Eligible source-paper population (`body NOT NULL` ∧ `raw ? 'reference'`) | **9,885,995**        |
| Total citation edges (`citation_edges`)                                 | 299,397,468          |
| Distinct `(source,target)` pairs covered by ≥1 context                  | **380,403**          |
| Edge coverage (distinct-pair / total)                                   | **0.127%**           |
| Target-side coverage on 1000 random `target_bibcode`s (`v_claim_edges`) | **11.4%** (114/1000) |
| `citation_contexts` heap size                                           | **1,451 MB**         |
| `citation_contexts` total relation size (heap + indexes + TOAST)        | **1,553 MB**         |
| Bytes per row (heap-only, 1451 MB / 825,115)                            | **1,844 B**          |
| Index `idx_citctx_source_target` size                                   | 43 MB                |
| Index `idx_citctx_target` size                                          | 23 MB                |

Probes:

```sql
-- row count
SELECT count(*) FROM citation_contexts;
-- distinct sources
SELECT count(DISTINCT source_bibcode) FROM citation_contexts;
-- eligible population
SELECT count(*) FROM papers
WHERE body IS NOT NULL AND raw::jsonb ? 'reference';
-- distinct edges covered
SELECT count(*) FROM citation_edges ce
WHERE EXISTS (
  SELECT 1 FROM citation_contexts cc
  WHERE cc.source_bibcode = ce.source_bibcode
    AND cc.target_bibcode = ce.target_bibcode
);
-- baseline 1000-random-target coverage
WITH random_targets AS (
  SELECT DISTINCT target_bibcode
  FROM citation_edges TABLESAMPLE SYSTEM (0.005) LIMIT 1000
)
SELECT count(*) AS sampled,
       count(*) FILTER (
         WHERE EXISTS (
           SELECT 1 FROM v_claim_edges v
           WHERE v.target_bibcode = rt.target_bibcode
         )
       ) AS covered
FROM random_targets rt;
-- table size and bytes/row
SELECT pg_size_pretty(pg_relation_size('citation_contexts')),
       pg_relation_size('citation_contexts') / 825115 AS bytes_per_row;
```

## Target state

- **Source papers processed**: ≥ 0.80 × 9,885,995 ≈ **7,908,796**.
  The remaining ≤20% accounts for papers whose `body` text has no
  parseable `[N]` markers (regex finds zero matches), papers whose
  `references` array is malformed, or papers whose all-zero context
  yield triggers no INSERTs.
- **Estimated post-run row count**:
  9,855,552 papers × 27.1 contexts/paper ≈ **267M rows**
  (lower bound: 210M if avg drops to 21/paper; upper bound: 300M if
  modern STEM papers run higher than the 27.1 historical avg).
  Estimate methodology: avg-rows-per-source from existing 30,443
  sources × remaining 9,855,552 sources. **Mark explicitly as
  estimate**; the only way to nail this down is to run a
  representative 10K-paper pilot under the same code path before
  greenlighting the full job.
- **Target-side coverage** on 1000 random `target_bibcode`s rises
  from baseline **11.4% → ≥50%**. The lift is non-linear because the
  long tail of `citation_edges` targets old papers that may not have
  body text in `papers` (and so will never be covered); the 50%
  number is the explicit acceptance gate, not an upper bound.
- **`ingest_log` row** for filename `'citctx_full_backfill_2026'` in
  status `'complete'` with `records_loaded` matching the
  source-paper count actually processed.

## Pipeline (as-is)

The pipeline is already implemented in
`src/scix/citation_context.py`. File:line references throughout:

1. **SELECT eligible papers** —
   `_SELECT_PAPERS` at
   [`src/scix/citation_context.py:291-301`](../../src/scix/citation_context.py).
   Filters `body IS NOT NULL AND raw::jsonb ? 'reference' AND NOT
   EXISTS (SELECT 1 FROM citation_contexts cc WHERE
   cc.source_bibcode = p.bibcode)`. The `NOT EXISTS` clause is the
   resume mechanism: a killed run resumes by re-running the same
   command, and only papers with no rows yet are reprocessed.
2. **Extract `[N]` markers** — `extract_citation_contexts` at
   [`src/scix/citation_context.py:133`](../../src/scix/citation_context.py)
   uses `_CITATION_RE` (line 60) with the `_word_boundary_window`
   helper at line 91 to build ~250-word context windows.
3. **Resolve `N → bibcode`** — `resolve_citation_markers` at
   [`src/scix/citation_context.py:175`](../../src/scix/citation_context.py)
   maps the 1-indexed marker numbers into the paper's `references`
   array, dropping out-of-range and non-string entries silently.
4. **~250-word context window** — set inside
   `extract_citation_contexts` via `_word_boundary_window`. Output
   is hard-capped at `_CONTEXT_TEXT_MAX_CHARS = 1000` at write time
   ([`src/scix/citation_context.py:289`](../../src/scix/citation_context.py))
   to match what `v_claim_edges` already truncates.
5. **`_enrich_with_sections`** —
   [`src/scix/citation_context.py:218`](../../src/scix/citation_context.py).
   Annotates each marker with the section it falls into. **Note**:
   the parent bead description references
   `papers_fulltext.sections` JSONB, but the current implementation
   instead invokes `parse_sections(body)` (defined at
   [`src/scix/section_parser.py:85`](../../src/scix/section_parser.py),
   called from
   [`src/scix/citation_context.py:274`](../../src/scix/citation_context.py))
   on the raw body. See
   [Open questions](#open-questions--tbd) — switching to the
   `papers_fulltext.sections` JSONB path is its own decision and
   should be confirmed before this PRD lands.
6. **SciBERT-SciCite intent classification** — already 100%
   backfilled across the existing 825K rows by
   `scripts/backfill_citation_intent.py` (parent bead 79n notes).
   Re-run after each context-extract checkpoint to incrementally
   classify newly-inserted rows.

The COPY-into-staging-and-INSERT pattern at
[`src/scix/citation_context.py:303-339`](../../src/scix/citation_context.py)
keeps the write loop hot: `_flush_contexts` materializes a batch
into a session-temp table and then issues one bulk
`INSERT...SELECT` per batch.

## Throughput plan

### Measured throughput (from parent bead 79n, 2026-04-25)

- **70 source-papers/sec** single-process on the RTX 5090 host. The
  bottleneck is CPU-bound regex + section parsing in
  `process_paper`, not the DB write path. Measurement note: this
  predates the 2026-04-26 hnsw.ef_search reduction and the
  2026-04-27 baseline numbers, but the single-process bottleneck is
  the per-paper extraction, which is independent of those changes.
- Single-process ETA at 70 pps:
  9,855,552 papers / 70 ≈ **140,793 sec ≈ 39 hours**.

### Recommended run shape

The single-process ETA is "barely tolerable" but not friendly to
the 2-3 day operator-attention budget called out in the parent
bead. Recommend two improvements before greenlight:

1. **Concurrent worker shards** — partition the eligible set by
   `hashtext(bibcode) % N` and run N workers, each scoped to its
   own shard. The COPY-into-temp-table-then-INSERT path is per-
   worker safe (each worker has its own temp table); the only
   contention is on the `citation_contexts` index updates from
   parallel inserts.
   - **Recommended N = 4** workers. Empirical reasoning: the host
     has ample CPU, the DB write path is not the bottleneck at
     single-worker rates, and 4× worker count brings ETA to
     **~10 hours** if extraction scales linearly. **Cap at 6**;
     beyond that, the index-update contention on a hot table
     starts dominating (this assumption is unmeasured — see
     [Open questions](#open-questions--tbd)).
   - This requires a CLI flag in
     `scripts/extract_citation_contexts.py`, e.g.
     `--shard <i>/<n>`, that injects `AND mod(hashtext(p.bibcode),
     <n>) = <i>` into the SELECT. **Implementation delta required
     before run.**
2. **Pilot pass first** — before kicking off the full run,
   process a 10,000-paper random sample and measure the actual
   contexts-per-paper ratio against the 27.1 historical estimate.
   If the modern-paper subset runs >35 contexts/paper, the row-
   count projection (and the 79n.3 storage plan) need revisiting.

### Recommended invocation

```bash
# inside a tmux pane (4-day budget per worker)
SHARD=0 N=4
scix-batch --mem-high 8G --mem-max 12G \
  python scripts/extract_citation_contexts.py \
    --allow-prod \
    --shard "${SHARD}/${N}" \
    --batch-size 1000

# repeat with SHARD=1, 2, 3 in separate panes
```

The batch size of 1000 is the existing default in `run_pipeline`
([`src/scix/citation_context.py:344`](../../src/scix/citation_context.py))
and was empirically validated by the existing 825K-row population.
Do not change without re-baselining.

## Operational guardrails

> **Cross-ref**: storage headroom, index rebuild, partition strategy,
> `v_claim_edges` materialization, MCP-side query latency, and
> backup-tier integration are owned by `scix_experiments-79n.3`.
> This PRD only covers the extraction-side guardrails.

### G1. `scix-batch` wrapping is mandatory

Per `CLAUDE.md` §Memory isolation, any script expected to run
longer than ~1 minute or allocate more than a few GB MUST be
wrapped in `scix-batch`. The full backfill takes hours and will
hold a streaming cursor against `papers` for its lifetime; gascity-
supervisor collateral OOM-kills are not acceptable. The
`scix-batch` wrapper sets `MemoryHigh=20G`, `MemoryMax=30G`, and
`ManagedOOMPreference=avoid` in a transient `systemd-run --scope`
unit.

### G2. Production-DSN gating (delta required)

`scripts/extract_citation_contexts.py` does **not currently** carry
the `--allow-prod` + `INVOCATION_ID` guard pattern that
`scripts/recompute_citation_communities.py` uses (verified by
`grep -n 'allow_prod\|INVOCATION_ID' scripts/extract_citation_contexts.py`
returning nothing on 2026-04-27). Before a multi-day production run
the script must be amended to mirror that pattern:

- Add `--allow-prod` argparse flag.
- Resolve effective DSN, `is_production_dsn(dsn) and not allow_prod`
  → exit 2 with a redacted error.
- `args.allow_prod and not os.environ.get("INVOCATION_ID")` →
  exit 2 telling the operator to relaunch via `scix-batch`.

This is a self-contained delta; no design discussion needed.
Reference implementation:
[`scripts/recompute_citation_communities.py:425-458`](../../scripts/recompute_citation_communities.py).

### G3. `ingest_log` integration (delta required)

`run_pipeline` does not currently write to `ingest_log`. The
acceptance criterion ("`ingest_log` records final state under
filename `'citctx_full_backfill_2026'`") requires an explicit log
write. Recommend two-step pattern:

- On pipeline start, INSERT `(filename='citctx_full_backfill_2026',
  status='in_progress')` if not already present.
- On pipeline end (in the `finally` block at
  [`src/scix/citation_context.py:439`](../../src/scix/citation_context.py)),
  UPDATE the row to `status='complete'`,
  `records_loaded=papers_processed`,
  `edges_loaded=total_inserted`, `finished_at=now()`.

When sharded, each worker writes its own filename
`citctx_full_backfill_2026_shard_<i>_of_<n>` and a final
post-run admin step rolls them into the canonical
`citctx_full_backfill_2026` row.

### G4. Resume strategy

Already implicit and battle-tested:

- The SELECT at
  [`src/scix/citation_context.py:291`](../../src/scix/citation_context.py)
  excludes any source paper that already has at least one row in
  `citation_contexts`. A killed run resumes by re-running the same
  command — already-processed papers skip out via the `NOT EXISTS`
  index-only scan against `idx_citctx_source_target`.
- Recommend logging `papers_processed` every 1000 papers (already
  done at
  [`src/scix/citation_context.py:416-424`](../../src/scix/citation_context.py))
  and snapshotting to `logs/citctx_backfill/<shard>.log` so resume-
  after-crash has visibility into what was reached.

### G5. Disk-pressure tripwire

Per `CLAUDE.md` §Storage tiering, `citation_contexts` lives on DS
(NVMe primary). On 2026-04-27 the DS partition has **285 GB free**
(85% used). At 1,844 B/row, the projected row count of ~267M
implies ~**450 GB** of additional heap+index growth — a **hard
overshoot of available DS headroom**. This PRD does not authorize
the full run while DS is in this state; storage planning is owned
by `scix_experiments-79n.3` and that planning must complete before
greenlight.

## Acceptance gates

Each gate is a deterministic SQL/probe an operator can run after
the backfill exits. All five must pass.

### A1. Source-paper coverage

```sql
WITH eligible AS (
  SELECT count(*) AS n FROM papers
  WHERE body IS NOT NULL AND raw::jsonb ? 'reference'
),
covered AS (
  SELECT count(DISTINCT source_bibcode) AS n FROM citation_contexts
)
SELECT covered.n::float / eligible.n AS pct
FROM eligible, covered;
```

PASS if `pct >= 0.80`.

### A2. Row-count growth landed in expected band

```sql
SELECT count(*) FROM citation_contexts;
```

PASS if result is within `[2.1e8, 3.0e8]`. Outside that band,
investigate (likely the contexts-per-paper rate has drifted from
the 27.1 baseline; record the new ratio in the parent bead's
`docs/citation_contexts_coverage.md`).

### A3. `v_claim_edges` queryable

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM v_claim_edges
WHERE source_bibcode = '2020ApJ...893...85B';
```

PASS if planner picks `idx_v_claim_edges_source_intent` and runtime
< 5 s. (`v_claim_edges` refresh cadence and any
materialization-strategy decisions are owned by `79n.3`.)

### A4. Target-side coverage uplift

Re-run the baseline probe from [Current state](#current-state-real-db-probes-2026-04-27):

```sql
WITH random_targets AS (
  SELECT DISTINCT target_bibcode
  FROM citation_edges TABLESAMPLE SYSTEM (0.005) LIMIT 1000
)
SELECT count(*) FILTER (
  WHERE EXISTS (
    SELECT 1 FROM v_claim_edges v
    WHERE v.target_bibcode = rt.target_bibcode
  )
)::float / count(*) AS coverage_pct
FROM random_targets rt;
```

PASS if `coverage_pct >= 0.50`.

### A5. `ingest_log` final state

```sql
SELECT status, records_loaded, edges_loaded, finished_at
FROM ingest_log
WHERE filename = 'citctx_full_backfill_2026';
```

PASS if `status = 'complete'` and `finished_at IS NOT NULL`.

## Open questions / TBD

1. **Section-enrichment source**: bead 79n.1 description references
   `papers_fulltext.sections` JSONB as the section-enrichment input,
   but `_enrich_with_sections` at
   [`src/scix/citation_context.py:218`](../../src/scix/citation_context.py)
   reads from `parse_sections(body)` (an inline parse, not the
   JSONB column). Decision needed before run: keep inline-parse, or
   refactor to read from `papers_fulltext.sections` (which would
   change throughput and add a JOIN). **Owner: parent of 79n.1.**
2. **Worker count cap**: the recommendation of N=4 (cap 6) is
   intuition-based, not measured. Run a 30-minute pilot with N ∈
   {1, 2, 4, 6} on disjoint shards and measure aggregate
   papers/sec; pick the largest N before per-worker pps starts
   degrading. **Methodology: track wall-clock to drain a 100K
   shard at each N.** Inputs come from the 79n.3 sibling decision
   on whether the index-rebuild cost from concurrent inserts is
   acceptable.
3. **Estimated row-count band**: the 210M–300M projection is from
   a single ratio (27.1 contexts/source). A 10K-paper representative
   pilot, sampled at the same `hashtext(bibcode) % N` modulus the
   full run will use, should be required before greenlight. The
   pilot's measured contexts/paper goes back into the 79n.3
   storage plan.
4. **Storage headroom**: DS has 285 GB free, projected need ~450
   GB. Full resolution lives in `scix_experiments-79n.3`, not
   here, but the throughput run cannot proceed without it.

## Cross-references

- Parent: `scix_experiments-79n` —
  [scope] citation_contexts covers only 0.27% of citation graph.
  Established the 70-papers/sec single-process baseline and the
  27 contexts/paper average.
- Sibling: `scix_experiments-79n.3` — operational rollout PRD:
  hot-table growth, index rebuild plan, partition strategy,
  `v_claim_edges` materialization strategy, MCP downtime planning,
  pg_dump/NAS backup-tier integration. **Storage headroom and any
  index-rebuild downtime owned by this bead.**
- Related: `docs/citation_contexts_coverage.md` — agent-facing
  coverage block reference; the `note` string here will need to be
  updated post-backfill to reflect the new edge-coverage figure.
- Related: `scripts/refresh_v_claim_edges.py` — must be invoked
  after backfill (and ideally on a checkpoint cadence during the
  run) for `v_claim_edges` to reflect new rows.
