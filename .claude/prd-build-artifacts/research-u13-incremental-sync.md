# Research — u13 incremental sync (PRD M10)

## Existing tier-1 / tier-2 linker shape

### `scripts/link_tier1.py`

- Exposes `run_tier1_link(conn, *, dry_run=False) -> int`
- Single SQL pass joining `papers.keywords` against `entities.canonical_name`
  and `entity_aliases.alias`, idempotent via `ON CONFLICT DO NOTHING`.
- Writes tier=1 rows with `link_type='keyword_match'`, `confidence=1.0`,
  `match_method='keyword_exact_lower'`.
- The whole SQL literal is marked `# noqa: resolver-lint` so the u03 AST
  lint tolerates it.
- Already scoped by `papers.keywords IS NOT NULL` — **no bibcode filter**
  built in, so the incremental runner has to restrict the joined `papers`
  set itself (we will build a temporary "new papers" filter CTE around it,
  or simpler: add a bibcode-prefix / watermark filter by using a temp view
  / passing a filter clause).

### `scripts/link_tier2.py`

- Exposes `run_tier2_link(conn, *, workers, bibcode_prefix, max_per_entity,
dry_run) -> Tier2Stats`.
- Streams papers via `iter_paper_batches(conn, bibcode_prefix, batch_size)`,
  which takes a `LIKE '<prefix>%'` filter on `bibcode`. Good — the
  incremental runner can leverage the same CTE/filter by passing a
  per-run prefix or (better) wrapping the paper iteration.
- Builds an Aho-Corasick automaton from `curated_entity_core`. We will add
  an optional `ac_automaton.pkl` load path (falling back to inline build
  if the file doesn't exist).
- All writes are `# noqa: resolver-lint` so the AST lint is already happy.

## Existing daily-sync pattern: `scripts/daily_sync.sh`

Current pipeline:

1. Harvest new records (harvest_daily.py)
2. Ingest new records (ingest.py)
3. Backfill body/refs
4. Re-ingest enriched
5. Embed

**No entity linking step exists.** M10 slots in between 4 and 5 (or after 5)
with a 5-minute budget circuit breaker — if it trips, daily sync continues
without blocking, watermark still advances, and `link_catchup.py` picks up
the missed papers out-of-band.

## Schema facts

- `papers.entry_date` is **TEXT** (ISO-8601 format: `2023-09-20T00:00:00Z`).
  `entry_date::timestamptz` cast works (prod sample verified). We use
  `NULLIF(entry_date, '')::timestamptz` defensively.
- No `ingested_at` / `inserted_at` column. Watermark MUST be on `entry_date`.
- No existing `alerts` table. We add it inline in migration 036.
- `document_entities` is the target table; writes honor
  `ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING`.
- Most recent migration on disk: **035**. PRD reservation says u07=032,
  u08=033, u10=034, u11=035 — so **036 is ours** (matches plan).

## AST lint

- `scripts/ast_lint_resolver.py` only scans files under `src/`. `scripts/*`
  are out of scope by design. Tier-1 and tier-2 scripts still carry
  `# noqa: resolver-lint` annotations for parity. We'll do the same on our
  new writer paths when we do any direct INSERT (we won't — we delegate to
  `run_tier1_link` and `run_tier2_link`).

## Test infra

- `tests/helpers.py::get_test_dsn()` returns None if `SCIX_TEST_DSN` unset
  or points at production — tests skip gracefully.
- Existing tier1 / tier2 fixtures use `test_u06_` / `test_u09_` bibcode
  prefixes. We'll use `test_u13_` for this unit's seed rows.

## Circuit breaker design shape

- 3-state FSM: closed / open / half_open.
- On each `call` the breaker checks elapsed wall-clock against
  `budget_seconds` (the 5-min budget is the time budget for the **whole**
  sync run, not per-call; see the `TimeBudget` wrapper).
- Alternative interpretation given the acceptance criterion ("forced
  10-minute delay trips the breaker and the sync completes without
  links"): the breaker is a **time budget guard**. If we've already spent
  `budget_seconds` inside the breaker, it trips to `open`, and subsequent
  `call()` invocations short-circuit (raise `CircuitBreakerOpen`).
- 2 consecutive trips on the same run (or across 2 back-to-back runs) →
  row into `alerts` with `severity='page'`.
