# Plan — u13 incremental sync (PRD M10)

## Files (6)

### 1. `migrations/036_link_runs_watermark.sql`

- `CREATE TABLE link_runs (run_id BIGSERIAL PK, max_entry_date TIMESTAMPTZ,
timestamp TIMESTAMPTZ DEFAULT now(), rows_linked INT,
status TEXT DEFAULT 'ok', trip_count INT DEFAULT 0)`.
- `CREATE TABLE alerts (id BIGSERIAL PK, severity TEXT NOT NULL,
source TEXT NOT NULL, message TEXT NOT NULL,
created_at TIMESTAMPTZ DEFAULT now())`.
- Index on `link_runs(timestamp DESC)` for "latest watermark" lookup.

### 2. `src/scix/circuit_breaker.py`

- `CircuitBreakerOpen` exception.
- `CircuitBreaker` dataclass-ish class:
  - `budget_seconds: float`
  - `state: Literal["closed","open","half_open"] = "closed"`
  - `_started_at: float | None`
  - `_trip_count: int`
  - `start()` — starts the wall-clock budget.
  - `check()` — raises `CircuitBreakerOpen` if elapsed > budget (flips to
    `open`), else no-op. Call before each unit of work.
  - `trip()` — force-trip.
  - `reset()` — back to `closed`.
  - `half_open()` — after the breaker has been open and we want one probe.
- Pure Python, no DB. Unit-testable without postgres.

### 3. `scripts/link_incremental.py`

- Reads last `max_entry_date` from `link_runs` (0001-01-01 fallback).
- Selects new papers `WHERE entry_date::timestamptz > $last_watermark`
  into a temp view / temp table (`_u13_incremental_papers`).
- Wraps work in a `CircuitBreaker(budget_seconds=300.0)`.
- Step 1: `run_tier1_link_scoped(conn, bibcodes)` — adapter over
  `link_tier1.run_tier1_link` that adds a `WHERE papers.bibcode = ANY(...)`
  predicate. Rather than re-writing the SQL, we use an even simpler
  approach: **temporarily filter** via a transient `CREATE TEMP TABLE
papers_incremental AS SELECT * FROM papers WHERE entry_date::tstz > wm`
  — then run a local variant of tier-1 SQL against that table (or, since
  the tier-1 SQL is a module constant, we run a tiny adapter that
  replaces `FROM papers p` → `FROM papers p WHERE p.bibcode IN (...)`).
  - Cleanest: add **bibcode-prefix-like filter** to tier-1 by writing a
    small wrapper `_run_tier1_for_bibcodes(conn, bibcodes)` that calls
    `link_tier1.run_tier1_link` **after** restricting through a temp
    table via `CREATE TEMP TABLE _u13_incremental_bibcodes (bibcode TEXT
PRIMARY KEY)` and then running an adapted INSERT. Simpler still: do
    the insert here with an explicit SQL that joins against
    `_u13_incremental_bibcodes`. Keep `# noqa: resolver-lint` — scripts
    are out of AST-lint scope anyway.
- Step 2: tier-2 — uses the existing `link_tier2.run_tier2_link` with a
  **per-run synthetic bibcode prefix** won't work for real ADS bibcodes.
  Instead: create a temp table scoping the `iter_paper_batches` — but
  that function is hard-coded to `SELECT bibcode, abstract FROM papers
...`. We'll write a small `_run_tier2_for_bibcodes(conn, bibcodes)` that
  inlines the tier-2 pipeline against our temp scope, reusing
  `link_tier2.fetch_entity_rows` + `build_automaton` +
  `link_abstract` + the `_INSERT_SQL` template from the script (which we
  import). The automaton load path is optional
  (`data/entities/ac_automaton.pkl`) — fallback builds inline.
- Budget trip: catch `CircuitBreakerOpen`, record `status='tripped'` on
  the row; watermark **still advances** (per AC3); write alert if this
  was the 2nd consecutive trip.
- Watermark-staleness: compute
  `now() - max(latest.max_entry_date) > interval '24 hours'` → write
  alert `severity='page'`, `source='watermark_staleness'`.
- Writes new `link_runs` row at end. Returns 0 on success (including
  budget-trip "graceful degradation").

### 4. `scripts/link_catchup.py`

- Finds papers with `entry_date::tstz <= <latest watermark>` that have
  zero tier=1/tier=2 rows in `document_entities` (the set skipped by a
  tripped breaker).
- Runs the same scoped tier-1/tier-2 helpers (imported from
  link_incremental) over that set. No circuit breaker — catch-up runs
  off-peak and is allowed to take as long as it needs.
- CLI: `python scripts/link_catchup.py --db-url ... --limit N`.

### 5. `tests/test_incremental_sync.py`

- Fixture: `test_u13_` bibcodes, 10 papers with abstracts and entry_dates,
  3 curated entities via `curated_entity_core`, aliases wired.
- Test A: fresh watermark (far past), run
  `link_incremental.main(["--db-url",dsn])` — assert `link_runs` row
  appears, `document_entities` rows for the fixture papers exist.
- Test B: force `budget_seconds=0.001`, assert the run completes without
  raising, watermark row is written with `status='tripped'`, zero rows
  linked.
- Test C: run catchup after Test B, assert previously-skipped papers now
  have entity rows.
- Test D: two consecutive trips → alerts row with `severity='page'`.
- Test E: forced stale watermark (manually insert a `link_runs` row with
  `max_entry_date = now() - 48h`) → staleness alert fires.

### 6. `tests/test_circuit_breaker.py`

- `CircuitBreaker` unit tests (no DB):
  - initial state is closed
  - after `start()` and elapsed < budget, `check()` is a no-op
  - after `start()` and elapsed > budget, `check()` raises
    `CircuitBreakerOpen` and transitions to `open`
  - `reset()` returns to closed, clears trip count
  - `trip()` explicit forces open
  - trip count increments on each trip

## Execution order

1. Write migration 036 + apply to `scix_test`.
2. Implement circuit_breaker.py + test_circuit_breaker.py (pure unit).
3. Implement link_incremental.py.
4. Implement link_catchup.py.
5. Write test_incremental_sync.py, fix as we go.
6. Run ast lint, run pytest.
7. Commit.
