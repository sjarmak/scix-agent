# Plan — u07 query_log + curated entity core

## 1. `migrations/032_core_promotion_log.sql`

- CREATE TABLE curated_entity_core (
  entity_id INT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
  query_hits_14d INT NOT NULL DEFAULT 0,
  promoted_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );
- CREATE TABLE core_promotion_log (
  id SERIAL PRIMARY KEY,
  entity_id INT NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('promote','demote')),
  query_hits_14d INT,
  reason TEXT,
  ts TIMESTAMPTZ NOT NULL DEFAULT now()
  );
- Indexes on entity_id + ts.
- Wrapped in BEGIN/COMMIT, IF NOT EXISTS, idempotent.

## 2. `src/scix/query_log.py`

- `log_query(tool, query, result_count, session_id=None, is_test=False, conn=None, dsn=None)`:
  - Open connection if not given (use SCIX_TEST_DSN if set else default DSN).
  - INSERT row into query_log with both legacy (`tool_name`, `success=True`)
    and new columns (`ts=now() default`, `tool`, `query`, `result_count`,
    `session_id`, `is_test`).
- `count_hits_since(entity_id_or_name, days)` helper (optional).

## 3. `src/scix/core_lifecycle.py`

- `CORE_MAX = 10_000`
- `promote(entity_id, query_hits_14d=0, reason='manual', conn=None)`:
  - Compute current core size.
  - If at cap, find lowest-query-hit entity and auto-`demote` it first
    (with reason='auto_demote_cap').
  - INSERT INTO curated_entity_core (upsert).
  - Log promote event to core_promotion_log.
- `demote(entity_id, reason='manual', conn=None)`:
  - DELETE from curated_entity_core.
  - Log demote event.
- Unit tests stub with a real DB (SCIX_TEST_DSN).

## 4. `scripts/backfill_query_log.py`

- argparse: `--source <path>`, `--dsn`, `--is-test` (bool), `--dry-run`.
- Read JSONL file where each line = `{tool, query, result_count, session_id, ts?}`.
- For each line, call `log_query(...)`.
- Return count inserted. Expose `run_backfill(path, conn) -> int` for tests.

## 5. `scripts/curate_entity_core.py`

- argparse: `--dsn`, `--output build-artifacts/curated_core.csv`,
  `--strat-output build-artifacts/curated_core_stratification.md`,
  `--window-days 14`, `--max 10000`.
- Three-pass ranking (pure SQL where possible):
  - Pass 1: "gap candidates" — entities with highest zero-result hit counts
    within the window (zero_result = `result_count = 0`).
    For tier-1 purposes we match the query string (case-insensitive, LIKE)
    against entity canonical_name / aliases — simplistic but sufficient for
    test fixtures. Entities without a row still get a (name, 0) placeholder
    with a TODO comment for Wikidata backfill (pass 2).
  - Pass 2: "Wikidata backfill gap closer" — for now, no-op stub with
    TODO-comment; PRD says future N2.
  - Pass 3: "unique ambiguity + ≥1 hit" — entities with
    `ambiguity_class='unique'` AND at least 1 query_log hit in window.
- Union results, dedupe by entity_id, rank by query_hits_14d DESC,
  source as tiebreaker, hard cap 10_000.
- Write CSV with the 7 columns listed in AC3.
- Write stratification markdown with per-source counts.
- Expose `run_curation(conn, output, strat_output, window_days, max_n)` for tests.

## 6. `tests/test_query_log.py`

- Skip if SCIX_TEST_DSN not set.
- Test: `log_query()` writes a row; assert `ts > now() - 5s`.
- Test: backfill from a fixture JSONL file writes ≥N rows.

## 7. `tests/test_curated_core.py`

- Skip if SCIX_TEST_DSN not set.
- Fixture `seeded_db`: TRUNCATE relevant tables, insert:
  - ~15 entities with various ambiguity_class, sources.
  - query_log rows for some with hits/zero-results in last 14d.
- Test AC3: run curation, CSV exists, row count ≤ 10K, required columns.
- Test AC4: core_lifecycle at exactly the cap auto-demotes lowest.
  Use a small cap override (monkeypatch CORE_MAX=5) for the test.
- Test AC5: stratification md exists and has per-source counts.

## Files summary

1. migrations/032_core_promotion_log.sql
2. src/scix/query_log.py
3. src/scix/core_lifecycle.py
4. scripts/backfill_query_log.py
5. scripts/curate_entity_core.py
6. tests/test_query_log.py
7. tests/test_curated_core.py
