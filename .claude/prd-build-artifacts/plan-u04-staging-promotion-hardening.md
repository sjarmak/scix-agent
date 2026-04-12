# Plan — u04 Staging Promotion Hardening

## Step 1: src/scix/harvest_promotion.py

- `PER_SOURCE_FLOORS` dict (VizieR=55000, GCMD=9000, PwC=7500, ASCL=3500, PhySH=3500, AAS=600, SPASE=200; SsODNet/CMR/SBDB=0 with `known_broken=True` annotation as an inner dict).
- `DEFAULT_CANONICAL_SHRINKAGE_MAX = 0.02`, `DEFAULT_ALIAS_SHRINKAGE_MAX = 0.05`, `ORPHAN_THRESHOLD = 1000`.
- `_PROMOTE_SQL` constant with `CREATE OR REPLACE FUNCTION promote_harvest(run_id BIGINT) RETURNS JSONB` body that:
  1. Acquires `pg_try_advisory_lock(hashtext('entities_promotion'))`. If fails, return `{"accepted": false, "reason": "lock_unavailable"}`.
  2. Validates column schema match via `information_schema.columns` for staging vs target (rejects with `schema_mismatch`).
  3. Computes diff in SQL:
     - `staging_canonical_count`, `prod_canonical_count` (production entities total).
     - Per-source staging count. Floor check vs hardcoded per-source floors (passed in as parameter `floors_json jsonb`).
     - Shrinkage: (prod - staging_for_sources_in_run) / prod for sources present in staging.
  4. Orphan check: for sources in staging run, collect entity_ids in production with >= `orphan_threshold` document_entities rows that have no matching natural key in staging. If any, reject with `orphan_violation`.
  5. If all pass, UPSERT entities_staging → entities (on conflict by natural key, update source_version, ambiguity_class, link_policy, discipline, properties, updated_at). UPSERT aliases and identifiers.
  6. Update `harvest_runs.status='promoted'`, return `{"accepted": true, "diff": {...}}`.
  7. On reject: set `harvest_runs.status='rejected_by_diff'`, return JSON with reason.
  8. Release advisory lock.
- Python-side: `ensure_promote_function(conn)` runs the CREATE OR REPLACE via a single SQL string.
- Python `promote_harvest(run_id, dsn=None, ...)` opens a connection, calls `ensure_promote_function`, then `SELECT promote_harvest(%s, %s::jsonb, %s::numeric, %s::numeric, %s::int)` to get JSON result, maps to frozen `PromotionResult`.

## Step 2: src/scix/llm_cost_ceiling.py

- `DEFAULT_DAILY_CAP_USD = 50.0`, `DEFAULT_PER_QUERY_CAP_USD = 0.01`.
- Haiku rates: `HAIKU_INPUT_PER_1M = 0.25`, `HAIKU_OUTPUT_PER_1M = 1.25`.
- `estimate_cost_usd(prompt_tokens, completion_tokens) -> float`.
- `_ensure_ledger(conn)` runs CREATE TABLE IF NOT EXISTS.
- `check_and_reserve(estimated_cost_usd, dsn=None, per_query_cap=..., daily_cap=...) -> bool`:
  - Return False if estimated > per_query_cap (do not touch ledger).
  - Atomically: `INSERT ... ON CONFLICT (day) DO UPDATE SET total_usd = total_usd + EXCLUDED.total_usd RETURNING total_usd`.
  - If new total > daily_cap, rollback with SAVEPOINT (or check first then insert in same tx). Simpler: SELECT current total, if + est > cap return False, else UPSERT.
  - Use `SELECT ... FOR UPDATE` of the current day row to avoid races.
- `record_actual(actual_cost_usd, dsn=None)`: we don't track reservations individually; simplification — treat `check_and_reserve` as charging the ledger by estimate, and `record_actual` as adjusting delta to actual. Store last-estimate in module-level thread-local or return a reservation token. Simplest: `record_actual(actual, estimate)` signature accepts both so caller can supply delta. Spec says signature is `record_actual(actual_cost_usd) -> None`, so we'll just add `actual` to ledger (caller is expected to have NOT already reserved, or to reserve then adjust). Cleanest: make `check_and_reserve` only check + peek (no write), and `record_actual` does the write. That's atomic-safe too but risks races where caller exceeds cap between check and record. Given the PRD language says "reserve" — implement it as: reserve writes estimate, record_actual writes (actual - last_estimate) delta. Use a module-level `_last_reservation_usd` thread-local. For tests, we'll test reserve path only.

## Step 3: scripts/replay_harvest.py

- CLI: `--source SRC --snapshot YYYY-MM-DD --dsn DSN`.
- Reads `data/entities/snapshots/{source}/{snapshot}.jsonl.gz`.
- Creates harvest_runs row (source=<src>, status='replayed'), gets run_id.
- For each line: INSERT into entities_staging, entity_aliases_staging, entity_identifiers_staging with staging_run_id=run_id.
- Returns run_id on stdout.
- Exposes `replay_snapshot(source, snapshot_date, dsn) -> int` for programmatic use.

## Step 4: Tests

- `tests/test_promote_harvest.py`:
  - Skip unless SCIX_TEST_DSN set and non-production.
  - Fixture cleans `entities_staging/*`, `entities/*`, `document_entities`, `harvest_runs` rows created by tests via scoped deletes (not TRUNCATE).
  - Test 1: clean promote — insert 10 entities into staging, no production, promote → accepted, rows in entities.
  - Test 2: canonical_shrinkage reject — seed production with 100 entities (source=TEST_SRC), staging with 50 entities (source=TEST_SRC) where floor is 0. >50% shrinkage > 2% → reject, reason contains "canonical_shrinkage", status='rejected_by_diff', staging rows preserved.
  - Test 3: orphan reject — seed production entity with 1500 document_entities rows, staging for same source lacks that entity's natural key → reject "orphan_violation".
- `tests/test_llm_cost_ceiling.py`:
  - Skip unless SCIX_TEST_DSN.
  - Test per-query cap: `check_and_reserve(0.02)` → False.
  - Test daily cap: reserve 50 small calls totaling $49.99, then 0.005 → True, then 0.01 → False (exceeds 50).
- `tests/test_replay_harvest.py`:
  - Skip unless SCIX_TEST_DSN.
  - Create fixture snapshot file with 3 entities + aliases + identifiers.
  - Replay into staging, then query staging tables; assert row counts match.
  - Round-trip: dump back out and compare JSONL content.

## Step 5: Commit

- `git add -A && git commit` with provided message.
