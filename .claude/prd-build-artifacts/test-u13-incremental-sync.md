# Test results — u13 incremental sync

## Command

```
SCIX_TEST_DSN=dbname=scix_test \
  pytest tests/test_incremental_sync.py tests/test_circuit_breaker.py -v
```

## Result

**24 passed in 0.25s**

### Integration (`test_incremental_sync.py` — 7 tests)

| Test                                                                         | Status |
| ---------------------------------------------------------------------------- | ------ |
| `TestIncrementalHappyPath::test_fresh_run_links_fixture_papers_under_60s`    | PASS   |
| `TestIncrementalHappyPath::test_second_run_finds_nothing_new`                | PASS   |
| `TestCircuitBreakerTrip::test_tiny_budget_trips_and_watermark_advances`      | PASS   |
| `TestCatchupBackfillsSkippedPapers::test_catchup_after_trip_populates_links` | PASS   |
| `TestTwoConsecutiveTripsPager::test_second_trip_emits_page_alert`            | PASS   |
| `TestWatermarkStaleness::test_forced_stale_watermark_fires_alert`            | PASS   |
| `TestWatermarkStaleness::test_fresh_watermark_does_not_fire`                 | PASS   |

### Unit (`test_circuit_breaker.py` — 17 tests)

All `CircuitBreaker` FSM transitions verified: initial closed state, budget
enforcement (within / at / over / zero / tiny 0.001s trip), explicit trip,
trip-count incrementation across cycles, reset, half-open probe success and
re-open, elapsed/remaining clock arithmetic.

## AST lint

```
python scripts/ast_lint_resolver.py src
```

Exit code **0** — no M13 resolver violations in `src/`. (All incremental
linker writes live under `scripts/` which is out of AST-lint scope by
design; `_INCREMENTAL_TIER1_SQL` still carries a `# noqa: resolver-lint`
annotation for parity with tier-1 and tier-2.)

## Acceptance criteria coverage

| AC                                                              | Covered by                                                                             |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1. migration creates `link_runs` with required columns          | `migrations/036_link_runs_watermark.sql`, applied to `scix_test`                       |
| 2. incremental runs in <60s, writes tier-1 + tier-2             | `test_fresh_run_links_fixture_papers_under_60s`                                        |
| 3. `CircuitBreaker` FSM; forced delay trips, watermark advances | `test_circuit_breaker.py` (17 tests) + `test_tiny_budget_trips_and_watermark_advances` |
| 4. `link_catchup.py` backfills skipped papers                   | `test_catchup_after_trip_populates_links`                                              |
| 5a. 2 consecutive trips → page alert                            | `test_second_trip_emits_page_alert`                                                    |
| 5b. stale watermark → page alert                                | `test_forced_stale_watermark_fires_alert` + `test_fresh_watermark_does_not_fire`       |
| 6. pytest target suite passes                                   | 24/24 pass                                                                             |
