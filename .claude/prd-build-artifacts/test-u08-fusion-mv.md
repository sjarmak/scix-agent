# Test results — u08-fusion-mv

## Command

```
SCIX_TEST_DSN=dbname=scix_test .venv/bin/python -m pytest tests/test_fusion_mv.py -v
```

## Outcome

19 passed, 0 failed in 0.15s.

## Coverage of acceptance criteria

| AC                                                                  | Covered by                                                                                                                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. migration creates MV, tier_weight fn, calibration log, indexes   | `TestMigrationFileText` (8 tests), `TestMaterializedView::test_unique_index_on_bibcode_entity`, `TestTierWeightFunction::test_declared_immutable_and_parallel_safe` |
| 2. REFRESH CONCURRENTLY succeeds on populated fixture               | `TestMaterializedView::test_refresh_concurrently_succeeds`, `TestMaterializedView::test_fused_matches_closed_form_five_tiers`                                       |
| 3. `mark_dirty`/`refresh_if_due` + rate limiter via fusion_mv_state | `TestRefreshHelper` (3 tests)                                                                                                                                       |
| 4. fused_confidence matches closed form within 1e-9 on ≥5 tiers     | `TestMaterializedView::test_fused_matches_closed_form_five_tiers`                                                                                                   |
| 5. `placeholder_2026-04-12` row inserted                            | `TestCalibrationLog::test_initial_row_present`                                                                                                                      |
| 6. 100-link entity top-20 query <100ms                              | `TestLatency::test_entity_topk_under_100ms`                                                                                                                         |
| 7. `pytest tests/test_fusion_mv.py` passes with 0 failures          | all 19 tests green                                                                                                                                                  |

## Bug fixed mid-phase

First run: 4 tests failed with `psycopg.ProgrammingError: can't change 'autocommit' now: connection in transaction status INTRANS`.

Root cause: `refresh_if_due` tried to flip the caller's connection to autocommit to run REFRESH CONCURRENTLY, but the shared `db_conn` pytest fixture was already in an INTRANS state. Fix: always open a dedicated short-lived autocommit sibling connection for the refresh, reusing the caller's DSN. Commit the caller's pending transaction first so the sibling sees fresh writes. Clean separation of concerns and removes the `_conn_ctx` helper entirely.

## AST lint (resolver-lint)

- `scripts/ast_lint_resolver.py src/` -> exit 0 (clean).
- `src/scix/fusion_mv.py` holds no `FROM document_entities_canonical` literal; only a `REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical` statement, which the lint does not forbid.
- `tests/test_fusion_mv.py` contains `INSERT INTO document_entities` and `SELECT ... FROM document_entities_canonical` literals. This matches existing convention in `tests/test_promote_harvest.py`, `tests/test_migrations.py`, `tests/test_link_entities.py`, `tests/test_tier1.py`: the M13 lint is enforced against `src/` only (see `test_ast_lint_resolver.test_current_src_has_no_violations`). The MV SELECT literals additionally carry `# noqa: resolver-lint` for defence in depth.
