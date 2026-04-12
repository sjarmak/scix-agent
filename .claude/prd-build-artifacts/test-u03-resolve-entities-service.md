# Test results — u03-resolve-entities-service

## Command

```
.venv/bin/python -m pytest \
    tests/test_resolve_entities.py \
    tests/test_resolve_entities_type_guard.py \
    tests/test_resolve_entities_invariant.py \
    tests/test_ast_lint_resolver.py \
    tests/bench_resolve_entities.py -v
```

## Result

```
============================== 31 passed in 4.72s ==============================
```

## Breakdown

| file                                      | tests                       | status |
| ----------------------------------------- | --------------------------- | ------ |
| tests/test_resolve_entities.py            | 12                          | PASS   |
| tests/test_resolve_entities_type_guard.py | 5                           | PASS   |
| tests/test_resolve_entities_invariant.py  | 1 (120 Hypothesis examples) | PASS   |
| tests/test_ast_lint_resolver.py           | 12                          | PASS   |
| tests/bench_resolve_entities.py           | 1                           | PASS   |

## AST lint against current src/

```
$ .venv/bin/python scripts/ast_lint_resolver.py src
EXIT=0
```

`src/scix/link_entities.py` (u02 batch writer) is granted a `# noqa:
resolver-lint` scoped exemption pending u10, which will migrate the batch
write path through `scix.resolve_entities`. All other writes to
`document_entities*` or reads from `document_entities_canonical` would be
blocked by CI.

## Benchmark report

Written to `build-artifacts/m13_latency.md`:

| lane          | mock_latency_ms | p50_ms | p95_ms | p99_ms | budget_p95_ms | status |
| ------------- | --------------- | ------ | ------ | ------ | ------------- | ------ |
| static        | 0.5             | 0.56   | 0.56   | 0.57   | 5.0           | PASS   |
| jit_cache_hit | 2.0             | 2.06   | 2.08   | 2.09   | 25.0          | PASS   |
| live_jit      | 15.0            | 15.07  | 15.07  | 15.08  | 80.0          | PASS   |
| local_ner     | 10.0            | 10.06  | 10.07  | 10.08  | 60.0          | PASS   |

All lanes well inside budget at u03 mock latencies. Budgets will be
retightened against real backends once u08 (static MV) and u10 (JIT cache

- live Haiku) land.

## Pre-existing test regression check

Ran `tests/test_link_entities.py tests/test_link_entities_collision.py` to
verify the noqa insertion in `src/scix/link_entities.py` didn't break
anything: **26 passed in 0.12s**.

## Acceptance criteria mapping

| #   | criterion                                                       | satisfied by                                                                 |
| --- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 1   | Single entry point `resolve_entities(bibcode, context)`         | `src/scix/resolve_entities.py`                                               |
| 2   | Sentinel-gated `EntityLinkSet` raises TypeError outside module  | `tests/test_resolve_entities_type_guard.py` (5 tests)                        |
| 3   | libcst AST lint + planted-violation test                        | `scripts/ast_lint_resolver.py`, `tests/test_ast_lint_resolver.py` (12 tests) |
| 4   | Unit tests per lane                                             | `tests/test_resolve_entities.py` (12 tests)                                  |
| 5   | Hypothesis ≥100 examples, cross-lane set-equality + ≤0.01 drift | `tests/test_resolve_entities_invariant.py` (max_examples=120)                |
| 6   | Benchmark with per-lane p95 + report + budget asserts           | `tests/bench_resolve_entities.py`, `build-artifacts/m13_latency.md`          |
| 7   | All pytest 0 failures; ast lint exit 0                          | Confirmed above                                                              |
