# Test report — u10-jit-lane

## Command

```
SCIX_TEST_DSN=dbname=scix_test .venv/bin/python -m pytest \
  tests/test_jit_bulkhead.py \
  tests/test_jit_cache.py \
  tests/test_jit_local_ner.py \
  tests/test_jit_router.py -v
```

## Result

17 passed in 1.66s

### Bulkhead (5 tests)

- `test_bulkhead_degrades_under_forced_latency` — 2.5s sleep, degrade < 0.41s wall.
- `test_bulkhead_degrades_on_vendor_error` — RuntimeError collapses to DEGRADED.
- `test_bulkhead_returns_value_on_fast_call` — happy path returns value.
- `test_bulkhead_reports_configured_limits`
- `test_bulkhead_rejects_invalid_config`

### Cache (4 tests)

- `test_put_then_get_round_trip` — INSERT via drain_once, SELECT via get.
- `test_queue_saturation_drops_and_alerts` — maxsize=1, 2nd drop fires
  pager alert (monkey-patched).
- `test_bulkhead_degraded_is_not_written_to_cache` — forced degrade,
  row count unchanged (criterion 6).
- `test_get_ignores_expired_rows` — expires_at < now() filtered.

### Local NER (3 tests)

- Shape echoes candidate set at confidence 0.75.
- Type validation rejects non-frozenset.
- p95 latency ≤ 275ms over 20 runs.

### Router (5 tests)

- Happy path -> LiveJITResult.
- Forced Haiku outage -> LocalNERResult (NOT static-core).
- Both lanes down -> STATIC_CORE_FALLBACK.
- Canary roll 0.01 -> LocalNERResult.
- Canary local failure -> still tries Haiku -> LiveJITResult.

## AST lint

```
.venv/bin/python scripts/ast_lint_resolver.py src
```

Exit 0. Cache INSERT is exempted with `# noqa: resolver-lint`; no other
jit-cache writes exist outside resolve_entities.py.

## Regression

`tests/test_ast_lint_resolver.py` and `tests/test_resolve_entities.py`
still pass (24 tests, no u03 regressions).

## Migration 034

Applied idempotently to `scix_test`:

- `document_entities_jit_cache` partitioned RANGE(expires_at), tier CHECK = 5.
- `document_entities_jit_cache_default` DEFAULT partition.
- Lookup index on (bibcode, candidate_set_hash, model_version).
- Expires-at index.
