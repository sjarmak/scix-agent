# Plan — u10-jit-lane

## Files

1. `migrations/034_jit_cache.sql` — partitioned table + default partition.
2. `src/scix/jit/__init__.py` — re-export public symbols.
3. `src/scix/jit/bulkhead.py` — async bulkhead context manager (semaphore +
   400ms wall budget + degraded sentinel).
4. `src/scix/jit/local_ner.py` — stub NER returning `LocalNERResult`.
5. `src/scix/jit/cache.py` — get/put with asyncio.Queue fire-and-forget,
   pager alert on 2 consecutive drops, DSN-backed writer coroutine.
6. `src/scix/jit/router.py` — Haiku primary, 5% canary, degrade fallback
   chain.
7. `scripts/jit_cache_cleanup.py` — idempotent TTL cleanup.
8. `tests/test_jit_bulkhead.py` — 2.5s degrade within 410ms; vendor error.
9. `tests/test_jit_cache.py` — queue drop + pager alert; degraded not written.
10. `tests/test_jit_local_ner.py` — p95 ≤ 275ms.
11. `tests/test_jit_router.py` — canary selection + fallback chain.

## Domain types

Expose `CachedLinkSet` dataclass (bibcode, entity_ids, confidences,
model_version, candidate_set_hash, expires_at) as return type from jit
modules. `resolve_entities` in a future PR will wrap these in real
`EntityLinkSet`. For THIS unit, jit modules are self-contained; no changes
to u03.

## Bulkhead design

- `JITBulkhead(concurrency=4, budget_ms=400)`.
- `async with bulkhead.acquire() as ctx:` — raises `BulkheadDegraded` on
  semaphore overflow OR on wait_for timeout.
- Helper `async def run(coro)` that does `asyncio.wait_for(coro, budget_ms/1000)`
  under the semaphore. Any exception → returns the sentinel `DEGRADED`.

## Cache design

- Module-level `asyncio.Queue(maxsize=1024)`.
- `put(row)` → `queue.put_nowait(row)`, on QueueFull increment
  `_consecutive_drops`; if ≥2, call `raise_alert()`.
- `get(bibcode, hash, model_version)` → direct `SELECT` through a psycopg
  connection (DSN from env).
- `start_writer(conn_factory)` / `stop_writer()` — background task drains
  queue and INSERTs rows.
- INSERT uses `# noqa: resolver-lint`.

## Router

- `route_jit(bibcode, candidate_set, *, rng=random.random)` → tries Haiku
  via the bulkhead. On degraded/error, tries local_ner. On local failure,
  returns `static_core_sentinel`.
- 5% canary: before calling Haiku, if `rng() < 0.05`, call local_ner
  instead. If local fails in canary, still fall through to Haiku.

## Migration 034

```
CREATE TABLE document_entities_jit_cache (
  bibcode TEXT NOT NULL,
  entity_id INT NOT NULL,
  link_type TEXT NOT NULL,
  confidence REAL,
  match_method TEXT,
  evidence JSONB,
  harvest_run_id INT,
  tier SMALLINT NOT NULL DEFAULT 5,
  tier_version INT NOT NULL DEFAULT 1,
  candidate_set_hash TEXT NOT NULL,
  model_version TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (bibcode, entity_id, link_type, candidate_set_hash, model_version, expires_at)
) PARTITION BY RANGE (expires_at);

CREATE TABLE document_entities_jit_cache_default
  PARTITION OF document_entities_jit_cache DEFAULT;
```

- Include `CHECK (tier = 5)` if feasible across partitions.
- Include index on `(bibcode, candidate_set_hash, model_version)`.
- Apply migration once at test session setup via idempotent `CREATE TABLE IF NOT EXISTS`.

## Tests

- All async tests use `asyncio.run(inner())` — no pytest-asyncio config.
- Cache test applies migration with `psycopg.connect("dbname=scix_test")`.
- Bulkhead test uses `asyncio.sleep(2.5)` as the forced-latency coroutine;
  asserts wall time < 0.41s.
- Local NER test runs 20 invocations, asserts 95th-percentile sleep ≤ 0.275s.
- Router test monkeypatches `call_live_jit` and `run_local_ner`.
