# M13 resolve_entities() per-lane latency (u03 mocks)

Samples per lane: **50**. All backends are u03 in-module 
mocks with deterministic injected latency — real backends arrive in 
u08 (static) and u10 (jit_cache / live_jit).

| lane | mock_latency_ms | p50_ms | p95_ms | p99_ms | budget_p95_ms | status |
|------|-----------------|--------|--------|--------|---------------|--------|
| static | 0.5 | 0.56 | 0.56 | 0.57 | 5.0 | PASS |
| jit_cache_hit | 2.0 | 2.06 | 2.07 | 2.07 | 25.0 | PASS |
| live_jit | 15.0 | 15.07 | 15.08 | 15.08 | 80.0 | PASS |
| local_ner | 10.0 | 10.06 | 10.07 | 10.08 | 60.0 | PASS |

These mock-level budgets will be retightened against real backends once u08 / u10 land. The 5ms / 25ms budgets for static / jit_cache match the PRD §M13 acceptance criteria against real pgvector.
