# Research — u12-m4-three-way-eval

## Existing infrastructure

- `src/scix/ir_metrics.py` — already implements `dcg_at_k`, `ndcg_at_k`, `recall_at_k`, `precision_at_k`, `mean_reciprocal_rank`, `compute_retrieval_score`, `aggregate_scores` (with `RetrievalScore`/`EvalReport` DTOs). Direct reuse target.
- `scripts/eval_retrieval_50q.py` — exists. Reuses its `ndcg_at_k`/`recall_at_k`/`mrr` + `_pilot_sample` path. For u12 we want a higher-level three-way runner, so we wrap/mock rather than reuse the module wholesale.
- `src/scix/resolve_entities.py` — M13 single entry point. `EntityResolveContext(candidate_set, mode, ttl_max, budget_remaining, model_version)`. Modes: `static`, `jit`, `live_jit`, `local_ner`, `auto`. Returns `EntityLinkSet` with `.entity_ids() -> frozenset[int]`, `.lane`, `.model_version`, `.candidate_set_hash`.
- `scripts/ast_lint_resolver.py` — forbids non-resolver files from SELECT FROM `document_entities_canonical`, INSERT/UPDATE/DELETE to `document_entities` / `document_entities_jit_cache`. Escape hatch: `# noqa: resolver-lint`. Our files MUST go through `resolve_entities()` for static and jit lanes.
- `src/scix/eval/` — `audit.py`, `llm_judge.py`, `wilson.py`, `__init__.py`. No metrics module. We add `metrics.py` and `lane_delta.py`.
- `src/scix/jit/` — bulkhead/cache/local_ner/router; backend for jit lane.
- `src/scix/fusion_mv.py` — write helpers for MV; reads go through resolver.

## Citation-chain lane design

PRD asks for Jaccard over entity-id sets from three sources:

1. `citation_chain` tool (graph-walk) — for u12 we synthesize an analog: entities reachable by walking citation edges N hops and pulling their entities via the resolver. Since the test must work against `scix_test` fixture and on mocked resolver state, we implement a thin "citation_chain_entities(bibcode)" that goes: bibcode -> citation neighbors -> union of their resolver(static) entity_ids. This avoids direct MV access.
2. `hybrid_search[enrich_entities=True]` — simulated here via `resolve_entities(bibcode, mode='static')` since hybrid search's enrichment path reads the canonical MV through the resolver.
3. `SELECT FROM document_entities_canonical` — M13 forbids; we instead call `resolve_entities(bibcode, mode='static')` and call that "lane C". This still tests the M13 contract.

For the three-lane comparison, the simpler approach endorsed by the prompt: use `mode='static'` vs `mode='jit'` vs citation-chain-analog. We implement exactly that.

## lane_delta_set (Wikidata backfill stub)

u12 in-scope: `lane_delta_set` is a stubbed empty set with TODO pointing to future N2 task. Arithmetic path subtracts from numerator AND denominator in adjusted Jaccard.

## Gate

- `numpy.percentile([1 - j for j in jaccards], 90)` — pass if ≤ 0.05.
- Fixture must have known divergence so test can assert computation.

## Files to create

1. `src/scix/eval/metrics.py` — thin facade over `scix.ir_metrics` plus three-way eval runner helpers.
2. `src/scix/eval/lane_delta.py` — Jaccard, lane_delta, adjusted Jaccard, gate computation.
3. `scripts/eval_three_way.py` — runs M4 three-config eval against fixture, writes `build-artifacts/m4_inhouse_eval.md`.
4. `scripts/eval_lane_consistency.py` — M4.5 lane Jaccard, writes `build-artifacts/m45_consistency.md` and `build-artifacts/m45_lane_delta.md`.
5. `tests/test_m4_eval.py` — unit test.
6. `tests/test_m45_lane_consistency.py` — unit test.

## Acceptance mapping

- AC1 — `eval_three_way.py` emits `build-artifacts/m4_inhouse_eval.md` with the three configs' nDCG@10/Recall@20/MRR on 5 fixture queries + 2 graph-walk tasks.
- AC2 — Disclaimer at top.
- AC3 — `eval_lane_consistency.py` writes m45_consistency.md with raw/adjusted/distribution/per-pair-divergence.
- AC4 — `m45_lane_delta.md` with one row per unreachable entity (empty body + header for stub).
- AC5 — Gate printed + test fixture with known divergence.
- AC6 — pytest passes.
