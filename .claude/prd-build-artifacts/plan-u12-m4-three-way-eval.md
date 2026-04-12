# Plan — u12-m4-three-way-eval

## Step 1: metrics module (`src/scix/eval/metrics.py`)

Thin re-export of `scix.ir_metrics` functions plus:

- `ThreeWayConfig` dataclass: name, description, retrieve fn
- `run_three_way_eval(queries, configs)` -> dict[config_name, EvalReport]
- `format_m4_report(results, graph_walk_results, disclaimer=...)` -> markdown string

## Step 2: lane_delta module (`src/scix/eval/lane_delta.py`)

- `jaccard(a: frozenset[int], b: frozenset[int]) -> float`
- `adjusted_jaccard(a, b, lane_delta_set) -> float` — subtract lane_delta from both numerator union and denominator union before computing
- `compute_lane_delta_set(bibcode) -> frozenset[int]` — stub returning empty set, TODO comment
- `per_bibcode_divergences(...)` — returns per-bibcode dict with raw/adjusted/pair divergences
- `gate_p90(divergences) -> tuple[float, bool]` — uses numpy.percentile, returns (p90, pass)

## Step 3: `scripts/eval_three_way.py`

- Connects to `scix_test` (via SCIX_TEST_DSN) but primary path uses fixture seeded via resolver mocks
- Three configs:
  - hybrid_baseline: no entity enrichment (returns empty entity set)
  - hybrid_plus_static: uses `resolve_entities(mode='static')`
  - hybrid_plus_jit: uses `resolve_entities(mode='jit')`
- For each of 5 queries + 2 graph-walk tasks, compute retrieval scores
- Write `build-artifacts/m4_inhouse_eval.md` with disclaimer header

Since u12 is scoped to a fixture-driven eval (not real corpus), the "retrieval" for each query is a seeded ordered list; entity enrichment just tags entity overlap. Metrics: standard IR metrics.

## Step 4: `scripts/eval_lane_consistency.py`

- For each test bibcode:
  - lane A: citation_chain analog — union of `resolve_entities(neighbor, mode='static').entity_ids()` for neighbors
  - lane B: `resolve_entities(bibcode, mode='static').entity_ids()` (hybrid_search enrich path)
  - lane C: `resolve_entities(bibcode, mode='static').entity_ids()` — same as B per PRD note about using resolver in lieu of raw SELECT
  - Actually: use mode='jit' for lane C to provide real divergence
- Compute pair-wise Jaccards, raw and adjusted
- Write m45_consistency.md + m45_lane_delta.md
- Print gate pass/fail

## Step 5: tests

- `tests/test_m4_eval.py`:
  - Test `metrics` wrappers compute correct nDCG/Recall/MRR
  - Test `run_three_way_eval` returns three configs with expected summary
  - Test the report writer includes disclaimer
  - Test `ThreeWayConfig` retrieval hook
- `tests/test_m45_lane_consistency.py`:
  - Test `jaccard`, `adjusted_jaccard` basics
  - Test `lane_delta_set` stub returns empty
  - Test `per_bibcode_divergences` on known fixture
  - Test `gate_p90` pass/fail thresholds
  - Test end-to-end runner writes all three artifact files

## Step 6: run pytest + ast_lint_resolver
