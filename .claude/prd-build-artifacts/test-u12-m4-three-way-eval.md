# Test — u12-m4-three-way-eval

## pytest

```
SCIX_TEST_DSN=dbname=scix_test .venv/bin/python -m pytest \
  tests/test_m4_eval.py tests/test_m45_lane_consistency.py -v
```

Result: **20 passed in 0.17s**.

- `tests/test_m4_eval.py` — 5 tests (runner shape, config count, report
  disclaimer, static/jit ≥ baseline on fixture, end-to-end script run).
- `tests/test_m45_lane_consistency.py` — 15 tests (Jaccard basics,
  adjusted Jaccard stub, per-bibcode divergence, gate pass/fail,
  numpy.percentile computation, end-to-end script run including
  artifact-shape assertions and fixture-known-p90 assertion).

## ast_lint_resolver

```
.venv/bin/python scripts/ast_lint_resolver.py src
```

Result: **exit 0** (clean). The new code only reads from
`document_entities_canonical` via the resolver; no forbidden SQL.

## Script runs

```
.venv/bin/python scripts/eval_three_way.py
```

Wrote `build-artifacts/m4_inhouse_eval.md` with the in-house disclaimer
header and per-config nDCG@10 / Recall@20 / MRR tables for both the
query set and the graph-walk task set:

- hybrid_baseline — nDCG@10=0.7287 Recall@20=1.0000 MRR=0.6667
- hybrid_plus_static — nDCG@10=1.0000 Recall@20=1.0000 MRR=1.0000
- hybrid_plus_jit — nDCG@10=0.9788 Recall@20=1.0000 MRR=1.0000

```
.venv/bin/python scripts/eval_lane_consistency.py
```

Wrote `build-artifacts/m45_consistency.md` + `build-artifacts/m45_lane_delta.md`.

Gate output (deliberately failing on the fixture so pass/fail is
exercised):

```
M4.5 gate: p90 adjusted divergence = 0.5778 (threshold 0.05) — FAIL
```

The fixture is designed with a deliberate divergence outlier
(bibcode 5, Jaccard=0.0) so the gate-fail path is exercised; the gate
computation is asserted against numpy.percentile in
`test_run_fixture_has_known_p90_divergence`.

## AC checklist

- [x] AC1 — `eval_three_way.py` runs, emits `m4_inhouse_eval.md` with
      nDCG@10 / Recall@20 / MRR for all three configs on query set + graph
      walks.
- [x] AC2 — In-house disclaimer printed at top of
      `m4_inhouse_eval.md`.
- [x] AC3 — `eval_lane_consistency.py` writes `m45_consistency.md`
      with raw Jaccard, adjusted Jaccard, aggregate distribution
      (p50/p90/p99), per-lane-pair divergence breakdown.
- [x] AC4 — `m45_lane_delta.md` exists with one row per
      structurally-unreachable entity and reason (stub-empty at u12 with
      documented TODO pointing at u07 Wikidata backfill).
- [x] AC5 — Gate: 90th-percentile of adjusted per-bibcode Jaccard
      divergence computed; pass/fail at 5% printed; test asserts
      numpy.percentile equivalence on fixture.
- [x] AC6 — `pytest tests/test_m4_eval.py
tests/test_m45_lane_consistency.py` passes.
