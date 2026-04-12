# Test report: u11-eval-harness

## pytest

```
SCIX_TEST_DSN=dbname=scix_test .venv/bin/python -m pytest tests/test_audit.py tests/test_llm_judge.py -v
```

Result: **21 passed, 0 failed**.

Covered:

- Wilson 95% CI anchor `(95, 100) → [0.887, 0.978]` (±0.005)
- Wilson degenerate cases (0/0, 10/10, 0/10, invalid raises)
- `write_audit_report` renders per-tier rows and the worked example
- `sample_stratified` integration test against `scix_test` — seeds 4 tiers × 3 rows, asserts every tier present in sample and per-tier cap holds
- `sample_stratified(n_per_tier=0)` returns `[]`
- `judge()` stub — 1 label per link, all in allowed set, deterministic, pass-through identity, empty input handled, `use_real=True` with no API key falls back to stub
- `cohens_kappa`: perfect agreement → 1.0, total disagreement (2 classes) → -1.0, classic 2x2 textbook example (0.4), single-label edge cases, empty input → 0.0, mismatched length raises

## AST lint

```
.venv/bin/python scripts/ast_lint_resolver.py src
```

Exit code: **0** (clean — no M13 violations).

## Fixture end-to-end

```
SCIX_TEST_DSN=dbname=scix_test PYTHONPATH=src .venv/bin/python scripts/run_audit.py --fixture
```

Logs:

```
seeded fixture: 12 papers, 8 entities, 24 document_entities rows
sample_stratified: drew 24 candidates across 4 tier(s) (n_per_tier=125)
Wrote audit report to build-artifacts/eval_report.md
wrote build-artifacts/eval_report.md
```

`build-artifacts/eval_report.md` contains per-tier Wilson 95% CI rows for
tiers 1/2/4/5 plus the worked example line.

## Migration 035

Applied to `dbname=scix_test` successfully. Verified columns:

```
tier       smallint
bibcode    text
entity_id  bigint
annotator  text
label      text       (CHECK label IN ('correct','incorrect','ambiguous'))
note       text
created_at timestamptz
```

Primary key `(tier, bibcode, entity_id, annotator)` per spec.
