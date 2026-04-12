# Test Report — u14-should-haves

## Commands

```bash
psql dbname=scix_test -f migrations/037_citation_consistency_and_disputes.sql
SCIX_TEST_DSN="dbname=scix_test" .venv/bin/python -m pytest \
    tests/test_citation_consistency.py \
    tests/test_query_expansion.py \
    tests/test_feedback.py -v
.venv/bin/python scripts/ast_lint_resolver.py src
```

## Results

**Migration 037**: applied cleanly to scix_test (BEGIN … COMMIT).

**pytest** — 15 passed, 0 failed:

- `test_citation_consistency.py` — 4/4 (half, quarter, no-outbound-None, unrelated-zero)
- `test_query_expansion.py` — 8/8 (determinism, distinct queries, id validity, k-clamping, empty index, <20ms latency, shape validation, tie-break stability)
- `test_feedback.py` — 3/3 (row written, empty reason dropped, append-only multi-row)

**AST lint** — exit 0. No M13 violations.

## Findings

1. psycopg typed-null parameter: `(%(tier)s IS NULL OR de.tier = %(tier)s)` failed with "could not determine data type of parameter $3" when tier was None. Fixed by adding `::smallint` cast on both uses.
2. `entities` table requires NOT NULL `source` column — fixture seeds with `source='u14_test'`.
3. Relocated venv has stale shebang; invoked pytest via `.venv/bin/python -m pytest` instead.
4. Latency test: `expand()` on 100-vector fixture finished well under 20ms (sub-millisecond in practice).
