# Test summary: coverage-bias-report (M1)

## Test run

```
$ .venv/bin/python -m pytest tests/test_report_full_text_coverage_bias.py -q
....................                                                     [100%]
20 passed in 0.18s
```

Combined with the existing `tests/test_coverage_bias.py`:

```
$ .venv/bin/python -m pytest tests/test_report_full_text_coverage_bias.py tests/test_coverage_bias.py -q
.........................................................                [100%]
57 passed in 0.25s
```

No regressions to the parent script's test suite (37 existing tests in
`test_coverage_bias.py` still pass).

## What's covered

### `kl_divergence` helper (10 tests)
- Identity D_KL(P || P) = 0
- Manually-derived value for [0.5,0.5] vs [0.25,0.75]
- Concentrated vs uniform > 0.5
- Asymmetry on a non-mirrored example
- Smoothing for zeros in P (no NaN)
- Smoothing for zeros in Q (finite)
- eps=0 + zero in Q → +inf (per definition)
- Length mismatch raises ValueError
- Empty inputs → 0
- Unnormalised counts equivalent to normalised probabilities

### `rows_to_distributions` / `build_facet_payload` (4 tests)
- Counts extraction from DistributionRow
- Required JSON keys per facet and per row (kl_divergence + counts +
  p_fulltext + q_corpus + ratio_p_over_q)
- KL > 0 when distributions diverge
- KL ≈ 0 when full-text % is identical across facet rows

### `build_payload` schema (2 tests)
- Top-level keys (generated_at, corpus_total, fulltext_total, facets, ...)
- Optional facets (community_semantic_medium=None) are omitted

### Dry-run end-to-end (4 tests)
- Writes JSON file matching schema
- Idempotent docs upsert (re-run does not duplicate the section)
- Generated section contains 3+ safe and 3+ unsafe numbered bullets
- Refuses to create the docs file from scratch

## Acceptance criteria verification

| AC | Verified | Evidence |
|---|---|---|
| 1. `test -x scripts/report_full_text_coverage_bias.py` | yes | `chmod +x` applied; `test -x` returned 0 |
| 2. JSON keys + KL per facet + counts | yes | inline assertion script over `/tmp/cb_test.json` printed `AC2 PASS` |
| 3. Docs section with 3+3 examples | yes | `grep -c` heading=1; numbered bullets 1./2./3. present in both safe and unsafe blocks |
| 4. `pytest tests/test_report_full_text_coverage_bias.py -q` | yes | 20 passed |
| 5. scix-batch wrapping in docstring | yes | lines 21–25 of the script |

## Manual smoke run

```
$ .venv/bin/python scripts/report_full_text_coverage_bias.py --dry-run \\
    --json-out /tmp/cb_test.json
15:09:09 INFO __main__: Wrote JSON to /tmp/cb_test.json
15:09:09 INFO __main__: Upserted agent-guidance section in
  /home/ds/projects/scix_experiments/docs/full_text_coverage_analysis.md
```

JSON shape (synthetic):
- top-level: corpus_total, dsn_redacted, facets, fulltext_pct,
  fulltext_total, generated_at, kl_divergence_basis
- facets: arxiv_class, bibstem, citation_bucket,
  community_semantic_medium, year
- per-facet: kl_divergence_vs_corpus_prior, row_count, rows
- per-row: label, total, with_body, without_body, pct_with_body,
  p_fulltext, q_corpus, ratio_p_over_q

Sample KL values from synthetic distributions:
- arxiv_class: 0.000010 nats (modern arxiv classes ≈99% full-text;
  near-zero divergence from corpus prior)
- year: 0.065139 nats (modest skew because pre-modern years dominate
  the abstract-only side)
- bibstem: 0.865283 nats (large skew — full-text-only filter would
  drop conference-abstract venues entirely)

The same script run against prod will produce real numbers; the
agent-guidance bullets are derived from whichever rows have the
highest / lowest pct_with_body in the live JSON.
