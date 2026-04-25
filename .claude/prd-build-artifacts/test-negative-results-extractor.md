# Test summary — negative-results-extractor (M3)

## Test run

```
$ .venv/bin/python -m pytest tests/test_negative_results.py -q
..........................                                               [100%]
26 passed in 0.13s
```

## Acceptance-criterion coverage

| AC# | Statement | Test(s) |
|-----|-----------|---------|
| 1 | `src/scix/negative_results.py` exports `detect_negative_results(body, sections) -> list[NegativeResultSpan]` | imported in tests; 11 high-tier param cases + section-guard + dataclass tests |
| 2 | `scripts/run_negative_results.py` accepts `--max-papers`, `--dry-run`, `--allow-prod`, `--since-bibcode`; refuses prod DSN without `--allow-prod` | `--help` lists all flags; manual smoke: `SCIX_DSN=dbname=scix ... --max-papers 1` returns "Refusing to run against production DSN" |
| 3 | Fixture is exactly 100 lines of `{text, label}` JSON with `label in {true, false}` | `test_gold_fixture_has_exactly_100_labeled_spans` |
| 4 | Precision >= 0.70 AND recall >= 0.60 on the gold fixture | `test_precision_recall_on_gold_fixture` — measured **precision=1.000**, **recall=0.925** |
| 5 | Output rows include a 250-char `evidence_span` field | `test_evidence_span_is_exactly_250_chars`, `test_evidence_span_padded_when_body_shorter_than_window` |
| 6 | Writes target `staging.extractions` with `extraction_type='negative_result'` (mocked psycopg) | `test_db_insert_targets_staging_extractions_with_correct_columns`, `test_db_insert_with_no_spans_still_writes_row_with_null_tier` |

## Precision/recall detail

```
tp=37  fp=0  fn=3  tn=60
precision = 1.000   (>= 0.70 required)
recall    = 0.925   (>= 0.60 required)
```

Three missed positives (recall slack):
- "These models are excluded at the 5-sigma level by the new measurements."
  (rejected_sigma regex requires 'rejected/excluded/ruled out' adjacent to a
  number+sigma; the trailing 'by the new measurements' breaks adjacency.)
- "We derive a 3-sigma upper limit on the column density of CH3OH."
- "We provide a 95% upper limit of 0.07 mJy on the unresolved emission."

The two upper-limit misses are because the verb is followed by the verb form
("derive a 3-sigma upper limit ON" — but the sigma token sits between
verb-determiner-and-'upper'). Tightening the upper_limit regex to allow
arbitrary intermediate tokens between verb and 'upper limit' would help
recall further but at clear precision risk. With the current bar (0.60
recall) the slack is already > 50% over the gate, so we leave it for
follow-up tuning.

Zero false alarms means the precision bar (0.70) has 30-percentage-point
headroom — good cushion for new patterns added in future tuning passes.

## No new migration required

`staging.extractions` (migration 015 + 049) accepts arbitrary
`extraction_type` strings; no CHECK constraint blocks `'negative_result'`.
`confidence_tier` is `SMALLINT`; we store the human-readable tier
('high'/'medium'/'low') inside the JSONB payload alongside the integer.
