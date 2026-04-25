# Test Summary — quant-claim-extractor (M4)

## Run

```
$ .venv/bin/python -m pytest tests/test_claim_extractor.py -q
..................................                                       [100%]
34 passed in 0.14s
```

```
$ ruff check src/scix/claim_extractor.py scripts/run_claim_extractor.py \
            tests/test_claim_extractor.py
All checks passed!
```

## Recall on the 50-snippet cosmology fixture

| Quantity      | Hits | Total | Recall | PRD threshold |
|---------------|------|-------|--------|---------------|
| H0            | 20   | 20    | 1.00   | 0.80          |
| Omega_m       | 15   | 15    | 1.00   | 0.80          |
| sigma_8       | 10   | 10    | 1.00   | 0.80          |
| Omega_b       | 3    | 3     | 1.00   | n/a           |
| Omega_Lambda  | 2    | 2     | 1.00   | n/a           |

All three PRD-required thresholds (>=0.80) are met with full recall on
the curated gold set. No misses observed in the canonicalisation tier.

## Acceptance criteria — verification

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `src/scix/claim_extractor.py` exports `extract_claims(body)->list[ClaimSpan]` with required fields | Pass |
| 2 | `scripts/run_claim_extractor.py` accepts `--max-papers --dry-run --allow-prod --since-bibcode`, refuses prod DSN | Pass (`test_cli_refuses_production_dsn_without_allow_prod`) |
| 3 | `tests/fixtures/quant_claims_cosmology_50.jsonl` is exactly 50 JSON lines with the schema | Pass (`test_fixture_has_exactly_50_lines`, `test_fixture_schema_is_consistent`) |
| 4 | Per-quantity recall >= 0.80 on H0/Omega_m/sigma_8 | Pass (`test_recall_per_quantity_on_cosmology_fixture`) |
| 5 | Handles unicode `±`, ASCII `+/-`, LaTeX `\\pm`, asymmetric `^{+a}_{-b}` | Pass (`test_extract_symmetric_uncertainty_forms`, `test_extract_asymmetric_uncertainty`) |
| 6 | Writes `staging.extractions` with `extraction_type='quant_claim'` (mocked psycopg) | Pass (`test_insert_claims_writes_quant_claim_extraction_type`) |

## LLM hook

Hook is defined as `llm_disambiguate(span)` and unconditionally raises
`NotImplementedError("Requires paid API; see CLAUDE.md
feedback_no_paid_apis")`. Verified by
`test_llm_disambiguate_raises_not_implemented`.

## Notes

- Tests use a local `MagicMock` for psycopg — no DB is touched, so the
  suite runs cleanly without `SCIX_TEST_DSN` set.
- The CLI guard test calls `main(["--dsn", "dbname=scix"])` with
  `get_connection` patched to abort if reached; verifying the script
  exits with code 2 *before* opening any connection.
