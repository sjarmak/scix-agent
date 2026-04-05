# Test Results: extraction-quality-eval

## Summary

22 tests passed, 0 failed. All acceptance criteria verified.

## Test Run

```
tests/test_eval_extraction_quality.py — 22 passed in 0.08s
```

## Coverage of Acceptance Criteria

| Criterion                                  | Test(s)                                                   | Status |
| ------------------------------------------ | --------------------------------------------------------- | ------ |
| File exists and is importable              | TestImportability::test_script_is_importable              | PASS   |
| Samples papers from extractions table      | TestSamplePapers (3 tests)                                | PASS   |
| Uses EntityResolver                        | TestEvaluateMentions (6 tests)                            | PASS   |
| Output includes precision and recall       | TestFormatReport::test_includes_precision_and_recall      | PASS   |
| Output includes match_method distribution  | TestFormatReport::test_includes_match_method_distribution | PASS   |
| Output includes unmatched mention examples | TestFormatReport::test_includes_unmatched_examples        | PASS   |
| Tests pass                                 | All 22 tests                                              | PASS   |
