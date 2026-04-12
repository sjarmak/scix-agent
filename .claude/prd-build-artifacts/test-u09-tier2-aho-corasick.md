# Test Results — u09 Tier-2 Aho-Corasick + Adeft

## Command

```
SCIX_TEST_DSN=dbname=scix_test \
  .venv/bin/python -m pytest tests/test_tier2.py tests/test_adeft.py -v
```

## Summary

- **22 tests collected, 22 passed, 0 failed, 0 skipped**
- Runtime: 0.87 s
- AST lint (`scripts/ast_lint_resolver.py src`): exit 0

## Coverage of acceptance criteria

| AC                                                        | Evidence                                                                                                             |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --- | ------------- |
| 1 `build_automaton` + `link_abstract` exports, pickleable | `TestAhoCorasickPicklable::test_automaton_roundtrip`                                                                 |
| 2 HST alone → no link; HST + long-form → link             | `TestAhoCorasickAmbiguityGate::test_homograph_alone_does_not_fire` + `::test_homograph_with_long_form_fires`         |
| 3 Adeft ≥ 90 % for ≥ 3 acronyms                           | `test_adeft_accuracy_ge_90pct[HST                                                                                    | JET | AI]` all pass |
| 4 End-to-end DB write + 25 000 cap + `llm_only` demotion  | `TestLinkTier2EndToEnd::test_run_writes_tier2_rows_honoring_ambiguity` + `::test_per_entity_cap_demotes_link_policy` |
| 5 `pytest tests/test_tier2.py tests/test_adeft.py` passes | 22 passed                                                                                                            |
| 6 AST lint still exits 0                                  | `ast_lint_resolver.py src` exit 0                                                                                    |

## Notes

- All homograph co-presence, boundary-safe matching, and `dry_run`
  rollback paths are covered.
- `scripts/link_tier2.py` writes use `# noqa: resolver-lint` for parity
  with `link_tier1.py`; scripts under `scripts/` are outside the AST
  lint scope by design so no waiver is needed beyond the comment.
- `pyahocorasick>=2.0` + `scikit-learn>=1.3` added to
  `pyproject.toml` base deps.
