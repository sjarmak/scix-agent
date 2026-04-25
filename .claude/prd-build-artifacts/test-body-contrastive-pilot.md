# Test Summary: body-contrastive-pilot (S1)

## Final result

```
$ .venv/bin/python -m pytest tests/test_train_body_abstract_contrastive.py -q
............                                                             [100%]
12 passed in 6.05s
```

12/12 pass. Smoke test wall-clock: **2.8 s** (budget 60 s).

## Acceptance criteria mapping

| AC | Test(s) | Status |
|----|---------|--------|
| 1. Script + required CLI flags (`--base-model`, `--cohort-size`, `--cohort-strategy`, `--output-model-dir`, `--dry-run`, `--max-steps`) | `TestImports::test_required_cli_flags_present` (parses `--help` and asserts each flag is present in the output) | PASS |
| 2. Runbook documents production procedure / 50-query eval / go-no-go | Manual: `docs/runbooks/train_body_abstract_contrastive.md` §4 / §5 / §6 written verbatim | PASS |
| 3. Results scaffold with required sections | Manual: `results/body_contrastive_pilot.md` includes Methodology / Cohort Selection / Smoke-Test Result / Pilot Result (TBD) / Recommendation (TBD) | PASS |
| 4a. Script imports cleanly | `TestImports::test_module_exposes_public_api` | PASS |
| 4b. Loss decreases | `test_smoke_training_decreases_loss` (avg first epoch loss > avg last epoch loss; deterministic via seeded torch + numpy + random) | PASS |
| 4c. Cohort-stratified sampling honours `--cohort-strategy stratified-by-arxiv-class` | `TestCohortPlan::test_stratified_strategy_proportional_to_corpus_prior` (asserts proportions 50/30/20 from a 0.5/0.3/0.2 mock) and `TestCohortPlan::test_stratified_strategy_handles_rounding_drift` | PASS |
| 5. Cohort selection reads M1 JSON (mocked file read) | `TestCoverageBiasIO::test_reads_m1_json_from_mocked_path` (writes a fake M1 payload to `tmp_path` and asserts loader returns the expected facets) and `TestCoverageBiasIO::test_planner_uses_loaded_payload` (end-to-end loader → planner) | PASS |

## Stability check

Ran the smoke test 8x in a row. After seeding `random` / `numpy` /
`torch.manual_seed` inside `run_smoke_training`, the loss curve is
fully deterministic and the assertion holds every time. Without
seeding, the per-step loss on a 32-pair / batch-4 / MiniLM smoke is
noisy enough that an unseeded `loss[0]` vs `loss[-1]` comparison
flakes — that flake is masked by the epoch-average comparison the
script now uses.

## Lint

```
$ ruff check scripts/train_body_abstract_contrastive.py tests/test_train_body_abstract_contrastive.py
All checks passed!
```

## Files changed

- `scripts/train_body_abstract_contrastive.py` (new, executable)
- `docs/runbooks/train_body_abstract_contrastive.md` (new)
- `results/body_contrastive_pilot.md` (new)
- `tests/test_train_body_abstract_contrastive.py` (new)
- `.claude/prd-build-artifacts/research-body-contrastive-pilot.md` (new)
- `.claude/prd-build-artifacts/plan-body-contrastive-pilot.md` (new)
- `.claude/prd-build-artifacts/test-body-contrastive-pilot.md` (this file)

No edits to `src/scix/*` per the task DON'Ts.
