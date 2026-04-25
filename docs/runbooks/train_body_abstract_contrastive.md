# Body↔Abstract Contrastive Training Runbook

S1 of [`prd_full_text_applications_v2.md`](../prd/prd_full_text_applications_v2.md).
Companion script: [`scripts/train_body_abstract_contrastive.py`](../../scripts/train_body_abstract_contrastive.py).

## 1. Overview

This runbook covers the body↔abstract contrastive fine-tune pilot. The
script ships production-ready and is exercised in CI by a sub-60-second
smoke test on synthetic pairs with a tiny model. The **real 100K pilot
is GPU-window deferred** — execute it inside an explicit GPU window
under `scix-batch` so it stays out of the gascity supervisor's OOM
blast radius (see `CLAUDE.md` § _Memory isolation_).

The training signal is free: each paper with full text contributes
`(abstract, random body paragraph)` as a positive pair, and
`MultipleNegativesRankingLoss` derives negatives from other in-batch
positives. Cohort selection is gated on M1's coverage-bias output so
the trained model is not over-fit to the 14.9M-paper full-text-skewed
distribution.

## 2. Prerequisites

| Dependency | Notes |
|---|---|
| Python venv at `.venv/` | Per `CLAUDE.md`. |
| `sentence-transformers >= 5.x` | Already pinned via `pyproject.toml [project.optional-dependencies] search`. |
| `torch` with CUDA 13.1 wheels | Smoke test only needs CPU. |
| `results/full_text_coverage_bias.json` | M1 output (commit `130f4c6`). Required when `--cohort-strategy stratified-by-arxiv-class`. |
| Postgres DSN with `papers.body` populated | Required only for the deferred real run. |
| `scix-batch` wrapper at `~/.local/bin/scix-batch` | Required for the real run; see `CLAUDE.md`. |

## 3. Smoke test (CPU, < 60 s)

The smoke path exists to prove the training loop converges before
spending a GPU window on it.

```bash
cd ~/projects/scix_experiments
.venv/bin/pytest tests/test_train_body_abstract_contrastive.py -q
```

What it asserts:

- The script imports cleanly without sentence-transformers being on the
  critical path of `import` (the dep is loaded lazily inside the
  training functions).
- `build_cohort_plan(..., strategy="stratified-by-arxiv-class")`
  reads M1 JSON (mocked via `tmp_path`) and produces per-arxiv-class
  allocations whose totals match `--cohort-size`.
- `run_smoke_training(...)` on 32 synthetic pairs + 5 steps + the
  `sentence-transformers/all-MiniLM-L6-v2` model produces a strictly
  decreasing loss between step 0 and the last step.

If the smoke test fails, do NOT schedule the real GPU run — the loop
itself is broken and a 30-minute pilot will only confirm that.

## 4. Production procedure (real 100K pilot — GPU window)

Wrap with `scix-batch` so the job is the first to be killed if it
busts memory limits and the gascity supervisor survives.

```bash
cd ~/projects/scix_experiments

scix-batch --mem-high 25G --mem-max 40G \
    .venv/bin/python scripts/train_body_abstract_contrastive.py \
        --base-model indus \
        --cohort-strategy stratified-by-arxiv-class \
        --cohort-size 100000 \
        --batch-size 64 \
        --epochs 1 \
        --output-model-dir models/body_contrastive_indus_$(date +%Y%m%d) \
        2>&1 | tee logs/body_contrastive_$(date +%Y%m%d).log
```

For an ablation against the random cohort:

```bash
scix-batch --mem-high 25G --mem-max 40G \
    .venv/bin/python scripts/train_body_abstract_contrastive.py \
        --base-model indus \
        --cohort-strategy random \
        --cohort-size 100000 \
        --output-model-dir models/body_contrastive_indus_random_$(date +%Y%m%d)
```

To swap the base model to SPECTER2:

```bash
scix-batch ... --base-model specter2 ...
```

### GPU window expectations

- Hardware: RTX 5090 (CUDA 13.1).
- The INDUS forward path runs at ~2–5K records/sec in the existing
  embedding pipeline. A 100K cohort × 1 epoch at batch 64 fits in a
  ~30 minute wall-clock window with margin for the optimiser overhead;
  budget 60 minutes end-to-end including model load and checkpoint
  flush.
- Co-existence rule: M2 (body NER) is the long-running GPU job. Do
  not start the contrastive pilot while body NER is mid-pass — see
  `ner_gpu_deployment.md` for the FCFS scheduling discipline.

### Output artifacts

- `models/body_contrastive_<ts>/` — sentence-transformers checkpoint
  directory written by `model.fit(output_path=...)`.
- `models/body_contrastive_<ts>/run.json` — written by
  `write_run_log(...)`. Records the CLI args, the per-stratum cohort
  plan, and `loss_first` / `loss_last` for quick auditing.
- Stdout/stderr training log captured via `tee` per the command
  template above. There is no W&B / MLOps integration on purpose:
  free, local, no paid APIs.

## 5. 50-query retrieval evaluation

Per `prd_full_text_applications_v2.md` §S1:

> _Fine-tuned model evaluated on the 50-query retrieval eval
> (`results/retrieval_eval_50q.json`); report in
> `results/body_contrastive_pilot.md` with nDCG@10 delta vs the base
> model and a go/no-go recommendation for a full-corpus retrain.
> Negative result is an acceptable ship._

Once the pilot model is checkpointed, run the eval against the
existing 50-query set, comparing the fine-tuned model to the unmodified
INDUS baseline. The eval invocation reuses the harness behind
`results/retrieval_eval_50q.json` and `results/retrieval_eval_50q.md`.
Persist the per-query results next to the model directory and update
`results/body_contrastive_pilot.md` (the "Pilot Result" section) with
the nDCG@10 delta and the per-query winner table.

## 6. Go / no-go criteria for full-corpus retrain

Recommendation rule for `results/body_contrastive_pilot.md`:

| Outcome | nDCG@10 delta vs INDUS baseline | Action |
|---|---|---|
| **Strong go** | ≥ +1.0 absolute | Schedule full-corpus retrain. |
| **Weak go** | +0.3 to +1.0 | Spot-check the 5 hardest queries; only retrain if no regressions appear in the long-tail head. |
| **No-go** | between −0.3 and +0.3 | Ship the pilot result as a negative finding. Do not retrain. |
| **Regression** | ≤ −0.3 | Investigate before any retrain — likely cohort skew or pair quality. |

In every case: human-in-the-loop spot-check of the 5 hardest queries is
required before a retrain. Even a +1.0 average can hide a catastrophic
regression on the head queries.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Coverage-bias JSON not found` | M1 output missing or moved | Re-run `scripts/report_full_text_coverage_bias.py` or pass an explicit `--coverage-bias-json`. |
| `NotImplementedError: Real-data pair construction is GPU-window deferred` | Forgot `--synthetic-pairs N` for a smoke run | Either pass `--synthetic-pairs 32` for the smoke loop, or implement the DB pair-builder per the script's TODO before re-running. |
| Smoke test loss does not decrease | Tiny model could not learn the synthetic alignment within 5 steps | Increase `--max-steps` to 10; if still flat, the loss-capture wiring is wrong — check `make_loss_capture`. |
| OOM kill during real run | Default `scix-batch` limits too tight for batch 64 + INDUS | Drop `--batch-size 32` or raise `scix-batch --mem-max`. |
