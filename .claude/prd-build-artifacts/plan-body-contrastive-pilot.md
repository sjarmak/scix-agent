# Plan: body-contrastive-pilot (S1)

## Script architecture: `scripts/train_body_abstract_contrastive.py`

```
main()
  └── parse_args()
        --base-model {indus,specter2,minilm-test} (default: indus)
        --cohort-size INT (default: 100000)
        --cohort-strategy {stratified-by-arxiv-class,random}
            (default: stratified-by-arxiv-class)
        --output-model-dir PATH (default: models/body_contrastive_<ts>)
        --coverage-bias-json PATH
            (default: results/full_text_coverage_bias.json)
        --dry-run (skip model load + training; print plan)
        --max-steps INT (default: None == one epoch)
        --batch-size INT (default: 32)
        --epochs INT (default: 1)
        --dsn STR (only consulted when --dry-run is false AND
                   --pairs-from-db is set; pilot is GPU-deferred so DB
                   path is documented but unused in this PR)
        --synthetic-pairs INT (smoke-test injection point; default: 0)
        --seed INT (default: 13)
        --log-every INT (default: 1)

  └── load_coverage_bias(path) -> dict
        Reads M1 JSON. If missing and --cohort-strategy == stratified-...,
        raise FileNotFoundError with actionable message.
        If --cohort-strategy == random, returns None (not consulted).

  └── build_cohort_plan(coverage, cohort_size, strategy, seed) -> list[CohortAllocation]
        Returns per-stratum counts:
        - random: single bucket {"label": "random", "n": cohort_size}
        - stratified-by-arxiv-class: walk facets.arxiv_class.rows,
          allocate n_i = round(cohort_size * q_corpus_i / sum(q_corpus_j)),
          fixing rounding drift on the largest bucket.

  └── load_model(base_model_id) -> SentenceTransformer
        - "indus"        -> "nasa-impact/nasa-smd-ibm-st-v2"
        - "specter2"     -> "allenai/specter2_base"
        - "minilm-test"  -> "sentence-transformers/all-MiniLM-L6-v2"

  └── make_pairs(synthetic_pairs, ...) -> list[InputExample]
        For S1 pilot real execution: queries papers + bodies, builds
        (abstract, random body paragraph from same paper) positives;
        MNRL uses other in-batch positives as implicit negatives, so we
        do NOT explicitly construct negatives (this is the standard
        MNRL recipe).
        For smoke test: synthesize N pseudo-pairs of the form
          ("paper {i} abstract {topic}", "paper {i} body sentence about {topic}")
        with topic-keyword overlap that lets the model learn the pairing.

  └── train(model, examples, batch_size, epochs, max_steps, output_dir)
        - Wraps MNRL in LossCapture(MultipleNegativesRankingLoss) so
          tests can assert step-0 vs step-N loss.
        - DataLoader(InputExample list, batch_size=batch_size, shuffle=True)
        - model.fit(train_objectives=[(dl, loss_capture)],
                    epochs=epochs,
                    steps_per_epoch=max_steps,    # if None, full epoch
                    output_path=str(output_dir),
                    show_progress_bar=False)
        - Returns LossCapture instance for inspection.

  └── write_run_log(output_dir, args, cohort_plan, loss_history)
        Writes run.json with the args, cohort plan, and loss[0]/loss[-1]
        so the runbook reader can audit deferred runs without re-loading
        a model.

Module-level helpers exposed for testing:
  - load_coverage_bias
  - build_cohort_plan
  - LossCapture (subclass of MultipleNegativesRankingLoss)
  - make_synthetic_pairs(n, seed)
  - run_smoke_training(model_id, n_pairs, max_steps, batch_size, output_dir)
        Single function used by smoke test; calls make_synthetic_pairs +
        load_model + train and returns (loss_first, loss_last).
```

## Smoke test plan: `tests/test_train_body_abstract_contrastive.py`

1. **test_script_imports_cleanly**
   - `import train_body_abstract_contrastive as tbac` (works because
     pyproject `pythonpath = ["src", "tests", "scripts"]`).
   - Assert callable: `parse_args`, `build_cohort_plan`,
     `load_coverage_bias`, `LossCapture`, `make_synthetic_pairs`,
     `run_smoke_training`.

2. **test_cohort_plan_stratified_honours_strategy**
   - Mock coverage-bias JSON with 3 strata at ratios 0.5/0.3/0.2.
   - `build_cohort_plan(coverage, 100, "stratified-by-arxiv-class",
     seed=0)` returns 3 buckets summing to 100, in proportion 50/30/20.
   - `build_cohort_plan(None, 100, "random", seed=0)` returns single
     bucket {label: "random", n: 100}.

3. **test_load_coverage_bias_reads_m1_json**
   - Use tmp_path + monkeypatch to write a fake M1 JSON file.
   - Assert `load_coverage_bias(path)` returns dict with
     `facets.arxiv_class.rows`.
   - Assert FileNotFoundError when path is missing AND strategy
     would need it (separate test using build_cohort_plan).

4. **test_smoke_training_decreases_loss** (the <60s test)
   - `loss_first, loss_last = run_smoke_training(
       model_id="sentence-transformers/all-MiniLM-L6-v2",
       n_pairs=32, max_steps=5, batch_size=8, output_dir=tmp_path)`
   - Assert `loss_last < loss_first` (with a small tolerance margin in
     case of stochastic noise — but in-batch MNRL on aligned pairs is
     reliable).
   - Marker: skip if sentence_transformers cannot import (defensive,
     in case CI image lacks the optional dep).

5. **test_help_flags_present**
   - Parse `--help` via argparse + capsys; assert all five required
     flags appear.

## Runbook structure: `docs/runbooks/train_body_abstract_contrastive.md`

Sections:
1. Overview (S1 of PRD v2; operational deferral; what ships now vs what
   needs a GPU window).
2. Prerequisites (.venv, sentence-transformers, optional CUDA, M1 JSON).
3. Smoke test invocation (CPU, <60s).
4. Production procedure
   - scix-batch wrapping for the real 100K pilot.
   - GPU window expectations (RTX 5090, ~30 min wall-clock budget for
     100K pairs at batch 64; estimate based on INDUS forward throughput
     observed in the embedding pipeline — cite ~2-5K rec/s).
   - Output artifacts: model dir, run.json, training logs.
5. 50-query eval invocation (per M5 PRD wording).
   - Document the eval invocation pattern referencing
     `results/retrieval_eval_50q.json` baseline.
6. Go/no-go criteria for full-corpus retrain.
   - Quantitative: nDCG@10 delta ≥ +1.0 absolute over INDUS baseline
     to recommend retrain; ≤ 0 means abandon.
   - Qualitative: even a positive delta requires manual spot-check on
     the 5 hardest queries from the eval set before scheduling a
     full-corpus retrain.

## Results scaffold: `results/body_contrastive_pilot.md`

Sections (per acceptance criterion 3):
- Methodology (filled now)
- Cohort Selection (filled now — references M1 strata)
- Smoke-Test Result (filled now — links to test file)
- Pilot Result (TBD — placeholder block)
- Recommendation (TBD — placeholder block)

## Acceptance matrix

| AC | Approach |
|---|---|
| 1. script + flags | `--help` smoke-asserted in test_help_flags_present |
| 2. runbook scix-batch + 50q eval + go/no-go | written into runbook sections 4/5/6 |
| 3. results md scaffold | written verbatim with TBD blocks |
| 4. pytest <60s passes | smoke uses MiniLM + 32 pairs + 5 steps; covers (a) imports, (b) loss decrease, (c) cohort strategy |
| 5. cohort reads M1 JSON | test_load_coverage_bias_reads_m1_json uses tmp_path mock |

## Out of scope (explicit)
- Real DB pull of bodies (script signposts but does not run it).
- Real 100K training (gated on GPU window).
- Eval harness invocation against retrieval_eval_50q (depends on
  M5/eval pipeline — runbook documents only).
- No edits to `src/scix/*`.
- No paid APIs; no W&B; logging is stdout-only.
