# Plan — ner-wiesp-eval

## Files
1. `scripts/eval_ner_wiesp.py` — executable as `python -m scripts.eval_ner_wiesp` and as `python scripts/eval_ner_wiesp.py`.
2. `tests/test_eval_ner_wiesp.py` — pytest unit tests with monkeypatched HF loaders.
3. `results/ner_wiesp_eval.json` — placeholder report regenerated from tiny fixture, documented as such.
4. `pyproject.toml` — add `seqeval>=1.2` under new optional extra `ner_eval`.

## Model pinning
- Hardcode SHA as module-level constant:
  ```python
  MODEL_NAME = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"
  MODEL_REVISION = "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"
  ```
  This is a full-length hex SHA. If a future pin check wants the actual latest,
  a docstring note tells reader how to refresh it (HF model page → commits).

## Script structure
```
argparse:
  --output PATH (default results/ner_wiesp_eval.json)
  --sample N (limit dataset rows, default 0=all)
  --fixture PATH (env var WIESP_TEST_FIXTURE overrides)
  --no-model (skip HF model download; only useful with fixture predictions)
functions:
  load_model_and_tokenizer(revision) -> (tokenizer, model, id2label)
  load_dataset(sample, fixture_path) -> list[Example]
  run_inference(model, tokenizer, examples, id2label) -> list[list[str]] (predicted tags per token)
  compute_metrics(preds, golds) -> dict (per-entity precision/recall/f1 + micro + macro)
  write_report(metrics, path, meta) -> None
```

## F1 computation
- Prefer `seqeval.metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)`.
- Fallback: pure-Python BIO entity extraction + per-type TP/FP/FN counting. This fallback is what tests exercise (no seqeval install needed in CI).

## Fixture schema
A JSON list of examples:
```json
[
  {"tokens": ["M87", "observed", "by", "Chandra"],
   "tags":   ["B-CelestialObject","O","O","B-Mission"],
   "pred":   ["B-CelestialObject","O","O","B-Mission"]}
]
```
When `pred` is present, inference is skipped — allows offline/mocked runs.

## Test plan (tests/test_eval_ner_wiesp.py)
- `test_compute_metrics_perfect`: identical gold/pred → precision=recall=f1=1.0.
- `test_compute_metrics_partial`: one missed, one spurious → F1 matches hand-computed expected.
- `test_compute_metrics_empty_predictions`: all-O predictions against gold with entities → precision=0, recall=0, f1=0, report schema still well-formed.
- `test_report_schema`: returned dict has keys `per_entity`, `summary` with `micro_f1`, `macro_f1`, `support`, etc.
- `test_run_inference_mocked`: monkeypatch `AutoTokenizer`/`AutoModelForTokenClassification.from_pretrained` to return fakes with fixed label outputs; assert inference returns expected tags.
- `test_load_dataset_from_fixture`: fixture path load returns parsed Example list.
- `test_main_end_to_end_fixture`: run `main()` with `--fixture` (has `pred` fields), reading out of tmp dir → writes JSON file with expected schema.

## Placeholder results/ner_wiesp_eval.json
- Generated from a 3-example fixture with mix of correct/incorrect predictions.
- Includes a `_note` field stating "fixture-based placeholder; regenerate on GPU+network with real WIESP test split".
- Includes `model_name`, `model_revision`, `dataset`, `n_examples`.

## Acceptance self-check
1. Script exists + executable ✓
2. Running produces JSON ✓ (via fixture path)
3. Pinned SHA visible as module constant ✓
4. Loads WIESP from HF datasets OR local cache (fixture) ✓
5. Unit test for F1 on synthetic example ✓
6. Tests run without network (monkeypatch) ✓
7. pytest passes ✓
