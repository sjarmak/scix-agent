# Research: body-contrastive-pilot (S1)

## Source PRD
- `docs/prd/prd_full_text_applications_v2.md` §S1 (lines 144–154).
- Pilot scope is documented; full 100K execution is GPU-window deferred.

## Inputs (already-shipped dependencies)
- M1 coverage-bias output lives at `results/full_text_coverage_bias.json`
  (commit 130f4c6). Schema observed:
  - top-level `facets.<facet_name>.rows[]`
  - each row: `{label, total, with_body, without_body, pct_with_body,
    p_fulltext, q_corpus, ratio_p_over_q}`
  - `arxiv_class` facet exists with 4 strata in dry-run sample
    (`cs.LG`, `hep-ph`, `astro-ph.SR`, `math.CO`).
- `papers.body` populated for 14.9M papers; section_parser available but
  pilot can sample paragraphs via simple split on blank lines.

## Reusable code patterns
- `scripts/pilot_embed_compare.py`:
  - Model registry pattern with `MODELS = {"indus": {...}, "specter2": {...}}`.
  - INDUS = `nasa-impact/nasa-smd-ibm-st-v2` (768d, sentence-transformers
    compatible).
  - SPECTER2 = `allenai/specter2_base` (transformers CLS pooling) — but
    `allenai/specter2` (the sentence-transformers wrapper) is a sibling
    that works with `SentenceTransformer(...)` directly.
  - `_prepare_text(title, abstract, prefix)` for "title [SEP] abstract"
    join — reuse pattern.
- `scripts/embed.py`: standard CLI shape with argparse + logging.basicConfig.
- `src/scix/db.py`: `is_production_dsn`, `get_connection`, `redact_dsn`,
  `DEFAULT_DSN` — but S1 pilot does not need DB writes (model training
  outputs go to `--output-model-dir`).

## sentence-transformers training surface (v5.3.0 installed)
- `from sentence_transformers import SentenceTransformer, InputExample`
- `from sentence_transformers.losses import MultipleNegativesRankingLoss`
- `from torch.utils.data import DataLoader`
- API: `model.fit(train_objectives=[(train_dl, train_loss)],
  epochs=N, steps_per_epoch=K, output_path=...)`
- For deterministic loss capture in tests: wrap loss in a thin subclass
  that records each forward pass loss into a list, OR use the
  `callback=` kwarg of `model.fit` (callback signature
  `(score, epoch, steps)` is for evaluator only — it does NOT receive
  loss). The tractable approach: subclass MNRL, override `forward` to
  append loss to an instance list.
- Tiny model for smoke: `sentence-transformers/all-MiniLM-L6-v2` (22 MB,
  384d, runs ~5 batches in <30s on CPU).

## Cohort selection
- M1 JSON `facets.arxiv_class.rows[*]` provides per-stratum counts.
- For `--cohort-strategy stratified-by-arxiv-class` (default): sample
  proportional to `q_corpus` (the corpus prior), so the trained model
  is not over-fit to the full-text-skewed `p_fulltext` distribution.
- For `--cohort-strategy random`: ignore strata, uniform sample.
- The cohort sampler returns a list of bibcodes; pair-builder is
  responsible for fetching abstract + body paragraphs.

## Operational deferral pattern
- Same pattern as `plan-body-ner-pilot.md` — script is shipped and CI-
  tested, but real execution is gated on a GPU window. Smoke test
  uses synthetic pairs to prove the loop converges.

## Eval invocation (M5 PRD wording)
- `prd_full_text_applications_v2.md` §S1 acceptance: "evaluated on the
  50-query retrieval eval (`results/retrieval_eval_50q.json`)".
- Existing eval lives at `results/retrieval_eval_50q.json` and
  `results/retrieval_eval_50q.md`; runbook will document the
  invocation pattern (script TBD by M5, but referenced for go/no-go
  framework).

## Files-in-scope checklist
- [x] `scripts/train_body_abstract_contrastive.py` (new)
- [x] `docs/runbooks/train_body_abstract_contrastive.md` (new)
- [x] `results/body_contrastive_pilot.md` (new — scaffold only)
- [x] `tests/test_train_body_abstract_contrastive.py` (new)
- Nothing in `src/scix/*` is touched (per DON'Ts).
