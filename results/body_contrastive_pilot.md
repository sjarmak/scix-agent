# Body↔Abstract Contrastive Pilot — Results

S1 of [`prd_full_text_applications_v2.md`](../docs/prd/prd_full_text_applications_v2.md).
Companion script:
[`scripts/train_body_abstract_contrastive.py`](../scripts/train_body_abstract_contrastive.py).
Companion runbook:
[`docs/runbooks/train_body_abstract_contrastive.md`](../docs/runbooks/train_body_abstract_contrastive.md).

## Methodology

Body text is a free contrastive signal that today's INDUS / SPECTER2
encoders never see. The hypothesis under test: a one-epoch fine-tune
on `(abstract, random body paragraph from the same paper)` positive
pairs improves dense retrieval on diverse scientific queries enough
to justify a full-corpus retrain.

- **Loss**: `sentence_transformers.losses.MultipleNegativesRankingLoss`.
  In-batch positives serve as negatives, so the negative pool grows
  for free with the batch size — no explicit `(abstract, body of a
  different paper)` construction required.
- **Base models considered**: INDUS (`nasa-impact/nasa-smd-ibm-st-v2`,
  768d, default) and SPECTER2 (`allenai/specter2_base`, 768d). Both
  expose 512-token context which is sufficient for an abstract +
  ~1 body paragraph.
- **Cohort size**: 100K papers per the PRD; smoke test uses 32
  synthetic pairs.
- **Pair construction**: positive = (abstract, paragraph randomly
  drawn from the same paper's `papers.body`). Negatives: implicit,
  via MNRL.
- **Eval**: 50-query retrieval set
  (`results/retrieval_eval_50q.json` baseline) — same harness used
  for the dual-model embedding and reranker work.

## Cohort Selection

Cohort selection respects M1's coverage-bias output
(`results/full_text_coverage_bias.json`) so the trained model is not
over-fit to the body-coverage skew. M1 quantifies skew across
`arxiv_class`, `year`, `citation_bucket`, `bibstem`, and
`community_semantic_medium`; we stratify on `arxiv_class` because:

1. arxiv_class is the most actionable proxy for "downstream agent
   query topic" — it maps cleanly onto MCP retrieval intent.
2. The M1 KL divergence on arxiv_class is small (≈1e-5 in the
   sample), so stratified sampling against the corpus prior is
   numerically stable.
3. Stratifying further (year × class × bibstem) starves small strata
   of pairs and inflates variance on the head queries.

The default strategy is `stratified-by-arxiv-class`: per-stratum
allocation is proportional to `q_corpus` (corpus prior), not
`p_fulltext` (full-text-skewed prior). The `random` strategy is
preserved for ablation.

Concretely, for the 100K pilot using the M1 sample distribution
(`cs.LG`, `hep-ph`, `astro-ph.SR`, `math.CO`), allocations are
~38.5K / 35.6K / 12.3K / 13.5K respectively (rounded to integer
counts; rounding drift goes to the largest bucket).

## Smoke-Test Result

The CI smoke test
([`tests/test_train_body_abstract_contrastive.py`](../tests/test_train_body_abstract_contrastive.py))
runs in well under 60 s on CPU. It uses
`sentence-transformers/all-MiniLM-L6-v2` on 32 synthetic pairs and
asserts:

- `loss_last < loss_first` after 5 training steps.
- `build_cohort_plan(..., strategy="stratified-by-arxiv-class")`
  produces per-arxiv-class allocations whose totals match the
  requested cohort size.
- The script imports cleanly without forcing
  `sentence_transformers` import at module load.

This proves the loop converges and that the wiring (loss capture,
DataLoader, output dir) does not regress without anyone noticing.
It does **not** prove anything about retrieval quality on real ADS
queries — that requires the GPU pilot below.

## Pilot Result (TBD: results pending GPU window)

> Status: **deferred pending GPU window**. The script and runbook are
> shipped; this section will be filled in by the operator who runs
> the 100K pilot.

Required content once the pilot runs:

- INDUS baseline nDCG@10 on the 50-query eval set.
- Fine-tuned model nDCG@10 on the 50-query eval set.
- Absolute and relative delta.
- Per-query win/loss table for the 5 hardest queries.
- Wall-clock training time and final loss curve.
- Cross-check on the SPECTER2 base if a parallel run is feasible
  inside the same window.

Open the run's `models/body_contrastive_<ts>/run.json` to inspect the
exact CLI args, cohort plan, and per-step loss that produced the
checkpoint.

## Recommendation (TBD: pending Pilot Result)

> Status: **deferred**. Recommendation will follow the go/no-go matrix
> in `docs/runbooks/train_body_abstract_contrastive.md` §6 once the
> pilot delta is measured.

Negative result is explicitly acceptable per the PRD; if the pilot
delta lands in the no-go band we ship this document as a negative
finding and do not retrain.
