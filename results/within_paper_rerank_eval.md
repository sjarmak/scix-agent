# search_within_paper section-level rerank — M5 eval

> ⚠️ **SUPERSEDED / INVALID (bead scix_experiments-4skc).** This eval is
> structurally void: it runs on synthetic IMRaD fixtures scored by a
> Python token-count stub, which puts the baseline at nDCG@3 = 1.0000 — a
> ceiling by construction. The negative 'delta' below measures nothing
> about real reranking and must not be cited as evidence for or against
> cross-encoder reranking. For a valid, real-corpus reranker eval (incl.
> the INDUS domain-tuned cross-encoder) see
> `results/retrieval_eval_50q_rerank_indus.md` and
> `results/indus_ranker_benchmark_m2.json`. A real section-level rerank
> redesign over OA bodies (gated on `papers_is_oa_or_preprint`) would be
> a separate bead.

## Methodology

- Fixture: `tests/fixtures/within_paper_rerank_gold_20.jsonl` (20 entries)
- Each entry has a synthetic IMRaD-style paper body, a query, and a
  hand-labeled `gold_section_idx`.
- Baseline: `search_within_paper(..., use_rerank=False)` — top-3 by
  per-section `ts_rank` (PostgreSQL or Python proxy fallback).
- Reranked: `search_within_paper(..., use_rerank=True)` with
  `SCIX_RERANK_DEFAULT_MODEL=minilm`
  (`cross-encoder/ms-marco-MiniLM-L-12-v2`).
- Metric: nDCG@3 with binary relevance, averaged across 20 queries.
- Latency metric: per-query wall-clock around the function, p95 over
  the 20-query batch, MiniLM model.

## Results

| Metric | Value |
| --- | --- |
| Baseline nDCG@3 (BM25 only) | 1.0000 |
| Reranked nDCG@3 (MiniLM)    | 1.0000 |
| Delta                       | +0.0000 |
| p95 latency (rerank, MiniLM)| 0.0 ms |
| Improvement threshold       | +0.05 |

## Recommendation

NO-GO (negative result) — section-level cross-encoder rerank improves nDCG@3 by only +0.0000 (< +0.05). Keep `SCIX_RERANK_DEFAULT_MODEL='off'` as the production default. The signature still defaults `use_rerank=True` so flipping the env is the only operator change needed if a future re-eval shows a different outcome.

NOTE: MiniLM weights were not loadable in this environment, so the rerank pass fell back to the ts_rank ordering. The reported delta therefore measures the no-rerank vs no-rerank case and is exactly 0 by construction. Re-run on a host with sentence-transformers + network/cache for the real number.
