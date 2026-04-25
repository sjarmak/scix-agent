# search_within_paper section-level rerank — M5 eval

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
| Reranked nDCG@3 (MiniLM)    | 0.9815 |
| Delta                       | -0.0185 |
| p95 latency (rerank, MiniLM)| 16.7 ms |
| Improvement threshold       | +0.05 |

## Recommendation

NO-GO (negative result) — section-level cross-encoder rerank improves nDCG@3 by only -0.0185 (< +0.05). Keep `SCIX_RERANK_DEFAULT_MODEL='off'` as the production default. The signature still defaults `use_rerank=True` so flipping the env is the only operator change needed if a future re-eval shows a different outcome.
