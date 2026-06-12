# 50-Query Rerank — UMBRELA LLM-Judge Second Opinion

> **Provenance**: in-house authored. Graded relevance from the `umbrela_judge` OAuth subagent (verbatim Castorini UMBRELA rubric, 0-3), dispatched via `claude -p` — no paid API. Seeds and the shared INDUS-hybrid pool match `scripts/eval_rerank_local_ab.py` (bead 4skc). Self-reported engineering signal, not an external benchmark.

**Generated**: 2026-06-12T07:12:13Z
**Seeds used**: 50
**Judge calls**: 1069 (failed: 0)
**Pool**: shared INDUS-hybrid (Qdrant dense + title/abstract BM25 + body BM25, RRF k=60, top-50); judged the dedup union of each config's top-10.

## Judge-graded vs citation relevance

| Config | judge nDCG@10 | citation nDCG@10 | mean judge score (top-10) | judged-relevant@10 (≥2) |
|--------|---------------|------------------|---------------------------|-------------------------|
| `hybrid_indus` | 0.7108 | 0.2242 | 1.9600 | 7.46 |
| `minilm` | 0.7766 | 0.2731 | 2.1000 | 8.18 |
| `indus_ranker` | 0.6716 | 0.1843 | 1.9160 | 7.24 |

> **Judged-pool nDCG**: the ideal DCG is over the per-seed *judged union* (every paper in any config's top-10), so these values measure ordering quality of the pooled-relevant set and are NOT comparable to the citation-nDCG absolutes in the 4skc memo.

## Score distribution in the judged top-10 (pooled over seeds)

| Config | score 0 | score 1 | score 2 | score 3 |
|--------|---------|---------|---------|---------|
| `hybrid_indus` | 26 | 101 | 240 | 133 |
| `minilm` | 0 | 91 | 268 | 141 |
| `indus_ranker` | 8 | 130 | 258 | 104 |

## Why does `indus_ranker` underperform `minilm`? (disagreement analysis)

Across 50 seeds where the two top-10s diverge, comparing the papers each ranker promotes that the other does not:

| Disjoint set | n papers | mean judge score | frac relevant (≥2) | hist 0/1/2/3 |
|--------------|----------|------------------|--------------------|--------------|
| `indus_ranker` only | 266 | 1.7519 | 0.6541 | 8/84/140/34 |
| `minilm` only | 266 | 2.0977 | 0.8308 | 0/45/150/71 |

**Δ mean judge score (indus-only − minilm-only) = -0.3458.** Negative confirms the citation-nDCG finding under a graded human-calibrated signal: the papers `indus_ranker` uniquely promotes into the top-10 are judged *less* relevant than the ones `minilm` promotes — the domain-tuned ranker is mis-ordering paper-to-paper relevance, not merely disagreeing with noisy citation edges.

## Interpretation

- Judge nDCG@10: `minilm` 0.7766 vs `indus_ranker` 0.6716 vs baseline `hybrid_indus` 0.7108.
- The UMBRELA judge **corroborates** 4skc: the domain-tuned reranker underperforms the generic one on graded relevance too.
- The model card for `nasa-impact/nasa-smd-ibm-ranker` describes MS-MARCO-style short-question→passage fine-tuning. The seed-as-query task here (full paper title+abstract → related-paper ranking) is a different relevance notion; the disagreement table above quantifies how that mismatch surfaces.

## Provenance details

- **host_python**: `3.12.3`
- **platform**: `Linux-6.17.0-19-generic-x86_64-with-glibc2.39`
- **device**: `cuda`
- **minilm_model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **indus_ranker_model**: `nasa-impact/nasa-smd-ibm-ranker`
- **judge_persona**: `umbrela_judge`
- **judge_prompt_version**: `umbrela_judge-v1`
- **rrf_k**: `60`
- **top_n_from_hybrid**: `50`
- **k_judged**: `10`
- **relevant_threshold**: `2`
- **qdrant_url**: `http://localhost:6633`
