# 50-Query Rerank A/B Eval — INDUS hybrid + cross-encoders

> **Provenance**: in-house authored. Seed bibcodes are loaded from `/home/ds/projects/scix_experiments/results/retrieval_eval_50q.json`; ground truth is re-derived from the live `citation_edges` table at run time. Metrics are self-reported and should be interpreted as an engineering signal, not an external benchmark.

**Generated**: 2026-04-25T15:46:28Z
**Queries usable**: 50
**Hybrid stack**: INDUS dense (HNSW) + BM25 (tsvector), RRF k=60, top-50 candidates fed to reranker.

## Configs

| Config | nDCG@10 | Recall@10 | Recall@20 | MRR | P@10 | p50 rerank ms | p95 rerank ms |
|--------|---------|-----------|-----------|-----|------|---------------|---------------|
| `hybrid_indus` | 0.3255 | 0.0334 | 0.0536 | 0.6239 | 0.2900 | 0.00 | 0.00 |
| `minilm` | 0.2802 | 0.0290 | 0.0484 | 0.5428 | 0.2500 | 74.92 | 113.46 |
| `bge_large` | 0.2699 | 0.0278 | 0.0449 | 0.5019 | 0.2520 | 570.64 | 766.08 |

## Statistical significance

Two pairwise paired Wilcoxon signed-rank tests on per-query nDCG@10 deltas. Bonferroni-corrected significance threshold: α=0.05 / 2 = 0.0250.

| Comparison | n | mean Δ nDCG@10 | Wilcoxon stat | p-value | significant |
|------------|---|----------------|---------------|---------|-------------|
| minilm vs hybrid_indus | 50 | -0.0453 | 337.50 | 0.042176 | no |
| bge_large vs hybrid_indus | 50 | -0.0556 | 304.00 | 0.025813 | no |

## Winner

**Winner**: `hybrid_indus` — nDCG@10 0.3255 (+0.0000 vs `hybrid_indus` baseline), p95 rerank latency 0.00 ms.

## Methodology

- For each seed bibcode (loaded from `/home/ds/projects/scix_experiments/results/retrieval_eval_50q.json`), build a single INDUS-hybrid candidate pool of top-50 via `scix.search.lexical_search` (BM25) + an INDUS dense lane (`pe.embedding` cosine, the legacy column populated by the production embedding pipeline) fused with `scix.search.rrf_fuse` (k=60). The pool is reused across configs so retrieval cost is paid once and only the rerank stage is timed.
- The reranker (where present) scores all candidates; baseline returns the RRF order untouched.
- Metrics computed over the truncated ranking via `scix.ir_metrics`. Recall@10/20, P@10, MRR, nDCG@10 are reported.
- Rerank latency is measured around the reranker callable only (weights are pre-warmed before the bench loop so the first scored query does not include weight materialization). Baseline p50/p95 are zero because no rerank runs.
- Ground truth is binary citation relevance: papers that cite or are cited by the seed (capped at 500 per direction). Pulled live from `citation_edges`.
- Note: this script reads `pe.embedding` directly rather than calling `scix.search.hybrid_search` because that path requires the halfvec(768) shadow column `pe.embedding_hv` (migration 053) which is not yet present on the production database. RRF fusion and lexical search remain unchanged.

## Provenance details

- **host_python**: `3.12.3`
- **platform**: `Linux-6.17.0-19-generic-x86_64-with-glibc2.39`
- **device**: `cuda`
- **bge_revision**: `55611d7bca2a7133960a6d3b71e083071bbfc312`
- **bge_local_dir**: `models/bge-reranker-large`
- **minilm_model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **rrf_k**: `60`
- **top_n_from_hybrid**: `50`
- **k_metric**: `10`
