# 50-Query Rerank A/B Eval — INDUS hybrid + cross-encoders

> **Provenance**: in-house authored. Seed bibcodes are loaded from `results/retrieval_eval_50q.json`; ground truth is re-derived from the live `citation_edges` table at run time. Metrics are self-reported and should be interpreted as an engineering signal, not an external benchmark.

**Generated**: 2026-06-12T03:08:38Z
**Queries usable**: 50
**Hybrid stack**: INDUS dense (Qdrant) + title/abstract BM25 + body BM25, RRF k=60, top-50 candidates fed to reranker.

## Configs

| Config | nDCG@10 | Recall@10 | Recall@20 | MRR | P@10 | p50 rerank ms | p95 rerank ms |
|--------|---------|-----------|-----------|-----|------|---------------|---------------|
| `hybrid_indus` | 0.2242 | 0.0236 | 0.0401 | 0.4873 | 0.1980 | 0.00 | 0.00 |
| `minilm` | 0.2731 | 0.0281 | 0.0462 | 0.5603 | 0.2360 | 41.67 | 73.19 |
| `bge_large` | 0.2440 | 0.0256 | 0.0434 | 0.4967 | 0.2180 | 311.68 | 361.62 |
| `indus_ranker` | 0.1843 | 0.0203 | 0.0391 | 0.3851 | 0.1740 | 108.01 | 150.22 |

## Statistical significance

3 pairwise paired Wilcoxon signed-rank tests on per-query nDCG@10 deltas. Bonferroni-corrected significance threshold: α=0.05 / 3 = 0.0167.

| Comparison | n | mean Δ nDCG@10 | Wilcoxon stat | p-value | significant |
|------------|---|----------------|---------------|---------|-------------|
| minilm vs hybrid_indus | 50 | +0.0488 | 385.00 | 0.199224 | no |
| bge_large vs hybrid_indus | 50 | +0.0197 | 439.00 | 0.513414 | no |
| indus_ranker vs hybrid_indus | 50 | -0.0400 | 342.00 | 0.074166 | no |

## Winner

**Winner**: `minilm` — nDCG@10 0.2731 (+0.0489 vs `hybrid_indus` baseline), p95 rerank latency 73.19 ms.

## Methodology

- For each seed bibcode (loaded from `results/retrieval_eval_50q.json`), build a single candidate pool of top-50 via `scix.search.hybrid_search` (the production path): INDUS dense lane served from Qdrant (`scix_indus_v2_papers_s1`) + title/abstract BM25 + body BM25, fused with RRF (k=60). The seed query vector is recomputed with the INDUS encoder (mean pooling). The pool is reused across all four configs so retrieval cost is paid once and only the rerank stage is timed.
- The reranker (where present) scores all candidates; baseline returns the RRF order untouched. Candidates are enriched with their full abstract (trimmed to 1000 chars) before reranking — `hybrid_search` returns only a 150-char snippet, too short for a cross-encoder.
- Metrics computed over the truncated ranking via `scix.ir_metrics`. Recall@10/20, P@10, MRR, nDCG@10 are reported.
- Rerank latency is measured around the reranker callable only (weights are pre-warmed before the bench loop so the first scored query does not include weight materialization). Baseline p50/p95 are zero because no rerank runs.
- Ground truth is binary citation relevance: papers that cite or are cited by the seed (capped at 500 per direction). Pulled live from `citation_edges`.
- **Pool-comparability note (bead 4skc M3):** the April run (06a6cc3) built the dense lane from the legacy `paper_embeddings.embedding` pgvector column, which was DROPPED in the ADR-013 Qdrant migration. Pool comparability with April is therefore impossible; this is a re-baseline on the current production stack. Absolute nDCG is NOT comparable to the April memo, but the four-way reranker A/B is airtight because every config shares one pool per seed.

## Provenance details

- **host_python**: `3.12.3`
- **platform**: `Linux-6.17.0-19-generic-x86_64-with-glibc2.39`
- **device**: `cuda`
- **bge_revision**: `55611d7bca2a7133960a6d3b71e083071bbfc312`
- **bge_local_dir**: `models/bge-reranker-large`
- **minilm_model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **indus_ranker_model**: `nasa-impact/nasa-smd-ibm-ranker`
- **rrf_k**: `60`
- **top_n_from_hybrid**: `50`
- **k_metric**: `10`
- **pool_source**: `hybrid_search (Qdrant dense scix_indus_v2_papers_s1 + title/abstract BM25 + body BM25, RRF) — re-baseline on production stack; paper_embeddings pgvector column dropped in ADR-013 Qdrant migration, so April pool comparability is not possible`
- **qdrant_url**: `http://localhost:6633`

## Gate decision (bead 4skc M5) — NO-GO

**Recommendation: NO-GO. Keep `SCIX_RERANK_DEFAULT_MODEL='off'`.**

The M5 gate for `indus_ranker` was: GO iff **Δ nDCG@10 ≥ +0.02 vs baseline AND
p < α/3 (0.0167)**. The INDUS domain-tuned cross-encoder
(`nasa-impact/nasa-smd-ibm-ranker`) instead **regresses** the baseline:

- Δ nDCG@10 = **−0.0400** (negative — fails the effect-size gate outright).
- Wilcoxon p = 0.074 (fails the significance gate; the regression is itself
  not statistically significant at α/3).
- It is the **worst** of all four configs on nDCG@10, MRR, Recall@10 and P@10,
  while costing ~108 ms p50 / 150 ms p95 of added rerank latency.

This is not a wiring artifact: M2
(`results/indus_ranker_benchmark_m2.json`) confirms the model is wired
correctly — on its own home benchmark (`nasa-smd-IR-benchmark`) it *beats*
first-stage BM25 (nDCG@10 0.7535 → 0.7590, MRR 0.7303 → 0.7363). The model
works; it simply does not transfer to paper-to-paper, citation-relevance
reranking on this corpus. Its fine-tuning was MS-MARCO-style passage QA (per
the model card), and the citation-GT seed-as-query task is a different
relevance notion than the short-question→passage task it was trained on.

### Secondary observations (NOT acted on by this bead)

- On this **re-baselined** Qdrant pool, the generic `minilm` is the nominal
  winner (+0.0489) and `bge_large` is mildly positive (+0.0198) — the
  *opposite* of the April run (06a6cc3), where both regressed
  (minilm −0.0453, bge −0.0556) on the old `pe.embedding` pool. **Neither is
  statistically significant** here (minilm p=0.199, bge p=0.513). The sign
  flip across pool definitions is direct evidence that paper-level rerank
  gains on this corpus are **pool-dependent and not robust**, so the
  production default rightly stays `off`. A minilm flip would need its own
  bead with a significant, pool-stable result — out of scope here.
- A graded-relevance second opinion (M4, UMBRELA LLM-judge) was **deferred**:
  the citation-nDCG result is decisive (the candidate regresses, p=0.074), so
  the gate does not hinge on it. Recommended as a follow-up bead to probe
  *why* a domain-tuned reranker underperforms a generic one here.
