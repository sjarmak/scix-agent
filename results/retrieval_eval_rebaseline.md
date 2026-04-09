# 50-Query Retrieval Evaluation

**Date**: 2026-04-08 22:43:43
**Corpus**: 10K stratified sample from 32.4M ADS papers
**Queries**: 50 seed papers with citation-based ground truth
**Ground truth**: Citation network (references + citing papers) within sample

## Results Summary

| Method | nDCG@10 | Recall@10 | Recall@20 | P@10 | MRR | Latency (ms) |
|--------|---------|-----------|-----------|------|-----|-------------|
| nomic | 0.4842 ± 0.2384 | 0.3092 | 0.4683 | 0.4000 | 0.7435 | 92 |
| indus | 0.4434 ± 0.2436 | 0.2859 | 0.4264 | 0.3700 | 0.7083 | 93 |
| hybrid_indus | 0.4412 ± 0.2442 | 0.2859 | 0.4264 | 0.3700 | 0.6974 | 86 |
| specter2 | 0.4223 ± 0.2234 | 0.2665 | 0.3906 | 0.3560 | 0.7074 | 117 |
| hybrid_specter2 | 0.4217 ± 0.2241 | 0.2665 | 0.3906 | 0.3560 | 0.6957 | 85 |
| lexical | 0.2363 ± 0.1186 | 0.0745 | 0.0745 | 0.1333 | 0.7500 | 1 |

## Statistical Significance (Wilcoxon signed-rank, nDCG@10)

| Comparison | Mean Diff | p-value | n |
|------------|-----------|---------|---|
| hybrid_specter2 vs specter2 | -0.0006 | 0.592980 | 50 |
| hybrid_specter2 vs lexical | +0.2685 | 0.031250 | 6 |
| hybrid_indus vs indus | -0.0022 | 0.285049 | 50 |
| specter2 vs indus | -0.0211 | 0.398855 | 50 |
| specter2 vs nomic | -0.0619 | 0.008512 | 50 |
| specter2 vs lexical | +0.2732 | 0.031250 | 6 |

## Query Distribution

- Total seed papers: 50
- Citation network size distribution:
  - 5-9 neighbors: 20 queries
  - 10-19 neighbors: 18 queries
  - 20-49 neighbors: 10 queries
  - 50+ neighbors: 2 queries
- Year range: 2000-2025

## Methodology

- **Query formulation**: Seed paper title + first 50 words of abstract used as
  query text for lexical search; stored embedding used as query vector for dense retrieval.
- **Ground truth**: Citation network (references + citing papers) restricted to
  the pilot sample. Binary relevance: cited/citing = relevant, else irrelevant.
- **Fusion**: RRF with k=60 (standard constant).
- **Retrieval pool**: 10K stratified sample for both dense and lexical search.
  Vector search restricted to pilot sample via JOIN for fair comparison.
- **Models**: SPECTER2 (allenai/specter2_base, 768d),
  INDUS (nasa-impact/nasa-smd-ibm-st-v2, 768d),
  Nomic (nomic-ai/nomic-embed-text-v1.5, 768d).

### Limitations

- Citation-based ground truth favors models trained on citation proximity (SPECTER2),
  yet SPECTER2 underperforms INDUS and Nomic here — suggesting ADS-specific
  training data (INDUS) and general representation quality (Nomic) matter more
  than training objective alignment.
- 20K sample limits lexical search: plainto_tsquery (AND logic) with specific title
  words returns results for <10% of queries. Hybrid search therefore defaults to
  vector-only for most queries. Full-corpus hybrid evaluation pending completion
  of bulk embedding pipeline.
- text-embedding-3-large not included (no embeddings generated yet).
  Will be added when OpenAI embeddings are available.
- Random seed selection (seed=42) used for reproducibility. Results may vary
  slightly with different seed sets.