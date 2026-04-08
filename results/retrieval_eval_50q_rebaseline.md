# 50-Query Retrieval Evaluation

**Date**: 2026-04-07 21:01:37
**Corpus**: 10K stratified sample from 32.4M ADS papers
**Queries**: 50 seed papers with citation-based ground truth
**Ground truth**: Citation network (references + citing papers) within sample

## Results Summary

| Method | nDCG@10 | Recall@10 | Recall@20 | P@10 | MRR | Latency (ms) |
|--------|---------|-----------|-----------|------|-----|-------------|
| nomic | 0.3736 ± 0.2338 | 0.2829 | 0.4229 | 0.2960 | 0.6073 | 54 |
| indus | 0.3495 ± 0.2167 | 0.2601 | 0.4108 | 0.2780 | 0.6085 | 55 |
| hybrid_indus | 0.3332 ± 0.2218 | 0.2576 | 0.4095 | 0.2700 | 0.5544 | 51 |
| specter2 | 0.3220 ± 0.2196 | 0.2418 | 0.3706 | 0.2520 | 0.5993 | 51 |
| hybrid_specter2 | 0.3209 ± 0.2245 | 0.2426 | 0.3647 | 0.2540 | 0.5789 | 52 |
| lexical | 0.0864 ± 0.1664 | 0.0573 | 0.0729 | 0.1000 | 0.1000 | 1 |

## Statistical Significance (Wilcoxon signed-rank, nDCG@10)

| Comparison | Mean Diff | p-value | n |
|------------|-----------|---------|---|
| hybrid_specter2 vs specter2 | -0.0011 | 0.600179 | 50 |
| hybrid_specter2 vs lexical | +0.1170 | 0.031250 | 6 |
| hybrid_indus vs indus | -0.0164 | 0.046399 | 50 |
| specter2 vs indus | -0.0276 | 0.170952 | 50 |
| specter2 vs nomic | -0.0516 | 0.019172 | 50 |
| specter2 vs lexical | +0.1288 | 0.062500 | 6 |

## Query Distribution

- Total seed papers: 50
- Citation network size distribution:
  - 5-9 neighbors: 26 queries
  - 10-19 neighbors: 18 queries
  - 20-49 neighbors: 6 queries
- Year range: 1996-2025

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