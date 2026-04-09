# 50-Query Retrieval Evaluation

**Date**: 2026-04-06 19:25:05
**Corpus**: 10K stratified sample from 32.4M ADS papers
**Queries**: 50 seed papers with citation-based ground truth
**Ground truth**: Citation network (references + citing papers) within sample

## Results Summary

| Method | nDCG@10 | Recall@10 | Recall@20 | P@10 | MRR | Latency (ms) |
|--------|---------|-----------|-----------|------|-----|-------------|
| nomic | 0.4593 ± 0.2367 | 0.2846 | 0.4479 | 0.3780 | 0.7433 | 36 |
| hybrid_indus | 0.4281 ± 0.2438 | 0.2624 | 0.4547 | 0.3540 | 0.7474 | 36 |
| indus | 0.4274 ± 0.2415 | 0.2624 | 0.4532 | 0.3540 | 0.7444 | 38 |
| hybrid_specter2 | 0.4035 ± 0.2417 | 0.2505 | 0.3879 | 0.3400 | 0.6879 | 36 |
| specter2 | 0.4024 ± 0.2410 | 0.2505 | 0.3864 | 0.3400 | 0.6895 | 37 |
| lexical | 0.1998 ± 0.1285 | 0.0457 | 0.0457 | 0.1000 | 0.7500 | 1 |

## Statistical Significance (Wilcoxon signed-rank, nDCG@10)

| Comparison | Mean Diff | p-value | n |
|------------|-----------|---------|---|
| hybrid_specter2 vs specter2 | +0.0010 | 0.654721 | 50 |
| hybrid_specter2 vs lexical | +0.0000 | N/A | 4 |
| hybrid_indus vs indus | +0.0007 | 0.654721 | 50 |
| specter2 vs indus | -0.0250 | 0.291249 | 50 |
| specter2 vs nomic | -0.0569 | 0.004370 | 50 |
| specter2 vs lexical | +0.0000 | N/A | 4 |

## Query Distribution

- Total seed papers: 50
- Citation network size distribution:
  - 5-9 neighbors: 16 queries
  - 10-19 neighbors: 24 queries
  - 20-49 neighbors: 8 queries
  - 50+ neighbors: 2 queries
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