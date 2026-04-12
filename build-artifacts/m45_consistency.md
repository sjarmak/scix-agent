# M4.5 — Lane Consistency Report

**n**: 5 bibcodes
**Mean raw Jaccard**: 0.7778
**Mean adjusted Jaccard**: 0.7778
**p50 adjusted divergence**: 0.0000
**p90 adjusted divergence**: 0.5778
**p99 adjusted divergence**: 0.6578
**Gate (p90 ≤ 0.05)**: FAIL

## Per-bibcode Jaccard

| bibcode | raw_chain_hybrid | raw_chain_static | raw_hybrid_static | adj_chain_hybrid | adj_chain_static | adj_hybrid_static | mean_adj_divergence |
|---------|------------------|------------------|-------------------|------------------|------------------|-------------------|---------------------|
| 2024M45..1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2024M45..2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2024M45..3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2024M45..4 | 1.0000 | 0.3333 | 0.3333 | 1.0000 | 0.3333 | 0.3333 | 0.4444 |
| 2024M45..5 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6667 |

## Aggregate distribution

| statistic | value |
|-----------|-------|
| n | 5 |
| mean_raw_jaccard | 0.7778 |
| mean_adj_jaccard | 0.7778 |
| p50_adj_divergence | 0.0000 |
| p90_adj_divergence | 0.5778 |
| p99_adj_divergence | 0.6578 |
| gate_threshold | 0.0500 |
| gate_passed | False |

## Per-lane-pair mean divergence

| pair | mean_adjusted_divergence |
|------|--------------------------|
| citation_chain_vs_hybrid | 0.0000 |
| citation_chain_vs_static | 0.3333 |
| hybrid_vs_static | 0.3333 |

