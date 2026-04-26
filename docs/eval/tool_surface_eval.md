# Tool-Surface Eval — V0 (current 18) vs V1 (consolidated 8) vs V2 (terse v1)

Compares agent tool-selection across three MCP surface variants. Each variant returns identical canned data so we measure *selection* independently from retrieval quality. Agent: claude `-p` (Sonnet via OAuth subagent), 3 runs per (variant, query) at default temperature.

## Headline

| variant | n_runs | tool_accuracy | param_accuracy | avg_mcp_calls | selection_consistency |
|---|---:|---:|---:|---:|---:|
| v0 | 90 | 94.4% | 94.4% | 1.01 | 93.3% |
| v1 | 90 | 96.7% | 96.7% | 1.02 | 93.3% |
| v2 | 90 | 95.6% | 83.3% | 1.03 | 96.7% |

## Per-intent cluster

| intent | n_q | v0 tool / param | v1 tool / param | v2 tool / param |
|---|---:|---:|---:|---:|
| citation_similarity | 6 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| citation_traverse | 6 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| entity_context | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| entity_lookup | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| facet_counts | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| find_gaps | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| graph_context | 3 | 66.7% / 66.7% | 100.0% / 100.0% | 100.0% / 100.0% |
| paper_blame | 3 | 100.0% / 100.0% | 33.3% / 33.3% | 0.0% / 0.0% |
| paper_claims | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| paper_metadata | 6 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| paper_read | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| paper_replications | 3 | 100.0% / 100.0% | 66.7% / 66.7% | 66.7% / 66.7% |
| search_chunk | 9 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| search_claim | 6 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| search_concept | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| search_paper | 12 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 33.3% |
| search_section | 12 | 66.7% / 66.7% | 100.0% / 100.0% | 100.0% / 75.0% |
| temporal_evolution | 3 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |

## Disagreements (queries where variants split)

_2 of 30 queries had at least one variant disagree on majority correctness._

| query_id | intent | query | v0 | v1 | v2 |
|---|---|---|---|---|---|
| q27 | search_section | Find paragraph-level evidence for early dark energy in pa... | ✗ | ✓ | ✓ |
| q55 | paper_blame | Trace claim claim-001 back to its earliest source paper | ✓ | ✗ | ✗ |

## Verdict

- Highest tool accuracy: **v1** (96.7%)
- Deltas vs v0: v1 vs v0: +2.2%, v2 vs v0: +1.1%
- Most consistent across runs: **v2** (96.7%)
- Fewest MCP calls per query: **v0** (1.01)

_Interpretation: a >5pt advantage on tool_accuracy with comparable or better consistency suggests the consolidated surface is at least as selectable. A drop in v2 vs v1 isolates the description-quality effect._
