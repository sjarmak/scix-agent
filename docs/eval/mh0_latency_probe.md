# MH-0 Day-1 Latency Probe

Per amendment A1 of `docs/prd/scix_deep_search_v1.md` — measures per-turn dispatcher overhead for `ClaudeSubprocessDispatcher` (`src/scix/eval/persona_judge.py:354`).

## Methodology

- Fixture: 5-turn replay of representative (query, bibcode, snippet) triples.
- Runs: 5 (total measured turns: 25).
- Each turn awaits `dispatcher.judge(triple)` and records `time.perf_counter()` delta — i.e. subprocess startup + OAuth handshake + persona context replay. *Tool execution is excluded from this probe by construction* (no tool calls inside the judge subagent).
- Acceptance (amendment A1): PASS iff p50 ≤ 3.0s AND p95 ≤ 6.0s.

## Measurements

| Metric | Value (s) | Threshold (s) | Status |
|---|---|---|---|
| p50 per-turn overhead | 15.341 | 3.0 | OVER |
| p95 per-turn overhead | 19.854 | 6.0 | OVER |

### Per-turn timings (seconds)

| run | turn | elapsed_s |
|---|---|---|
| 0 | 0 | 16.475 |
| 0 | 1 | 15.341 |
| 0 | 2 | 14.873 |
| 0 | 3 | 16.557 |
| 0 | 4 | 18.111 |
| 1 | 0 | 16.626 |
| 1 | 1 | 20.194 |
| 1 | 2 | 14.998 |
| 1 | 3 | 15.904 |
| 1 | 4 | 14.223 |
| 2 | 0 | 18.495 |
| 2 | 1 | 14.023 |
| 2 | 2 | 16.725 |
| 2 | 3 | 16.252 |
| 2 | 4 | 15.882 |
| 3 | 0 | 14.950 |
| 3 | 1 | 14.528 |
| 3 | 2 | 16.758 |
| 3 | 3 | 23.962 |
| 3 | 4 | 13.156 |
| 4 | 0 | 14.392 |
| 4 | 1 | 14.495 |
| 4 | 2 | 14.825 |
| 4 | 3 | 13.613 |
| 4 | 4 | 12.955 |

## Verdict

**FAIL**

Per-turn dispatcher overhead exceeds the amendment A1 acceptance budget. See *Pivot Recommendation* below.

## Pivot Recommendation

**Pivot Recommendation (per amendment A1):** the per-turn dispatcher overhead exceeds the acceptance budget. v1 MUST NOT proceed with the MH-7 30-turn investigation harness as currently shaped. Two options are pre-authorized in the PRD:

1. **Replan MH-7** with a persistent dispatcher / batched tool calls / shorter turn budget (each carries the build-stall risk surfaced in premortem narrative N2).
2. **Pivot v1 to standalone MCP tools** — ship `claim_blame` and `find_replications` as direct MCP tools without the investigation loop; the deep-search persona defers to v2.

This pivot is pre-authorized; no Phase-4 negotiation under sunk-cost pressure is required. Update `docs/prd/scix_deep_search_v1.md` and open a follow-up PRD for the chosen path.
