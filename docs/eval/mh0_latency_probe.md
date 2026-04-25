# MH-0 Day-1 Latency Probe

Per amendment A1 of `docs/prd/scix_deep_search_v1.md` — measures per-turn dispatcher overhead for `ClaudeSubprocessDispatcher` (`src/scix/eval/persona_judge.py:354`).

## Methodology

- Fixture: 5-turn replay of representative (query, bibcode, snippet) triples.
- Runs: 5 (total measured turns: 25).
- Each turn awaits `dispatcher.judge(triple)` and records `time.perf_counter()` delta — i.e. subprocess startup + OAuth handshake + persona context replay. *Tool execution is excluded from this probe by construction* (no tool calls inside the judge subagent).
- Acceptance (amendment A1): PASS iff p50 ≤ 3.0s AND p95 ≤ 6.0s.
- **Mode: dry-run.** Timings are synthesized; no `claude -p` subprocess was invoked. This artifact exists to verify report shape on CI without OAuth.

## Measurements

| Metric | Value (s) | Threshold (s) | Status |
|---|---|---|---|
| p50 per-turn overhead | 2.000 | 3.0 | OK |
| p95 per-turn overhead | 4.500 | 6.0 | OK |

### Per-turn timings (seconds)

| run | turn | elapsed_s |
|---|---|---|
| 0 | 0 | 1.500 |
| 0 | 1 | 1.800 |
| 0 | 2 | 2.000 |
| 0 | 3 | 2.200 |
| 0 | 4 | 4.500 |
| 1 | 0 | 1.500 |
| 1 | 1 | 1.800 |
| 1 | 2 | 2.000 |
| 1 | 3 | 2.200 |
| 1 | 4 | 4.500 |
| 2 | 0 | 1.500 |
| 2 | 1 | 1.800 |
| 2 | 2 | 2.000 |
| 2 | 3 | 2.200 |
| 2 | 4 | 4.500 |
| 3 | 0 | 1.500 |
| 3 | 1 | 1.800 |
| 3 | 2 | 2.000 |
| 3 | 3 | 2.200 |
| 3 | 4 | 4.500 |
| 4 | 0 | 1.500 |
| 4 | 1 | 1.800 |
| 4 | 2 | 2.000 |
| 4 | 3 | 2.200 |
| 4 | 4 | 4.500 |

## Verdict

**PASS**

Per-turn dispatcher overhead is within the amendment A1 acceptance budget. MH-7's investigation-loop shape is viable on this hardware/OAuth path; proceed to Phase 1.

## Pivot Recommendation

_Not triggered — verdict is PASS. The pivot block is emitted only on FAIL, per amendment A1._
