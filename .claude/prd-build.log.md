# PRD Build Log

## 2026-04-25 (scix-deep-search-v1)

- **Start**: PRD `docs/prd/scix_deep_search_v1.md` → integration branch `prd-build/scix-deep-search-v1` (created from main @ 8618478e)
- **Decomposition complete** — 9 units across 4 layers (Layer 0: 5, Layer 1: 2, Layer 2: 1, Layer 3: 1). MH-0/1/2/3/4/5/6/7 + flagship/red-team coverage.
- **Layer 0 implement dispatch** — 5 agents launched in parallel (mh0-latency-probe, research-scope-type, intent-backfill-pipeline, v-claim-edges-migration, correction-events-ingest).
- **research-scope-type** — implement SUCCESS (b215005), review PASS (22/22 tests). Landed onto main (also tracked on prd-build/scix-deep-search-v1).
- **mh0-latency-probe** — implement SUCCESS (5fb856d), review PASS (11/11 tests). Landed.
- **v-claim-edges-migration** — implement SUCCESS (d3da316), review FAIL r1 (daily_sync.sh missing --allow-prod), fix r2 SUCCESS+PASS (166ed33). Landed.
- **intent-backfill-pipeline** — implement SUCCESS (c3bbcc0), review PASS (13/13 tests). Landed.
- **correction-events-ingest** — implement SUCCESS (dca7564), review PASS (20/20 tests, gold-200 coverage 100%). Landed.
- **Layer 0 complete (5/5)** → Layer 1 dispatch (claim-blame-and-replications + deep-search-persona-harness in parallel).
- **deep-search-persona-harness** — implement SUCCESS (ebddc19), review PASS (28/28 tests). Landed.
- **Note**: unrelated upstream commit `da85d08 feat(concepts): multi-vocabulary concept_search router (dbl.7)` landed on main mid-session, touching src/scix/mcp_server.py — claim-blame-and-replications agent rebased internally and landed cleanly.
- **claim-blame-and-replications** — implement SUCCESS (c25de6b), review PASS (32/32 tests, EXPECTED_TOOLS=15). Landed.
- **Layer 1 complete (2/2)** → Layer 2 dispatch (citation-grounded-gate).
- **citation-grounded-gate** — implement SUCCESS (a31534f), review PASS (37/37 tests, adversarial false-clean = 0/10). Landed.
- **Layer 2 complete (1/1)** → Layer 3 dispatch (flagship-and-red-team-tests).
- **flagship-and-red-team-tests** — implement SUCCESS (9dd920a), review PASS (4 flagship tests + 50/50 red-team cases scored). Landed.
- **Layer 3 complete (1/1)**. Final test suite: 168 unit + 4 integration = 172 passed, 9 skipped (DB-gated on SCIX_TEST_DSN). Build DONE.

## 2026-04-25 (cross-encoder-reranker-local)

- **Start**: PRD `docs/prd/prd_cross_encoder_reranker_local.md` → integration branch `prd-build/cross-encoder-reranker-local` (running concurrently with vjf option 2 worktree on entity backfill — independent surfaces).
- 16:42 — Decomposition complete: 5 units across 4 layers.
  - Layer 0: bge-reranker-option (M2, small)
  - Layer 1: rerank-ab-eval (M1, medium), cpu-rerank-bench (N1, small) — depend on M2
  - Layer 2: mcp-rerank-default (M3, medium) — depends on M1
  - Layer 3: rerank-rollout-canary (M4+S1, small) — depends on M3; M4 and S1 fused because both touch deploy/README.md.
- **bge-reranker-option** — implement SUCCESS (8339ee2 on main, FF-merged from worktree by harness), review PASS (8/8 tests offline, scope clean — search.py + tests/test_rerank.py only).
- **Layer 0 complete (1/1)** → Layer 1 dispatch (rerank-ab-eval + cpu-rerank-bench in parallel).
- Side-channel: vjf agent (entity backfill) landed independently as 8d53d5d on main; unrelated dbl.3 GLiNER (ff097ef) also landed mid-session — neither overlaps the reranker scope.
- **rerank-ab-eval** — implement SUCCESS on worktree branch (25b4253), cherry-picked to main as 06a6cc3, review PASS (all 6 ACs + methodology). **Winner: hybrid_indus (no rerank).** MiniLM Δ=-0.0453 (p=0.042), bge_large Δ=-0.0556 (p=0.026); neither passes Bonferroni 0.025. **M4 gate FAILS.**
- **cpu-rerank-bench** — first agent abandoned (wrote script ~12KB, never ran/committed, ended on a "wait for monitor" line). Re-dispatched fresh, second agent SUCCESS on worktree (9f4b5b7), cherry-picked to main as a595088, review PASS. MiniLM CPU p95 988.5 ms (NOT suitable per 400 ms threshold); bge-large CPU p95 6466.2 ms (definitively unviable).
- **Layer 1 complete (2/2)** → Layer 2 dispatch (mcp-rerank-default with default_value='off' per M4-FAIL).
- Side-channel: 4b54054 (ops MH-1 + MH-3 substrate rollout) landed on main between cherry-picks; non-overlapping.
- **mcp-rerank-default** — implement SUCCESS (1ea69b7 on main, 15/15 new tests + 80/80 adjacent MCP tests green), review PASS (all 7 ACs). Wired SCIX_RERANK_DEFAULT_MODEL with default='off'; use_rerank=True param; lazy singleton (no model construction when off); top_k<=20 guard with PRD comment; tools/list description carries the M4-FAIL rationale.
- **Layer 2 complete (1/1)** → Layer 3 dispatch (rerank-rollout-canary).
- **rerank-rollout-canary** — implement SUCCESS (bf02490 on main), review PASS (all 6 ACs; canary exits 0; 20-pair fixture committed; deploy/README.md has 'Rerank rollout' + 'Daily canaries' sections).
- **Layer 3 complete (1/1)**. Verification: `pytest tests/test_rerank.py tests/test_mcp_search.py -q` = 23/23 passed; adjacent MCP test files 80/80 still green (verified during M3 review). Build DONE.
- Final landed sequence on main (5 commits): 8339ee2 (bge-reranker-option), 06a6cc3 (rerank-ab-eval), a595088 (cpu-rerank-bench), 1ea69b7 (mcp-rerank-default), bf02490 (rerank-rollout-canary). Integration branch `prd-build/cross-encoder-reranker-local` FF'd to bf02490.
- **Net result**: M1 ablation showed both rerankers REGRESS quality — SCIX_RERANK_DEFAULT_MODEL ships as 'off'. Forward-compatible wiring + canary in place for when a domain-tuned reranker lands.

## 2026-04-18 (community-detection-v2)

- **Start**: PRD `docs/prd/prd_community_detection_v2.md` → integration branch `prd-build/community-detection-v2` (created from main @ f5437179)
- **Decomposition complete** — 7 units across 3 layers (Layer 0: 3, Layer 1: 3, Layer 2: 1). Must-haves M1-M5 + should-haves S1, S2. N1 ablation deferred (nice-to-have, no downstream dep).
- **Layer 0 implement** — M1 (uat-loader-fix), M2 (semantic-communities), M3 (citation-recompute) all SUCCESS.
- **Layer 0 review** — M1 PASS, M3 PASS, M2 FAIL r1 (gitignored artifacts — criteria 5 & 6 required committed sample JSON); M2 r2 SUCCESS+PASS after committing `docs/prd/artifacts/semantic_communities*.sample.json`.
- **Layer 0 land** — All 3 units landed onto `prd-build/community-detection-v2`. 9/9 tests pass (3 uat loader + 4 semantic + 2 citation).
- **Layer 1 implement** — M4 (community-labels), S1 (uat-descendant-search), S2 (coverage-report) all SUCCESS. S1 agent stalled before commit; orchestrator committed the verified diff from the worktree.
- **Layer 1 review** — M4 PASS, S1 PASS, S2 PASS.
- **Layer 1 land** — All 3 units landed. 18/18 tests pass (Layer 0 + Layer 1 combined).
- **Layer 2 implement** — M5 (mcp-community-signals) SUCCESS on first attempt.
- **Layer 2 review** — M5 PASS. 25/25 PRD tests pass. 163 broader MCP tests pass; 15 pre-existing errors in test_mcp_e2e.py (unrelated AttributeError, not a regression).
- **Layer 2 land** — M5 already on integration branch. Final state: 7/7 units landed, zero evictions, single pass.
- **PRD build complete** — `prd-build/community-detection-v2` @ 75956ea ready to merge to main.

## 2026-04-18

- **Start**: PRD `docs/prd/prd_pgvectorscale_migration_benchmark.md` → integration branch `prd-build/pgvectorscale-migration-benchmark`
- **Decomposition complete** — 10 units across 3 layers (Layer 0: 5, Layer 1: 4, Layer 2: 1)
- Layer 0 implement — all 5 agents SUCCESS (env-metadata-capture, pgvectorscale-install-docs, copy-indus-script, hnsw-baseline-runner, streamingdiskann-builder)
- Layer 0 review — all 5 PASS
  - hnsw-baseline-runner: PASS with advisory security note on `--index-name` DDL interpolation (operator-only CLI, deferred cleanup)
- Layer 0 land — all 5 merged to integration; 78/78 layer-0 tests pass
- Layer 1 implement — 3 parallel agents SUCCESS (retrieval-quality-bench, concurrent-stress-bench, cold-start-bench)
- Layer 1 review — all 3 PASS
- Layer 1 land — 100/100 layer-1 tests pass on integration
- Layer 2 implement — filtered-query-bench SUCCESS (51 tests)
- Layer 2 review — PASS
- Layer 2 land — merged; 151/151 layer-1+2 tests pass
- Layer 3 implement — migration-decision-doc SUCCESS (docs/prd/pgvectorscale_migration_decision.md + prd_pgvectorscale_migration_build.md stub)
- Layer 3 review — PASS
- Layer 3 land — merged
- **PRD build complete** — 10/10 units landed on `prd-build/pgvectorscale-migration-benchmark`; 229/229 tests green

## 2026-04-17

- **Start**: PRD `docs/prd/prd_full_text_100pct_coverage.md` → integration branch `prd-build/full-text-100pct-coverage`
- Archived prior `dag.json` → `dag.previous-build.json`
- Decomposition complete — **11 units across 3 layers**
  - Layer 0 (7 units, parallel): mig-046-canonical-bibcode, mig-047-fulltext-failures, mig-048-suppress-and-versions, ads-body-parser, adr-006-addendum, section-schema-contract, suppress-list-config
  - Layer 1 (3 units, parallel): route-tree-module, mcp-tool-contracts-doc, sibling-fallback-read-fulltext
  - Layer 2 (1 unit): read-paper-response-builder
- Scope: code-buildable units from the PRD. Operational units (R6a GPU benchmark, R8 human labeling, R11 data-pull study, R14 pilot batch, R5 Haiku batch spend, R9 batch precompute, Q5 legal memo, R12 agentic eval) are deferred — require hardware/humans/$/lawyers, not code agents.
