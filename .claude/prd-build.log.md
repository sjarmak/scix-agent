# PRD Build Log

## 2026-04-18 (community-detection-v2)

- **Start**: PRD `docs/prd/prd_community_detection_v2.md` → integration branch `prd-build/community-detection-v2` (created from main @ f5437179)
- **Decomposition complete** — 7 units across 3 layers (Layer 0: 3, Layer 1: 3, Layer 2: 1). Must-haves M1-M5 + should-haves S1, S2. N1 ablation deferred (nice-to-have, no downstream dep).
- **Layer 0 implement** — M1 (uat-loader-fix), M2 (semantic-communities), M3 (citation-recompute) all SUCCESS.
- **Layer 0 review** — M1 PASS, M3 PASS, M2 FAIL r1 (gitignored artifacts — criteria 5 & 6 required committed sample JSON); M2 r2 SUCCESS+PASS after committing `docs/prd/artifacts/semantic_communities*.sample.json`.
- **Layer 0 land** — All 3 units landed onto `prd-build/community-detection-v2`. 9/9 tests pass (3 uat loader + 4 semantic + 2 citation).
- **Layer 1 implement** — M4 (community-labels), S1 (uat-descendant-search), S2 (coverage-report) all SUCCESS. S1 agent stalled before commit; orchestrator committed the verified diff from the worktree.
- **Layer 1 review** — M4 PASS, S1 PASS, S2 PASS.
- **Layer 1 land** — All 3 units landed. 18/18 tests pass (Layer 0 + Layer 1 combined).

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
