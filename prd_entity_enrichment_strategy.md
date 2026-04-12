# PRD: Entity Enrichment Strategy for SciX — Tiered Linkage Across 32.4M ADS Papers

**Source bead:** `scix_experiments-xz4`
**Date:** 2026-04-12
**Status:** Risk-annotated by `/premortem` 2026-04-12 → Scope-down decision 2026-04-12 (see §Decision). Ready for `/prd-build`.
**Premortem artifact:** `premortem_entity_enrichment_strategy.md`
**Build investigation bead:** `scix_experiments-d35`

---

## Decision (2026-04-12) — Path B (Ambitious, Non-Blocking Validation)

The premortem flagged five P0 mitigations (D1–D5). After review, the build path is **Path B: full ambitious scope with technical mitigations as hard gates and human-validation mitigations demoted to non-blocking parallel tracks.**

### What is hard-required (blocks build)

- **D1 — M11 restructured into M11a/M11b/M11c.** Async bulkhead + persisted JIT cache (`tier=5`) + local NER fallback. See §Requirements M11a/b/c below.
- **D3 — Dependency Hardening.** New §Dependency Hardening section with real per-dependency inventory (populated from live `harvest_runs` data), staging+promotion gates, S4 shadow-diff promoted to blocking CI gate with auto-rollback, dual-vendor LLM path.
- **D5 — Canonical `resolve_entities()` service.** New requirement M13. Single internal function both graph-analytics and retrieval MCP tools call through. Lane choice is internal to the service, not a tool-author decision.
- **Technical ceilings from P1/P2.** Hard per-day dollar cap on M11 JIT; circuit-breaker paging after 2 consecutive trips; MV refresh rate-limited to ≤1/hour via dirty-flag; load-test milestone between M11 ship and default-on; per-entity linkage cap retained on M6.

### What is demoted to Parallel Validation Tracks (non-blocking)

- **D2 — Demand validation via external scientist interviews.** Runs in parallel with the build. If ≥2 of ≥5 interviewed scientists express unprompted demand for non-astronomy concept access, the full build continues as-is. If demand signal comes back negative, M3.5+/M6/M11 are retroactively deprioritized in a follow-on amendment. **This is signal, not a gate.**
- **D4 — External pre-registration of the M4 eval set.** The in-house 50-query set + 20 graph-walk tasks is used for the first M4 run. ADS-librarian and non-astronomy-scientist review of the eval set is a parallel track; if it completes before the M4 gate decision, its feedback is incorporated; if not, the in-house eval runs anyway and the external review feeds into a quarterly re-run.

**Why the demotion:** The project is not under deadline pressure, and human-annotation loops must not be load-bearing for build decisions. D2 and D4 remain valuable as feedback signals, but they are framed as parallel tracks that can amend the PRD retroactively, not as P0 gates.

### What stays unchanged from `/converge`

- Two-lane architecture (static curated core + JIT retrieval enrichment) — per §Convergence Refinements.
- M1, M2, M3, M5, M8, M9, M10, M12 — unchanged.
- M7 deletion (LLM as closed-set disambiguator via M11 JIT, not batch extractor) — unchanged.
- M6 scoped to ~6K curated core, not full 90K — unchanged.
- Advocate B's preserved dissent: if the three-way eval shows the static lane adds no benefit beyond `paper_uat_mappings` + JIT, M6/M12 collapse to JIT-only in a follow-on amendment.

---

## Premortem Risk Annotations (2026-04-12)

Five independent failure-lens agents projected this PRD forward to October 2026 and wrote narratives from a failed state. Every lens returned Critical or High severity with High likelihood on one-shot runs. **The PRD is not ready for build in its current form.** Full details in the premortem artifact; this section is the minimum the build team must absorb before proceeding.

### Top 3 risks

1. **Integration & Dependency collapse (Score 12, Critical/High).** PRD treats 9 external ontology harvesters + Anthropic + Wikidata as stable despite the 2026-04-05 harvester-break incident. A Haiku 4.5 → 4.6 deprecation, an ontology shrinkage, or a Wikidata rate-limit change during the build window breaks M11 JIT and/or invalidates the curated core.
2. **Operational coupling on MCP response path (Score 12, Critical/High).** M11/M8/M10 specified with independent local SLOs. In production the shared asyncio event loop, MV-refresh-under-daily-sync-locks, and the silent circuit breaker couple into a cascade that starves unrelated MCP tools and blocks ADASS camera-ready.
3. **Proxy-based validation failure (Score 12, Critical/High).** Team-authored query_log + team-authored 50-query eval + team-authored 20 graph-walk tasks produce a circular validation that ADASS reviewers will reject: "the benchmark does not establish research scientists want this access pattern."

### Cross-cutting themes

- **Theme A — JIT response-path coupling** surfaces in 3 of 5 lenses (Technical, Operational, Scale) chained on the same physical component (M11 inline on MCP response path with no persistence, no bulkhead, no cost ceiling).
- **Theme B — Self-generated validation** surfaces in 3 of 5 (Scope, Technical, Scale) — the entire defense rests on evaluations the builders author.
- **Theme C — Silent failure modes** surface in 3 of 5 (Operational, Integration, Technical) — breaks that don't page.
- **Theme D — Single-vendor LLM concentration** (Integration, Scale) — deleting M7 in favor of JIT removed the degraded-mode buffer without replacing it.

### P0 mitigations (D1–D5) — status after 2026-04-12 Decision

> **Read §Decision (above) first.** D1, D3, D5 are **hard-required** and implemented in the updated requirements (M11a/M11b/M11c, §Dependency Hardening, M13). D2 and D4 are **demoted to non-blocking Parallel Validation Tracks** — the original text below is preserved unedited for audit trail, but the binding language ("Gate:", "before M6 runs", "must be") is superseded by §Parallel Validation Tracks.

These were originally framed as hard amendments required before `/prd-build`:

**D1. Restructure M11 into M11a/M11b/M11c**

- M11a: Async bulkhead with 400ms hard budget + per-tool executor pool; degrade to static-core filter on timeout/vendor error.
- M11b: Persist JIT results to `document_entities_jit_cache` partition with `tier=5`, TTL 14 days, keyed by `(bibcode, candidate_set_hash, model_version)`. M8 fusion MV includes `tier=5`.
- M11c: Local SciBERT/INDUS-NER inference path (promoted from N3 to must-have) as degraded-mode fallback when Anthropic is unavailable, deprecated, or over budget. Budget GPU time on existing RTX 5090.

> **D2 (DEMOTED per §Decision — non-blocking Parallel Validation Track 1).** Original wording preserved below; the ~~`Gate:`~~ language is superseded and is NOT a build blocker.
>
> D2. Add M0 Demand Validation before M3.5. Interview ≥5 external research scientists with structured think-aloud protocol. Record verbatim whether entity-layer navigation appears in their workflows. ~~**Gate:** if fewer than 2 express unprompted demand for non-astronomy concept access, the PRD scopes down to M1 + M2 + M3 only (schema hardening + bug fix + Tier 1 audit) and defers M3.5+ indefinitely.~~ → See §Parallel Validation Tracks Track 1 for the non-blocking protocol.

**D3. Dependency Hardening section.** New PRD subsection with (a) inventory table per-dependency (SLA, deprecation cadence, fallback path, last-known-good snapshot, cost-of-1-week-outage), (b) all ~~9~~ **live** harvesters write to staging with promotion gate, (c) S4 converted from alert to blocking CI gate with auto-rollback on >2% shrinkage, (d) ~~30-day pre-ADASS freeze window as explicit milestone~~ **(removed per §Decision — no deadline pressure, freeze window was an ADASS-specific artifact)**, (e) dual LLM path (Haiku primary + local NER degraded).

> **D4 (DEMOTED per §Decision — non-blocking Parallel Validation Track 2).** Original wording preserved below; the "Before M6 runs" and "must be … locked" language is superseded and is NOT a build blocker.
>
> D4. External pre-registration of M4 eval set. ~~Before M6 runs: ≥200 externally-logged MCP queries (not team-authored); 50-query set + 20 graph-walk tasks reviewed and locked by ≥1 ADS librarian + ≥1 non-astronomy scientist; documented as immutable for the PRD duration.~~ → See §Parallel Validation Tracks Track 2. The new **M4.5 consistency acceptance** (three-lane Jaccard check) remains hard-required and is specified in §Requirements M4.5 — that part is NOT demoted.

**D5. Single canonical entity resolution service.** Define one internal function `resolve_entities(bibcode, context) -> EntityLinkSet` that both graph-analytics and retrieval MCP tools call. Lane choice (static vs JIT cache vs live JIT vs local NER) is internal to the resolver, not a tool-author decision. Unblocks D1. Implemented as M13. Property-test invariant: same `(bibcode, candidate_set_hash, model_version)` yields identical `EntityLinkSet` regardless of which lane served it (see M13 acceptance).

### P1/P2 mitigations

See premortem artifact §Mitigation Priority List for the full ranked list: hard per-day dollar ceiling on JIT, pager alerts on circuit breaker + watermark staleness, MV refresh rate-limited to 1/hour via dirty flag, quarterly re-run of M4 eval, 10K cap on curated core with ambiguity re-classification on promotion, load-test milestone before default-on flip.

### Preserved Advocate B dissent — handled via retrospective amendment path

The premortem's Scope lens reached Advocate B's conclusion from a different angle: if the demand validation in D2 (now Track 1) returns negative, the entity-linking project may be the wrong shape and M3.5+/M6 could collapse to a JIT-only follow-on. Per §Decision, this is handled as a **non-blocking retrospective amendment** — if Track 1 returns negative, a follow-on PRD amendment re-scopes M3.5+/M6; in-flight build work does not pause. ~~Build leadership must pre-agree to honor D2's gate~~ (superseded — D2 is not a gate).

---

## Convergence Refinements (2026-04-12)

Three advocates debated the diverge output: (A) full tiered pipeline, (B) pivot to JIT query-time extraction, (C) curated 5–10K core + reuse `paper_uat_mappings`. A and C's synthesis offers independently converged on a **two-lane architecture**; B's graph-analytics objection was acknowledged but not fully conceded.

### Resolved consensus

- **Schema hardening is strategy-independent**: M1 (`tier`/`tier_version`/`ambiguity_class`/`link_policy`, composite PK) and M2 (fix `link_entities.py:70` collision-drop bug) ship regardless. Non-negotiable.
- **Embeddings belong at query time, not as a batch tier.** S3 promoted to must-have; no Tier 4 batch run.
- **Tier 3 LLM must be a closed-set disambiguator**, never a free-text extractor (hallucination + shadow-ontology risk).
- **Full 23.3M-abstract × 90K-entity Aho-Corasick sweep is scope overreach.** `paper_uat_mappings` (2.3M rows, 363× `document_entities`) already handles the astronomy concept layer; duplicating it is net-negative. Real delta is in non-astronomy + cross-domain concepts.
- **M4 as originally written is insufficient.** ±1 nDCG@10 over 50 queries is within noise; the original formulation measured a constructed entity-filter boost rather than the intrinsic value of linkage. Refined to a three-way eval.
- **Tier 3 top-by-citation sampling is replaced with stratified `(year, arxiv_class)` quota** to avoid under-representing 2023–2026 papers in the ADASS narrative.
- **Resolved Open Question #2**: `paper_uat_mappings` _does_ already solve the astronomy concept layer. The entity graph's marginal value lives in non-astronomy ontologies (GCMD, SPASE, PhySH, ASCL, PwC) plus cross-domain concepts (missions, instruments, techniques, phenomena).

### Emerged architecture: Two lanes

**Static lane** (for graph analytics): a curated 5–10K "high-query-value" entity core, linked exhaustively at ≥0.97 precision. Used by `citation_chain`, `co_citation_analysis`, community detection, Leiden builds — anywhere extracting on-demand against 299M edges is infeasible. Reuses `paper_uat_mappings` as the astronomy backbone. Backfilled from Wikidata for non-astronomy gaps.

**JIT lane** (for retrieval path): query-time closed-set Haiku disambiguation over top-K retrieved papers. Candidate list derived from the Tier 1 SQL pass + ambiguity-filtered entity names + UAT mappings. Exposed as an `enrich_entities=True` flag on existing hybrid search — no new MCP tool (stays under 13-tool cap). Replaces Tier 3 batch entirely. ~$0.01/query, scales with traffic not corpus.

**Decisive three-way eval (refined M4)**: on the same 50-query set + ≥20 agent graph-walk tasks, same day, compare (a) hybrid baseline, (b) hybrid + static-core filter, (c) hybrid + JIT enrichment. Let data pick which lane matters for which query class. This replaces the original M4's ±1-point-threshold gate.

### Requirement deltas vs original PRD (see §Requirements for originals)

| Req                                 | Change                                                                                                                                                                                                                                                                                                                                                                                                                   | Reason                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| M1, M2, M5, M8, M9, M10             | Keep unchanged                                                                                                                                                                                                                                                                                                                                                                                                           | Infrastructure, strategy-independent                                                                                 |
| M3                                  | Keep unchanged                                                                                                                                                                                                                                                                                                                                                                                                           | Tier 1 SQL pass + 200-sample audit is still the first cheap measurement                                              |
| **M3.5 (new)**                      | Query-log-driven curated core selection, 3 passes (fail-to-resolve query ranking → Wikidata gap backfill → `ambiguity_class=unique` with query evidence). Target ~6K. Promotes N1 from nice-to-have.                                                                                                                                                                                                                     | Replaces scaling-to-90K with precision-on-queried-subset                                                             |
| **M4 (refined)**                    | Three-way eval: baseline vs static-core filter vs JIT enrichment, on 50-query set + ≥20 agent graph-walk tasks. Gate on data, not on a single nDCG threshold. Cross-references M4.5.                                                                                                                                                                                                                                     | Original was theater per B's critique                                                                                |
| **M4.5 (new, D4 residual)**         | Three-lane Jaccard consistency check with Wikidata-backfill structural-delta adjustment. Gate at 5% 90th-percentile divergence. Technical portion of D4 that remains hard-required after D4's human-review portion was demoted to Parallel Validation Track 2.                                                                                                                                                           | Failure Narrative 4 invariant needs a runtime gate, not only the M13 property test                                   |
| M5                                  | Keep unchanged; applied as a filter on the full 90K regardless of which entities enter the curated core                                                                                                                                                                                                                                                                                                                  | Ambiguity classification is cheap and useful for both lanes                                                          |
| **M6 (restricted)**                 | Tier 2 Aho-Corasick runs over the ~6K curated core only, not the full 90K. Wall time ~2h (was ~24h). Per-entity cap retained. Adeft-style per-acronym disambiguation (S2 promoted to M-tier) applied to top-20 ambiguous astro acronyms.                                                                                                                                                                                 | Precision-first scope; reuses UAT for astronomy rather than duplicating                                              |
| **M7 (deleted)**                    | Tier 3 LLM stratified batch is **deleted**. Budget ($1,500) reallocates to M11 infrastructure + M4 eval runs.                                                                                                                                                                                                                                                                                                            | JIT replaces it; closed-set disambiguation at query time is the same shape, cheaper and fresher                      |
| **M11 → M11a/M11b/M11c (D1)**       | Restructured per §Decision: M11a async bulkhead (400ms hard budget, per-tool executor pool, degrade on timeout/vendor); M11b persisted `document_entities_jit_cache` with `tier=5` and TTL 14d; M11c local SciBERT/INDUS-NER fallback on RTX 5090 (promoted from N3). No new MCP tool; response-path coupling broken by the bulkhead; JIT lane writes back so M8 fusion sees `tier=5`.                                   | D1 hard mitigation — moves JIT off the shared event loop and off single-vendor LLM concentration                     |
| **M12 (new)**                       | Dual-lane routing: graph-analytics MCP tools consume static curated core; retrieval tools consume JIT enrichment on-demand. Contract documented in MCP tool schemas and enforced through M13.                                                                                                                                                                                                                            | Explicit architectural boundary so neither lane silently takes over the other                                        |
| **M13 (new, D5)**                   | Single canonical `resolve_entities(bibcode, context) -> EntityLinkSet` service. Sole write path for `document_entities` + `document_entities_jit_cache`; sole read path for `document_entities_canonical`. Enforced at the Python type level via `@dataclass(frozen=True)` + `_ResolverToken` sentinel and AST-level CI lint. Property test asserts lane-invariance. Per-lane p95 latency budgets baked into acceptance. | D5 hard mitigation — Failure Narrative 4 "two-truths" collapse prevented by a single resolver with a typed invariant |
| S1                                  | Keep unchanged (citation-consistency precision proxy is free and serves both lanes)                                                                                                                                                                                                                                                                                                                                      | -                                                                                                                    |
| **S2 → promoted to M6-integrated**  | Per-acronym Adeft classifiers for top-20 ambiguous astro acronyms are no longer optional; they run as post-filter to M6 Tier 2 on the curated core                                                                                                                                                                                                                                                                       | Single highest-leverage precision intervention per prior art                                                         |
| **S3 → promoted to M11-integrated** | Query-time entity expansion via NN on entity-description embeddings is the embedding path; part of M11 JIT lane                                                                                                                                                                                                                                                                                                          | Correct role for embeddings in entity work                                                                           |
| S4                                  | Keep (ontology version pinning + shadow-diff)                                                                                                                                                                                                                                                                                                                                                                            | Harvester regression already bit us once                                                                             |
| S5                                  | Keep (researcher feedback loop)                                                                                                                                                                                                                                                                                                                                                                                          | Needed for trust in either lane                                                                                      |
| **N1 → promoted to M3.5**           | Query-log ranking drives curated core selection                                                                                                                                                                                                                                                                                                                                                                          | Supply-side vs demand-side ontology bias must be corrected at selection time, not post-hoc                           |
| **N2 → promoted to should-have**    | Wikidata backfill for non-astronomy gaps identified by M3.5                                                                                                                                                                                                                                                                                                                                                              | Explicit plan to close the 58% LLM-pilot-unresolved gap                                                              |
| **N3 → promoted to M11c (D1c)**     | Local SciBERT/INDUS-NER promoted from benchmark-only to must-have fallback. Routing uses a continuous 5% canary share and a human-ack primary/fallback swap, NOT an automatic F1-gap flip.                                                                                                                                                                                                                               | Eliminates single-vendor LLM concentration; canary prevents degraded path from rotting                               |

### Preserved dissent

Advocate B remains unconvinced that any static layer is warranted — their position is that the static curated core should be deleted and JIT should serve all paths including graph analytics. Advocate C's counter (JIT cannot scale to 299M-edge Leiden builds) is compelling but unmeasured. **If the M4 three-way eval shows the static lane adds no graph-analytics benefit beyond `paper_uat_mappings` + JIT, B's position wins ex post** and M6/M12 collapse to JIT-only. The PRD commits to honoring that outcome.

### ADASS narrative refinement

Old framing: "We linked 32M papers to 90K canonical entities from 9 ontologies." Weak — coverage number without measured value.

New framing: "We show that a ~6K high-query-value curated entity core, linked at ≥0.97 precision, produces a measurable retrieval lift on held-out queries and enables graph-analytics queries over a non-astronomy concept axis that UAT cannot express. Tail queries are served by query-time JIT disambiguation at $0.01/query. We document the dual-lane architecture as a reference pattern for agent-navigable scientific corpora." Defensible under reviewer questioning.

---

## Problem Statement

The SciX corpus has 32.4M papers and 299M citation edges, but only 0.012% (3,852 papers, 6,373 edges) are linked to any canonical entity in the entity graph — despite 90,287 entities + 10,567 aliases + 80,947 identifiers from 7 ontologies (GCMD, SPASE, SSODNet, VizieR, ASCL, PhySH, PwC, AAS, ADS) being loaded. The existing `link_entities.py` pipeline has never run at scale; the current edges are a pilot only.

Two concurrent realities shape the design:

1. **Entity linkage is not yet on any retrieval code path.** Hybrid search (SPECTER2 + text-3-large/INDUS + BM25 via RRF) works today on 32M papers without touching `document_entities`. `paper_uat_mappings` already contains ~2.3M paper→concept edges for astronomy and is the de-facto concept layer. Any investment in scaled entity linking must justify a measurable downstream retrieval or agent-task improvement, not just a coverage number.
2. **If linkage is built, precision is non-negotiable.** Research scientists consume the graph via MCP tools; false links poison graph queries and erode trust. Prior art (PubTator 3, BERN2, SciSpacy, BLINK/ReFinED) is uniformly hybrid (dictionary + specialized normalizers + disambiguator), and short-name / homonym entities (Mercury, JET, HST, HD, NGC, R, AI) dominate the failure modes — an estimated 6–8K of the 90K entities are "dangerous" without a disambiguation layer.

This PRD defines a tiered strategy that (a) starts with Tier 1 keyword matching at essentially zero cost, (b) gates every subsequent tier on measured downstream value, (c) commits to a schema that supports per-tier isolation and rollback from day one, and (d) commits to an evaluation harness that measures precision against silver labels + a 500-example human audit + downstream retrieval nDCG delta.

---

## Goals & Non-Goals

### Goals

- Reach defensible coverage and precision on the SciX entity graph before ADASS 2026 submission, with per-tier accountability.
- Make the entity graph **measurably improve** a downstream metric (retrieval nDCG@10, agent task success, or MCP query relevance) on a held-out eval set.
- Fix the existing silent-collision bug in the caching resolver (`link_entities.py:70`) and schema gaps (`document_entities` missing `tier`, `tier_version`, `ambiguity_class`) before any backfill.
- Ship an incremental linker that stays inside the daily-sync budget (<5 min per 1K new papers).
- Produce eval numbers (precision per tier, recall on silver set, downstream retrieval delta) with explicit in-house-authored disclaimer until §Parallel Validation Tracks Track 2 (external eval pre-registration) lands. External review feeds into quarterly re-runs and can retroactively amend claims; build decisions run on the in-house eval.

### Non-Goals

- Replacing hybrid search with an entity-centric retrieval path. Entity linkage is a **supplementary** graph layer, not a primary retrieval mechanism.
- Building a ground-truth eval set from scratch. We combine silver labels, adversarial sampling, and a small human audit (n ≤ 500) calibrated with LLM-as-judge.
- Running LLM extraction over the full 32M corpus. Tier 3 is bounded to a stratified sample with explicit budget cap.
- Adding new MCP tools that push the total past 13 (the already-PRDed consolidation target). Entity access stays inside existing hybrid-search filters.
- Deprecating `paper_uat_mappings`. It remains the concept layer for astronomy; the entity graph complements it.

---

## Requirements

### Must-Have

**M1. Schema hardening before any backfill.**

- Requirement: Add `tier SMALLINT NOT NULL`, `tier_version INT NOT NULL DEFAULT 1`, and `entities.ambiguity_class` enum (`unique | domain_safe | homograph | banned`) and `entities.link_policy` enum (`open | context_required | llm_only | banned`). Change `document_entities` PK to `(bibcode, entity_id, link_type, tier)`.
  - **Acceptance:** Migration 026 applied; `\d document_entities` shows new columns and updated PK; `INSERT` conflict test writes two rows with different `tier` values without collision; `DELETE WHERE tier = 2` removes exactly one tier's rows.

**M2. Fix the silent collision-drop bug in caching resolver.**

- Requirement: `src/scix/link_entities.py:70` currently does `if key and key not in cache`, silently letting the first ontology own every ambiguous name. Replace with a collision-aware cache that stores all candidates and surfaces ambiguity to the disambiguator.
  - **Acceptance:** A unit test with two entities sharing a lowercase canonical name asserts the resolver returns both candidates, not one; `git log` shows the fix in a separate commit before any backfill runs.

**M3. Tier 1 — keyword → entity exact match, executed to completion with a precision audit.**

- Requirement: Run a single SQL pass joining `papers.keywords` against `entities.canonical_name` + `entity_aliases.alias` into `document_entities` with `tier=1`, `link_type='keyword_match'`, `confidence=1.0`. Then sample 200 of the resulting links stratified across ontologies and arxiv_class; manually label.
  - **Acceptance:** At least 3M new `tier=1` rows written; audit report at `build-artifacts/tier1_audit.md` with precision ≥ 0.95 on the 200-sample set (Wilson 95% CI ±3%). If precision < 0.95, the Tier 1 dictionary is dirty and Tier 2 is blocked until fixed.

**M3.5. Curated high-query-value entity core (formerly N1, promoted via /converge; addresses missing `query_log` pre-requisite).**

- Sub-requirement **M3.5.0 — `query_log` table bootstrap (pre-requisite).** The `query_log` table does not exist in the production DB as of 2026-04-12. Before M3.5 can rank entities by query volume, this table must be created and populated. Two-step process:
  1. Create `query_log(id BIGSERIAL PK, ts TIMESTAMPTZ, tool TEXT, query TEXT, result_count INT, session_id TEXT NULL, is_test BOOL NOT NULL DEFAULT false)` via migration 029. The `is_test` flag lets the curation pass exclude team test harnesses from ranking.
  2. Backfill from existing MCP tool-call structured logs (present since the MCP consolidation merge `f334961` on 2026-04-11). If structured logs are incomplete, the migration also wires a PostToolUse logging hook into the MCP server so that all future tool calls land in `query_log` going forward. M3.5 can proceed on partial data once at least **14 days of non-test-harness queries** have accumulated (configurable `min_days_of_traffic` with default 14).
  - **Acceptance:** Migration 029 applied; `SELECT count(*) FROM query_log WHERE NOT is_test AND ts >= now() - interval '14 days'` returns ≥ 500 rows before M3.5 core-selection runs; M11a load test (which depends on `query_log`) is also unblocked by this step.

- Sub-requirement **M3.5.1 — Core selection.** Three-pass ranking: (pass 1) entities with the highest count of zero-result queries from `query_log WHERE NOT is_test` are flagged as "gap" candidates; (pass 2) for every gap candidate without an existing `entities` row, Wikidata backfill adds the entity via N2 (cross-domain concept harvest); (pass 3) entities with `ambiguity_class='unique'` and ≥1 `query_log` hit are accepted into the curated core. Target size ≤ 10,000 (hard cap, not soft target — see §Dependency Hardening §LLM Cost Ceilings for the rationale on capping core size).
  - **Acceptance:** `build-artifacts/curated_core.csv` with exactly one row per accepted entity and columns `(entity_id, canonical_name, source, ambiguity_class, pass_triggered, query_hits_14d, zero_result_hits_14d)`; total row count ≤ 10,000; stratification across `source` (at least 6 of 7 live harvesters represented) reported in `build-artifacts/curated_core_stratification.md`.

- Sub-requirement **M3.5.2 — Promotion / demotion lifecycle.** When S5 feedback adds an entity to the core, M5 (ambiguity classification) is re-run on the new entity before it is accepted. Core size cannot exceed 10,000; the lowest-query-hit entity is demoted when a new one is promoted.
  - **Acceptance:** `core_promotion_log` table populated on every change; unit test asserts a core over 10K triggers a demotion.

**M4. Downstream value gate before Tier 2 build-out.**

- Requirement: Run the existing 50-query retrieval eval (`scripts/eval_retrieval_50q.py`) in two configurations: (a) current hybrid search, (b) hybrid search with an entity-filter boost sourced from the new Tier 1 rows. Report nDCG@10, Recall@20, MRR deltas.
  - **Acceptance:** Delta report at `build-artifacts/tier1_retrieval_delta.md`. If delta < 1 nDCG point on ≥3 of 50 queries, Tier 2-4 are deferred and the PRD is re-scoped toward JIT extraction (see Design Considerations §C1).

**M5. Ambiguity class annotation for all 90K entities.**

- Requirement: Populate `entities.ambiguity_class` offline: `banned` for any name/alias that matches the top-20K English words at WordFreq Zipf ≥ 3.0 or is ≤ 2 characters; `homograph` for any name that collides with another entity's canonical or alias across the full graph; `domain_safe` for entities with ≥ 6-character names unique in the graph and a single ontology of origin; `unique` otherwise.
  - **Acceptance:** `SELECT ambiguity_class, count(*) FROM entities GROUP BY 1` returns numbers; report at `build-artifacts/ambiguity_audit.md` with random 50-example spot-check per class.

**M6. Tier 2 — Aho-Corasick over abstracts, gated and ambiguity-aware.**

- Requirement: If M4 passes, build a `pyahocorasick` automaton over entities with `ambiguity_class ∈ {unique, domain_safe}` only. Additionally include `homograph` entities but only fire when the paper's abstract also contains a disambiguating long-form alias from the same entity (e.g. "HST" fires only when "Hubble Space Telescope" is also present). Run over 23.3M abstracts in parallel (shared automaton via pickle + fork). Write with `tier=2`, `link_type='ac_dictionary'`, confidence = calibrated logistic on 7-feature context vector.
  - **Acceptance:** End-to-end wall time ≤ 24h on project hardware; total new rows reported; precision ≥ 0.90 on a 500-example adversarial sample focused on short-name/homograph surface forms; per-entity linkage cap enforced (`≤ log₂(32M) × 1000 = 25K` papers per entity; entities exceeding the cap auto-demoted to `link_policy='llm_only'`).

**M7. ~~Tier 3 — LLM as closed-set disambiguator~~ (DELETED per /converge — see §Convergence Refinements requirement deltas).**

- Status: **DELETED.** The batch-LLM Tier 3 path is replaced by M11a/M11b/M11c (query-time JIT disambiguation with persisted cache and local NER fallback). The original $1,500 batch budget reallocates to M11 infrastructure + M4 eval runs per the deltas table. This entry is retained as a placeholder so Must-Have numbering stays stable for existing cross-references; do not implement M7 as written.

**M8. Fusion materialized view for MCP queries.**

- Requirement: `document_entities_canonical` MV computes fused confidence via noisy-OR: `fused = 1 - exp(SUM(ln(1 - c_t × w_t)))` with calibrated `tier_weight(tier)` IMMUTABLE SQL function. Initial placeholder weights (replaced by M9 calibration output once the 500-sample audit labels a per-tier sample): Tier 1: 0.98, Tier 2: 0.85, Tier 3: 0.92 (deprecated slot), Tier 4: 0.50, **Tier 5 (JIT): 0.88** (placeholder until M9 calibration lands). After M9 completes, `tier_weight()` is recomputed from the empirical per-tier precision and the MV is refreshed once with the corrected weights. Indexes on `(entity_id, fused_confidence DESC)` and `(bibcode)` — and a **UNIQUE index on `(bibcode, entity_id)`** required for `REFRESH MATERIALIZED VIEW CONCURRENTLY`. Refresh is performed exclusively via `REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical` (never a plain REFRESH) so that reads (citation_chain, graph-analytics tools) are not blocked by the refresh.
  - **Acceptance:** MV exists; unique index on `(bibcode, entity_id)` exists; sample queries `SELECT * FROM document_entities_canonical WHERE entity_id = X ORDER BY fused_confidence DESC LIMIT 20` return in <100ms on a row with ≥10K links; refresh time < 15 min (concurrent refresh is slower than plain refresh — acceptable tradeoff); a coupled-load test ("daily sync + `REFRESH ... CONCURRENTLY` + JIT canary + 1h query_log replay running simultaneously") shows citation_chain p95 stays within 15% of baseline throughout the refresh window; `tier_weight()` function has a history row in a `tier_weight_calibration_log` table recording each calibration version and source (placeholder vs M9-output).

**M9. Evaluation harness: 500-example human audit + silver set + LLM-as-judge.**

- Requirement: Create `entity_link_audits` table keyed by `(tier, bibcode, entity_id, annotator, label, note)`. Annotate 125 stratified examples per tier (500 total) at ~45 sec/example. Compute Wilson 95% CIs per tier. Run Claude Opus (different family than Tier 3 extractor) as LLM-judge over 10K random links; calibrate against the 500-human set (Cohen's κ ≥ 0.6 required before trusting judge).
  - **Acceptance:** `entity_link_audits` populated; per-tier precision with Wilson CI published in `build-artifacts/eval_report.md`; LLM-judge calibration report showing κ; downstream retrieval nDCG delta from M4 re-run with all tiers.

**M10. Incremental daily-sync integration with circuit breaker.**

- Requirement: `scripts/link_incremental.py` runs Tier 1 + Tier 2 inline on new papers from the daily sync watermark. Pre-built automaton loaded from `data/entities/ac_automaton.pkl`. Circuit breaker: if linkage exceeds 5 min budget, the day's sync completes without entity links and schedules a catch-up job. A single-row watermark table `link_runs(run_id, max_entry_date, timestamp, rows_linked)` tracks progress. Circuit breaker trips are **paged** after 2 consecutive occurrences; watermark-staleness alert fires when `now() - max_entry_date > 24h`.
  - **Acceptance:** 1K-paper incremental run completes in <60s; a forced 10-min delay scenario trips the circuit breaker and sync completes; watermark table advances; catch-up job runs successfully; pager alert fires on the second consecutive trip; watermark-staleness alert fires under a forced stale watermark.

**M11a. JIT async bulkhead on retrieval path (D1a).**

- Requirement: When `hybrid_search(enrich_entities=True)` fires, JIT disambiguation runs inside a per-tool `asyncio.Semaphore` bulkhead with a **400ms hard wall budget** and a dedicated executor pool isolated from other MCP tools' event loop work. On timeout, vendor error, or bulkhead exhaustion, the call **degrades to static-core filter** (the curated 5–10K entity set from M6) and returns successfully with a structured `degraded_mode: true` field in the response envelope. The bulkhead budget is configurable per-tool and defaults to 4 concurrent JIT calls per MCP worker process.
  - **Acceptance:** Load test replays 24h of query_log with JIT enabled against a fault-injected Anthropic endpoint (synthetic 2.5s p99 latency + 5% error rate); MCP response-path p95 for unrelated tools (`citation_chain`, `vector_search`, `bm25_search`) stays within 10% of baseline; `degraded_mode=true` rate under fault injection is measured and reported; zero event-loop starvation events observed in `asyncio.debug` traces.

**M11b. JIT result persistence with `tier=5` cache partition (D1b).**

- Requirement: **Only successful JIT or M11c local-NER disambiguation results** write back to a partitioned table `document_entities_jit_cache` with `tier=5`, keyed by `(bibcode, candidate_set_hash, model_version)` and TTL 14 days (cleaned by a daily cron). **Bulkhead-degraded static-core responses are NOT written to the cache** — they pass through unchanged so the M8 fusion MV does not double-count static-core entities under `tier=5`. On a cache hit the JIT layer skips the Anthropic call entirely. The M8 fusion MV **includes tier=5** with `tier_weight(5)` supplied by the M9 calibration run (the 0.88 initial placeholder is replaced by the empirical precision measurement once M9's 500-sample audit labels tier=5 rows; the placeholder is only used between M11 ship and M9 completion). JIT writes are async and never block the MCP response (fire-and-forget via a bounded queue of depth 1024). **Queue drops are wired to M10's alert rail**: 2 consecutive `jit_cache_write_queue_drops` events page the on-call (same mechanism as the M10 circuit-breaker trip alert).
  - **Acceptance:** Cache hit rate ≥ 30% measured on the replayed 24h query_log after 7 days of warmup (depends on M3.5.0 query_log bootstrap); `document_entities_canonical` MV returns **numerically consistent** results for any `(bibcode, entity_id)` present in both static and JIT lanes (defined as: `|fused_confidence_static - fused_confidence_with_jit| ≤ 0.01` under identical tier inputs); TTL cleanup job removes rows older than 14 days; fire-and-forget queue drops under backpressure and fires an alert after 2 consecutive drops; unit test asserts that a bulkhead-degraded response does NOT produce a cache write.

**M11c. Local SciBERT/INDUS-NER fallback + continuous-exercise canary (D1c).**

- Requirement: Promote N3 (local NER inference) to must-have. Deploy SciBERT-NER and INDUS-NER as local inference services on the RTX 5090 (batch mode for M6 + single-query mode for M11 fallback). **Routing policy (fixed topology, not F1-gap-driven):**
  1. **Haiku remains primary** for JIT disambiguation calls. This is the steady-state topology regardless of benchmark outcome.
  2. **Local NER receives a continuous canary share** of 5% of JIT traffic (configurable, default 5%) _regardless_ of the Haiku-vs-local F1 gap. This guarantees the degraded path is exercised under production load at all times — Failure Narrative 1's "degraded path rots because it is never invoked" mode is prevented by the canary, not by a benchmark flip.
  3. **Local NER handles all overflow**: when M11a's bulkhead degrades (timeout / vendor error / concurrency saturation) or when the per-day JIT dollar cap (§Dependency Hardening §LLM Cost Ceilings) is exceeded, the call routes to local NER instead of the static-core fallback. Static-core fallback is the last-line mode only when both Haiku and local NER are unavailable.
  4. **Primary/fallback swap is a human-ack decision, not automatic.** The benchmark report below is an _input_ to a human decision to reverse the primary/fallback assignment (e.g., if local NER proves uniformly superior), not a trigger. A swap is a config-flag change with an explicit bead note and PR review.
- Benchmark requirement: Run both local models against Haiku 4.5 on the 500-example human audit set (from M9). Report F1 per `(entity_type × year × arxiv_class)` stratum with confusion matrix.
  - **Acceptance:** Local NER single-query p95 latency ≤ 250ms on RTX 5090; benchmark report at `build-artifacts/local_ner_vs_llm.md` with F1 per stratum + confusion matrix; routing config flag wired with values `{haiku_primary, local_primary}` and a **5% canary share enforced independently** of which value is set; canary traffic share verified by a week of production logs showing ≥ 4.5% and ≤ 5.5% actual local-NER calls; GPU peak utilization ≤ 60% during a coupled-load test (JIT canary + embedding pipeline running concurrently, measured via `nvidia-smi` sampled every 10s); unit test asserts that under a forced Haiku outage, routing falls back to local NER (not to static-core), and only falls back to static-core when local NER is also force-failed.

**M12. Dual-lane routing contract (explicit acceptance).**

- Requirement: Graph-analytics MCP tools (`citation_chain`, `co_citation_analysis`, `bibliographic_coupling`, `community_detection`) consume the **static curated core** (tier 1/2/M3.5) via `document_entities_canonical`. Retrieval MCP tools (`hybrid_search[enrich_entities=True]`, `vector_search`, `bm25_search`) consume **JIT enrichment** via the M13 resolver. The contract is enforced through M13 and validated by an **M4.5 consistency acceptance** (§Requirements M4.5 below): for the same 50 bibcodes, `citation_chain(bib)`, `hybrid_search[enrich_entities=True]`, and `SELECT FROM document_entities_canonical` return the same entity set modulo a declared lane-delta report; fails the gate if divergence > 5%.
  - **Acceptance:** MCP tool schemas document which lane they consume; M13 is the only code path that writes or reads `document_entities_canonical`; M4.5 report at `build-artifacts/lane_consistency.md`.

**M13. Canonical `resolve_entities()` service (D5).**

- Requirement: Define one internal Python service `src/scix/resolve_entities.py` exposing a single function:

  ```python
  def resolve_entities(
      bibcode: str,
      context: EntityResolveContext,  # includes candidate_set, mode, ttl_max, budget_remaining
  ) -> EntityLinkSet:
      ...
  ```

  Both graph-analytics and retrieval MCP tools call this service. Lane choice (static curated core vs `document_entities_jit_cache` hit vs live JIT call vs M11c local NER) is internal to the resolver. All entity writes to `document_entities` and `document_entities_jit_cache` go through this service; all reads go through `document_entities_canonical` via this service. **No MCP tool constructs `EntityLinkSet` directly** — enforced at the Python type level, not via grep: `EntityLinkSet` is a `@dataclass(frozen=True)` whose `__init__` takes a module-private `_ResolverToken` sentinel argument. Only `resolve_entities.py` imports and constructs the sentinel; any caller that attempts to build an `EntityLinkSet` outside the module gets a `TypeError` at import time. This converts the "single entry point" contract from a lint convention into a type-system guarantee.

- Per-lane latency budgets (built into the service and asserted in benchmarks):
  - **Static lane** (Postgres read on `document_entities_canonical` via an index hit): p95 ≤ 5ms.
  - **`document_entities_jit_cache` hit** (partitioned table read): p95 ≤ 25ms.
  - **Live JIT call** (Haiku through the M11a bulkhead + write-back to M11b cache): p95 ≤ 450ms (400ms bulkhead + 50ms overhead).
  - **M11c local NER** (single-query RTX 5090 inference): p95 ≤ 275ms (250ms inference + 25ms overhead).
- **Property test (Failure Narrative 4 invariant)**: for a fixed `(bibcode, candidate_set_hash, model_version)` input, `resolve_entities()` returns an `EntityLinkSet` that is **set-equal** across all four lanes (static-hit, JIT-cache-hit, live-JIT, local-NER) — the internal lane choice is invisible to the caller. Differences in _confidence values_ are allowed within ±0.01 per entity; the _set of entities returned_ must match exactly. Tested as a hypothesis-style property test with ≥ 100 randomized inputs.
  - **Acceptance:**
    1. `EntityLinkSet` is `@dataclass(frozen=True)` with a module-private `_ResolverToken` sentinel; a unit test in a separate test module imports `EntityLinkSet` and asserts that `EntityLinkSet(...)` raises `TypeError` without the sentinel.
    2. AST-level lint (libcst walker, wired to CI) scans `src/` for any write targeting `document_entities` or `document_entities_jit_cache`, or any read from `document_entities_canonical`, and fails the build if any such access originates outside `src/scix/resolve_entities.py`. Grep is retained as a last-line pre-commit check but is not the enforcement mechanism.
    3. Unit tests cover all four internal lanes with mock backends.
    4. Integration test calls `resolve_entities()` from a graph-analytics tool and a retrieval tool and asserts lane choice matches mode.
    5. Per-lane latency budgets above are measured in a benchmark suite (`tests/bench_resolve_entities.py`) and published to `build-artifacts/m13_latency.md`; regression > 10% over baseline fails CI.
    6. Property test passes with ≥ 100 randomized inputs across all four lanes (hypothesis).
    7. M13 is the sole entry point documented in §Dependency Hardening §Internal Contracts.

**M4.5. Three-lane consistency acceptance (extends M4, retains technical validity regardless of D4 demotion).**

- Requirement: As part of the M4 three-way eval, add a consistency check: for the same 50 bibcodes drawn from the eval set, compare the entity sets returned by `citation_chain(bib)`, `hybrid_search(bib, enrich_entities=True)`, and `SELECT entity_id FROM document_entities_canonical WHERE bibcode=$1`. Compute Jaccard per bibcode on the **entity-id set** (not confidence-thresholded — every entity returned by any lane is included). Aggregate distribution.
- **Lane-structural delta declaration.** Before the gate runs, compute a `lane_delta_set` per bibcode: entities present in the static lane via Wikidata backfill (N2) that are structurally unreachable from JIT's candidate derivation (Tier 1 near-misses + UAT). Entities in `lane_delta_set` are subtracted from both numerator and denominator of the Jaccard computation. The `lane_delta_set` is itself published in `build-artifacts/m45_lane_delta.md` with one row per excluded entity and the reason.
- Divergence threshold: **5% at the 90th percentile** of the adjusted per-bibcode Jaccard. Gate failure triggers a bug hunt on M13 (not an architectural retreat). The 5% budget is for genuine lane disagreements, not for Wikidata-backfill structural differences.
  - **Acceptance:** Report at `build-artifacts/m45_consistency.md` with per-bibcode Jaccard (raw and adjusted), aggregate distribution, divergence breakdown by lane pair, and root-cause analysis of any >5% adjusted divergences; `build-artifacts/m45_lane_delta.md` exists with one row per structurally-unreachable entity; a failing run attributes the failure to a specific M13 lane and references the M13 property test.

### Should-Have

**S1. Cross-citation consistency precision proxy.**

- Requirement: For each `(bibcode, entity_id)` pair, compute a network-locality signal: fraction of the paper's outbound citations that also link to the same entity. Store as `document_entities.citation_consistency`. Use as a free precision proxy — pairs with ≥2 citing neighbors linked to the same entity are almost certainly correct; pairs with 0 are suspect.
  - **Acceptance:** Column populated for all tiers; distribution reported; precision on high-consistency subset measured vs random sample.

**S2. Per-acronym Adeft-style disambiguator for top-20 ambiguous astro acronyms.**

- Requirement: For HST, JET, HD, NGC, IC, M31, R, CORE, MAGIC, SAGE, ATLAS, CMS, IRAS, WISE, ROSAT, SOFIA, GAIA, ALMA, MARS, AI: train a char-n-gram TF-IDF + logistic regression per acronym on co-occurring terms from unambiguous long-form papers as positives. Integrate as a post-filter to Tier 2 Aho-Corasick.
  - **Acceptance:** ≥ 90% accuracy on 50 held-out labeled examples per acronym; integration gates drop-in replacement of plain AC matches.

**S3. Query-time entity expansion (not batch).**

- Requirement: Add an MCP-internal helper (not a new tool) that maps a user query string to a set of entity IDs via nearest-neighbor on entity-description embeddings (90K vectors, indexed once). Hybrid search's entity filter uses this as an optional boost. This is the **correct** role for embeddings in entity work; it is not a 4th tier.
  - **Acceptance:** Given 10 sample queries from `query_log`, the top-5 expanded entities are manually rated relevant ≥ 70% of the time; latency ≤ 20ms per query.

**S4. Ontology version pinning + shadow-diff on re-harvest.**

- Requirement: Each entity row records its source ontology version. Before accepting a new harvest, compute diffs (alias count, canonical set, identifier set) against the previous version; alert on >2% shrinkage. Add `entities.supersedes_id` self-FK so renamed entities transparently redirect.
  - **Acceptance:** Migration 027; harvester re-run dry-mode prints diff; nightly integrity check fails on orphaned `document_entities.entity_id` post-supersede.

**S5. Feedback loop for researcher-reported bad links.**

- Requirement: MCP responses that include entity links expose a `report_incorrect_link(bibcode, entity_id, reason)` affordance. Reports land in `entity_link_disputes` and are surfaced in the weekly eval report.
  - **Acceptance:** Table exists; MCP tool response schema includes the affordance; weekly report shows dispute rate per tier.

### Nice-to-Have

**N1. Curated "high-query-value" entity subset derived from `query_log`.**

- Requirement: Rank entities by actual MCP query volume; produce a curated 5–10K "high-query-value" set that Tier 2/3 prioritize. Identify ontology gaps (missions, instruments, techniques, phenomena under-represented in the 90K).
  - **Acceptance:** Ranking written to `build-artifacts/entity_query_value.csv`; gap report lists top-100 queried terms with no matching entity.

**N2. Ontology expansion from Wikidata Q-items for cross-domain concepts.**

- Requirement: Harvest a curated Wikidata subset (missions, spacecraft, instruments, techniques, phenomena) and merge via `entity_clusters` table so cross-ontology duplicates (Mars in SSODNet + GCMD + Wikidata) share a cluster ID.
  - **Acceptance:** Cluster count < entity count; resolver returns clusters not raw entities.

**N3. SciBERT/INDUS-NER local inference path as cost-free alternative to Tier 3.**

- Requirement: Benchmark SciBERT-NER and INDUS-NER against Claude Haiku on the 500-example human audit set. If F1 gap ≤ 0.05, local NER replaces Tier 3 Haiku batches for future runs.
  - **Acceptance:** Benchmark report at `build-artifacts/local_ner_vs_llm.md`.

---

## Dependency Hardening (D3)

This section is a hard-required deliverable. It replaces the ambient assumption that 9 ontology harvesters + Anthropic Haiku + Wikidata are stable substrates. The live DB audit (2026-04-12) shows the real risk surface is **7 live harvesters with ≥1 entity + 3 broken + 3 never-run** — not the nominal 9. Every entry below reflects **live DB state as of 2026-04-12**, not aspirational state.

### Per-dependency inventory (D3a)

| Dependency                                | Live entities          | Last successful run | Status                                                                         | Fallback path                                                                   | Last-known-good snapshot                  | Cost of 1-week outage                                  |
| ----------------------------------------- | ---------------------- | ------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| **VizieR**                                | 62,388                 | 2026-04-05          | stale 7d; populated                                                            | Frozen snapshot in `data/entities/snapshots/vizier-20260405.jsonl` (to produce) | Snapshot TBD                              | Low — snapshot serves reads; re-harvest blocked 1 week |
| **GCMD**                                  | 10,481                 | 2026-04-05          | stale 7d; **zombie run in `running` state**                                    | Kill zombie; snapshot; re-harvest on staging                                    | Snapshot TBD                              | Medium — NASA ESO concepts, non-astronomy delta        |
| **PwC (Papers With Code)**                | 8,635                  | 2026-04-11          | **fresh** ✓                                                                    | Snapshot required per S4                                                        | Snapshot TBD                              | Medium — ML methods coverage                           |
| **ASCL**                                  | 3,958                  | 2026-04-11          | **fresh** ✓ (post 2026-04-05 parser fix)                                       | Snapshot required per S4                                                        | Snapshot TBD                              | Low — software catalog                                 |
| **PhySH**                                 | 3,891                  | 2026-04-11          | **fresh** ✓ (post 2026-04-05 parser fix)                                       | Snapshot required per S4                                                        | Snapshot TBD                              | Low — physics subject headings                         |
| **AAS (keywords)**                        | 680                    | 2026-04-05          | stale 7d                                                                       | Local snapshot                                                                  | Stable                                    | Low                                                    |
| **SPASE**                                 | 229                    | 2026-04-05          | stale 7d; **zombie run in `running` state**                                    | Kill zombie; re-harvest on staging                                              | Snapshot TBD                              | Low — small footprint                                  |
| **SsODNet**                               | 20                     | 2026-04-05          | **effectively broken** — only 20 entities vs expected ~10⁵ solar-system bodies | Re-investigate parser; consider deprioritizing                                  | n/a                                       | Low — currently contributing ~0 to linking             |
| **CMR**                                   | 0                      | 2026-04-05          | **broken** — harvest runs complete but no entities produced                    | Re-investigate parser                                                           | n/a                                       | Low — currently zero signal                            |
| **SBDB**                                  | 0                      | 2026-04-05          | **broken** — 4 runs complete, no entities                                      | Re-investigate parser                                                           | n/a                                       | Low — currently zero signal                            |
| **ads_data** (seed)                       | 5                      | —                   | seeded manually                                                                | n/a                                                                             | Stable                                    | Low                                                    |
| **PDS4**                                  | 0                      | never run           | **not harvested**                                                              | Out of scope unless D2 surfaces demand                                          | n/a                                       | n/a                                                    |
| **SPDF**                                  | 0                      | never run           | **not harvested**                                                              | Out of scope unless D2 surfaces demand                                          | n/a                                       | n/a                                                    |
| **Wikidata**                              | 0                      | never run           | **aspirational in original PRD; not implemented**                              | N2 backfill deferred until M3.5 runs                                            | n/a                                       | Medium — non-astronomy cross-domain gap closer         |
| **Anthropic Haiku API (M11 JIT primary)** | —                      | n/a                 | stable vendor; Haiku 4.6 deprecation window plausible                          | M11c local NER on RTX 5090                                                      | Pinned model version per request          | Critical if no fallback; Low with M11c                 |
| **SciBERT-NER + INDUS-NER (M11c)**        | —                      | not deployed        | must deploy as part of this build                                              | Haiku primary (reversed degraded mode)                                          | Model weights pinned in `data/models/`    | n/a                                                    |
| **pgvector 0.8.2**                        | 32.4M paper embeddings | continuous          | stable; iterative scan + halfvec in use                                        | None — underpins retrieval                                                      | Binary snapshot per pg_basebackup cadence | Critical                                               |

**Key findings from the live audit (informed the D3 decisions below)**:

1. **3 harvesters are effectively broken** (SsODNet, CMR, SBDB) and contribute near-zero entity signal. M3 (Tier 1 audit) and M5 (ambiguity classification) run without them. Fixing these harvesters is filed as separate beads, not part of this PRD.
2. **2 harvesters have zombie `running`-state rows** (GCMD, SPASE) from earlier incomplete runs. Startup checks must clean these before build.
3. **VizieR dominates the entity population at 69%** (62,388 of 90,287). Any D3 failure of VizieR is Critical regardless of other harvester health.
4. **`query_log` table does not exist in the DB**. The original PRD's M3.5 (query-log-driven curated core selection) cannot run until the table is created and populated from MCP conversation history. M3.5 gains a sub-requirement to bootstrap this table from existing MCP session logs or ship the table as part of the MCP consolidation's tool-call logging (pre-requisite).
5. **The original PRD's "9 harvesters" risk model is nominal, not real.** The real risk surface is: **7 live harvesters with meaningful coverage** (VizieR, GCMD, PwC, ASCL, PhySH, AAS, SPASE), **3 broken** (SsODNet=20 entities, CMR=0, SBDB=0, all reporting `completed` status), **3 never-run** (PDS4, SPDF, Wikidata), plus the `ads_data` seed (5 entities, manually managed). Forward references to "9 harvesters" in the preserved premortem annotations (§Premortem Risk Annotations) and the rejected "Old framing" sentence in §ADASS narrative refinement are historical-record text, not live assumptions.

### Staging + promotion gate (D3b)

All **7 live harvesters** (VizieR, GCMD, PwC, ASCL, PhySH, AAS, SPASE) write to a `entities_staging` / `entity_aliases_staging` / `entity_identifiers_staging` set of tables, not directly to production. When the 3 broken harvesters (SsODNet, CMR, SBDB) are repaired under separate beads, they inherit the staging gate automatically. A `promote_harvest(run_id)` stored procedure runs the S4 shadow-diff (alias count, canonical set, identifier set) vs the previous production snapshot. Promotion is rejected and rolled back if any of the following fire:

- **>2% shrinkage in canonical count** (S4 blocking gate — promoted from alert).
- **>5% shrinkage in alias count.**
- **Canonical count below the per-source floor** defined in the inventory table (e.g., VizieR floor = 55,000; GCMD = 9,000; PwC = 7,500; ASCL = 3,500; PhySH = 3,500; AAS = 600; SPASE = 200). Floors are set at ~90% of the current live counts; broken harvesters (SsODNet=20, CMR=0, SBDB=0) have floor=0 with a "known-broken, repair bead pending" annotation until the repair lands. This catches the "harvester completes but produces zero rows" class of failure (relevant to first runs after a parser fix).
- **Any entity with >1000 existing `document_entities` rows disappears** (would orphan links).
- **Schema mismatch** (new required column, removed column, type change).

On rejection, the staging data is preserved for inspection, the harvest run's status is set to `rejected_by_diff`, and an alert is paged to the build owner. On acceptance, the promotion writes atomically under a single transaction with advisory lock `pg_try_advisory_lock('entities_promotion')`.

**Acceptance**: Migration 028 creates staging tables + `promote_harvest()` function; unit test fires a synthetic >2% shrinkage and asserts rejection; unit test fires a clean harvest and asserts promotion.

### Ontology version pinning + last-known-good snapshots (D3b continued)

Before the first production build, every live harvester's current state is captured as a versioned snapshot in `data/entities/snapshots/<source>-<YYYYMMDD>.jsonl.gz` committed via DVC (git-LFS fallback). Harvest runs record `source_version` in `harvest_runs.config` (pinned to the upstream ontology's release tag or commit hash where available, dated snapshot otherwise). Re-running against a pinned snapshot is a single `scripts/replay_harvest.py --source <src> --snapshot <date>` command.

**Acceptance**: **7 snapshots** exist in `data/entities/snapshots/` — one per live harvester (VizieR, GCMD, PwC, ASCL, PhySH, AAS, SPASE); `replay_harvest.py` round-trips a snapshot back into staging and produces zero diff vs the current production snapshot of the same date; `harvest_runs.config->>'source_version'` is populated for all future runs. Broken harvesters (SsODNet, CMR, SBDB) are excluded from snapshot requirements until their respective repair beads land.

### LLM cost ceilings and circuit breakers (D3c + P1-9)

- **Per-day absolute dollar cap on M11 JIT**: `$50/day` default (configurable). Enforced at the `resolve_entities()` M13 layer via a Redis counter (keyed by `YYYY-MM-DD`). Over-ceiling calls degrade to M11c local NER (not to static-core fallback, which would hide the cost signal).
- **Per-query cost ceiling**: `$0.01/query` hard cap **enforced at runtime** by `resolve_entities()` M13. Before the Haiku API call fires, the resolver estimates cost as `(candidate_set_size × avg_candidate_tokens + query_tokens) × haiku_input_rate + expected_output_tokens × haiku_output_rate`. Calls whose estimate exceeds the ceiling are rejected and routed to M11c local NER instead (same policy as over-day-cap overflow). Unit test asserts a forced over-ceiling call routes to local NER. Runtime cost telemetry is recorded alongside each JIT call for post-hoc calibration.
- **Vendor SKU watch**: Haiku model version is pinned per request in `resolve_entities()`. A weekly cron fetches the Anthropic SDK's model registry and alerts if the pinned SKU shows `deprecated=true`.
- **M11c local path is deployed from day one** so degraded mode is tested in CI, not discovered in production.

**Acceptance**: Daily cost panel at `build-artifacts/m11_cost.csv` (one row per day, columns: `day, jit_calls, jit_cost_usd, local_ner_calls, cache_hits, degraded_mode_rate`); forced ceiling overflow in a staging run routes to local NER and is logged; Anthropic SDK registry check is wired into a weekly GitHub Action.

### Internal Contracts

- `resolve_entities()` (M13) is the only code path that writes `document_entities` or `document_entities_jit_cache`, or reads `document_entities_canonical`. Enforced by code review + grep-based lint.
- The M8 fusion MV is refreshed exclusively by a dirty-flag-driven cron (not per-batch), capped at ≤1 refresh/hour (P1-11 hardening).
- Harvest promotion (`promote_harvest`) acquires `pg_advisory_lock('entities_promotion')` to serialize against any reader that may be running a Leiden rebuild.
- No MCP tool constructs an `EntityLinkSet` directly; all paths go through M13.

---

## Parallel Validation Tracks (non-blocking, D2 + D4 demoted)

These tracks run in parallel with the build. They are **signals that amend the PRD retroactively**, not gates that block it.

### Track 1 — D2 Demand Validation (was P0, now parallel)

- **Protocol**: Interview ≥5 external research scientists using a structured think-aloud protocol on existing SciX MCP tools. Record verbatim whether entity-layer navigation appears in their workflows. No priming about entities in the recruiting script.
- **Runs in parallel with**: M1, M2, M3, M5 (strategy-independent infra). Target kickoff: week 2 of build.
- **Non-blocking outcome model**:
  - **Positive** (≥2 of ≥5 express unprompted demand for non-astronomy concept access): no PRD amendment; M3.5/M6/M11 proceed unchanged.
  - **Mixed** (1 of 5, or demand for astronomy-only concept layer): amend PRD to collapse the curated core to the astronomy subset and continue M11; `paper_uat_mappings` serves the astronomy path and entity graph layer serves only what came through the interview signal.
  - **Negative** (0 of 5 express unprompted demand): file a retrospective amendment bead; M3.5/M6 are deprioritized in a follow-on PRD revision. Build-in-progress continues on M1/M2/M3/M5/M10/M11-infrastructure until the retrospective amendment lands, at which point the remaining work is re-scoped. **Nothing is paused mid-flight on this signal.**
- **Owner**: Human track, scheduled separately from the build. Build does not block on it.

### Track 2 — D4 External Pre-registration of M4 Eval Set (was P0, now parallel)

- **In-house baseline**: The M4 three-way eval runs first against the in-house 50-query set + 20 graph-walk tasks. Results are reported in `build-artifacts/m4_inhouse_eval.md` with an explicit "in-house-authored, not externally reviewed" disclaimer.
- **Parallel external review**: In parallel with the M4 in-house run, the 50-query set is sent to ≥1 ADS librarian + ≥1 non-astronomy scientist for review. Reviewers may add up to 50 additional queries (capped for protocol control). The combined externally-reviewed set is locked as "M4 v2".
- **Quarterly re-run**: M4 is re-run quarterly on the combined set (P2-12 mitigation). Any divergence > 1 nDCG point between in-house and externally-reviewed eval triggers an investigation bead, not a PRD amendment.
- **Non-blocking**: M4 gate decisions proceed on the in-house run if the external review is not yet complete by the gate deadline. The external review then feeds into the next quarterly re-run.

### Track 3 — S5 Researcher Feedback Loop (already Should-Have)

- Unchanged from the original PRD. `report_incorrect_link` affordance in MCP responses; reports land in `entity_link_disputes`; weekly report shows dispute rate per tier. Part of the Should-Have requirements, already non-blocking by design.

---

## Design Considerations

### C1. Static pre-linking vs query-time JIT extraction

**Tension:** The contrarian lens argued — with direct code evidence (`document_entities` is on zero search paths; `paper_uat_mappings` already has 2.3M rows; `search.py` never reads `document_entities`) — that static pre-linking is the wrong abstraction and just-in-time extraction over retrieved top-K at query time is cheaper, higher-precision, and ZFC-aligned.

**Resolution in this PRD:** M4 (downstream value gate) is the hinge. Tier 1 is cheap enough to run regardless. If M4 shows Tier 1 moves no downstream metric, this PRD's Tier 2-4 are deferred and the project pivots to JIT extraction as a new PRD. If M4 shows meaningful delta, Tiers 2-4 proceed. This turns the contrarian argument into a measurable experiment rather than a philosophical dispute.

### C2. Hybrid (dictionary + disambiguator) vs LLM-first

**Tension:** None in prior art — production systems are uniformly hybrid. But the SciX team's instincts from the embedding pipeline lean LLM-maximalist.

**Resolution:** Tier 3 LLM is explicitly a **closed-set disambiguator** (pick from candidate list) not an extractor. Hallucination is structurally prevented, not prompt-engineered away. Aligned with BLINK/ReFinED candidate-gating pattern.

### C3. Embedding tier: batch pre-linking vs query-time expansion

**Tension:** First-principles math says paper-embedding NN is topical-drift noise; prior art says dense retrieval gated behind a similarity cutoff; contrarian says don't do it at all.

**Resolution:** No batch Tier 4. Embeddings enter via S3 (query-time expansion only). This is both cheaper and better-precision than any batch alternative.

### C4. Fusion formula: noisy-OR over max or DS

**Resolution:** Noisy-OR. Closed form, handles non-independence, degrades gracefully, requires only scalar per-tier weights. Concrete formula in M8.

### C5. Sampling for Tier 3: stratified by (year, arxiv_class), not top-by-citation

**Resolution:** Top-by-citation systematically under-represents recent research — exactly where ADASS reviewers look. Stratified sampling is a one-line fix with high leverage.

### C6. Precision bar inversion

**Surprise from Eval lens:** The biggest eval risk is Tier 1 being quietly dirty, not the exotic tiers. First eval dollar should audit Tier 1. M3 enforces this; M4 gates everything else on Tier 1 results.

### C7. Ambiguity handling — hard blocklist + context required, not heuristic scoring

**Failure-mode insight:** Estimated 6–8K of 90K entities are dangerously short or common words. At 32M papers even a 1-in-1000 FP rate = ~220M bogus edges. M5 forces every entity into one of four `ambiguity_class` buckets with hard policy. M6 enforces context-required firing for homographs.

---

## Open Questions

1. **What nDCG@10 delta does entity coverage actually produce?** This is the M4 hinge. Until measured, all tier planning is speculative.
2. **Does `paper_uat_mappings` (2.3M rows) already solve the astronomy concept layer?** If yes, the entity graph's astronomy contribution is marginal; non-astronomy ontologies (GCMD, SPASE, PhySH, ASCL) are the real delta. Worth confirming before scoping M6.
3. **Can Tier 1 precision clear 0.95 on the 200-sample audit?** If not, the dictionary is dirty (likely from ASCL/PhySH/PwC harvester bugs) and Tier 2 must wait for cleanup.
4. **Is a 500-example human audit enough?** Per-tier Wilson CIs at ±4% are sufficient for go/no-go decisions but thin for a published precision claim. We may need n=1000 for ADASS — budget ~16 human-hours.
5. **Will the pyahocorasick automaton fit CoW under multiprocessing fork on Linux?** Python reference counting breaks CoW; may need `multiprocessing.shared_memory` with a serialized trie. Benchmark needed.
6. **~~Does the team have stomach for the contrarian pivot if M4 fails?~~ RESOLVED (2026-04-12).** The pivot is now structured as a non-blocking amendment path, not a pre-committed gate. If M4 shows no delta, a retrospective amendment bead is filed; in-flight build work does not pause. See §Parallel Validation Tracks.
7. **Cross-ontology entity clustering (N2 / C7): scope in or out?** Mars appearing as 4 rows in 4 ontologies is a real UX problem but requires an `entity_clusters` table and resolver changes. Candidate for a follow-on PRD — defer until after M13 ships.
8. **How does the new MCP tool count stay below 13?** Entity access via filters on existing search tools, not new top-level tools. Confirmed aligned with the 2026-04-11 MCP consolidation merge (`f334961`).
9. **`query_log` table does not exist in the DB as of 2026-04-12.** M3.5 (query-log-driven curated core selection) and M11 (for cost/hit-rate reporting) both reference it. Resolution: M3.5 gains a pre-requisite sub-task to bootstrap `query_log` from existing MCP session logs OR wire it into the MCP tool-call logging path. This is tracked as part of M3.5 acceptance, not a separate PRD.
10. **3 harvesters are effectively broken** (SsODNet: 20 entities, CMR: 0, SBDB: 0) despite reporting `completed` status. Repair is **out of scope for this PRD** (tracked as separate harvester beads); the PRD proceeds with 6 live + 1 aspirational (Wikidata via N2) ontologies as the real risk surface, not the nominal 9. This reality is reflected in §Dependency Hardening.
11. **2 harvesters have zombie `running`-state rows in `harvest_runs`** (GCMD, SPASE). Cleanup on build start is added as a pre-requisite sub-task of M1.

---

## Research Provenance

### Lenses contributing

1. **Prior art (scientific entity linking systems)** — anchored the hybrid-architecture convergence (PubTator 3, BERN2, SciSpacy), surfaced Adeft/BELHD for short-name disambiguation, provided DEAL @ WIESP 2022 as reusable astro NER benchmark, flagged INDUS NER as too weak to be a primary linker (sub-0.34 F1 on CLIMATE-CHANGE-NER).
2. **First-principles technical design** — validated pyahocorasick throughput envelope (3–4 hours for 23M abstracts on 16 cores), proposed noisy-OR fusion formula with closed form, proved embedding tier is net-negative for linking, specified canonical MV, designed incremental watermarked architecture (not triggers).
3. **Evaluation without ground truth** — built the multi-source eval plan (silver from ADS keywords + 500-human audit + LLM-as-judge), grounded sample sizes in Wilson CI math, surfaced cross-citation consistency as free precision proxy, flagged Tier 1 audit as highest-ROI eval spend.
4. **Failure modes / adversarial** — enumerated 20+ specific ambiguous surface forms, exposed the `link_entities.py:70` silent-collision bug, flagged missing `tier` column as catastrophic, surfaced prompt-injection attack surface on abstracts, flagged top-by-citation sampling bias against recent research, proposed hard per-entity linkage cap.
5. **Contrarian / devil's advocate** — verified via grep that `document_entities` is on zero search paths, documented that `paper_uat_mappings` already has 363× more concept edges, argued for JIT extraction + curated smaller entity set, forced M4 to exist as a measurable gate on further investment.

### Convergence / divergence summary

Five independent lenses converged on: (1) hybrid architecture, (2) schema tier column, (3) closed-set LLM disambiguator, (4) ambiguity class annotation, (5) downstream-metric eval gate, (6) Tier 1 first, (7) no batch embedding tier. The strongest divergence was whether static pre-linking is the right abstraction at all — resolved by making M4 a hinge gate that empirically decides.

### Non-obvious insights preserved

- `link_entities.py:70` silent collision bug (from Failures) — single line, catastrophic if unfixed.
- `paper_uat_mappings` 2.3M rows (from Contrarian) — the astronomy concept layer already exists and must not be duplicated worse.
- Adeft per-acronym logistic regression (from Prior art) — the one published technique directly targeting Mercury/JET/HST.
- Noisy-OR fusion closed form (from First-principles) — strictly better than max for non-independent tiers.
- Cross-citation consistency as free precision proxy (from Eval) — uses existing 299M citation edges.
- Stratified (year, arxiv_class) sampling for Tier 3 (from Failures) — corrects against the recent-research under-representation that would undermine the ADASS narrative.
