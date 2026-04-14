# Premortem: Entity Enrichment Strategy for SciX

**Date:** 2026-04-12
**Source PRD:** `prd_entity_enrichment_strategy.md` (refined by `/converge`)
**Method:** 5 independent failure-lens agents, each writing from October 2026 after the project has failed

---

## Risk Registry

Risk Score = severity × likelihood (Critical=4, High=3, Medium=2, Low=1 × High=3, Med=2, Low=1).

| #   | Failure Lens             | Severity | Likelihood | Score | Root Cause                                                                                                                                                                                                           | Top Mitigation                                                                                                                                             |
| --- | ------------------------ | -------- | ---------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Integration & Dependency | Critical | High       | 12    | Treated 9 external ontologies + Anthropic + Wikidata as stable substrates despite the 2026-04-05 harvester-break warning                                                                                             | Dependency inventory with pinned snapshots; local NER fallback (N3 → must-have); 30-day pre-ADASS freeze window                                            |
| 2   | Operational              | Critical | High       | 12    | M11/M8/M10 specified as independent requirements with local SLOs — nobody modeled shared asyncio event loop + MV refresh under daily-sync locks + silent circuit breaker as a coupled failure surface                | Move JIT off MCP response path with async bulkhead; rate-limit MV refresh; page on circuit breaker trips + watermark staleness                             |
| 3   | Scope & Requirements     | Critical | High       | 12    | Measured "whether the built thing works on a proxy" (internal query log + team-authored 50 queries + team-authored 20 graph-walk tasks) instead of "whether research scientists want entity-layer navigation at all" | Add M0 demand-validation; pre-register eval set with external reviewer; minimum 200 externally-sourced queries before M3.5                                 |
| 4   | Technical Architecture   | Critical | Medium     | 8     | Two-lane split drawn on consumer axis (tools) rather than entity identity axis; M8 fusion MV has no slot for JIT results, making "dual-lane" a two-truths architecture with no reconciliation contract               | Assign JIT `tier=5` with persisted cache partition; single canonical `resolve_entities()` service both lanes call through; add M4.5 consistency acceptance |
| 5   | Scale & Evolution        | High     | High       | 9     | Both lanes' scaling constants set against April 2026 traffic/corpus; no hard ceilings wired to the ADASS success metric (agent traffic) that trip before the architecture breaks                                     | Hard dollar cap + degraded-mode on JIT; cap curated core at 10K; quarterly re-run of M4 eval so dissent can win ex post                                    |

**All five risks land in the critical/high band. This is not a comfortable risk registry — it is a design that needs structural hardening before implementation starts.**

---

## Cross-Cutting Themes

### Theme A: Response-path coupling of JIT is the single biggest vulnerability

Surfaced by **Technical** (no reconciliation contract with the static lane), **Operational** (blocks shared asyncio loop, starves unrelated MCP tools), and **Scale** ($0.01/query scales with traffic and produces $50K/month bills at 8× traffic growth). Three independent lenses converge on the same physical component: M11 JIT inlined on the MCP response path.

**Combined severity if exploited:** Critical. A single Anthropic latency incident takes down the full 13-tool MCP surface, corrupts the graph-analytics path via fusion MV mismatches, and produces an unbounded bill at the moment the ADASS paper drives new traffic. The three failures are not independent — they chain.

### Theme B: Validation rests entirely on self-generated proxies

Surfaced by **Scope** (team-authored 50-query eval + team-authored 20 graph-walks + team-biased curated core selection from unrepresentative query log), **Technical** (no consistency eval across lanes), **Scale** (M4 is a one-shot gate, not quarterly). The PRD's entire defense rests on evaluations the same people who designed the system generate.

**Combined severity:** Critical. ADASS reviewers will notice; external researchers already don't use entity-layer navigation; the curated core selection biases match the eval biases, producing a circular validation.

### Theme C: Silent failure modes without paging or playbooks

Surfaced by **Operational** (circuit breaker trips logged but not paged; watermark staleness has no alert), **Integration** (S4 shadow-diff has alert without remediation playbook or rollback path), **Technical** (lane divergence is invisible because JIT results are never written back). Three independent failures where something breaks but nothing pages.

**Combined severity:** High → Critical under cascade. Silent failures have the longest mean-time-to-detection, so they compound over days before anyone looks.

### Theme D: Single-vendor LLM dependency concentration

Surfaced by **Integration** (deleting M7 batch in favor of JIT consolidated all LLM risk onto Anthropic at query time — no offline fallback) and **Scale** (no cost ceiling). The `/converge` decision to delete Tier 3 batch was rational on architectural grounds but removed a degraded-mode buffer without replacing it.

**Combined severity:** Critical during any Anthropic incident or SKU deprecation within 6 months. Haiku 4.5 → 4.6 migration is already plausible in this window.

---

## Mitigation Priority List

Ranked by (failure modes addressed × severity) / implementation cost.

### P0 — Must address before implementation starts

1. **Add `tier=5` for JIT and persist JIT results to `document_entities_jit_cache`** (partition with TTL). Addresses: Technical (reconciliation), Operational (amortize cost across repeated queries), Scale (cap cost via cache hits). **Effort: Medium** (schema + cache layer). Addresses 3 of 5 failure modes.

2. **Move JIT off MCP response path with async bulkhead + per-tool executor pools.** JIT fires with a bounded async budget; on timeout, degrade to static-core filter. Addresses: Operational (event-loop starvation), Technical (lane independence), Scale (bounded latency). **Effort: Medium**. Addresses 3 of 5.

3. **Promote N3 (local SciBERT/INDUS-NER) to Must-Have as degraded-mode fallback for M11.** Addresses: Integration (vendor risk), Scale (cost explosion), Operational (latency regression under Anthropic incidents). **Effort: Medium** (existing RTX 5090 + model download). Addresses 3 of 5.

4. **Add M0 Demand Validation before M3.5.** Interview ≥5 external research scientists with think-aloud protocol. Kill or rescope PRD if fewer than 2 express unprompted demand for non-astronomy concept access. Addresses: Scope (primary), Scale (avoids building unused system that then needs to scale). **Effort: Low** (1 week of interviews). Addresses 1 primary + 1 secondary.

5. **Pre-register the M4 eval set with external reviewer (ADS librarian + non-astronomy scientist) before M6 starts.** ≥200 externally-sourced queries, locked before curated core selection. Addresses: Scope (validation integrity), Technical (consistency eval), Scale (defensible quarterly re-runs). **Effort: Low**. Addresses 3 of 5.

### P1 — Address before M6 Tier 2 run

6. **Dependency inventory section in PRD with per-dependency SLA, deprecation window, fallback path, last-known-good snapshot, cost-of-1-week-outage.** Block build on any dependency missing all five. Addresses: Integration (primary). **Effort: Low**. Addresses 1 primary.

7. **Pin every external ontology to a versioned local snapshot (S3/DVC/git-LFS); harvesters write to staging with promotion gate.** Addresses: Integration (primary), Scope (curated core stability). **Effort: Medium**. Addresses 2 of 5.

8. **Convert S4 shadow-diff from alert to CI gate with automatic rollback on >2% shrinkage.** Addresses: Integration, Operational (silent failure). **Effort: Low**. Addresses 2 of 5.

9. **Hard dollar cap on JIT spend (per-day absolute ceiling, not per-query).** Above ceiling → fail over to static-lane-only with explicit degraded-mode response. Addresses: Scale (primary), Integration (vendor concentration). **Effort: Low**. Addresses 2 of 5.

### P2 — Address before ship

10. **Page on M10 circuit breaker trips after 2 consecutive; watermark-staleness alert at now()-max_entry_date > 24h.** Addresses: Operational (silent failure). **Effort: Low**.

11. **Rate-limit + de-duplicate MV refresh to once/hour via dirty-flag counter, not per-batch.** Addresses: Operational. **Effort: Low**.

12. **Quarterly re-run of M4 three-way eval** so Advocate B's "JIT-only" dissent can win ex post if the static lane's advantage erodes under organic growth. Addresses: Scope, Scale. **Effort: Low** (cron).

13. **30-day pre-ADASS dependency freeze window** (no harvester re-runs, no model SKU upgrades, no Wikidata refreshes, frozen snapshots only). Addresses: Integration. **Effort: Low**.

14. **Cap curated core at 10K entities; re-run M5 ambiguity classification on every S5-driven promotion.** Addresses: Scale, Technical (ambiguity drift). **Effort: Low**.

15. **Load-test milestone between M11 ship and "default on"**: replay 24h of query_log through full MCP surface with JIT enabled, measure cross-tool contention. Addresses: Operational. **Effort: Medium**.

---

## Top Design Modifications

### D1. Restructure M11 from "inline JIT on MCP response path" to "async JIT with persisted cache and local fallback"

**Change:** M11 becomes three coupled sub-requirements:

- M11a: JIT Haiku call fires with 400ms hard async budget, per-tool executor pool (bulkhead); on timeout or vendor error, degrade to static-core filter.
- M11b: JIT results persisted to `document_entities_jit_cache` partition with `tier=5`, TTL 14 days, keyed by `(bibcode, candidate_set_hash, model_version)`. M8 fusion MV includes JIT tier.
- M11c: Local SciBERT/INDUS-NER inference path (promoted from N3) serves as degraded-mode fallback when Anthropic is unavailable, deprecated, or over budget.

**Addresses:** Technical (1), Operational (2), Integration (3 partial), Scale (5).
**Effort:** Medium-High (~2 weeks).

### D2. Add M0 Demand Validation as a hard gate before M3.5

**Change:** Interview ≥5 external research scientists with structured think-aloud protocol on existing SciX MCP tools. Record verbatim whether entity-layer navigation appears in their workflows. Gate: if fewer than 2 express unprompted demand for non-astronomy concept access, PRD terminates (or scopes down to fix-silent-bug + schema-hardening only, deferring M3.5+).

**Addresses:** Scope (3, primary); secondary: all others by reducing blast radius if the project is the wrong one.
**Effort:** Low (~1 week).

### D3. Dependency Hardening section (new PRD subsection)

**Change:** Add a Dependency Hardening section with:

- Inventory table: each dependency's SLA, deprecation cadence, fallback path, last-known-good snapshot, cost-of-outage.
- Staging + promotion-gate architecture for all 9 harvesters.
- S4 shadow-diff converted from alert to blocking CI gate with auto-rollback.
- 30-day pre-ADASS freeze window as an explicit milestone.
- Dual LLM path: Haiku primary + local SciBERT/INDUS-NER degraded mode (same as D1).

**Addresses:** Integration (1, primary), Scale (5 partial), Operational (2 partial).
**Effort:** Medium (~1 week for inventory + staging architecture).

### D4. External pre-registration of M4 eval set

**Change:** Before M6 runs, the 50-query eval set + 20 agent graph-walk tasks must be:

- Sourced from ≥200 externally-logged MCP queries (not team-authored).
- Reviewed and locked by ≥1 ADS librarian + ≥1 non-astronomy scientist.
- Documented as immutable for the duration of the PRD (any change requires amendment).

M4 also gains a **consistency acceptance** (M4.5): for the same 50 bibcodes, `citation_chain(bib)`, `hybrid_search[enrich_entities=True]`, and `SELECT FROM document_entities_canonical` return the same entity set modulo a declared lane-delta report; fails the gate if divergence > 5%.

**Addresses:** Scope (3, primary), Technical (4, consistency), Scale (5, enables defensible quarterly re-runs).
**Effort:** Low (~3 days for protocol + sourcing).

### D5. Single canonical entity resolution service

**Change:** Define one internal function `resolve_entities(bibcode, context) -> EntityLinkSet` that both graph-analytics and retrieval tools call. Lane choice (static vs JIT vs cache) is internal to the resolver, not a tool-author decision. All entity writes go through this service; all reads query `document_entities_canonical` through it; JIT cache hits are transparent.

**Addresses:** Technical (4, primary), Operational (2, reduces integration surface), Scope (consistency for eval).
**Effort:** Medium (~1 week, but unblocks D1 and D4).

---

## Full Failure Narratives

### Failure Narrative 1: Integration & Dependency Failure

**What happened:** In mid-May 2026, Anthropic shipped Haiku 4.6 and announced a 90-day deprecation window for Haiku 4.5 Batches, coinciding with a tool-use schema tweak that broke M11's closed-set disambiguator. Migrated, but new model's tool-use latency p95 regressed from 380ms to 720ms — blowing M11's ≤500ms budget. S4 shadow-diff fired weekly: GCMD v20.1 shrank canonical set 4%; SSODNet changed its REST endpoint to require OAuth (silent HTML login returns); PhySH's maintainers froze the ontology. In September, M3.5 Wikidata backfill hit new 30-day rate limits (post-July DDoS), cutting throughput 12×. M4 ran on a curated core that was 38% stale. Static lane lost to JIT not for architectural reasons but because 38% of entities pointed at retired terms. Two weeks to ADASS, couldn't rebuild against pinned versions, paper withdrawn.

**Root cause:** Treated 9 harvesters + Anthropic + Wikidata as stable despite prior warnings; S4 was a guard without a rebuild-on-break SLA.

**Severity:** Critical | **Likelihood:** High

### Failure Narrative 2: Operational Failure

**What happened:** M11 shipped behind `enrich_entities=True` flag, defaulted off. By July the internal demo agent flipped it on by default, exercising it on ~60% of MCP calls. Dev-tested p95=500ms held against 5-row candidate lists; production had 40–80 entities from `entity_aliases` joins on broad queries, and Haiku tool-use tail-spiked to 2.5s on regional slow minutes. JIT inlined on MCP response path blocked the shared asyncio event loop, starving `citation_chain`, `vector_search`, `bm25_search` — on-call chased "pgvector HNSW regression" for four days. `document_entities_canonical` MV refresh, triggered per-batch by incremental linker, took locks that stalled daily-sync upserts; M10 5-min circuit breaker tripped 11 days in a row, nobody noticed catch-up wedged on stale watermark. September: curated core 11 days behind, JIT spiking, Leiden rebuild read half-refreshed snapshot mixing tier=1/tier=2 rows inconsistently, camera-ready pulled.

**Root cause:** M11/M8/M10 specified as independent with local SLOs; shared asyncio loop + MV-under-daily-sync-locks + silent circuit breaker never modeled as coupled failure surface.

**Severity:** Critical | **Likelihood:** High

### Failure Narrative 3: Scope & Requirements Failure

**What happened:** M3.5 ran query-log mining in May. `query_log` contained ~3 months of team test harnesses, 2 friendly beta testers, internal eval runs — unrepresentative. Fail-to-resolve ranking surfaced astronomy acronyms already in `paper_uat_mappings` plus mission/instrument names the team had manually tested. M6 linked ~6K core at 0.97 precision on schedule. June M4 ran on the same 50-query set reused from MCP consolidation — keyword-shaped retrieval probes authored by the team. 20 graph-walk tasks hand-written by the curated core designer, unknowingly biased. Static lane +0.3 nDCG on 4 queries, JIT +0.4 on 6, baseline noise on the rest. Both lanes survived the gate. August: ADASS draft circulated to 2 external research scientists; both said "I use `search_papers` and `citation_chain` — I don't think about entities, I've never once wanted to filter a search by GCMD concept." Reviewer 2 rejected: "evaluation does not establish that research scientists using MCP tools have the access patterns this architecture optimizes for. The 50-query benchmark appears to be retrieval-shaped, not navigation-shaped."

**Root cause:** Measured whether the built thing worked on a proxy (team-authored evals) instead of whether entity-layer navigation was wanted at all.

**Severity:** Critical | **Likelihood:** High

### Failure Narrative 4: Technical Architecture Failure

**What happened:** M1/M2 shipped clean in April. M3 Tier 1 produced ~4.1M rows, cleared 0.95 precision. June M4 three-way eval: JIT +0.8, static +0.3, within noise on 50-query set, but 20 agent graph-walks showed static winning on multi-hop citation traversals. Both lanes greenlit per M12. M6 ran 2h as promised. Then the seam frayed: M12's dual-lane contract was un-enforceable at runtime. Agents composing tool chains (retrieval → citation walk on retrieved IDs) produced graphs where half the nodes had JIT-lane entities from Haiku's closed-set and half had static-core entities, with different `entity_id` populations because JIT's candidates came from UAT+Tier1 near-misses while static had Wikidata backfill JIT never saw. M8 noisy-OR was per-tier with no lane slot — JIT results were never written back. MV and query-time path answered different questions for the same `(bibcode, entity_id)`. By August Leiden on static lane and agent graph-walks on retrieval lane produced contradictory neighborhoods for the same papers. September: reviewer-rehearsal question "show me the entity graph for bibcode X" returned three different answers. Paper pulled mid-September.

**Root cause:** Two-lane split on consumer axis (tools) rather than entity identity axis; fusion MV had no JIT slot; dual-lane became two-truths with no reconciliation contract.

**Severity:** Critical | **Likelihood:** Medium

### Failure Narrative 5: Scale & Evolution Failure

**What happened:** Dual-lane shipped clean in May. M4 validated static core: +2.1 nDCG@10 on graph-walks JIT couldn't match. July ADASS preprint posted; agent traffic 8×'d within six weeks. MV refresh drifted from <5min to 45min under 40K papers/day back-harvest load (planned 1K/day). Composite PK write amplification under concurrent tier-2 upserts blocked readers. CONCURRENT refresh doubled bloat; autovacuum permanent catch-up on 180M-row `document_entities`. JIT at modeled 5K queries/day hit 180K/day with agent loops doing 20–50 sub-queries/task — Haiku bill crossed $50K/month. Emergency cache-and-rate-limit patch introduced staleness, corrupted two downstream benchmarks, forced camera-ready errata. Curated "6K" core grew to 34K via S5 promotions + M3.5 refreshes; Mercury/JET/HST-class homographs returned. Leiden over `document_entities_canonical` stopped converging at resolution 0.001 (one giant component containing 71% of linked papers). Graph-analytics tools returned results contradicting M12's static-lane contract.

**Root cause:** Both lanes' scaling constants set against April 2026 traffic/corpus; no hard ceilings wired to the ADASS success metric (agent traffic) that trip before the architecture breaks.

**Severity:** High | **Likelihood:** High

---

## Summary for the user

Every one of five independent failure lenses returned Critical or High severity with High likelihood on a one-shot run. This registry tells the team the PRD is not ready for build in its current form. The dominant vulnerability is **JIT on the MCP response path coupled with no persistence contract and no local fallback** — three lenses chained on the same physical component. The second is **validation resting entirely on team-authored proxies**, which ADASS reviewers will notice.

The P0 mitigations (D1–D5 above) are not optional. They should be applied as PRD amendments before any M6 / M11 work begins.
