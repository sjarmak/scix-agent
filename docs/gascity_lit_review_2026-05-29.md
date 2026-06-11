# SciX Literature Review — Multi-Agent Orchestration (for the Gas City cross-architecture investigation)

> **Provenance.** Reconstructed from the `BIB` bibliography embedded in
> `.claude/workflows/gascity_research_synthesis.js`, the workflow that produced
> `/home/ds/gas-city/docs/gascity_improvement_program_2026-05-29.md` (run
> `wf_b53ea849-d5f`, 2026-05-29). This is the **"just-completed SciX lit review
> (last 6 months)"** referenced in that workflow — captured here as a durable
> standalone artifact because it previously existed only as an inline JS constant.
>
> **Retrieval lane.** Searches ran on the SciX MCP **BM25 / lexical lane only** —
> the INDUS dense lane was down (`idx_embed_hnsw_indus` dropped; see the
> pgvectorscale cutover blocker). Queries were keyword-rich to compensate; recall
> on semantically-phrased topics is therefore lower than a hybrid run would give.
> Filters: `year_min:2025`, `arxiv_class ∈ {cs.AI, cs.SE, cs.MA, cs.DC, cs.LG}`.
>
> **Verification status.** All 15 bibcodes were verified to be real, resolvable
> SciX-corpus papers during the 2026-05-29 session. Two anchors were independently
> re-resolved with full author attribution on 2026-05-31 (see §"Load-bearing
> anchors"). The literature here is **motivational / framing only** — in the
> downstream synthesis no P0/P1 priority rests on a citation; every priority rests
> on an on-host verified code finding.

---

## How this maps to the Gas City investigation

The investigation asked: *what do recent multi-agent / orchestration papers
suggest we change in Gas City (formulas, subagent design, routing, observability,
eval)?* The corpus clusters into eight themes, each touching a Gas City concept:

| Theme | Gas City surface it informs |
|---|---|
| Runtime substrate & execution-trace meta-agents | mayor + beads-as-trace; the proposed state-transition view |
| Formal config semantics / typed composition | `city.toml` validation, `gc doctor` linter |
| Multi-agent failure taxonomy & single-agent skepticism | eval honesty gate, MAST retrospective |
| Distributed / parallel agentic workflows | formulas/orders dispatch model |
| Coordination protocols | MCP/A2A message passing between agents |
| Dynamic vs fixed role formation | "zero hardcoded roles" principle |
| Model routing / cost-aware tiering | opus/sonnet/haiku routing, cost-aware dispatch |
| Scheduling / serving & security | supervisor reconcile loop, multi-agent attack surface |

---

## The corpus (15 papers)

### A. Runtime substrate & execution-trace meta-agents
- **`2026arXiv260510913Y` — Shepherd: a runtime substrate for meta-agents with a
  formalized execution trace.** Directly analogous to the mayor + beads-as-trace
  model; motivates a meta-agent that reads a formalized trace. Informs the proposed
  `(issue_id, from_status, to_status, actor, session, ts)` state-transition view
  and the "Shepherd meta-agent over trace" research direction.

### B. Formal config semantics / typed composition
- **`2026arXiv260411767L` — λ_A: a typed lambda calculus for LLM agent
  composition** (well-formedness / termination guarantees). Motivated a `city.toml`
  termination/validation idea — *down-ranked in synthesis* to a cheap runtime
  config linter (formal-methods ceremony judged YAGNI vs a ~50-line `gc doctor`).
- **`2026arXiv260413120K` — AgentForge: execution-grounded multi-agent automated
  software engineering** (mandatory verification step). Backs the "every change
  ships with a verification gate" stance behind the eval-release program.

### C. Multi-agent failure & single-agent skepticism *(load-bearing)*
- **`2025arXiv250313657C` — "Why Do Multi-Agent LLM Systems Fail?" (MAST failure
  taxonomy).** The template for labeling our closed-bead corpus: 14 failure modes
  in 3 categories (specification issues, inter-agent misalignment, task
  verification), built via an LLM-as-Judge pipeline. Its *fixed-taxonomy* nature is
  why the proposed dashboard panel carries a "Bitter-Lesson-fragile" caveat.
- **`2025arXiv250312029C` — "Is Multi-Agent Debate (MAD) the Silver Bullet?"**
  Empirical finding: structured multi-agent debate yields minimal/inconsistent
  gains over a strong single-agent baseline on code summarization & translation.
  The evidentiary basis for the **single-agent baseline honesty gate** (a formula
  ships `enabled=false` unless it beats its 1-agent baseline).

### D. Distributed / parallel agentic workflows
- **`2026arXiv260515132R` — APWA: a distributed architecture for parallelizable
  agentic workflows.** Framing for how independent formulas/orders fan out.
- **`2025arXiv250307675Y` — DynTaskMAS: dynamic task-graph async/parallel
  multi-agent systems.** Closest analogue to the formulas/orders dispatch model;
  motivates dynamic task graphs over static schedules.

### E. Coordination protocols
- **`2026arXiv260409744Q` — MPAC: a multi-principal agent coordination protocol**
  (extends MCP + A2A). Relevant to cross-agent message passing in beads.
- **`2026arXiv260507935X` — TraceFix: repairing agent coordination protocols with
  TLA+ counterexamples.** Pairs with λ_A as the formal-verification angle; same
  synthesis verdict — interesting, but not worth the ceremony yet.

### F. Dynamic vs fixed role formation
- **`2026arXiv260422446Y` — "From Skills to Talent": heterogeneous agents
  organized as a company (dynamic vs fixed roles).** The literature backing for
  the "zero hardcoded roles" principle and the dynamic-role-formation research
  direction.
- **`2026arXiv260513850H` — a 2-D framework for AI agent design patterns**
  (cognitive function × execution topology). A taxonomy lens for classifying our
  formulas/molecules.

### G. Communication & security
- **`2026arXiv260421794Y` — "Learning to Communicate": end-to-end optimization of
  multi-agent language systems.** Motivates treating inter-agent communication as
  optimizable rather than fixed.
- **`2026arXiv260423459H` — "Architecture Matters for Multi-Agent Security".**
  Flags that orchestration topology itself is an attack surface — relevant to the
  multi-principal / external-intake packs.

### H. Model routing / scheduling / tiering
- **`2025arXiv250215964N` — Minions: cost-efficient on-device↔cloud LM
  collaboration.** Backs model-tiering / cost-aware routing (the opus/sonnet/haiku
  decision and the deferred TRACER cost-router).
- **`2025arXiv250213965L` — Autellix: a serving engine for LLM agents as general
  programs** (scheduling). Framing for how the supervisor schedules agent work.

---

## Load-bearing anchors (re-resolved 2026-05-31)

Only two papers actually shaped P0/P1 priorities in the downstream synthesis; both
were independently re-resolved against the SciX corpus with full attribution:

| Cited as | Resolves to | What it informed |
|---|---|---|
| **MAST** `2025arXiv250313657C` | Cemri et al. (2025), *Why Do Multi-Agent LLM Systems Fail?* | P0 MAST-taxonomy-on-bead-corpus retrospective; P2 failure-taxonomy dashboard panel |
| **MAD** `2025arXiv250312029C` | Chun et al. (2025), *Is Multi-Agent Debate the Silver Bullet? …Code Summarization and Translation* | Program-#2 single-agent baseline honesty gate; P1 multi-agent-vs-single A/B |

Both are 2025 SE-domain empirical studies whose findings argue *against*
unconditional multi-agent fan-out — consistent with the synthesis's net-negative
("cut speculative tooling") bias.

> **Note on HARBOR / TRACER.** These names appear in the downstream synthesis doc
> but are **not** in this bibliography. HARBOR did not resolve to any SciX-corpus
> paper (treated as non-load-bearing). TRACER is an internal codename for the
> proposed cost-aware router, not a citation — nearest real corpus work would be
> CascadeDebate `2026arXiv260412262C` and sequential LLM routing
> `2026arXiv260412385Z`.

---

## Caveats for re-use
- **BM25-only recall gap.** This was a lexical-lane run. Re-running the literature
  phase after the INDUS dense lane is rebuilt (post pgvectorscale cutover) would
  likely surface semantically-phrased work this keyword sweep missed.
- **Motivational, not load-bearing.** By design, no engineering priority depends
  on any citation here — use these to frame *method*, not to justify *shipping*.
- **Two augmentation passes not captured here.** The workflow also ran two live
  `mcp__scix__search` augmentation agents (`lit-observability-eval`,
  `lit-routing-modelchange`) at synthesis time; their additional hits live only in
  the run transcript, not in this BIB. A full re-capture would re-run those.

---

## Addendum — dense-lane rerun (2026-06-11)

This review was built **2026-05-29, during the lexical-only window** (INDUS
dense lane down 05-29 → 06-11). Post-restoration rerun of the 7 section
queries, hybrid vs lexical-only (`results/litreview_rerun_2026-06-11.md`):

- **§C (multi-agent failure, load-bearing): unchanged** — hybrid top-15 ≡
  lexical top-15. The review's central skepticism corpus was never at risk.
- **§H (model routing/tiering) changes materially** — dense adds 8 on-topic
  papers the lexical run missed: `2026arXiv260407494M` *Triage: Routing
  Software Engineering Tasks to Cost-Effective LLM Tiers* (closest single
  paper to the Gas City tiering problem); `2024arXiv241010347D` *A Unified
  Approach to Routing and Cascading for LLMs*; `2025arXiv250410681S`
  *EMAFusion*; `2024arXiv240702348K` *Agreement-Based Cascading*;
  `2024arXiv240600060W` *Cascade-Aware Training*; `2025arXiv250219335R`
  *Confidence Tuning for cascades*; `2022arXiv220511747K` *BabyBear*.
- Other sections: dense-added items are largely pre-LLM SOA/grid/web-service
  orchestration (2003–2013) — conceptual archaeology, mostly outside this
  review's deliberate LLM-era scope; no graded misses.
