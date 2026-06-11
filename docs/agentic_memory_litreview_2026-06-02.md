# Agentic Memory Systems — Consolidated Literature Review

*Prepared 2026-06-02 for ramp-up into an applied-research role on agentic-memory **evaluation** (benchmark design, evaluation methodology, synthetic-data pipelines, open-source harnesses) across semantic / episodic / procedural memory and multi-session / multi-turn trajectories.*

**Method.** Built with the SciX MCP (32M-paper arXiv mirror + NASA ADS). One `lit_review` seed (93-paper working set, citation-expanded), then two fanned-out waves of parallel sub-agent research: Wave 1 = 8 thematic branches (~110 papers); Wave 2 = (a) full-text design extraction of three keystone papers, (b) a forgetting/consolidation deep-dive, (c) temporal-evolution charting. ~130 unique papers.

**Reading caveat.** This field is dominated by **2026 zero-citation preprints**; the SciX `citation_contexts` substrate covers ~0% of edges for this topic, so claims are verified from abstracts / extracted sections, not citation-graph corroboration. Every quantitative figure below is an author self-report. Interactive companion: `docs/agentic_memory_explorer.html`.

---

## 0. The shape of the field (temporal evolution)

Per-year on-topic volume in the topical working set: **2022 ≈ 0 (the modern framing did not exist)** → **2023: 31** → **2024: 33** → **2025: 20\*** → **2026: 39** (year only ~5 months old). The 2025 dip is a sampling artifact (citation-anchored seed expansion favors older, more-cited papers; 2025–26 papers carry citation_count 0 because they're too recent) — raw hybrid-search ranking is overwhelmingly 2026, so the acceleration is real and steepening.

Keystone citation trajectories (citations-per-year, from the graph):

| Paper | 2023 | 2024 | 2025 | 2026\* | Shape |
|---|---|---|---|---|---|
| Voyager (`2023arXiv230516291W`) | 175 | **402** | 101 | 90 | highest-cited; 2024 peak |
| Reflexion (`2023arXiv230311366S`) | 193 | **256** | 71 | 64 | foundational; 2024 peak |
| CoALA (`2023arXiv230902427S`) | 23 | **110** | 20 | 7 | concept-scaffold boom→migration |
| LoCoMo (`2024arXiv240217753M`) | — | 21 | 17 | 24 | durable benchmark adoption |
| LongMemEval (`2024arXiv241010813W`) | — | 2 | 13 | **31** | sharp monotonic ramp |
| A-MEM (`2025arXiv250212110X`) | — | — | 5 | **64** | explosive uptake |
| Mem0 (`2025arXiv250419413C`) | — | — | 0 | **78** | highest single-year keystone |
| Zep (`2025arXiv250113956R`) | — | — | — | 33 | temporal-KG, MemGPT successor |

**Benchmark frontier by year:**
- **2023 — cognitive-architecture & self-reflection.** No dedicated memory benchmark; agents improvise memory. Reflexion (episodic self-reflection), Voyager (lifelong skill library), CoALA (the taxonomy), MemGPT (`2023arXiv231008560P`, memory-as-OS).
- **2024 — dedicated conversational-memory benchmarks arrive.** LoCoMo + LongMemEval make multi-session recall measurable; structural-memory ablations; embodied long/short-term memory (KARMA). Question shifts from "can agents act" to "can agents remember correctly."
- **2025 — production memory architectures vs the 2024 benchmarks.** A-MEM, Mem0, Zep; position papers ("Episodic Memory is the Missing Piece", `2025arXiv250206975P`); lifelong-learning roadmaps.
- **2026 — stress-testing, governance & self-evolving memory.** Diagnostic/adversarial benchmarks (MemConflict, STALE, EvoMemBench, MemGround); memory as an attack surface (memory hijacking/Trojans); architecture specialization (graph/hierarchical/efficiency).

---

## 1. Conceptual foundations & the memory taxonomy

- **CoALA** (Sumers et al. 2023, `2023arXiv230902427S`, 154 cites) — canonical semantic/episodic/procedural framing; modular memory + structured action space + decision loop.
- **"From Storage to Experience"** survey (Luo et al. 2026, `2026arXiv260506716L`) — the maturity axis: **Storage → Reflection → Experience** (trajectory preservation → refinement → abstraction). Names the field's split personality: "oscillating between operating-system engineering and cognitive science."
- **Taxonomy is contested:** the Missing Knowledge Layer (`2026arXiv260411364R`) argues CoALA/JEPA lack a typed Knowledge layer; **ZenBrain** (`2026arXiv260423878B`) proposes 7 layers.
- **"Survey on Evaluation of LLM-based Agents"** (Yehudai et al. 2025, `2025arXiv250316416Y`) — states plainly that **memory evaluation lags planning/tool-use and is benchmark-fragmented**. The strongest one-line justification for the role.

**Takeaway:** instrument *per-memory-type* probes, not one aggregate recall number. The taxonomy debate is itself the argument for a clean decomposed benchmark.

---

## 2. Memory architectures — a structure spectrum

`flat text → typed/semi-structured → pairwise KG → hypergraph → hierarchical tree`

- **Temporal KG:** **Zep/Graphiti** (`2025arXiv250113956R`) — de facto baseline; DMR 94.8% vs MemGPT 93.4%, +18.5% LongMemEval at ~90% lower latency. **APEX-MEM** (`2026arXiv260414362B`) semi-structured + temporal reasoning.
- **Hypergraph:** **HyperMem** (`2026arXiv260408256Y`) — n-ary hyperedges for multi-participant events pairwise graphs fragment.
- **Hierarchical/tree:** MemTree (`2024arXiv241014052R`); **LinkedIn hiring agent** production tiered semantic memory (`2026arXiv260426197X`).
- **Specialized retrieval (recall@k is insufficient):** Memanto information-theoretic (`2026arXiv260422085M`); MemCog "memory-as-cognition" iterative retrieval (`2026arXiv260528046L`); Thought-Retriever stores reasoning traces (`2026arXiv260412231F`); cooperative keyword paging (`2026arXiv260412376L`).
- **Cross-system comparative method:** **GRAVITY** (`2026arXiv260501688S`) — plug-in evaluated across 5 host memory systems on LongMemEval+LoCoMo (+7.5–10.1%), gains *inversely correlated with baseline strength*. The harness pattern to emulate.
- **Skeptic:** "Do We Still Need GraphRAG?" (`2026arXiv260409666F`) — flat RAG vs GraphRAG head-to-head.

---

## 3. Procedural memory & skill libraries

Two representations compete and **fail differently**: executable code-skill libraries (Voyager `2023arXiv230516291W`; SkillClaw `2026arXiv260408377M`; Skill1 RL co-evolution `2026arXiv260506130S`; SkillDroid compiled GUI skills `2026arXiv260414872C`) vs natural-language workflows/manuals (Agent Workflow Memory `2024arXiv240907429Z`; AutoManual `2024arXiv240516247C`).

Dedicated benchmarks (new, unconsolidated):
- **SkillEvolBench** (`2026arXiv260524117L`) — see §10 deep-dive. Finding: **raw-trajectory reuse often beats distilled skills** (lossy-abstraction bottleneck); larger libraries add drift/clutter.
- **SEA-Eval** (`2026arXiv260408988J`) — scores skill/tool *accumulation across* tasks, not per-episode.
- **ImplicitMemBench** (`2026arXiv260408064Q`) — implicit/procedural memory (experience → automated behavior).
- **"Harness Updating Is Not Harness Benefit"** (`2026arXiv260530621L`) — methodological keystone: updating a store ≠ benefiting from it.

**Opportunity:** the canonical procedural-memory harness with frozen held-out splits **does not yet exist**. AWM's widening train-test-gap protocol and Voyager's cross-world transfer are reusable templates.

---

## 4. Reflection & experience-stage memory (memory that improves the policy)

- **Reflexion** (`2023arXiv230311366S`, 531 cites) — lineage anchor; cross-attempt-improvement protocol.
- **Write/read decomposition** is now standard: Reflective Memory Management (`2025arXiv250308026T`), "Learning How and What to Memorize" (`2026arXiv260500702X`), TSUBASA (`2026arXiv260407894Z`).
- **Retrieved → generated/abstracted experience:** CLEAR (`2026arXiv260407487L`, open-source; AppWorld 72.6→81.2%, WebShop 0.68→0.74), HiExp (`2026arXiv260408124H`).
- **Memory co-evolves with capability:** SEARL (`2026arXiv260407791F`), Mem2Evolve (`2026arXiv260410923C`), Skill1.
- **Parametric vs non-parametric consolidation:** "Beyond Inference-Only Deployment" (`2026arXiv260524657D`) — per-user weight consolidation vs context compaction, head-to-head.
- **Ready multi-task benchmark:** BEHEMOTH/CluE (`2026arXiv260411610Y`) — 18 datasets, downstream-*utility* metric (+9.04% rel).

---

## 5. Benchmarks — fragmented but mapped

Canonical pair (verified composition):
- **LoCoMo** (`2024arXiv240217753M`, 38 cites) — 10 conversations, 1,813 questions, ~35 sessions; persona + temporal-event-graph synthetic pipeline; QA = single-hop/multi-hop/temporal/open-domain; human 87.9 F1. **Saturating** (Synthius-Mem 94.4% acc / 99.6% adversarial; APEX-MEM 88.9%).
- **LongMemEval** (`2024arXiv241010813W`, 15 cites) — 500 questions, five abilities (extraction, multi-session reasoning, temporal, knowledge-update, abstention); commercial assistants drop ~30% across sessions.

Expansion frontier (prior art for novel tasks):

| Axis | Benchmark | bibcode |
|---|---|---|
| Multi-party / speaker attribution | GroupMemBench | `2026arXiv260514498Y` |
| Implicit / procedural | ImplicitMemBench | `2026arXiv260408064Q` |
| Continuous lifelog / wearable | LifeMem/EgoMem | `2026arXiv260411182Z` |
| Multimodal recall | MemLens / MemEye | `2026arXiv260514906R` / `2026arXiv260515128G` |
| Strategic (when to use a fact) | StratMem-Bench | `2026arXiv260426243W` |
| Memory conflicts | MemConflict / Selective-QA | `2026arXiv260520926T` / `2026arXiv260530087Y` |
| Stale-memory / belief revision | **STALE** | `2026arXiv260506527C` |
| Failure-mode stress test | MemFail | `2026arXiv260526667G` |
| Fully-synthetic, fixed-answerer | EngramaBench | `2026arXiv260421229A` |
| Real human anchor | REALTALK (21-day) | `2025arXiv250213270L` |

**EngramaBench** holds the answering model fixed (GPT-4o) and varies only the memory architecture — the clean controlled comparison.

**Headline gap:** no unified harness spans the full taxonomy; abstention/refusal is a first-class LongMemEval ability that LoCoMo leaderboards routinely omit.

---

## 6. Evaluation methodology — where the real research lives

**White-box / failure-attribution:**
- **MemTrace** (`2026arXiv260528732D`) — traces failures to the responsible stage (write/retrieve/generate).
- **AuthTrace** (`2026arXiv260525382W`) — isolates evidence-construction.
- **MemConflict** (`2026arXiv260520926T`) — answer-correctness and retrieval-correctness *diverge* (see §10).

**Validity threats to defend a leaderboard against:**
- **Scoring-target sensitivity — TIAP / "Same Ranking, Different Winner"** (`2026arXiv260524060P`, see §10) — switching the credited memory form flips rankings on 83–94% of queries. *The methodology must-read.*
- **Contamination — NumLeak** (`2026arXiv260530393K`) — public benchmarks leak into pretraining → argues for leak-controlled synthetic data.
- **Trilemma — WebForge** (`2026arXiv260410988Y`); **production divergence — AlphaEval** (`2026arXiv260412162L`).
- **LLM-as-judge reliability:** surveys (`2024arXiv241115594G` 106 cites; `2024arXiv241205579L`) + fine-tuned judges don't transfer (`2024arXiv240302839H`, 58 cites). Judge calibration on memory-specific question types is largely unvalidated.
- **Richer-than-binary correctness:** graded support relations (`2026arXiv260408082S`).

**Bar for a credible harness:** (a) make scoring target explicit + report sensitivity; (b) report retrieval/ranking diagnostics alongside answers; (c) report CIs/variance (almost never done); (d) validate judges against human labels on memory question types.

---

## 7. Synthetic data generation — pipelines + the realism trap

- **Templates:** AgenticAI-DialogGen (`2026arXiv260412179M`, topic→KG→persona→dialogue+QA); Graph2Counsel (`2026arXiv260420382M`, profile-graph-conditioned clinical dialogue); UniToolCall (`2026arXiv260411557L`, 22k tools / 390k structurally-controlled tool trajectories with distractor settings); VeriSim (`2026arXiv260440441M`, configurable realistic-noise injection).
- **Realism trap — OmniBehavior** (`2026arXiv260408362C`) — LLM user simulators converge to a "positive average person" (persona homogenization, Utopian bias, lost long-tail) and plateau as context grows. Pair with **REALTALK** as the real-data yardstick.
- **Generator auditing — QDC** (`2024arXiv241202980H`) — Quality drives in-distribution, Diversity drives OOD generalization, Complexity helps both. User-simulator survey (`2026arXiv260424977N`); persona survey (`2024arXiv240601171T`, 44 cites).

**Gap:** no agreed score for whether a synthetic multi-session memory set is realistic — itself a publishable contribution.

---

## 8. Memory security & governance

- **Poisoning the store:** AgentPoison (`2024arXiv240712784C`, 33 cites — canonical, optimized retrieval trigger); Hijacking Agent Memory (`2026arXiv260529960W`, poisons via normal conversation; defines the **RSR@k / ASR / Benign-Accuracy** triad); ShadowMerge (`2026arXiv260509033L`, first *graph*-memory attack, cross-user contamination); Phantom (`2024arXiv240520485C`) + Morris-II worm (`2024arXiv240302817C`).
- **Unintended drift:** "When Routine Chats Turn Toxic" (`2026arXiv260506731X`) — Harm-Score benchmark + StateGuard writeback auditing.
- **Availability:** RAG jamming / blocker docs induce over-refusal (`2024arXiv240605870S`).
- **Defenses → provenance/lineage:** MemLineage (`2026arXiv260514421O`); Agent-BOM audit graph (`2026arXiv260506812L`); layered framework (`2026arXiv260423338C`).
- **Privacy-in-action:** PrivacyLens (`2024arXiv240900138S`, 11 cites) — contextual-integrity over action trajectories; seed-to-trajectory pipeline.
- **Foundational threat model:** indirect prompt injection (Greshake et al. 2023, `2023arXiv230212173G`, 134 cites).

**Gap:** no standardized open memory-security harness; **episodic/procedural-memory security is unaddressed** (almost all attacks hit semantic/retrieval memory). Adversarial robustness belongs in the core eval suite.

---

## 9. Applications & personalization

- **Personalization → evolving memory:** TSUBASA (`2026arXiv260407894Z`), Reflective Memory Management (`2025arXiv250308026T`), PASK proactive intent-aware memory (`2026arXiv260408000X`).
- **Multi-user isolation:** Multi-User LLM Agents (`2026arXiv260408567Y`) — per-user partitioning + cross-user leakage (essentially unbenchmarked).
- **Multimodal:** PersonaVLM (`2026arXiv260413074N`), MMSkills (`2026arXiv260513527Z`).
- **Coding/GUI:** Memory Transfer Learning across domains (`2026arXiv260414004K`); SkillDroid (`2026arXiv260414872C`) — **efficiency (inference-cost) is a co-equal metric** to accuracy.
- **Healthcare (longitudinal):** four-layer coherence/continuity/adaptation/agency framework (`2026arXiv260412019G`); Clinical World Model + Skill-Mix (`2026arXiv260408226S`); Evo-MedAgent (`2026arXiv260414475S`).
- **Harness-as-unit:** SemaClaw harness engineering (`2026arXiv260411548Z`).

---

## 10. Design deep-dives (full-text extraction of three keystones)

### SkillEvolBench (`2026arXiv260524117L`) — procedural-skill benchmark
- **Tasks:** 180 = 6 environments × 5 families × 6 roles. A *family* = a recurring procedural capability (shared latent procedure). The **6-role arc splits into two phases**: ACQUISITION (library updates allowed) = {canonical, enriched, variant}; DEPLOYMENT (library **frozen**) = {context-shift (implicit invocation), adversarial (shortcut resistance), composition (multi-skill interaction)}.
- **Data generation:** source-driven + human-curated, **no reuse of existing benchmark instances**. Cluster real workflows → retain families meeting real-world-relevance + procedural-fit + verifiable-evolvability → author a **gap-exposed curated skill** (supports canonical but deliberately leaves enriched/variant/adversarial/composition unresolved) → manual review. Skill-evolution loop runs per environment with a fresh env-scoped library reset between envs (no cross-env leakage); a **host-side Skill-Author call, separate from the solving agent**, emits library edits.
- **Metrics:** verifier returns outcome (public+hidden tests) + **process** score (brittle-strategy detection: hard-coded constants, swallowed exceptions, skipped validation). Success-rate decomposition: **LSR** (acquisition), **RSR** (replay after freeze = local recovery, not transfer), **ESR** (frozen deployment) → decomposed into **CSSR / ARSR / CompSR**. Report pp deltas vs **No-Skill** and vs **Raw-Trajectory** baselines.
- **Numbers:** evaluated across 10 model configs × 3 harnesses. Self-Generated (Opus): LSR +5.5pp, RSR +10.0pp vs No-Skill but ESR/CSSR/CompSR *decrease*; many Raw-Trajectory deltas negative (−10 to −13pp). Central finding: **local procedural adaptation, not reliable reusable skill formation.**

### TIAP / "Same Ranking, Different Winner" (`2026arXiv260524060P`) — scoring-target audit
- **Object:** the *scoring target* — which stored form gets retrieval credit when one source turn spawns multiple derived memories. Three formal targets: **Raw** (exact source turn), **Source** (any descendant linked to the source), **Canonical** (transformed serving memories). Raw and Canonical are disjoint subsets of Source.
- **Method:** fixed-output **rescoring of saved top-k ranked traces** under each target (no retrieval re-run). Stages: construct targets → rescore → target-sensitivity + semantic audit. Crucial controls: **shared-query-subset** (only score queries where every compared target has ≥1 eligible ID — prevents confounding with difficulty); **gold/evidence labels kept out of the memory writer** (no leakage); k=60 justified by recall plateau after ~k=40.
- **Metrics:** nDCG; `Δ_t(A,B)=M_t(A)−M_t(B)`; **winner flip** = sign of Δ changes as target t varies; 3,000-bootstrap 95% CIs. Semantic audit of all **1,902** contested credits via 5-model majority vote at temp 0, validated on a 115-case human subset (**Fleiss κ = 0.83**).
- **Numbers:** switching *only* the target changes nDCG on **83.4–94.0%** of shared queries; flips winners on Mem0 & MemoryOS transfer runs; reverses parser-density recommendations. Relaxed source-linked credit fully justified only **29.2%** of the time (39.6% partial, 31.2% unsupported). Coverage confound: Canonical-eligible queries are easier under Raw by +0.086 nDCG. Ships **MTEL-Mem** reproducibility layer.

### MemConflict (`2026arXiv260520926T`) — conflict-aware benchmark
- **Tasks:** memory validity as query-conditioned **fitness-for-use** along 3 dimensions → 3 conflict types: **Dynamic** (later true update supersedes earlier state — temporal validity), **Static** (later *false* contradiction must NOT overwrite an invariant — factual correctness), **Conditional** (multiple values valid under different conditions — contextual applicability). Gold = `argmax F(v|α,H_t,q_t)` at the history prefix (unique maximizer by construction).
- **Data generation (5-step, Algorithm 1, all calls gpt-5.0-mini):** Persona-Hub seed → structured profile (invariant / dynamic / conditional attrs) → timeline simulation Jan-2022..Dec-2025 monthly with conflict construction (validation rule V_d enforces marital/child/cooldown constraints; conflict endpoints not re-mentioned between, to isolate) → **related-entity distractor injection** that leaves the answer unchanged → two-stage dialogue (synopsis → multi-turn realization, every info item explicitly realized & correctly attributed) → **human expert verification** → query+label construction only once the instance is evaluable.
- **Metrics:** macro-averaged across types. Black-box **AA**; white-box **SEH@K** (gold memory in top-K) and **SRS** (log-discounted rank); diagnostics **UOCS** (update order, dynamic) and **CRS** (conflict recognition, static). The **SEH@K-vs-SRS gap** and **CRS-vs-AA divergence** localize failure (missing vs low-ranked vs ineffective use).
- **Numbers:** 12 instances, avg 52 sessions / 2,349 turns / ~204K tokens / 124 queries each. MemOS best avg AA 0.554 (static hardest). **CRS ≤ 0.25 for all systems** — they answer correctly *without* recognizing the contradiction. White-box: MemOS best SEH@3 0.671 / SRS 0.588; SEH@3 > SRS broadly. No system dominates all conflict types; longer histories, distractors, implicit queries, larger conflict distance all degrade.

**Cross-cutting design principles (all three):**
1. **Freeze-then-evaluate** — separate the write/consolidation phase from a frozen read phase so gains can't come from test-time adaptation.
2. **Separate the credit/scoring definition from retrieval behavior** — pin Raw/Source/Canonical, test ranking robustness to it.
3. **Two-level (black-box + white-box) evaluation** — answer correctness diverges from whether the right evidence was retrieved/ranked or the right procedure followed.
4. **Control set / ablation baselines mandatory** — No-Skill, Raw-Trajectory, coverage-matched shared-query subsets, one-factor-at-a-time sweeps.
5. **Controlled synthesis from structured seeds + human verification** — generate then validate; never trust raw synthetic.
6. **Calibrate LLM judges against humans before scaling** (Fleiss κ reported; human verification gate).

**Pitfalls to avoid:** target/credit non-invariance; coverage confound; answer-only evaluation hiding memory failures; mistaking local recovery / raw reuse for durable memory formation; data leakage through the memory writer; undetected shallow shortcuts/distractors without process & hard-negative design; uncontrolled retrieval depth.

---

## 11. Forgetting, consolidation & obsolescence (the thinnest, highest-leverage area)

- **STALE** (`2026arXiv260506527C`) — the most on-target measurement paper. 400 expert-validated conflict scenarios / 1,200 queries (to 150K tokens), three dimensions: **State Resolution** (detect a belief is outdated), **Premise Resistance** (reject queries falsely presupposing a stale state), **Implicit Policy Adaptation** (apply the updated state downstream). **Best frontier model only 55.2% accuracy.**
- **Adaptive Memory Crystallization** (`2026arXiv260413085K`) — Liquid–Glass–Crystal consolidation via Itô SDE / Fokker-Planck; **67–80% catastrophic-forgetting reduction, +34–43% forward transfer, −62% memory footprint** on Meta-World MT50 / Atari / MuJoCo (continual-RL metrics imported).
- **Learning to Forget / H2-EMV** (`2026arXiv260411306B`) — LM-driven selective forgetting under learned NL rules: **−45% memory, −35% query compute, no QA-accuracy loss, +70% second-round accuracy.** Evidence forgetting can be net-positive when relevance-driven.
- **Eviction-with-recall:** Cooperative Memory Paging (`2026arXiv260412376L`) — keyword bookmarks make evicted content recoverable.
- **Consolidation systems:** SCM sleep-consolidation + algorithmic forgetting (`2026arXiv260420943S`); Evolve knowledge lifecycle (`2026arXiv260423424H`); NeuSymMS self-curation (`2026arXiv260517596S`); Time-is-Not-a-Label continuous phase rotation for obsolescence (`2026arXiv260411544W`).
- **Stability-plasticity** is quantified only at the mechanism level in deep RL (`2025arXiv250408000L`); LLM-agent papers invoke it narratively. Roadmap: Lifelong Learning of LLM Agents (`2025arXiv250107278Z`).

**How forgetting is measured today (and isn't):** STALE-style 3-dimensional accuracy and AMC-style forward-transfer/forgetting-% (borrowed from continual RL) are the *closest* things to standards — and they are **not shared across papers**. A-MEM, EvolveMem, NeuSymMS, ZenBrain describe consolidation/self-curation/forgetting layers but report only end-task quality, never a forgetting rate, retention curve, or negative-transfer delta. **There is no standardized retention-curve / forgetting-rate protocol for textual agent memory, and no work measures whether pruning removes the *right* stale items (precision/recall of obsolescence decisions).**

---

## 12. Synthesis — prioritized opportunities for the role

1. **A unified, multi-type, white-box memory harness.** Nothing spans semantic+episodic+procedural with stage-attributed diagnostics, explicit scoring targets (TIAP), fixed-answerer controls (EngramaBench), cross-system evaluation (GRAVITY), and reported CIs. Flagship open-source contribution.
2. **The canonical procedural-memory benchmark with frozen splits + attribution controls.** SkillEvolBench/SEA-Eval are days-old and unconsolidated; "Harness Updating ≠ Benefit" names the confound. Adopt the freeze-then-deploy arc + No-Skill/Raw-Trajectory controls.
3. **A forgetting / negative-transfer / obsolescence metric suite.** The thinnest area: STALE (55.2% ceiling) + AMC's continual-RL metrics are the only hard numbers; standardize a retention-curve + obsolescence-precision/recall protocol for textual memory.
4. **A synthetic-data realism metric.** OmniBehavior/REALTALK prove the gap; build a fidelity score + controllable conflict/distractor injection (MemConflict recipe) + QDC auditing.
5. **A shared memory-security harness** (RSR@k / ASR / Benign + over-refusal + cross-user leakage + provenance-violation) covering episodic/procedural stores.
6. **Unbenchmarked production axes:** multi-user isolation, proactive memory use, multimodal long-horizon recall.

---

## Reading order (12 papers)

**Foundations:** CoALA (`2023arXiv230902427S`) · "From Storage to Experience" (`2026arXiv260506716L`) · Survey on Evaluation of LLM-based Agents (`2025arXiv250316416Y`)
**Systems:** Zep/Graphiti (`2025arXiv250113956R`) · A-MEM (`2025arXiv250212110X`) · Mem0 (`2025arXiv250419413C`)
**Procedural:** Voyager (`2023arXiv230516291W`) · Agent Workflow Memory (`2024arXiv240907429Z`) · SkillEvolBench (`2026arXiv260524117L`)
**Methodology (must-reads):** TIAP/"Same Ranking, Different Winner" (`2026arXiv260524060P`) · MemConflict (`2026arXiv260520926T`) · "Harness Updating Is Not Harness Benefit" (`2026arXiv260530621L`)

*Bonus (forgetting):* STALE (`2026arXiv260506527C`).

---

*Bibcodes resolve at `https://ui.adsabs.harvard.edu/abs/<bibcode>` and (arXiv) `https://arxiv.org/abs/<id>`. MemGPT and LoCoMo are cited by name where their bibcode is ambiguous in the mirror; LoCoMo resolves to `2024arXiv240217753M`, MemGPT to `2023arXiv231008560P`.*

---

## 13. Addendum — dense-lane rerun (2026-06-11)

This review was built **2026-06-02, during the lexical-only window** (the INDUS
dense lane was down 2026-05-29 → 06-11). After restoration (Qdrant
`scix_indus_v2_papers_s1`, canary Δ=0.000 vs exact), the 9 theme queries were
re-run hybrid vs lexical-only (`results/litreview_rerun_2026-06-11.md`,
`scripts/litreview_rerun_2026-06-11.py`). Dense reshapes ~half of every
top-15. Graded misses worth folding in:

- **Thesis-level:** `2025arXiv250206975P` — *Position: Episodic Memory is the
  Missing Piece for Long-Term LLM Agents*. Directly on §1's framing.
- **§8 security (was thin):** `2025arXiv250303704D` *A Practical Memory
  Injection Attack against LLM Agents*; `2025arXiv250213172W` *Unveiling
  Privacy Risks in LLM Agent Memory*; `2025arXiv250111739D` *Episodic memory
  in AI agents poses risks that should be studied and mitigated*.
- **§11 forgetting (called "thinnest, highest-leverage" above):** Science
  neuroscience anchors `2016Sci...352..305S` (*A pathway for forgetting*),
  `2019Sci...365R1260S`, `2020Sci...370S1428S` (consolidation), plus
  `2020arXiv200207111U` (adversarial false-memory formation in continual
  learners).
- **§5 benchmark lineage:** `2021arXiv210707567X` *Beyond Goldfish Memory*;
  `2022arXiv221008750B` *Keep Me Updated!* — the pre-wave long-term-dialogue
  benchmarks.
- **§2 architectures:** `2024arXiv240704363A` *AriGraph* (KG world models +
  episodic memory); `2025arXiv250308026T` *Reflective Memory Management for
  Long-term Personalized Agents*.
- **§1 lineage:** `2022arXiv220109305L` (ACT-R vs Soar),
  `2022arXiv221202098K` / `2022arXiv220401611K` (human-like memory-systems
  machines, 2022 — predating the "modern framing did not exist in 2022" claim
  in §0, which should be read as *agentic-LLM* framing specifically).

Method note: dense's systematic contribution here is **recent low-citation
2024–26 papers phrased differently from the query** — exactly the bias this
review flagged in §0 (citation-anchored expansion favors older, cited work).
Core conclusions stand; §8 and §11 change materially.
