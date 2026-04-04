# PRD: Entity Extraction and Normalization for Cross-Community Methodology Transfer

_Revised after 5-agent diverge, 30-idea brainstorm, 3-agent converge debate, and 5-agent premortem._

## Problem Statement

The SciX agent-navigable knowledge layer has 32.4M papers with embeddings, citation graph analytics, and community structure — but no structured entity data. The goal is to enable agents to answer questions like "which methods from community A are unused in community B?" This requires not just extracting entities from text, but normalizing them into a canonical vocabulary that supports cross-corpus aggregation.

The current extraction approach (Claude Haiku with tool-use, `src/scix/extract.py`) costs ~$40 for 10K papers but $500-$17K at full corpus scale. More critically, it produces raw strings with no normalization — "MCMC", "Markov chain Monte Carlo", and "MCMC sampling" would be three separate entities, making cross-community queries useless.

Three rounds of analysis converge on two key insights:

1. **Normalization is harder than extraction**, and the normalization architecture must be designed first — not as an afterthought.
2. **The existing infrastructure (SPECTER2 + Leiden communities + existing metadata) may already answer the cross-community transfer question without entity extraction.** This must be tested before committing to extraction.
3. **Three of four entity types are enumerable** and solvable with dictionary matching. Only methods require model-based extraction — and methods should be discovered statistically (TF-IDF), then validated, not extracted blind.

## Tensions Identified and Resolved

### Tension 1: Do we need entity extraction at all?

The converge debate reached consensus: **embeddings for discovery, entities for explanation.** SPECTER2 cross-community similarity can surface methodology transfer candidates (discovery), but agents need named entities to explain what they found (explanation). Both are needed. The question is sequencing — test the embedding baseline first.

**Decision**: Run the "Parallel Evidence Sprint" (see Architecture) to generate evidence from both tracks simultaneously. Entity extraction proceeds only if the embedding baseline demonstrably fails to surface known transfers, or if the 10K pilot reveals high-value entities that TF-IDF misses.

### Tension 2: Entity types — revised taxonomy

The converge debate resolved this unanimously: **replace "materials" with "software packages."**

- Software packages (astropy, emcee, stan) are the most enumerable, most transferable, and most easily extracted entity type. They appear in references, acknowledgments, and DOIs.
- Materials are rarely the transferable element across astronomy communities.
- ASCL.net provides 3,200+ entries with canonical names, aliases, and Zenodo DOIs.

**Revised taxonomy**: methods, instruments, datasets, software packages.

**Decision**: Split by enumerability. Instruments + datasets + software are CLOSED-SET — use dictionary matching from IVOA, NASA, ASCL.net registries. Zero LLM cost, >95% precision. Methods are OPEN-SET — discover with community-differential TF-IDF, validate with the 10K pilot, scale extraction only where needed.

### Tension 3: Sequential vs parallel phasing

The converge debate reached strong consensus: **Phase 0 and Phase 1 have zero dependencies and must run in parallel.** The original PRD's sequential gating wasted a week for no technical reason.

**Decision**: Run all four evidence tracks simultaneously in a 1-week "Parallel Evidence Sprint."

### Tension 4: Normalization architecture

The premortem identified a critical flaw: **SPECTER2 document embeddings cannot be used for entity-phrase deduplication** (document similarity ≠ entity-string similarity). This is a category error that would corrupt the normalization pipeline.

**Decision**: Use a phrase-level sentence transformer (all-MiniLM-L6-v2 or similar) for entity string embeddings, NOT SPECTER2. Validate on known synonym pairs before committing to dedup architecture. Normalization is scoped to methods only — instruments/datasets/software arrive pre-normalized from registries.

### Tension 5: Graph propagation risk

The premortem flagged that graph propagation as the PRIMARY coverage mechanism amplifies every upstream error 10x. Hub papers (reviews with 500+ citations) contaminate propagation across community boundaries.

**Decision**: Treat graph propagation as a **scoring signal**, not an assignment mechanism. Propagated entities get confidence scores and require lightweight validation before being written as assignments. Extract from a larger seed set (bridge papers + cluster exemplars) to reduce dependence on propagation quality.

### Tension 6: TF-IDF terms vs typed entities

The converge debate surfaced a key empirical question: what fraction of community-distinctive TF-IDF terms are actually methods? If >80% are methods, statistical discovery suffices. If <50%, semantic typing via extraction is essential.

**Decision**: The 10K Haiku pilot serves a dual purpose: (1) open-ended extraction for ground truth, and (2) classification of top TF-IDF candidates as method/non-method. This answers the question empirically in Week 1.

## Goals & Non-Goals

### Goals

- Enable cross-community methodology transfer discovery queries
- Extract and normalize methods, instruments, datasets, and software packages from paper abstracts
- Build a canonical entity vocabulary with alias resolution
- Process the full 32M-paper corpus at <$500 total cost
- Design for incremental updates (new papers get entities on ingest)
- Validate baselines before committing to extraction at scale

### Non-Goals

- Full-text extraction at corpus scale (Phase 2+ enhancement)
- Relation extraction (method USED-FOR task) — entity extraction first
- Building a complete scientific methods ontology — emergent vocabulary, not top-down
- Real-time extraction during agent sessions (batch pipeline)

## Requirements

### Must-Have

- **Parallel Evidence Sprint (Week 1)**: Four parallel tracks producing comparable evidence
  - Acceptance: All four tracks complete within 7 days. Decision gate document produced comparing results across tracks with recommendation for Phase 2 scope.

- **Track A — Dictionary matching for closed-set entities**: Registry matching across 32M abstracts
  - Acceptance: ≥5,000 entries from IVOA + NASA + ASCL.net registries (instruments, datasets, software). Matching completes in <2 hours. Precision ≥0.95 on 200-paper sample. Results stored in `entity_mentions` with FK to canonical entities.

- **Track B — Community-differential TF-IDF**: Statistical method discovery
  - Acceptance: Top-100 distinctive n-grams computed for each Leiden community (medium resolution). Output includes frequency ratio (community vs corpus). Evaluated against 10K pilot: what fraction of top TF-IDF terms are actual methods?

- **Track C — Cross-community SPECTER2 similarity baseline**: Embedding-based transfer detection
  - Acceptance: Given a paper or search term in community A, returns top-10 semantically similar papers from community B with <200ms latency. Tested on 10 known methodology transfer cases with ≥7/10 surfaced.

- **Track D — 10K Haiku pilot with dual purpose**: Ground truth extraction + TF-IDF validation
  - Acceptance: 10K papers extracted with normalization-aware schema (canonical_name field). Precision ≥0.85 and recall ≥0.70 on 100-paper manually reviewed subset. Top 500 TF-IDF candidate terms classified as method/non-method.

- **Entity normalization schema**: PostgreSQL tables with phrase-level embeddings (NOT SPECTER2)
  - Acceptance: `entities` table seeded from registries. `entity_mentions` with FK. `entity_aliases` mapping raw strings to canonical IDs. Dedup validated on 50 known synonym pairs with ≥90% accuracy using all-MiniLM-L6-v2 (not SPECTER2).

### Should-Have

- **Bridge paper extraction at scale**: GLiNER or fine-tuned model on ~3M cross-community papers
  - Acceptance: Processing rate ≥15K papers/hour. F1 ≥0.50 on 500-paper evaluation set. Gated on: (1) Leiden OOM resolved, (2) canary batch of 500 papers validates extraction quality, (3) Week 1 decision gate recommends scaling.

- **Graph propagation as scoring signal**: Confidence-scored entity propagation through citation edges
  - Acceptance: Propagated entities include confidence score (0-1). Hub paper contamination mitigated (review papers weighted down). Spot-check: ≥70% of propagated assignments agree with direct extraction on 1K validation set.

- **Fine-tuned local model**: QLoRA fine-tune on pilot output
  - Acceptance: F1 improvement ≥0.10 over zero-shot GLiNER. Training <8 hours on RTX 5090. Throughput ≥20K papers/hour.

### Nice-to-Have

- **Cross-community method transfer MCP tool**: Agent-facing tool combining entity data with community structure
  - Acceptance: Returns methods ranked by (frequency_in_source / frequency_in_target). Latency <500ms.

- **Methods-section targeted extraction**: Full-text methods sections only
  - Acceptance: Section detection ≥85% accuracy. Token reduction ≥90% vs full body. F1 improvement ≥0.05 over abstract-only.

- **Temporal method diffusion tracking**: When did method X cross from community A to B?
  - Acceptance: Given a method entity, returns timeline of first appearance per community with year granularity.

## Design Considerations

### Key Trade-offs

1. **Enumerable vs open-ended entities**: 3 of 4 types (instruments, datasets, software) are closed-set and solvable with dictionary matching. Only methods require model-based extraction. Architecture must treat them differently.

2. **Discovery vs explanation**: Embeddings discover transfer candidates (unnamed). Entities explain them (named). Both needed, but discovery should precede and scope the explanation effort.

3. **Normalization granularity**: Start with flat alias table. Add hierarchy only if flat resolution proves insufficient for cross-community queries.

4. **Propagation risk**: Graph propagation amplifies errors. Use as scoring signal, not assignment mechanism. Hub papers (reviews) must be weighted down.

### Architecture: Parallel Evidence Sprint

```
Week 1: Parallel Evidence Sprint ($15, all tracks independent)
  ├─ Track A: Dictionary match instruments/datasets/software (IVOA+NASA+ASCL.net) → 32M papers
  ├─ Track B: Community-differential TF-IDF → distinctive n-grams per Leiden community
  ├─ Track C: Cross-community SPECTER2 similarity → embedding-based transfer baseline
  └─ Track D: 10K Haiku pilot → ground truth extraction + TF-IDF term classification

  → DECISION GATE: Compare all four tracks. Three outcomes:
     1. Baselines sufficient → skip model-based method extraction for now
     2. Extraction adds clear value → proceed to Phase 2 on bridge papers
     3. Mixed → run GLiNER on 50K sample to break tie

Phase 2: Targeted Scale (2-4 weeks, ~$50-200, GATED on Week 1 results + Leiden OOM fix)
  ├─ Prerequisite: Solve Leiden OOM (streaming algorithm or external-memory)
  ├─ Prerequisite: Canary batch (500 papers) validates extraction quality
  ├─ GLiNER or fine-tuned model on ~3M bridge papers
  ├─ Graph propagation as SCORING signal (confidence scores, hub dampening)
  ├─ Phrase-level embedding dedup (all-MiniLM, NOT SPECTER2)
  └─ Cascade for low-confidence: local LLM → cloud LLM

Phase 3: Enrich (2-4 weeks, ~$50-100)
  ├─ Methods-section targeted extraction from full text
  ├─ Temporal method diffusion tracking
  ├─ Cross-community method transfer MCP tool
  └─ Shared-reference clustering (papers citing same methodology paper)

Phase 4: Research (ongoing, $0)
  ├─ Sparse autoencoder on SPECTER2 (interpretable method features)
  ├─ Author mobility tracking (researchers carry methods between communities)
  └─ Contrastive abstract pairs (same foundation, different communities)
```

### Key architectural shifts from v1

1. **Parallel Evidence Sprint replaces sequential phasing.** The converge debate unanimously agreed: Phase 0 and Phase 1 have zero dependencies. Running them in parallel produces 4x the evidence in the same wall-clock time.

2. **Extract from high-value subset, propagate to corpus.** Bridge papers (~3M) are the extraction targets, not all 32M. Graph propagation amplifies sparse extraction — but as a scoring signal, not assignment mechanism (premortem mitigation).

3. **Dictionary matching covers 3 of 4 entity types.** Instruments, datasets, and software packages are closed-set. Only methods require model-based extraction. This is the hybrid-pragmatist insight that all three converge debaters agreed on.

4. **TF-IDF discovers, extraction validates.** Community-differential TF-IDF surfaces method candidates statistically. The 10K pilot validates which candidates are actually methods. This inverts the traditional approach (extract first, analyze later) and is more cost-effective.

5. **Phrase-level embeddings for normalization, not SPECTER2.** The premortem caught that SPECTER2 document embeddings can't deduplicate entity phrases. A sentence transformer (all-MiniLM-L6-v2) is the correct embedding model for entity strings.

## Premortem Risk Registry

| #   | Risk                                                                                                   | Severity | Likelihood | Score | Mitigation                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------ | -------- | ---------- | ----- | ------------------------------------------------------------------------------------------------------------- |
| 1   | **Dependency breakage**: GLiNER/Haiku API/CUDA silent changes corrupt extraction                       | Critical | High       | 12    | Pin model weights (SHA), snapshot registries locally, canary batch before every scale run                     |
| 2   | **Operational cascade**: OOM kills PostgreSQL, loses UNLOGGED table, multi-day GPU job crashes         | Critical | High       | 12    | Disable auto-reboot, per-1K checkpointing, cgroup memory limits, convert UNLOGGED immediately after bulk load |
| 3   | **Wrong taxonomy**: Entity types don't match what astronomers find useful for transfer discovery       | Critical | High       | 12    | 50-paper annotation with domain experts before entity type selection; "usefulness gate" before Phase 2        |
| 4   | **Hub contamination in propagation**: Reviews spread noisy labels globally across community boundaries | Critical | High       | 12    | Propagation as scoring (not assignment); hub dampening; larger seed set reduces propagation dependence        |
| 5   | **SPECTER2 ≠ entity dedup**: Document embeddings can't cluster entity phrases                          | Critical | Medium     | 8     | Use phrase-level model (all-MiniLM); validate on known synonyms before committing                             |
| 6   | **Leiden OOM unresolved**: Bridge paper identification requires community boundaries that don't exist  | High     | High       | 9     | Solve before Phase 2: streaming Louvain (~12GB), or external-memory, or 128GB+ machine as one-time job        |
| 7   | **Seed vocabulary too sparse**: UAT/IVOA/ASCL cover <50% of entity surface area                        | High     | Medium     | 6     | Cluster-first schema (extract raw, cluster, assign canonical post-hoc) instead of vocabulary-first            |

### Mitigations built into architecture

- **Canary batch (500 papers)** before every scale run — catches silent model/API breakage
- **Usefulness gate** between Week 1 and Phase 2 — validates that entities enable actionable transfer queries
- **Phrase-level embeddings** for entity dedup — addresses the SPECTER2 category error
- **Propagation as scoring** — prevents hub contamination from corrupting assignments
- **Parallel Evidence Sprint** — produces evidence for decision-making before major investment
- **Pinned dependencies** — model weights (SHA hash), registry snapshots (local JSONL), locked PyTorch/CUDA versions

## Cost Summary

| Phase                   | Duration      | API Cost         | GPU Cost | What You Get                                                                                                                 |
| ----------------------- | ------------- | ---------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Week 1: Evidence Sprint | 5-7 days      | ~$15 (Haiku 10K) | ~$5      | Dictionary entities for 3 types across 32M papers, TF-IDF method candidates, embedding baseline, ground truth, decision gate |
| Phase 2: Scale          | 2-4 weeks     | ~$50-200         | ~$20     | 3M bridge papers extracted, propagation scores for 32M, normalized entity vocabulary                                         |
| Phase 3: Enrich         | 2-4 weeks     | ~$50-100         | ~$20     | Full-text methods, temporal tracking, transfer discovery MCP tool                                                            |
| **Total**               | **5-9 weeks** | **~$115-315**    | **~$45** | **Normalized entity vocabulary for 32M papers**                                                                              |

Note: Phase 2 is GATED on Week 1 results. If baselines are sufficient, Phase 2 scope shrinks or is deferred. Total timeline and cost may be significantly lower.

## Open Questions

### Resolved by converge debate

- ~~Do we need entity extraction at all?~~ → Both embeddings and entities are needed. Embeddings for discovery, entities for explanation. Test baselines first.
- ~~Should Phase 0 and Phase 1 be sequential?~~ → No. Run in parallel. Zero dependencies between them.
- ~~Is the entity taxonomy right?~~ → Replace "materials" with "software packages." Three of four types are enumerable.

### Resolved by premortem

- ~~Can SPECTER2 be used for entity dedup?~~ → No. Use phrase-level sentence transformer. Document ≠ entity similarity.
- ~~Is graph propagation safe as primary coverage?~~ → No. Use as scoring signal with confidence thresholds and hub dampening.

### Still open

1. **Does SPECTER2 cross-community similarity surface methodology transfer?** The single highest-leverage experiment. Testable in Week 1.
2. **What fraction of TF-IDF distinctive terms are actual methods?** Determines whether statistical discovery suffices or semantic typing is essential. Answered by the dual-purpose Haiku pilot.
3. **What F1 does GLiNER achieve zero-shot on astronomy entities?** No benchmark exists. The 10K pilot provides ground truth. If GLiNER >0.6 F1, fine-tuning may be unnecessary.
4. **What fraction of papers are cross-community bridge papers?** Defines Phase 2 extraction scope. Must be measured on live graph after Leiden OOM is resolved.
5. **Can the Leiden OOM be solved?** Blocking prerequisite for Phase 2. Options: streaming Louvain, external-memory igraph, or rent a 128GB machine for a one-time computation.

## Research Provenance

### Diverge (5 agents)

- **Prior Art**: astroBERT, WIESP2022-NER/DEAL, GLiNER, Dagdelen et al. 500-example threshold, Astro-NER F1=0.51
- **Cost Architecture**: 4-tier cascade, $5K all-Haiku vs $50-200 cascaded, all-Haiku cheaper than expected
- **Normalization**: No methods ontology exists, SIMBAD-style alias architecture, normalization-during-extraction
- **Local LLM**: RTX 5090 at 5,841 tok/s, Qwen2.5-14B-GPTQ-Int4, QLoRA fine-tuning feasible
- **Contrarian**: Existing embeddings may solve the problem; normalization is the real bottleneck

### Brainstorm (30 ideas, all rated)

Top 5: #8 Bridge paper interrogation (15/15), #4 Community-differential TF-IDF (14/15), #1 Citation-graph propagation (13/15), #23 Multi-abstract consensus (13/15), #2 SPECTER2 neighborhood voting (12/15). Key pattern: 6 of top 12 ideas require zero model inference.

### Converge debate (3 agents, 2 rounds)

- **Embedding-First**: Won on sequencing (baselines first) and the normalization risk argument
- **Extraction-First**: Won on explainability (agents need named entities) and the bridge-paper insight
- **Hybrid-Pragmatist**: Won on taxonomy fix (software > materials), parallel execution, and the TF-IDF-as-discovery framing
- **Consensus**: Parallel Evidence Sprint in Week 1, dictionary matching for 3/4 types, TF-IDF discovers + pilot validates, extraction gated on evidence

### Premortem (5 agents)

- **Technical**: SPECTER2 for entity dedup is a category error → use phrase-level model
- **Dependency**: No version pinning = silent breakage at scale → pin everything, canary batches
- **Operational**: Solo operator + multi-day jobs + no monitoring → cgroups, checkpointing, disable auto-reboot
- **Scope**: Taxonomy never validated with astronomers → usefulness gate before scaling
- **Scale**: Hub contamination in propagation → scoring signal, not assignment; larger seed set

Full brainstorm session: `.brainstorm/cc4545895ddd/`
