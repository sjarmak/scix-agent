# PRD: Entity Extraction v3 — Refined Pipeline with Risk Mitigations

## Problem Statement

The 10K Haiku pilot demonstrated that LLM-based entity extraction works but has systematic quality issues: ~25% entity duplication from normalization gaps, cross-type leakage (methods as instruments), noisy materials category dominated by generic substances, and cohort bias toward cross-disciplinary papers rather than astronomy. Meanwhile, divergent research revealed that (a) ADS metadata fields already cover most closed-set entities, (b) purpose-built NER models from the ADS team exist on HuggingFace, (c) OpenAlex provides free topic enrichment via DOI linking, and (d) the existing citation intent graph is already a knowledge graph that flat entity lists should integrate with, not duplicate.

The premortem exercise identified critical risks: PostgreSQL overload from mixing extraction writes with MCP reads, dependency fragility from unpinned HuggingFace models, operational gaps (no checkpointing/disk budgeting), and scope failure from building extraction infrastructure before validating that agents need it.

## Goals & Non-Goals

### Goals

- Provide agents with structured entity data (instruments, datasets, software, methods) that enhances navigation beyond what keyword search already offers
- Integrate entity data with the existing citation graph and community structure
- Maintain sub-second MCP query latency throughout extraction/normalization
- Stay within $1K API budget for LLM-based extraction

### Non-Goals

- Building a standalone entity search engine (entities augment graph navigation, not replace it)
- Extracting from all 32M papers (target astronomy corpus; expand only if validated)
- Perfect entity normalization (80/20 rule: deterministic normalization first, defer expensive dedup)
- Replacing ADS metadata curation (use metadata as primary source, NER fills gaps only)

## Requirements

### Must-Have (Phase 0: Validate Before Building)

- **Instrument MCP tools with query logging**
  - Acceptance: After 2 weeks, produce a report showing top 50 entity queries by agents, query failure rates, and which entity types are requested vs. available

- **ADS metadata coverage analysis**
  - Acceptance: For 1,000 sample astronomy papers, compare entities available in facility[]/data[]/keyword_norm[] vs. NER extraction. Report the delta — what NER adds that metadata doesn't

- **OpenAlex DOI linking**
  - Acceptance: `SELECT count(*) FROM papers WHERE openalex_id IS NOT NULL` returns >50% of papers with DOIs; topic labels queryable via MCP tool

### Must-Have (Phase 1: Low-Cost, High-Value)

- **Deterministic string normalization**
  - Acceptance: Running normalization on 10K pilot extractions reduces unique entity count by >30%; "density functional theory" and "density-functional theory" resolve to same canonical form

- **Entity specificity scoring**
  - Acceptance: Entities with specificity score below threshold (e.g., "water", "carbon") are auto-flagged; top-100 most frequent entities reviewed and classified as keep/filter

- **Pin all HuggingFace models to commit SHAs with local cache**
  - Acceptance: `models/` directory contains cached weights; pipeline runs successfully with network disconnected

- **Paper-level checkpointing in extraction pipeline**
  - Acceptance: Kill extraction at 50% progress, restart, verify it resumes from checkpoint (not from beginning)

- **Batches API idempotency + cost accumulator**
  - Acceptance: Submitting same cohort twice does not create duplicate extractions; pipeline halts at 80% budget with clear message

### Should-Have (Phase 2: Targeted NER)

- **Evaluate nasa-smd-ibm-v0.1_NER_DEAL on WIESP2022-NER test set**
  - Acceptance: F1 scores reported per entity type; comparison with Haiku extraction on same 500-paper sample

- **Run NER only on papers where ADS metadata has gaps**
  - Acceptance: NER cohort is <40% of full astronomy corpus; coverage report shows NER fills >70% of metadata gaps

- **Separate staging schema for extraction writes**
  - Acceptance: Extraction pipeline writes to `staging.extractions`; production `public.extractions` only receives batch-loaded canonical results; MCP tool latency unchanged during extraction runs

- **Entity provenance tracking**
  - Acceptance: Each entity record includes source (metadata/NER/LLM/OpenAlex), confidence tier, and extraction version; MCP tool can filter by provenance

### Nice-to-Have (Phase 3: Graph Integration)

- **Entity inheritance through citation chains**
  - Acceptance: For papers where entity X has method-intent citation, propagated entities are stored with provenance="citation_propagation" and lower trust tier

- **Cross-community differential TF-IDF on entities**
  - Acceptance: For each Leiden community, top-10 distinctive entities are computed and queryable via MCP

- **Contrastive entity pair extraction**
  - Acceptance: Pairs like (MCMC, compared_with, nested_sampling) stored for papers that compare methods

- **SIMBAD/NED object crossmatch**
  - Acceptance: Celestial object entities resolve to SIMBAD identifiers with canonical names

## Design Considerations

### Execution Order (Risk-Informed)

The premortem revealed that the biggest risk is building before measuring. The plan is explicitly sequenced:

1. **Instrument** (2 weeks) — Log MCP queries to understand what agents need
2. **Validate** (1 week) — ADS metadata coverage analysis + OpenAlex DOI linking
3. **Normalize** (1 week) — Deterministic normalization on existing 10K pilot data
4. **Evaluate** (1 week) — Head-to-head: ADS metadata vs. NER vs. Haiku on 500 papers
5. **Decide** — Based on instrumentation + evaluation, scope the NER investment
6. **Extract** (2-4 weeks) — Run targeted NER only where needed, with operational guardrails
7. **Integrate** (2 weeks) — Graph integration features from Phase 3

### Database Architecture

Write-heavy extraction workloads MUST be isolated from read-heavy MCP queries:

- Extraction + normalization in `staging` schema
- Batch-load canonical results into `public` on completion
- Partition `extraction_entity_links` by entity_type from day one
- Cap similarity blocking at 200 entities per block

### Dependency Management

- All HuggingFace models pinned to commit SHA, cached in `models/`
- `transformers` and `torch` pinned to exact versions
- Nightly canary test: 10 papers through each NER model, 10-record Batches API submission, 10 OpenAlex DOI lookups
- Adapter interfaces for each external dependency

### Operational Guardrails

- Disk budget calculation before any full run (estimate: extractions JSONB + indexes + WAL)
- WAL management: `max_wal_size` bounded, `wal_compression` enabled
- Paper-level checkpoints every 10K papers
- Batches API cost accumulator with 80% budget circuit breaker
- Staged rollout: validate at 1M papers before committing to full corpus

## Open Questions

- What is the actual F1 of nasa-smd-ibm-v0.1_NER_DEAL on WIESP2022-NER test split?
- What % of ADS papers have DOIs that match OpenAlex work IDs?
- Are Leiden communities populated? (community_id columns may be NULL)
- Should materials category be replaced with software (aligns with ASCL + WIESP2022-NER)?
- What additional models exist at huggingface.co/adsabs beyond v0.1?

## Research Provenance

### Divergent Research (5 independent agents)

1. **Prompt Engineering**: Found few-shot examples actively teach wrong behavior; proposed exclusion criteria and canonical_name field
2. **Normalization**: Designed 3-stage pipeline; discovered UAT covers only topics, not entities; deterministic normalization handles bulk of 25% duplication
3. **Cohort Selection**: Quantified bias (astronomy = 9.3% of corpus); proposed stratified sampling at $100/100K papers
4. **Alternative Architectures**: Found astroBERT-NER-DEAL and nasa-smd-ibm models; token-classification beats generative LLMs at NER
5. **Contrarian**: The project already builds a knowledge graph via citation contexts + intent; OpenAlex provides 26B free triples; flat entity lists may not serve agent needs

### Brainstorm (30 ideas, top 7 by score ≥13/15)

| Score | Idea                                        | Why it matters                                                        |
| ----- | ------------------------------------------- | --------------------------------------------------------------------- |
| 14/15 | Entity type from section context            | Section position resolves cross-type ambiguity without classification |
| 14/15 | Entity inheritance through citation chains  | Propagate entities through method-intent citations                    |
| 14/15 | Entity specificity scoring                  | Information-theoretic noise filter                                    |
| 13/15 | Bibgroup as ground truth labels             | Free evaluation set for instrument extraction                         |
| 13/15 | Cross-community differential TF-IDF         | Graph structure defines what entities matter per community            |
| 13/15 | Entity extraction via paper metadata fields | ADS already curates facility/data/keywords                            |
| 13/15 | Entity provenance chain                     | Trust tiers enable quality-aware agent queries                        |

### Premortem (5 failure narratives, all Critical/High)

| Failure Lens           | Root Cause                                             | Top Mitigation                             |
| ---------------------- | ------------------------------------------------------ | ------------------------------------------ |
| Technical Architecture | Normalization quadratic blocking in shared PG instance | Separate staging DB; cap block sizes       |
| Integration/Dependency | Unpinned HF models and API changes                     | Pin SHAs locally; nightly canary tests     |
| Operational            | No disk budget, no checkpointing, no batch idempotency | Operational guardrails before any full run |
| Scope/Requirements     | Never measured agent needs before building             | Instrument MCP tools; ship OpenAlex first  |
| Scale/Evolution        | Single PG for everything; no partition strategy        | Partition by entity type; load test at 1M  |

### Key Convergence

All 5 premortem agents rated their failure as Critical/High, and 3 of 5 independently identified PostgreSQL as the single point of failure. The scope agent's insight — "instrument before building" — was the most impactful resequencing recommendation, shifting the plan from build-first to validate-first.
