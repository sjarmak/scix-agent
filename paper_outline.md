# Agent-Navigable Scientific Knowledge: Graph Intelligence and Semantic Retrieval on the ADS Citation Corpus

## Target Venues

- **Primary**: ADASS 2026 (Astronomical Data Analysis Software and Systems)
- **Secondary**: arXiv preprint, cs.DL + astro-ph.IM
- **Stretch**: JCDL 2027 (Joint Conference on Digital Libraries)

---

## Abstract (~250 words)

Scientific literature grows faster than researchers can navigate it. AI agents are emerging as research tools, but they lack infrastructure: existing search APIs return ranked lists, not navigable knowledge structures. We present an agent-native knowledge layer built on NASA ADS metadata covering 32.4M papers and 299M citation edges spanning 1800-2026. Our system combines three capabilities absent from existing scientific search: (1) corpus-scale graph analytics — PageRank, HITS, and Leiden community detection revealing the structural topology of science; (2) multi-model semantic retrieval — SPECTER2 for document similarity fused with text-embedding-3-large for natural-language queries via reciprocal rank fusion; and (3) agent session state — working-set management that enables systematic literature exploration rather than isolated searches. These capabilities are exposed through a 22-tool MCP (Model Context Protocol) server, enabling AI agents to map research communities, detect literature gaps, and conduct structured reviews. We evaluate retrieval quality, graph topology, and community structure on the full ADS corpus. Ingesting the complete 1800-2026 corpus resolves 99.6% of citation edges (vs. 17.8% for a 6-year window), transforming a fragmented graph into a connected knowledge structure. We discuss implications for ADS infrastructure: precomputed embeddings and graph metrics as data products, and MCP endpoints as a complement to REST APIs. Code and precomputed metrics are released as open-source contributions to the ADS ecosystem.

---

## 1. Introduction (2 pages)

### The navigation problem

- Scientific output doubles every ~12 years; >3M papers/year across disciplines
- Researchers spend 20-30% of time on literature search (cite: Tenopir et al.)
- AI agents can systematically explore, but need structured knowledge, not just search results

### The infrastructure gap

- ADS, Semantic Scholar, Google Scholar: excellent human-facing search
- REST APIs return ranked lists — no graph topology, no session memory, no community structure
- MCP protocol emerging as standard for AI tool use — but no scientific literature MCP exists

### Our contribution

- Agent-native navigation layer on 32.4M ADS papers
- Three layers: graph intelligence, semantic retrieval, session state
- 22 MCP tools enabling agent-driven research workflows
- Open-source, designed for upstream integration into ADS

---

## 2. Related Work (1.5 pages)

### Scientific search systems

- ADS (Kurtz et al. 2000, Accomazzi et al. 2015): Solr-based, citation tracking, myADS
- Semantic Scholar (Ammar et al. 2018): ML-powered, TLDR, citation intent
- Google Scholar: broad coverage, limited API, no graph analytics

### Scientific embeddings

- SPECTER (Cohan et al. 2020): citation-proximity training on Semantic Scholar
- SPECTER2 (Singh et al. 2023): multi-task adapters, improved zero-shot
- SciBERT (Beltagy et al. 2019): scientific pretraining
- Gap: all trained for document-to-document, not query-to-document retrieval

### Citation network analysis

- PageRank on citation graphs (Chen et al. 2007)
- Community detection in science (Waltman & van Eck 2012, Leiden algorithm)
- Bibliographic coupling (Kessler 1963), co-citation analysis (Small 1973)
- Gap: not integrated into search systems; computed offline for bibliometrics, not navigation

### AI agents for research

- Literature review agents (emerging area, 2024-2025)
- Retrieval-augmented generation for scientific QA
- MCP protocol (Anthropic 2024): standard for tool use
- Gap: no MCP server for scientific literature; no agent-optimized knowledge infrastructure

---

## 3. Data & Infrastructure (2 pages)

### 3.1 Corpus

- 32,390,237 papers from NASA ADS, years 1800-2026
- 23.3M with abstracts, 299.3M citation edges
- Harvested via ADS API v1, stored as JSONL (~140GB raw)
- Fields: bibcode, title, abstract, authors, affiliations, keywords, citations, references, arxiv_class, doctype, etc.

### 3.2 PostgreSQL + pgvector architecture

- Single-server PostgreSQL 16 + pgvector 0.8.0+
- No Elasticsearch, no separate vector DB — everything in PostgreSQL
- Tables: papers, citation_edges, paper_embeddings, paper_metrics, extractions, uat_concepts
- Design choice: simplicity over distributed complexity; 62GB RAM suffices

### 3.3 Edge resolution

- **Key finding**: A 6-year window (2021-2026) resolves only 17.8% of citation edges
- Full corpus (1800-2026) resolves 99.6% — from fragmented graph to connected knowledge
- Table: edge resolution by decade of target paper
- Implication: graph analytics on partial corpora are fundamentally misleading

### 3.4 Ingestion pipeline

- Idempotent JSONL → PostgreSQL via COPY, tracked by ingest_log
- 227 files processed in 4 hours; supports .jsonl, .jsonl.gz, .jsonl.xz
- IngestLog pattern: skip-if-complete, resumable on crash

---

## 4. Semantic Retrieval Layer (2.5 pages)

### 4.1 SPECTER2 embeddings

- 768d CLS pooling on "title [SEP] abstract"
- 24M papers embedded (all with title); HNSW index via pgvector
- Strength: encodes citation proximity (papers that cite each other cluster)
- Weakness: trained for doc-to-doc similarity, not query-to-doc retrieval

### 4.2 text-embedding-3-large (dual-model)

- 1024d Matryoshka truncation via OpenAI API
- Strength: trained for asymmetric query-document retrieval
- Halfvec (float16) quantization: ~37GB + ~24GB for dual HNSW indexes

### 4.3 Hybrid search with RRF fusion

- Three-signal fusion: BM25 (tsvector) + SPECTER2 vector + OpenAI vector
- Reciprocal Rank Fusion with k=60
- Circuit breaker: OpenAI failure → graceful fallback to SPECTER2 + lexical

### 4.4 Evaluation

- 50-query benchmark across 4 categories (topical, method-specific, citation-proximal, cross-domain)
- Metrics: Recall@10, NDCG@10, MRR
- Latency: semantic search p95 < 10ms, combined query p95 < 200ms
- Comparison: SPECTER2-only vs OpenAI-only vs fusion vs lexical-only

### 4.5 Embedding as infrastructure (ADS integration discussion)

- Cost of computing embeddings: ~$0 (SPECTER2 local) or ~$65 (OpenAI batch for 5M)
- Cost of NOT sharing: every downstream consumer re-embeds the corpus
- Proposal: ADS serves precomputed embeddings as a data product alongside metadata
- Precedent: Semantic Scholar serves SPECTER embeddings via API

---

## 5. Graph Intelligence Layer (2.5 pages)

### 5.1 Full-graph analytics

- PageRank on 32.4M nodes, 298M edges
- HITS hub/authority scores
- Table: top-20 papers by PageRank (sanity check against known landmarks)

### 5.2 Community detection

- Leiden algorithm on giant component
- Three resolutions: coarse (~20 communities), medium (~200), fine (~2000)
- Calibration methodology: binary search on resolution parameter targeting community count

### 5.3 Parallel community signals

- **Citation community**: Leiden on giant component — "papers that cite each other"
- **Semantic community**: k-means on SPECTER2 embeddings — "papers with similar content"
- **Taxonomic community**: arXiv class + UAT hierarchy — "papers in the same field"
- Each signal has clean semantics; agents choose the appropriate lens
- Coverage: citation ~65% (connected only), semantic ~95% (has embedding), taxonomic ~90%

### 5.4 Giant component analysis

- Before full ingest: ~45K components, 1.7M isolated nodes
- After full ingest: [MEASURE — expected: 1 dominant giant component, >90% of nodes]
- Impact on community quality: [MEASURE — modularity, community size distribution]

### 5.5 Comparison with partial-corpus analytics

- Side-by-side: PageRank on 5M (2021-2026) vs 32M (full corpus)
- Do the same papers rank highest? How does community structure change?
- Key argument: partial-corpus graph analytics produce misleading results

---

## 6. Agent Navigation Interface (2 pages)

### 6.1 MCP tool design

- 22 tools across 5 categories:
  - Search: semantic_search, hybrid_search, lexical_search, faceted_search
  - Paper: get_paper, get_citations, get_references
  - Graph: co_citation_analysis, bibliographic_coupling, citation_chain, temporal_evolution, graph_metrics, explore_community
  - Entity: entity_search, entity_profile
  - Session: add_to_working_set, get_working_set, get_session_summary, find_gaps, clear_working_set, concept_search
- Tool count discipline: premortem identified >15 tools degrades agent selection accuracy

### 6.2 Session state architecture

- In-memory working set per session (stdio transport)
- WorkingSetEntry: bibcode + provenance (source_tool, source_context, tags)
- find_gaps: SQL query finding cross-community bridge papers
- Implicit vs explicit curation: design tradeoff (discussed)

### 6.3 Entity extraction

- LLM-based (Haiku): methods, datasets, instruments, materials
- JSONB storage with GIN index for containment queries
- Pilot validation: precision/recall on gold-standard set
- Cost analysis: $40 pilot (10K papers), $500-$17K full corpus (fine-tune vs LLM)

### 6.4 Agent workflow demonstration

- Concrete example: "Map the research landscape of exoplanet atmospheres"
- Agent transcript showing multi-tool composition
- Steps: hybrid_search → explore_community → find_gaps → citation_chain → temporal_evolution
- Comparison: same task attempted with ADS API-only agent

---

## 7. Evaluation (2 pages)

### 7.1 Retrieval benchmarks

- Table: latency p50/p95/p99 for each search type
- Table: recall@10 comparison across models and fusion strategies
- Ground truth: ADS search results as relevance proxy

### 7.2 Graph metrics validation

- PageRank sanity check: do highly-cited papers rank highest?
- Community coherence: do communities align with arXiv class labels?
- Parallel signal agreement: how often do citation/semantic/taxonomic communities agree?

### 7.3 Edge resolution impact

- Before/after: graph diameter, component count, giant component coverage
- Community detection quality vs corpus completeness

### 7.4 Agent task evaluation

- 10 research tasks (qualitative)
- Metrics: papers found, coverage breadth, gap identification, time to completion
- Comparison: MCP-equipped agent vs ADS-API-only agent

---

## 8. Infrastructure Implications for ADS (1.5 pages)

### 8.1 Embeddings as a data product

- SPECTER2 embeddings: compute once, serve to all consumers
- Options: bulk download, API endpoint, or pgvector-backed search
- Precedent: Semantic Scholar serves embeddings; ADS would be the first for astronomy
- Update model: embed on ingest (same-day), rebuild HNSW index weekly

### 8.2 Graph metrics as a data product

- PageRank, HITS, community assignments as columns on paper records
- Weekly batch recomputation (full graph in ~30 min on adequate hardware)
- Expose via existing ADS API: add `pagerank`, `community_id` to search results
- Enable: "sort by influence" in the ADS UI, community browsing

### 8.3 MCP server as a service

- ADS hosts an MCP endpoint alongside REST API
- Agents connect directly — no need to harvest and re-index
- Real-time: new papers get embeddings and graph metrics on ingest
- Session state: server-side working sets for multi-turn agent research

### 8.4 What changes for ADS

- New compute: GPU for embedding (one-time + incremental), CPU for weekly graph batch
- New storage: paper_embeddings table (~80GB for SPECTER2 + OpenAI), paper_metrics (~5GB)
- New endpoint: MCP server (Python process, minimal infrastructure)
- Unchanged: Solr search, existing REST API, myADS, ORCID integration

---

## 9. Discussion (1 page)

### What works

- Full-corpus ingestion transforms graph quality (82% → 0.4% dangling)
- Hybrid search compensates for individual model weaknesses
- Parallel community signals give agents multiple navigation lenses

### Limitations

- Embedding 24M+ papers takes days on a single GPU; ADS infrastructure would amortize this
- Leiden on 32M nodes needs >43GB RAM; requires memory management or distributed approach
- Entity extraction quality varies by subfield; not yet validated at scale
- Session state design assumes agent curation that agents may not spontaneously perform

### Premortem learnings

- Cost estimates for LLM extraction were 3-5x too low (prompt overhead)
- Pilot cohorts biased toward high-citation papers don't represent the long tail
- Tool count matters: >15 tools degrades agent tool selection accuracy

---

## 10. Conclusion & Future Work (0.5 pages)

### Summary

- First MCP server for scientific literature navigation
- 32.4M papers, 299M edges, 22 tools, 3 community signals
- Designed for upstream integration into ADS/SciX

### Future work

- Full-text embeddings (not just abstracts) — requires ADS full-text access
- Cross-disciplinary expansion (beyond astro-ph to all of ADS)
- Agent benchmark suite as a shared resource
- Incremental graph updates (streaming PageRank, online community detection)

---

## Appendices

### A. MCP Tool Specification

Complete JSON schema for all 22 tools.

### B. SQL Schema

Migrations 001-009.

### C. Benchmark Results

Full latency and recall tables.

### D. Premortem Risk Analysis

Condensed from the 3-agent premortem (infrastructure, agent utility, data quality).

### E. Reproduction

Instructions for harvesting, ingesting, embedding, and running the MCP server.
Open-source repository: [URL]

---

## Figures

1. Architecture diagram (corpus → PostgreSQL → MCP server → agent)
2. Edge resolution curve (% resolved vs years of corpus ingested)
3. Community structure visualization (coarse communities, colored by arXiv class)
4. Embedding space t-SNE/UMAP (SPECTER2, colored by community)
5. Retrieval comparison (bar chart: recall@10 across models)
6. Agent workflow transcript (annotated screenshot or sequence diagram)
7. PageRank distribution (log-log plot, power law)
8. Parallel community signal agreement matrix (Sankey or confusion matrix)
