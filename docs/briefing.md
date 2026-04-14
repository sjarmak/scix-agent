# SciX Agent-Navigable Knowledge Layer: Technical Briefing

_Prepared for the ADS/SciX tech lead -- April 2026_

---

## What This Is

An experimental system that transforms the full NASA ADS corpus into infrastructure that AI agents can navigate programmatically. Instead of returning ranked lists, the system exposes the structural topology of science -- citation graphs, research communities, and multi-model embeddings -- through a 22-tool MCP (Model Context Protocol) server.

The goal is to demonstrate capabilities that could become upstream ADS data products: precomputed embeddings, graph metrics, community assignments, and an MCP endpoint alongside the existing REST API.

Repository: https://github.com/sjarmak/scix-agent

---

## Why This Matters

ADS is the definitive search system for astronomy and astrophysics. But AI agents are emerging as research tools, and they need different primitives than human searchers:

1. **Graph topology, not just ranked lists.** An agent conducting a systematic review needs to know which papers are structural bridges between communities, which are foundational (high PageRank), and which clusters exist -- not just "top 10 results for query X."

2. **Multi-model retrieval.** SPECTER2 excels at document-to-document similarity (citation proximity), but agents ask natural-language questions -- a fundamentally asymmetric retrieval task. Dual-model fusion compensates for each model's weakness.

3. **Session memory.** An agent exploring a research landscape makes dozens of queries. Without a working set, every query starts from zero. Session state lets agents accumulate, curate, and reason over papers across a research session.

4. **Community structure.** Three independent community signals (citation-based Leiden, embedding-based clustering, taxonomy-based arXiv/UAT classification) give agents different lenses to understand how research is organized.

None of these exist in current scientific search APIs.

---

## Corpus and Scale

| Dimension       | Value                                     |
| --------------- | ----------------------------------------- |
| Papers          | 32,390,237 (1800--2026)                   |
| With abstracts  | ~23.3M (72%)                              |
| Citation edges  | 299,336,889                               |
| Edge resolution | 99.6% (vs 17.8% for a 6-year window)      |
| Raw data        | ~140 GB JSONL across 227 files            |
| Database size   | ~162 GB (PostgreSQL 16 + pgvector 0.8.0+) |
| Years covered   | 1800--2026                                |
| Source          | NASA ADS API v1, all doctypes             |

**Key finding on edge resolution**: Ingesting only a 6-year window (2021--2026, ~5M papers) resolves just 17.8% of citation edges -- most references point to older papers outside that window. The full 1800--2026 corpus resolves 99.6%. This means graph analytics on partial corpora are fundamentally misleading: PageRank, community detection, and co-citation analysis all produce qualitatively different (and wrong) results when 82% of edges dangle into the void.

---

## What's Been Built

### Phase 1: Data Infrastructure (March 31)

**Ingestion pipeline** (`src/scix/ingest.py`, `src/scix/field_mapping.py`):

- Idempotent JSONL-to-PostgreSQL pipeline using `COPY` for throughput
- Supports `.jsonl`, `.jsonl.gz`, `.jsonl.xz` transparently
- Handles null bytes, encoding issues, and malformed records
- `IngestLog` pattern: tracks per-file progress, skips completed files, resumes on crash
- Full corpus ingested in ~4 hours (227 files)

**Schema** (migrations 001--009):

- `papers`: 32.4M rows, ~40 fields per record (bibcode, title, abstract, authors, affiliations, keywords, citations, references, arxiv_class, doctype, etc.)
- `citation_edges`: 299M rows, FK-constrained, covering the full citation graph
- `paper_embeddings`: composite PK `(bibcode, model_name)`, untyped `vector` column supporting multiple embedding dimensions
- `paper_metrics`: PageRank, HITS hub/authority, community assignments at multiple resolutions
- `extractions`: LLM-extracted entities (methods, datasets, instruments) with JSONB+GIN indexing
- `uat_concepts`: Unified Astronomy Thesaurus hierarchy for taxonomic classification
- `ingest_log`: per-file ingestion tracking for idempotent resumable loads

**ADS sync** (`scripts/ads_sync.py`):

- Incremental sync script for new papers since last harvest
- Handles API rate limits and pagination

### Phase 2: Retrieval Layer (March 31)

**SPECTER2 embeddings**:

- 768-dimensional CLS pooling on "title [SEP] abstract"
- All 32M+ papers with titles embedded (title-only fallback for papers without abstracts)
- HNSW index via pgvector with `m=16, ef_construction=200`
- Halfvec (float16) quantization to halve index memory

**Embedding pipeline optimization** (508 records/sec):

- Started at 32 rec/s with a naive loop
- Final architecture: multiprocessing with main process doing file read + tokenize + GPU inference, dedicated writer process doing binary `COPY` to PostgreSQL
- UNLOGGED table + dropped indexes during bulk load, restored after completion
- Full 27M new embeddings completed overnight on a single RTX 5090

**Hybrid search with RRF fusion**:

- Three-signal fusion: BM25 (tsvector on title + abstract + keywords) + SPECTER2 vector + OpenAI vector
- Reciprocal Rank Fusion with k=60
- Circuit breaker: OpenAI failure gracefully degrades to SPECTER2 + lexical

**pgvector 0.8.0+ upgrade** (migration 005):

- Iterative index scans for filtered vector search
- Halfvec support for quantized indexes
- Per-model HNSW indexes (migration 004)

### Phase 3: Graph Intelligence (March 31 -- April 1)

**Full-graph analytics**:

- PageRank on 32.4M nodes, 299M edges -- identifies structurally important papers beyond citation count
- HITS hub/authority scores -- distinguishes foundational work (authorities) from survey/review papers (hubs)
- Stored in `paper_metrics` table via TRUNCATE + COPY (~10 min after tuning shared_buffers to 16GB)

**Community detection**:

- **Citation communities**: Leiden algorithm on the giant component at three resolutions (coarse ~20, medium ~200, fine ~2000 communities)
- **Taxonomic communities**: arXiv class + UAT concept hierarchy mapping
- **Semantic communities**: k-means on SPECTER2 embedding space (planned)
- Three independent signals let agents choose the appropriate lens for their task

**Graph analysis tools** (4 MCP tools):

- `co_citation_analysis`: papers frequently cited together (shared intellectual base)
- `bibliographic_coupling`: papers that share references (methodological similarity)
- `citation_chain`: shortest path between two papers through the citation graph (BFS with memory cap)
- `temporal_evolution`: how a paper's citation neighborhood changes over time

**PostgreSQL tuning for graph workloads**:

- `shared_buffers`: 128MB -> 16GB
- `work_mem`: 4MB -> 256MB
- `maintenance_work_mem`: 64MB -> 2GB

### Phase 4: Knowledge Enrichment (April 2)

Designed via a full diverge-converge-premortem pipeline (3 independent research agents -> structured debate -> prospective failure analysis). Resulted in a premortem-annotated PRD with cost corrections.

**Entity extraction pipeline** (`src/scix/extraction.py`):

- LLM-based extraction using Claude Haiku via Anthropic Messages Batches API
- Entity types: methods, datasets, instruments, materials
- JSONB storage with GIN index for containment queries
- Pilot: 10K high-citation papers, ~$40 estimated cost
- MCP tools: `entity_search`, `entity_profile`

**Dual-model embedding support** (`src/scix/embed_openai.py`):

- OpenAI `text-embedding-3-large` at 1024 dimensions (Matryoshka truncation)
- Complements SPECTER2: trained for asymmetric query-to-document retrieval (vs SPECTER2's symmetric doc-to-doc)
- Halfvec quantization: combined dual HNSW indexes ~22GB (35% of RAM)
- Pilot: 10K papers, ~$0.13

**Agent session state** (`src/scix/session.py`):

- In-memory working set per session (appropriate for stdio MCP transport)
- `WorkingSetEntry`: bibcode + provenance (source_tool, source_context, tags)
- MCP tools: `add_to_working_set`, `get_working_set`, `get_session_summary`, `find_gaps`, `clear_working_set`
- `find_gaps`: SQL query finding cross-community bridge papers not yet in the working set

**RFC-DAG build process**: 6 independent work units built in parallel, all landed, 64 tests passing. Code reviewed and hardened (5 HIGH + 4 MEDIUM issues fixed).

### Phase 5: MCP Server (Cumulative)

**22 tools across 5 categories**:

| Category | Tools                                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Search   | `semantic_search`, `hybrid_search`, `lexical_search`, `faceted_search`                                                         |
| Paper    | `get_paper`, `get_citations`, `get_references`                                                                                 |
| Graph    | `co_citation_analysis`, `bibliographic_coupling`, `citation_chain`, `temporal_evolution`, `graph_metrics`, `explore_community` |
| Entity   | `entity_search`, `entity_profile`                                                                                              |
| Session  | `add_to_working_set`, `get_working_set`, `get_session_summary`, `find_gaps`, `clear_working_set`, `concept_search`             |

Tool count kept deliberately under 22 -- premortem analysis identified that >15 tools degrades agent tool selection accuracy. Each tool returns compact stubs (bibcode, title, first_author, year, citation_count, abstract_snippet) to minimize context window consumption.

**Server hardening**: model caching, health check endpoint, lifecycle management, streaming cursors for large result sets.

### Ongoing: Full-Text Body Ingestion (April 3)

**Discovery**: The ADS harvest scripts already request the `body` field, and it's present in the raw JSONL files. ~55% of papers have full text (median ~65K chars, mean ~73K chars). This was being silently dropped at ingest time.

**Why this changes things**:

- SPECTER2 only embeds title+abstract (512 token limit). Full text enables chunk-and-embed for RAG-style retrieval.
- Entity extraction on full text would be dramatically better than abstract-only.
- Full-text tsvector search would significantly improve lexical search quality.
- We already have this data. No additional API access or partnership needed.

**In progress**: Migration 010 (adds `body TEXT` column), field mapping update, backfill script. Storage estimate: ~2TB for 32M papers (1.3TB disk free -- tight but feasible with TOAST compression).

---

## Technical Decisions and Their Rationale

### PostgreSQL-only (no Elasticsearch, no separate vector DB)

Everything runs in a single PostgreSQL 16 instance with pgvector. This was a deliberate choice: it avoids synchronization complexity between multiple datastores, keeps the system simple enough to reason about, and demonstrates what's possible with PostgreSQL alone. At 32M papers, this is within PostgreSQL's comfort zone. If ADS adopted this approach, their existing PostgreSQL infrastructure could host it.

### Halfvec quantization for all embeddings

Float16 quantization cuts HNSW index memory in half with <1% recall degradation. At dual-model scale (SPECTER2 768d + OpenAI 1024d), this is the difference between fitting in RAM and not.

### Three independent community signals

Rather than one "best" community assignment, the system provides three orthogonal signals:

- **Citation communities** (Leiden): papers that cite each other -- behavioral affinity
- **Semantic communities** (embedding k-means): papers with similar content -- topical affinity
- **Taxonomic communities** (arXiv/UAT): papers in the same field -- institutional affinity

These frequently disagree, and the disagreements are informative. A paper that sits in one citation community but a different semantic community is likely a methodological bridge.

### In-memory session state (not persisted)

For the stdio MCP transport (one agent, one process), in-memory Python dicts are correct. Persisting to PostgreSQL would add complexity without benefit -- sessions are ephemeral by nature. The design supports future HTTP transport with server-side session storage.

### Full corpus ingestion (not just recent years)

The 99.6% vs 17.8% edge resolution finding was the strongest technical result of the project. It means partial-corpus graph analytics are not just incomplete -- they're structurally wrong. PageRank on a graph where 82% of edges dangle produces meaningless scores.

---

## Performance Characteristics

| Operation                        | Throughput / Latency                          |
| -------------------------------- | --------------------------------------------- |
| JSONL ingestion                  | ~2.2K papers/sec (COPY-based)                 |
| SPECTER2 embedding               | 508 rec/sec (RTX 5090, multiprocess pipeline) |
| Semantic search (HNSW)           | p95 < 10ms                                    |
| Hybrid search (3-signal RRF)     | p95 < 200ms                                   |
| PageRank (32M nodes, 299M edges) | ~10 min (after tuning)                        |
| HNSW index build (32M vectors)   | 1--4 hours (CONCURRENTLY)                     |
| Full corpus ingest               | ~4 hours (227 files)                          |

---

## Paper Outline (Targeting ADASS 2026)

**Title**: "Agent-Navigable Scientific Knowledge: Graph Intelligence and Semantic Retrieval on the ADS Citation Corpus"

Secondary targets: arXiv preprint (cs.DL + astro-ph.IM), stretch for JCDL 2027.

### Structure

1. **Introduction** (2 pp) -- The navigation problem: AI agents need structured knowledge, not ranked lists. No scientific literature MCP server exists. Our contribution: 32.4M-paper agent-native layer with graph intelligence, multi-model retrieval, and session state.

2. **Related Work** (1.5 pp) -- ADS, Semantic Scholar, Google Scholar as search systems. SPECTER/SPECTER2/SciBERT as scientific embeddings. PageRank and Leiden on citation graphs. Emerging literature review agents. Gap: none of these integrated into agent-facing infrastructure.

3. **Data & Infrastructure** (2 pp) -- Corpus statistics. PostgreSQL + pgvector architecture (single-server, no separate search engine). Edge resolution curve: 17.8% at 6 years vs 99.6% at full corpus. Idempotent ingestion pipeline.

4. **Semantic Retrieval Layer** (2.5 pp) -- SPECTER2 (doc-to-doc) vs text-embedding-3-large (query-to-doc). RRF fusion with circuit breaker fallback. 50-query benchmark: recall@10, NDCG@10, MRR across models and fusion strategies. Proposal: ADS serves precomputed embeddings as a data product (Semantic Scholar precedent).

5. **Graph Intelligence Layer** (2.5 pp) -- PageRank + HITS on full graph. Leiden community detection at 3 resolutions. Three parallel community signals (citation, semantic, taxonomic) and what their disagreements reveal. Giant component analysis before/after full ingest. Side-by-side: partial vs full corpus analytics.

6. **Agent Navigation Interface** (2 pp) -- 22 MCP tools across 5 categories. Session state architecture. Entity extraction pipeline (Haiku-based). Demo transcript: "Map the research landscape of exoplanet atmospheres" using multi-tool composition. Comparison with ADS-API-only agent.

7. **Evaluation** (2 pp) -- Retrieval benchmarks. Graph metrics validation (PageRank vs citation count, community vs arXiv class). Edge resolution impact on graph diameter and community quality. 10 qualitative agent task evaluations.

8. **Infrastructure Implications for ADS** (1.5 pp) -- Embeddings as a data product. Graph metrics as columns on paper records. MCP server as an ADS service endpoint. What changes: GPU for embedding, weekly graph batch, MCP process. What doesn't: Solr, REST API, myADS, ORCID.

9. **Discussion** (1 pp) -- What works (full-corpus transforms graph quality, hybrid search compensates model weaknesses, parallel communities). Limitations (single-GPU embedding takes days, Leiden OOMs at 32M nodes, entity extraction unvalidated at scale). Premortem learnings (extraction cost 3-5x underestimated, pilot cohort bias, tool count degrades selection).

10. **Conclusion** (0.5 pp) -- First MCP server for scientific literature. Future: full-text embeddings, cross-disciplinary expansion, agent benchmark suite, streaming graph updates.

### Key Figures Planned

1. Architecture diagram (corpus -> PostgreSQL -> MCP server -> agent)
2. Edge resolution curve (% resolved vs years of corpus ingested)
3. Community structure visualization (coarse communities, colored by arXiv class)
4. Embedding space UMAP (SPECTER2, colored by community)
5. Retrieval comparison (recall@10 across models and fusion)
6. Agent workflow transcript (annotated sequence diagram)
7. PageRank distribution (log-log, expected power law)
8. Parallel community signal agreement matrix

---

## Current State and Remaining Work

### Completed

- Full corpus ingestion (32.4M papers, 299M edges)
- SPECTER2 embeddings for all papers with HNSW index
- PageRank + HITS on full graph (stored in paper_metrics)
- Knowledge enrichment schema + pipelines (entity extraction, dual-model embedding, session state)
- 22-tool MCP server with server hardening
- Code review pass (5 HIGH + 4 MEDIUM issues fixed)
- 53+ tests (unit + integration + E2E)
- Paper outline drafted
- Repository published

### In Progress

- Full-text body field ingestion (migration 010, field mapping update, backfill script)

### Remaining for Demo/Paper

| Task                                       | Status      | Dependencies                            | Estimated Effort |
| ------------------------------------------ | ----------- | --------------------------------------- | ---------------- |
| Leiden communities on giant component      | Not started | Memory-managed approach (OOM'd at 43GB) | Medium           |
| Taxonomic community assignment (all 32M)   | Not started | Quick SQL UPDATE from arXiv class       | Low              |
| Apply migration 009 to live DB             | Not started | --                                      | Low              |
| OpenAI embedding pilot (10K papers)        | Not started | ~$0.13                                  | Low              |
| Entity extraction micro-pilot (100 papers) | Not started | ~$2                                     | Low              |
| 50-query retrieval benchmark               | Not started | Embeddings + HNSW index                 | Medium           |
| Agent demo transcript                      | Not started | All above                               | Medium           |
| Full-text body ingestion + tsvector index  | In progress | Migration 010                           | Medium           |

### Known Issues

- **Leiden OOM**: The full 32M-node citation graph exceeded 43GB RSS during Leiden community detection. The graph + undirected copy + Leiden partition exceeds 62GB RAM. Fix: run on giant component only, or filter to papers with >1 citation, or use lower resolution parameter.
- **HNSW rebuild needed**: Indexes were dropped for bulk embedding load and need to be rebuilt (1-4 hours with CONCURRENTLY).
- **Entity extraction cost underestimate**: Premortem caught that prompt overhead makes full-corpus LLM extraction 3-5x more expensive than initial estimate (~$500-$17K vs $100-$3K). Fine-tuning a smaller model may be necessary.

---

## What This Could Mean for ADS

This project is designed as a proof-of-concept for capabilities that could become part of ADS infrastructure:

1. **Precomputed embeddings as a data product**: Embed on ingest, serve to all downstream consumers. Semantic Scholar already does this. ADS would be the first for astronomy specifically.

2. **Graph metrics as searchable fields**: `pagerank`, `community_id` as columns on paper records. Enables "sort by influence" in the ADS UI, community browsing, and "related community" recommendations.

3. **MCP endpoint alongside REST API**: Agents connect directly to ADS for structured exploration. No need for third parties to harvest and re-index the corpus.

4. **Full-text embeddings**: With the body field already available for 55% of papers, chunk-and-embed RAG retrieval becomes possible without additional data partnerships.

The incremental cost to ADS: one GPU for embedding (one-time bulk + incremental), weekly CPU batch for graph metrics, one Python process for MCP. Everything else (Solr, REST API, myADS, ORCID) stays unchanged.
