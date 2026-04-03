# PRD: Embedding + Hybrid Retrieval + Agent Navigation Architecture

## Problem Statement

We have 5M+ scientific papers (NASA ADS metadata) ingested into PostgreSQL 16 with pgvector 0.6.0, including titles, abstracts, authors, affiliations, keywords, citations, references, and rich metadata. The goal is to make this corpus navigable by AI agents — not just searchable by humans. Agents need programmatic query primitives that combine semantic similarity, lexical matching, citation graph traversal, and structural metadata to answer complex research questions like "What are recent advances in gravitational wave detection?" or "Find papers that challenge the conclusions of X."

The infrastructure layer (schema, ingestion, sync) is complete. This PRD defines the retrieval and navigation layer: what embeddings to generate, how to search them, and what tools to expose to agents.

## Goals & Non-Goals

### Goals

- Enable AI agents to navigate the scientific corpus through a composable set of MCP tools
- Achieve high-quality retrieval via hybrid search (dense vectors + BM25 + re-ranking)
- Support citation graph exploration (forward/backward citations, co-citation, bibliographic coupling)
- Handle papers without abstracts (~15-20%) gracefully with quality-aware degradation
- Keep the entire system within PostgreSQL (no separate Elasticsearch or vector DB)
- Design for embedding model evolution (versioning, blue-green transitions)

### Non-Goals

- Full-text paper search (we have metadata only, not full papers)
- Human-facing search UI (agent-first, API-only)
- Real-time embedding of new papers (batch pipeline, daily/weekly refresh)
- ColBERT or other multi-vector late-interaction models (impractical at 5M scale in pgvector)
- Fine-tuning embedding models on ADS data (evaluate off-the-shelf first)

## Requirements

### Must-Have (Phase 1: Core Retrieval)

1. **SPECTER2 embeddings (768d) for all papers**
   - Input: `title [SEP] abstract` (do NOT include authors/keywords in embedding text)
   - Title-only embedding for papers without abstracts, flagged with `input_type` column
   - Local GPU inference (~2-3 hours on A100 for 5M papers, ~$5-10 one-time cost)
   - HNSW index with `m=16, ef_construction=200`

2. **Hybrid search: pgvector + BM25 with RRF fusion**
   - Install pg_search (ParadeDB) for BM25 scoring on title + abstract + keywords
   - Reciprocal Rank Fusion combining vector and lexical results (k=60)
   - Fallback: native tsvector + GIN with weighted fields (title=A, abstract=B)

3. **Cross-encoder re-ranking**
   - Re-rank top-20 hybrid results with `cross-encoder/ms-marco-MiniLM-L-12-v2`
   - ~50-200ms latency (acceptable for agent queries)
   - Run on CPU (no GPU required for inference on 20 candidates)

4. **Upgrade pgvector to 0.8.0+**
   - Enable iterative index scans for filtered vector search
   - Support halfvec quantization for future storage optimization

5. **7 core MCP tools**
   - `semantic_search(query, filters?, limit?)` — pgvector cosine similarity, returns compact stubs
   - `keyword_search(terms, filters?, limit?)` — BM25 full-text search, returns compact stubs
   - `get_paper(bibcode)` — full metadata for one paper
   - `get_citations(bibcode, limit?)` — forward citations (who cites this), returns stubs
   - `get_references(bibcode, limit?)` — backward references (what this cites), returns stubs
   - `get_author_papers(author_name, year_range?)` — papers by author, returns stubs
   - `facet_counts(field, filters?)` — distribution counts (papers per year, per arxiv_class, etc.)

6. **Compact stub format for all paper-list responses**
   ```json
   {
     "bibcode": "...",
     "title": "...",
     "first_author": "...",
     "year": 2024,
     "citation_count": 42,
     "abstract_snippet": "First 150 chars..."
   }
   ```

### Should-Have (Phase 2: Graph Intelligence)

7. **Precomputed graph metrics**
   - PageRank, HITS hub/authority scores on citation_edges
   - Leiden community detection at 3 resolutions (coarse ~50, medium ~1000, fine ~10000)
   - Stored in `paper_metrics` table, refreshed weekly
   - Community labels auto-generated from top-K distinctive keywords

8. **4 advanced MCP graph tools**
   - `co_citation_analysis(bibcode, min_overlap?)` — query-time SQL join on citation_edges
   - `bibliographic_coupling(bibcode, min_overlap?)` — papers sharing references
   - `citation_chain(source, target, max_depth?)` — shortest citation path (bounded BFS)
   - `temporal_evolution(query_or_bibcode, year_range)` — citation/publication trends over time

9. **UAT concept hierarchy**
   - Load Unified Astronomy Thesaurus (2,122 concepts, SKOS) into PostgreSQL
   - Map existing ADS keywords to UAT concepts
   - Enable hierarchical queries ("all papers about stellar evolution" includes subtopics)

10. **Embedding versioning infrastructure**
    - `source_hash` column (SHA-256 of input text) to detect stale embeddings
    - Partial HNSW indexes per `model_name` for blue-green model transitions
    - Support untyped `vector` column for models with different dimensionalities

### Nice-to-Have (Phase 3: Knowledge Enrichment)

11. **General-purpose embedding (text-embedding-3-large at 1024d Matryoshka)**
    - Complements SPECTER2 for natural-language agent queries
    - ~$65 via OpenAI batch API for 5M papers
    - Evaluate against SPECTER2 on agent query benchmarks before committing

12. **LLM entity extraction**
    - Methods, datasets, instruments, materials from abstracts
    - Haiku-class model for bulk extraction, structured JSON output
    - Store in `extractions` table with GIN index on JSONB payload
    - Start with top-10K cited papers, expand if agents use it

13. **Agent session state**
    - Server-side working set of papers the agent has examined
    - `get_working_set()` and `add_to_working_set(bibcodes)` tools
    - Avoids re-fetching previously seen papers, manages context budget

## Design Considerations

### Key Tensions

**Domain-specific vs general-purpose embeddings**: SPECTER2 excels at citation-structure-aware similarity but was trained on CS + biomedical, not astrophysics. General-purpose models (text-embedding-3-large) have seen scientific text but don't capture citation signals. Start with SPECTER2 (proven on scientific tasks, local inference, no vendor lock-in), add general-purpose only if evaluation shows gaps.

**Precomputation vs query-time**: PageRank and communities are global, stable metrics — precompute them. Co-citation and bibliographic coupling are local, query-dependent — compute on demand via SQL joins. The citation_edges indexes support both patterns.

**PostgreSQL-native vs specialized systems**: At 5M papers, PostgreSQL handles both BM25 and vector search competitively. The operational simplicity of one database outweighs the marginal performance gain of Elasticsearch + Qdrant. Revisit at 50M+ scale.

**Embedding storage**: 768d float32 × 5M = ~15GB for SPECTER2. Adding a 1024d general-purpose model doubles this to ~35GB. Both fit comfortably in 64GB RAM. Halfvec quantization (pgvector 0.7.0+) can halve storage if needed.

### Retrieval Pipeline Architecture

```
Agent Query
    │
    ├──► semantic_search (pgvector HNSW, top-60)
    │                                              ├──► RRF Fusion (top-20) ──► Cross-encoder Re-rank (top-10) ──► Agent
    └──► keyword_search  (pg_search BM25, top-60)
```

### Schema Additions

```sql
-- Embedding metadata
ALTER TABLE paper_embeddings ADD COLUMN input_type TEXT NOT NULL DEFAULT 'title_abstract';
ALTER TABLE paper_embeddings ADD COLUMN source_hash TEXT;

-- Graph metrics
CREATE TABLE paper_metrics (
    bibcode TEXT PRIMARY KEY REFERENCES papers(bibcode),
    pagerank FLOAT,
    hub_score FLOAT,
    authority_score FLOAT,
    community_id_coarse INTEGER,
    community_id_medium INTEGER,
    community_id_fine INTEGER,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Community metadata
CREATE TABLE communities (
    community_id INTEGER NOT NULL,
    resolution TEXT NOT NULL,
    label TEXT,
    paper_count INTEGER,
    top_keywords TEXT[],
    PRIMARY KEY (community_id, resolution)
);

-- BM25 full-text index (pg_search)
CALL paradedb.create_bm25(
    index_name => 'papers_bm25',
    table_name => 'papers',
    key_field => 'bibcode',
    text_fields => paradedb.field('title', boost => 2.0)
                || paradedb.field('abstract')
                || paradedb.field('keywords', tokenizer => paradedb.tokenizer('keyword'))
);

-- Fallback native tsvector
ALTER TABLE papers ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(abstract, '')), 'B')
    ) STORED;
CREATE INDEX idx_papers_tsv ON papers USING GIN (tsv);
```

## Open Questions

1. **SPECTER2 on astrophysics**: How well does it perform on ADS-specific retrieval? Need head-to-head eval on 10K papers with known citation relationships.
2. **Citation graph completeness**: 2021-2026 window means many citations point outside corpus. Should we import OpenAlex historical data to complete the graph?
3. **pg_search stability**: ParadeDB is production-ready but adds Rust dependency. Acceptable? Or start with native tsvector?
4. **Entity extraction ROI**: At $50K+ for 5M abstracts, need to validate that agents actually use extracted entities before investing. Pilot on 10K high-cited papers first.
5. **Filtered vector search benchmarks**: Need to test pgvector 0.8.0 iterative scans on real data with selective filters (year + arxiv_class).

## Research Provenance

This PRD synthesizes findings from 5 independent research agents exploring:

| Lens                          | Key Contribution                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| **Embedding Models**          | SPECTER2 vs general-purpose comparison, Matryoshka insight, dual-embedding strategy  |
| **Retrieval Architecture**    | pgvector scaling benchmarks, RRF hybrid search SQL patterns, pg_search evaluation    |
| **Agent Navigation**          | PaperQA2-informed MCP tool taxonomy, facet counting as navigation primitive          |
| **Chunking & Representation** | Title+abstract validation, anti-pattern of metadata in embeddings, versioning schema |
| **Knowledge Structure**       | UAT discovery, Leiden communities, precompute vs query-time graph metrics            |

**Strongest convergence**: Hybrid search + cross-encoder reranking, title+abstract as embedding input, PostgreSQL-native architecture sufficient at 5M scale.

**Key divergence**: Whether to start with one embedding model (SPECTER2) or two (+ general-purpose). Resolved: start with one, evaluate, add if needed.
