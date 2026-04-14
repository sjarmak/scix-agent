# SciX Experiments

Agent-navigable knowledge layer on the full NASA ADS corpus. Transforms 32.4M scientific papers and 299M citation edges into infrastructure that AI agents can navigate programmatically via a 22-tool MCP server.

## What This Does

Instead of returning ranked lists, the system exposes the structural topology of science -- citation graphs, research communities, and multi-model embeddings -- through [Model Context Protocol](https://modelcontextprotocol.io/) tools.

Key capabilities:

- **Hybrid search**: SPECTER2 (doc-to-doc similarity) + text-embedding-3-large (query-to-doc retrieval) + BM25 fused via Reciprocal Rank Fusion
- **Graph intelligence**: PageRank, HITS, Leiden community detection at multiple resolutions on the full citation graph
- **Entity extraction**: LLM-based extraction of methods, datasets, instruments from abstracts and full text
- **Session state**: Working sets that let agents accumulate and reason over papers across a research session
- **Full-text search**: Body text available for ~55% of papers with GIN-indexed tsvector

## Architecture

Single PostgreSQL 16 instance with pgvector 0.8.2. No separate search engine or vector database.

| Dimension       | Value                            |
| --------------- | -------------------------------- |
| Papers          | 32,390,237 (1800--2026)          |
| With abstracts  | ~23.3M (72%)                     |
| Citation edges  | 299,336,889                      |
| Edge resolution | 99.6%                            |
| Database size   | ~162 GB                          |
| Embeddings      | SPECTER2 (768d) + OpenAI (1024d) |

## Project Structure

```
src/scix/                     -- Python package (47 modules)
  mcp_server.py               -- MCP server (22 tools across 5 categories)
  search.py                   -- Hybrid search with RRF fusion
  db.py                       -- DB helpers (connection pool, IndexManager, IngestLog)
  ingest.py                   -- JSONL -> PostgreSQL via COPY
  field_mapping.py            -- ADS JSONL -> SQL field mapping + transforms
  embed.py                    -- SPECTER2 embedding pipeline
  graph_metrics.py            -- PageRank, HITS, community detection
  extract.py                  -- LLM entity extraction
  session.py                  -- Agent working set management
  sources/                    -- OpenAlex, ar5iv, S2 source modules
  jit/                        -- JIT entity resolution (cache, router, NER)
  eval/                       -- Retrieval evaluation framework
scripts/                      -- CLI tools (78 scripts)
  ingest.py                   -- Corpus ingestion CLI
  embed_fast.py               -- GPU embedding pipeline
  harvest_*.py                -- External data harvesters
  eval_*.py                   -- Evaluation scripts
  link_*.py                   -- Entity linking scripts
  setup_db.sh                 -- Idempotent database creation
migrations/                   -- PostgreSQL schema (44 migrations)
tests/                        -- pytest suite (107 test files)
docs/                         -- Documentation
  prd/                        -- Product requirement docs
  premortem/                  -- Premortem risk analyses
  ADR/                        -- Architecture decision records
  paper_outline.md            -- ADASS 2026 paper outline
  briefing.md                 -- Technical briefing document
  figures/                    -- Data visualizations
results/                      -- Evaluation results (JSON + reports)
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector 0.8.2
- (Optional) NVIDIA GPU for embedding pipeline

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For embedding pipeline
pip install -e ".[embed]"

# For MCP server
pip install -e ".[mcp]"

# For graph analytics
pip install -e ".[graph]"
```

### Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
#   ADS_API_KEY=<your ADS API key>
#   SCIX_DSN=dbname=scix
```

### Database

```bash
# Create database and apply all migrations
scripts/setup_db.sh

# Ingest ADS metadata
python scripts/ingest.py ads_metadata_by_year_picard/
```

### Running the MCP Server

```bash
python -m scix.mcp_server
```

## Testing

```bash
# Set test DSN to avoid hitting production database
export SCIX_TEST_DSN="dbname=scix_test"

# Run all tests
pytest

# Run only unit tests (no database required)
pytest -m "not integration"
```

The default DSN (`dbname=scix`) points at the production database with 32M papers. Integration tests that write data require `SCIX_TEST_DSN` to be set.

## MCP Tools

| Category | Tools                                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Search   | `semantic_search`, `hybrid_search`, `lexical_search`, `faceted_search`                                                         |
| Paper    | `get_paper`, `get_citations`, `get_references`                                                                                 |
| Graph    | `co_citation_analysis`, `bibliographic_coupling`, `citation_chain`, `temporal_evolution`, `graph_metrics`, `explore_community` |
| Entity   | `entity_search`, `entity_profile`                                                                                              |
| Session  | `add_to_working_set`, `get_working_set`, `get_session_summary`, `find_gaps`, `clear_working_set`, `concept_search`             |

## Key Technical Decisions

- **PostgreSQL-only**: Everything in one instance -- avoids sync complexity between datastores. At 32M papers, well within PostgreSQL's comfort zone.
- **Dual-model embeddings**: SPECTER2 for citation-proximity similarity, text-embedding-3-large for asymmetric query-document retrieval. Fused via RRF.
- **Full corpus**: Ingesting only a 6-year window resolves 17.8% of citation edges. The full 1800-2026 corpus resolves 99.6%. Graph analytics on partial corpora are fundamentally misleading.
- **Halfvec quantization**: Float16 cuts HNSW index memory in half with <1% recall loss.

## Performance

| Operation                        | Throughput / Latency   |
| -------------------------------- | ---------------------- |
| JSONL ingestion                  | ~2.2K papers/sec       |
| SPECTER2 embedding               | 508 rec/sec (RTX 5090) |
| Semantic search (HNSW)           | p95 < 10ms             |
| Hybrid search (3-signal RRF)     | p95 < 200ms            |
| PageRank (32M nodes, 299M edges) | ~10 min                |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

ADS metadata is subject to the [ADS terms of service](https://ui.adsabs.harvard.edu/help/terms/).
