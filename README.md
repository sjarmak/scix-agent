# SciX Experiments

Agent-navigable knowledge layer on the full NASA ADS/SciX corpus. Transforms 32.4M scientific papers and 299M citation edges into infrastructure that AI agents can navigate programmatically via a 22-tool MCP server.

## What This Does

Instead of returning ranked lists, the system exposes the structural topology of science -- citation graphs, research communities, and multi-model embeddings -- through [Model Context Protocol](https://modelcontextprotocol.io/) tools.

Key capabilities:

- **Hybrid search**: INDUS (domain-specific scientific similarity) + text-embedding-3-large (query-to-doc retrieval) + BM25 fused via Reciprocal Rank Fusion
- **Graph intelligence**: PageRank, HITS, Leiden community detection at multiple resolutions on the full citation graph
- **Entity extraction**: LLM-based extraction of methods, datasets, instruments from abstracts and full text
- **Session state**: Working sets that let agents accumulate and reason over papers across a research session
- **Full-text search**: Body text available for ~55% of papers with GIN-indexed tsvector

## Architecture

Single PostgreSQL 16 instance with pgvector 0.8.2. No separate search engine or vector database.

| Dimension       | Value                         |
| --------------- | ----------------------------- |
| Papers          | 32.4M (1800--2026)            |
| With abstracts  | 23.3M (72%)                   |
| With full text  | 6.0M (19%)                    |
| Citation edges  | 299M                          |
| Edge resolution | 99.6%                         |
| Embeddings      | INDUS (768d) + OpenAI (1024d) |

**Discipline coverage** (papers may belong to multiple):

| Collection    | Papers |
| ------------- | ------ |
| Physics       | 17.1M  |
| Earth science | 13.1M  |
| General       | 5.8M   |
| Astronomy     | 3.0M   |

**Content coverage:**

| Field        | Coverage |
| ------------ | -------- |
| Title        | >99%     |
| Affiliations | 96%      |
| DOI          | 87%      |
| Abstract     | 72%      |
| Cited papers | 54%      |
| Keywords     | 49%      |
| References   | 40%      |
| Full text    | 19%      |

## Project Structure

```
src/scix/                     -- Python package (47 modules)
  mcp_server.py               -- MCP server (22 tools across 5 categories)
  search.py                   -- Hybrid search with RRF fusion
  db.py                       -- DB helpers (connection pool, IndexManager, IngestLog)
  ingest.py                   -- JSONL -> PostgreSQL via COPY
  field_mapping.py            -- ADS/SciX JSONL -> SQL field mapping + transforms
  embed.py                    -- INDUS / SPECTER2 embedding pipeline
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
schema.sql                    -- Consolidated PostgreSQL schema
tests/                        -- pytest suite (107 test files)
docs/                         -- Documentation
  ADR/                        -- Architecture decision records
  paper_outline.md            -- ADASS 2026 paper outline
  figures/                    -- Data visualizations
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
#   ADS_API_KEY=<your NASA ADS/SciX API key>
#   SCIX_DSN=dbname=scix
```

### Database

```bash
# Create database and apply schema
scripts/setup_db.sh

# Ingest ADS/SciX metadata
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

**Search & Discovery**

| Tool             | Description                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `search`         | Search the corpus by natural-language query. Modes: hybrid (INDUS + OpenAI + BM25 via RRF), semantic, or keyword.        |
| `concept_search` | Retrieve papers tagged with a Unified Astronomy Thesaurus (UAT) concept, with optional expansion to descendant concepts. |
| `facet_counts`   | Distribution of paper counts grouped by year, doctype, arxiv_class, database, bibgroup, or property.                     |

**Paper Access**

| Tool         | Description                                                                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_paper`  | Full metadata for a paper by bibcode: title, abstract, authors, affiliations, keywords, citation counts. Optionally includes linked entities.  |
| `read_paper` | Read or search inside a paper's full-text body. Supports section selection (introduction, methods, results, etc.) and in-paper keyword search. |

**Citation Graph**

| Tool                  | Description                                                                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `citation_graph`      | Walk the citation graph: forward (papers that cite it), backward (references), or both. Optionally includes surrounding citation context sentences. |
| `citation_similarity` | Find structurally related papers via co-citation (cited together) or bibliographic coupling (shared references).                                    |
| `citation_chain`      | Trace the shortest citation path between two papers (BFS up to 5 hops).                                                                             |
| `temporal_evolution`  | Citations-per-year for a paper, or publications-per-year for a search query.                                                                        |

**Entities**

| Tool             | Description                                                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entity`         | Look up scientific entities (methods, datasets, instruments, materials). Search for papers mentioning an entity, or resolve a free-text mention to canonical records. |
| `entity_context` | Full profile of a known entity by ID: canonical name, type, external identifiers, aliases, related entities, paper count.                                             |

**Graph Analytics & Session**

| Tool            | Description                                                                                                                                                                         |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph_context` | Citation-graph analytics for a paper: PageRank, HITS hub/authority, community membership at coarse/medium/fine resolution. Optionally returns sibling papers in the same community. |
| `find_gaps`     | Surface papers in unexplored communities that cite papers you already inspected. Reads from implicit session state across `get_paper` calls.                                        |

## Performance

MCP tool latency at 32M papers:

| Operation                    | Latency     |
| ---------------------------- | ----------- |
| Semantic search (HNSW)       | p95 < 10ms  |
| Hybrid search (3-signal RRF) | p95 < 200ms |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

NASA ADS/SciX metadata is subject to the [ADS terms of service](https://ui.adsabs.harvard.edu/help/terms/).
