# Research: consolidate-mcp-tools

## Current State

### mcp_server.py (~1499 lines)

- **list_tools()**: 25 tools registered (semantic_search, keyword_search, get_paper, get_citations, get_references, get_author_papers, facet_counts, co_citation_analysis, bibliographic_coupling, citation_chain, temporal_evolution, get_paper_metrics, explore_community, concept_search, entity_search, entity_profile, add_to_working_set, get_working_set, get_session_summary, find_gaps, clear_working_set, get_citation_context, health_check, read_paper_section, search_within_paper, get_openalex_topics, document_context, entity_context, resolve_entity)
- **\_dispatch_tool()**: Routes by name string matching (if/elif chain)
- **Session state**: `_session_state = SessionState()` singleton
- **Connection pooling**: psycopg_pool with per-tool timeouts

### session.py

- SessionState class with working_set (dict[str, WorkingSetEntry]) and seen_papers (set[str])
- Methods: add_to_working_set, get_working_set, is_in_working_set, clear_working_set, get_session_summary

### search.py functions

- `hybrid_search(conn, query_text, query_embedding=None, model_name='indus', ...)` -> SearchResult
- `vector_search(conn, query_embedding, model_name, ...)` -> SearchResult
- `lexical_search(conn, terms, ...)` -> SearchResult
- `get_paper(conn, bibcode)` -> SearchResult
- `get_citations(conn, bibcode, limit)` -> SearchResult
- `get_references(conn, bibcode, limit)` -> SearchResult
- `get_paper_metrics(conn, bibcode)` -> SearchResult
- `explore_community(conn, bibcode, resolution, limit)` -> SearchResult
- `get_document_context(conn, bibcode)` -> SearchResult
- `get_entity_context(conn, entity_id)` -> SearchResult

### Test files

- `test_mcp_server.py`: Unit tests for \_dispatch_tool, HNSW guard, health_check, shutdown (~440 lines)
- `test_mcp_session.py`: Entity/session tool tests (~278 lines)

## Consolidation Plan

### Target: 13 tools

search, concept_search, get_paper, read_paper, citation_graph, citation_similarity, citation_chain, entity, entity_context, graph_context, find_gaps, temporal_evolution, facet_counts

### Mappings

- search <- semantic_search + keyword_search + hybrid_search
- citation_graph <- get_citations + get_references
- citation_similarity <- co_citation_analysis + bibliographic_coupling
- entity <- entity_search + resolve_entity
- get_paper <- get_paper + document_context + get_openalex_topics (include_entities param)
- read_paper <- read_paper_section + search_within_paper
- graph_context <- get_paper_metrics + explore_community
- find_gaps <- find_gaps (now reads from implicit session state)

### Removed from list_tools

- add_to_working_set, get_working_set, get_session_summary, clear_working_set
- health_check, entity_profile, get_citation_context (deprecated aliases)
- get_author_papers (deprecated alias)

### Session changes

- Auto-track: any tool returning bibcodes adds to "seen" set
- get_paper adds to "focused" set
- find_gaps reads from focused set
- Add focused_papers set to SessionState
