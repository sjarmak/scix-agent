---
name: scix-mcp
description: Use the SciX MCP server for scientific literature research over 32M NASA ADS papers — hybrid semantic+lexical search (INDUS + body BM25 via RRF), citation graph traversal (299M edges, PageRank, communities), entity extraction, and session-aware working sets. Invoke when the user asks about astronomy/astrophysics papers, citations, authors, or wants to explore the scientific literature.
origin: scix-experiments
---

# SciX MCP — Scientific Literature Research

A local-intelligence MCP over the full NASA ADS corpus (32M papers, 299M citation edges, 14.9M paper bodies). Adds dense semantic retrieval, citation graph analytics, and entity extraction on top of ADS metadata.

## When to Activate

- User asks to find papers on a topic (astronomy, astrophysics, space science)
- User wants to trace citations or references of a paper
- User asks about an author's work, research lineage, or impact
- User asks "what's the state of research on X" — multi-hop exploration
- User wants to find gaps in a research area
- User wants to explore a specific paper's body text (methods, equations buried in full-text)
- User mentions bibcodes (ADS identifiers like `2019ApJ...875L...1E`)

## Tool Overview (13 tools)

| Category | Tools | When to use |
|----------|-------|-------------|
| **Search** | `search`, `concept_search` | Find papers by query. Default: `search` (hybrid mode). Use `concept_search` for formal taxonomy terms (UAT concepts). |
| **Paper access** | `get_paper`, `read_paper` | Get full metadata / read sections/body text of a known bibcode. |
| **Citation graph** | `citation_graph`, `citation_chain`, `citation_similarity` | Neighbors, multi-hop chains, co-citation / bibliographic-coupling similarity. |
| **Entity system** | `entity`, `entity_context` | Resolve and explore named entities (instruments, methods, datasets, objects). |
| **Structure** | `graph_context`, `find_gaps`, `temporal_evolution`, `facet_counts` | Community/topic exploration, gap analysis, field-over-time trends. |

## Typical Workflows

### 1. Topic survey → citation exploration
```
search("dark matter halo mass function", limit=20)
  → pick top-cited bibcodes
citation_chain(bibcode, direction="references", depth=2)
  → foundational prior work
citation_chain(bibcode, direction="citations", depth=2)
  → recent extensions
```

### 2. Body-only discovery (new capability)
Use when a technique/equation is likely buried in the body text (methods section) rather than the abstract. `search` with `mode="hybrid"` (default) already includes body BM25 as a 4th RRF signal. No extra step needed — it just surfaces papers the title/abstract search misses.

### 3. Author impact
```
search("first_author:'Hogg, David W.'", limit=50)  # via filter
temporal_evolution(author="Hogg, David W.")
```

### 4. Entity-driven research
```
entity(action="search", query="JWST NIRSpec")
entity(action="profile", entity_id=<id>)
entity_context(entity_id=<id>, action="papers")
```

### 5. Gap finding
```
search("exoplanet atmosphere retrieval", limit=30)
  → add top 10 to working set (use `add_to_working_set` in session)
find_gaps(working_set_id)
  → under-cited or under-researched angles
```

## Filter Syntax

`search` and related tools accept filters:
- `year_min` / `year_max` (integers)
- `arxiv_class` (e.g., `"astro-ph.EP"`, `"astro-ph"`)
- `doctype` (e.g., `"article"`, `"review"`)
- `first_author` (string, exact match on normalized name)

## Ranking Model (hybrid search)

4 signals fused via Reciprocal Rank Fusion (k=60):
1. **INDUS dense embedding** (768d, NASA's domain-specific) — semantic similarity
2. **text-embedding-3-large** (3072d, OpenAI) — optional, skipped if not available
3. **Title+abstract BM25** (GIN on `papers.tsv`) — exact term matching on metadata
4. **Body BM25** (GIN expression index on `papers.body`) — matches full-text body, ~47% coverage

Body matches are ranked by the title/abstract tsvector to keep latency sub-second. For most queries, the hybrid total is <1s.

## Performance Characteristics

| Operation | Typical latency |
|-----------|-----------------|
| Keyword/BM25 search | 50-500ms |
| Hybrid search (all 4 signals) | 300-900ms |
| Citation chain (depth 1-2) | 20-100ms |
| Citation chain (depth 3+) | 1-10s |
| Entity resolution | <50ms |
| Body BM25 alone | 400-800ms |
| `read_paper` for full body | 100-300ms |

## Working Set / Session

If the MCP server supports session state, build a working set with `add_to_working_set` for iterative refinement. `find_gaps`, `citation_similarity`, and some summaries operate over the working set. Clear with `clear_working_set` to start fresh.

## Connection Config

For Claude Desktop / Claude Code:
```json
{
  "mcpServers": {
    "scix": {
      "url": "<https://...trycloudflare.com>/mcp/",
      "headers": {"Authorization": "Bearer <token>"}
    }
  }
}
```

Local stdio (if running the server locally):
```json
{
  "mcpServers": {
    "scix": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "scix.mcp_server"],
      "cwd": "/path/to/scix_experiments"
    }
  }
}
```

## Dos and Don'ts

**Do:**
- Start with `search` in default (hybrid) mode — it combines all 4 signals
- Pass a year filter for recent work (e.g., `year_min=2023`)
- Chain: `search` → pick bibcode → `citation_chain` or `read_paper` for depth
- Use `find_gaps` on a working set of 10-20 papers for recommendation quality

**Don't:**
- Make many parallel calls — rate limit is 60/min per token
- Query with just one word (use phrases for hybrid ranking to shine)
- Assume every paper has a body — ~47% coverage. Body BM25 contributes where available, doesn't when not.
- Expect `concept_search` to work on arbitrary phrases — it maps to the UAT (Unified Astronomy Thesaurus) taxonomy only.

## Corpus Scale Facts

- **Papers:** 32.4M (full ADS metadata)
- **Citation edges:** 299M
- **INDUS embeddings:** 32M papers (complete coverage)
- **Body text:** 14.9M papers (~47% coverage, ~65K chars avg)
- **Entity dictionary:** ~90K entities (instruments, datasets, methods, objects, missions, software)
- **Body GIN index:** 37 GB partial expression index on `papers.body`

## Differences from `nasa-ads-mcp`

The public `prtc/nasa-ads-mcp` is a thin API wrapper (10 tools, stateless, every call hits ADS). SciX MCP is a local-intelligence layer:
- Hybrid semantic+lexical search (nasa-ads-mcp has Solr keyword only)
- Full citation graph analytics (nasa-ads-mcp has citation_count only)
- Entity extraction + knowledge graph (nasa-ads-mcp has none)
- Sub-second local latency (nasa-ads-mcp bound by ADS API round-trips)
- Body full-text search (nasa-ads-mcp has none)
