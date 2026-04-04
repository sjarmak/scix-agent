"""MCP server exposing 22 tools for agent navigation of the SciX corpus.

Uses the `mcp` Python SDK to register tools. Each tool is a thin wrapper
around functions in search.py. Connection pooling via psycopg.pool for
production-grade performance.

Usage:
    python -m scix.mcp_server
    # Or via MCP client configuration pointing to this module
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

import psycopg

from scix import search
from scix.db import DEFAULT_DSN
from scix.embed import _model_cache, clear_model_cache, embed_batch, load_model
from scix.session import SessionState, WorkingSetEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool (singleton, lazy-initialized)
# ---------------------------------------------------------------------------

_pool = None


def _get_pool():
    """Get or create the connection pool (singleton)."""
    global _pool
    if _pool is not None:
        return _pool

    try:
        from psycopg_pool import ConnectionPool

        dsn = os.environ.get("SCIX_DSN", DEFAULT_DSN)
        min_size = int(os.environ.get("SCIX_POOL_MIN", "2"))
        max_size = int(os.environ.get("SCIX_POOL_MAX", "10"))
        timeout = float(os.environ.get("SCIX_POOL_TIMEOUT", "30.0"))

        _pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
        )
        logger.info(
            "Connection pool created: min=%d, max=%d, timeout=%.1fs",
            min_size,
            max_size,
            timeout,
        )
        return _pool
    except ImportError:
        logger.warning(
            "psycopg_pool not available; falling back to single connections. "
            "Install with: pip install 'psycopg[pool]'"
        )
        return None


@contextmanager
def _get_conn() -> Generator[psycopg.Connection, None, None]:
    """Get a connection from the pool (or create a one-off if no pool)."""
    pool = _get_pool()
    if pool is not None:
        with pool.connection() as conn:
            yield conn
    else:
        conn = psycopg.connect(os.environ.get("SCIX_DSN", DEFAULT_DSN))
        try:
            yield conn
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tool timeout configuration
# ---------------------------------------------------------------------------

# Per-tool timeout in seconds (configurable via env vars)
TOOL_TIMEOUTS: dict[str, float] = {
    "semantic_search": float(os.environ.get("SCIX_TIMEOUT_SEMANTIC", "30")),
    "keyword_search": float(os.environ.get("SCIX_TIMEOUT_KEYWORD", "10")),
    "get_paper": float(os.environ.get("SCIX_TIMEOUT_PAPER", "5")),
    "get_citations": float(os.environ.get("SCIX_TIMEOUT_CITATIONS", "10")),
    "get_references": float(os.environ.get("SCIX_TIMEOUT_REFERENCES", "10")),
    "get_author_papers": float(os.environ.get("SCIX_TIMEOUT_AUTHOR", "15")),
    "facet_counts": float(os.environ.get("SCIX_TIMEOUT_FACETS", "10")),
    "health_check": float(os.environ.get("SCIX_TIMEOUT_HEALTH", "3")),
    "co_citation_analysis": float(os.environ.get("SCIX_TIMEOUT_COCITATION", "15")),
    "bibliographic_coupling": float(os.environ.get("SCIX_TIMEOUT_COUPLING", "15")),
    "citation_chain": float(os.environ.get("SCIX_TIMEOUT_CHAIN", "20")),
    "temporal_evolution": float(os.environ.get("SCIX_TIMEOUT_TEMPORAL", "10")),
    "get_paper_metrics": float(os.environ.get("SCIX_TIMEOUT_METRICS", "5")),
    "explore_community": float(os.environ.get("SCIX_TIMEOUT_COMMUNITY", "10")),
    "concept_search": float(os.environ.get("SCIX_TIMEOUT_CONCEPT", "15")),
    "entity_search": float(os.environ.get("SCIX_TIMEOUT_ENTITY_SEARCH", "10")),
    "entity_profile": float(os.environ.get("SCIX_TIMEOUT_ENTITY_PROFILE", "5")),
    "find_gaps": float(os.environ.get("SCIX_TIMEOUT_FIND_GAPS", "15")),
    "get_citation_context": float(os.environ.get("SCIX_TIMEOUT_CITATION_CONTEXT", "5")),
    "read_paper_section": float(os.environ.get("SCIX_TIMEOUT_READ_SECTION", "5")),
    "search_within_paper": float(os.environ.get("SCIX_TIMEOUT_SEARCH_WITHIN", "10")),
}


def _set_timeout(conn: psycopg.Connection, tool_name: str) -> None:
    """Set statement_timeout for this tool's query."""
    timeout_sec = TOOL_TIMEOUTS.get(tool_name, 30)
    timeout_ms = int(timeout_sec * 1000)
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_json(result: Any) -> str:
    """Serialize a SearchResult to JSON with timing metadata.

    All paper-returning results are annotated with ``in_working_set``.
    """
    if isinstance(result, search.SearchResult):
        output: dict[str, Any] = {
            "papers": _annotate_working_set(result.papers),
            "total": result.total,
            "timing_ms": result.timing_ms,
        }
        if result.metadata:
            output["metadata"] = result.metadata
        return json.dumps(output, indent=2, default=str)
    return json.dumps(result, indent=2, default=str)


def _parse_filters(filters: dict[str, Any] | None = None) -> search.SearchFilters:
    """Parse a filter dict into a SearchFilters instance."""
    if not filters:
        return search.SearchFilters()
    return search.SearchFilters(
        year_min=filters.get("year_min"),
        year_max=filters.get("year_max"),
        arxiv_class=filters.get("arxiv_class"),
        doctype=filters.get("doctype"),
        first_author=filters.get("first_author"),
    )


# ---------------------------------------------------------------------------
# Session state (singleton for the server process)
# ---------------------------------------------------------------------------

_session_state = SessionState()


def _annotate_working_set(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add 'in_working_set' boolean to each paper dict."""
    return [
        {**paper, "in_working_set": _session_state.is_in_working_set(paper.get("bibcode", ""))}
        for paper in papers
    ]


# ---------------------------------------------------------------------------
# Model pre-loading and lifecycle
# ---------------------------------------------------------------------------


def _init_model_impl() -> None:
    """Eagerly load SPECTER2 model into cache at server startup.

    Non-fatal: if torch/transformers are not installed, semantic_search
    will be unavailable but all other tools still work.
    """
    try:
        device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
        load_model("specter2", device=device)
        logger.info("SPECTER2 model pre-loaded on %s", device)
    except ImportError:
        logger.warning(
            "torch/transformers not installed — semantic_search will be unavailable. "
            "Install with: pip install transformers torch"
        )
    except Exception:
        logger.exception("Failed to pre-load SPECTER2 model")


def _shutdown() -> None:
    """Clean up resources: close connection pool, clear model cache."""
    global _pool
    clear_model_cache()
    if _pool is not None:
        try:
            _pool.close()
            logger.info("Connection pool closed")
        except Exception:
            logger.exception("Error closing connection pool")
        _pool = None


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


def create_server():
    """Create and configure the MCP server with 22 tools (7 core + 4 graph + 3 intelligence + 7 entity/session + health_check).

    Eagerly pre-loads the SPECTER2 model so semantic_search is fast from
    the first call.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError(
            "mcp SDK is required for the MCP server. " "Install with: pip install mcp"
        )

    _init_model_impl()

    server = Server("scix")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="semantic_search",
                description=(
                    "Search for papers by semantic similarity to a natural language query. "
                    "Uses SPECTER2 embeddings and pgvector cosine similarity."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query",
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "year_min": {"type": "integer"},
                                "year_max": {"type": "integer"},
                                "arxiv_class": {"type": "string"},
                                "doctype": {"type": "string"},
                            },
                        },
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="keyword_search",
                description=(
                    "Search for papers using keyword/full-text search. "
                    "Uses PostgreSQL tsvector with weighted fields "
                    "(title=A, abstract=B, keywords=C)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "terms": {
                            "type": "string",
                            "description": "Search terms (plain text or PostgreSQL tsquery syntax)",
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "year_min": {"type": "integer"},
                                "year_max": {"type": "integer"},
                                "arxiv_class": {"type": "string"},
                                "doctype": {"type": "string"},
                            },
                        },
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["terms"],
                },
            ),
            Tool(
                name="get_paper",
                description="Get full metadata for a paper by its ADS bibcode.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode (e.g., '2024ApJ...962L..15J')",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="get_citations",
                description=(
                    "Get forward citations -- papers that cite the given paper. "
                    "Returns compact stubs ordered by citation count."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="get_references",
                description=(
                    "Get backward references -- papers that the given paper cites. "
                    "Returns compact stubs ordered by citation count."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="get_author_papers",
                description=(
                    "Get papers by an author (case-insensitive partial match). "
                    "Optionally filter by year range."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "author_name": {"type": "string"},
                        "year_min": {"type": "integer"},
                        "year_max": {"type": "integer"},
                    },
                    "required": ["author_name"],
                },
            ),
            Tool(
                name="facet_counts",
                description=(
                    "Get distribution counts for a field. "
                    "Supported: year, doctype, arxiv_class, database, bibgroup, property."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": [
                                "year",
                                "doctype",
                                "arxiv_class",
                                "database",
                                "bibgroup",
                                "property",
                            ],
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "year_min": {"type": "integer"},
                                "year_max": {"type": "integer"},
                                "arxiv_class": {"type": "string"},
                                "doctype": {"type": "string"},
                            },
                        },
                        "limit": {"type": "integer", "default": 50},
                    },
                    "required": ["field"],
                },
            ),
            Tool(
                name="co_citation_analysis",
                description=(
                    "Find papers frequently co-cited with a given paper. "
                    "Two papers are co-cited when a third paper cites both. "
                    "Use this to discover intellectually related work that citing "
                    "authors consider together."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper to analyze",
                        },
                        "min_overlap": {
                            "type": "integer",
                            "default": 2,
                            "description": "Minimum number of shared citing papers",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="bibliographic_coupling",
                description=(
                    "Find papers that share references with a given paper. "
                    "Two papers are bibliographically coupled when they cite the same works. "
                    "Use this to find methodologically similar or contemporaneous research."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper to analyze",
                        },
                        "min_overlap": {
                            "type": "integer",
                            "default": 2,
                            "description": "Minimum number of shared references",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="citation_chain",
                description=(
                    "Find the shortest citation path between two papers. "
                    "Walks forward along citation edges (A cites B cites C...). "
                    "Returns the ordered path of papers, or empty if no path exists "
                    "within max_depth hops. Use this to trace intellectual lineage."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_bibcode": {
                            "type": "string",
                            "description": "Starting paper (the one that cites)",
                        },
                        "target_bibcode": {
                            "type": "string",
                            "description": "Destination paper (the one being cited)",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of hops (1-5)",
                        },
                    },
                    "required": ["source_bibcode", "target_bibcode"],
                },
            ),
            Tool(
                name="temporal_evolution",
                description=(
                    "Show temporal trends. If given a bibcode, shows citations received "
                    "per year. If given search terms, shows publication volume per year. "
                    "Use this to understand how a topic or paper's impact evolves over time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode_or_query": {
                            "type": "string",
                            "description": (
                                "A bibcode (for citation trends) "
                                "or search terms (for publication volume)"
                            ),
                        },
                        "year_start": {"type": "integer"},
                        "year_end": {"type": "integer"},
                    },
                    "required": ["bibcode_or_query"],
                },
            ),
            Tool(
                name="get_paper_metrics",
                description=(
                    "Get precomputed graph metrics for a paper: PageRank, "
                    "HITS hub/authority scores, and Leiden community assignments "
                    "at 3 resolutions with labels."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="explore_community",
                description=(
                    "Find what community a paper belongs to and return sibling "
                    "papers (same community) ranked by PageRank. Use this to "
                    "discover related work in the same research neighborhood."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper",
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["coarse", "medium", "fine"],
                            "default": "coarse",
                            "description": "Community resolution level",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="concept_search",
                description=(
                    "Search for papers by Unified Astronomy Thesaurus (UAT) concept. "
                    "Accepts a concept name (e.g., 'Galaxies', 'Exoplanets') or URI. "
                    "With include_subtopics=true, also returns papers matching "
                    "descendant concepts in the hierarchy."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "UAT concept label or URI",
                        },
                        "include_subtopics": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include papers from descendant concepts",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="entity_search",
                description=(
                    "Search for papers containing a specific entity in their extractions. "
                    "Uses JSONB containment (@>) on the GIN-indexed payload column."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Extraction type to search within (e.g., 'entities', 'keywords')",
                        },
                        "entity_name": {
                            "type": "string",
                            "description": "Entity name to search for",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["entity_type", "entity_name"],
                },
            ),
            Tool(
                name="entity_profile",
                description=(
                    "Get all extracted entities for a paper by bibcode. "
                    "Returns extraction types, versions, and payloads."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="add_to_working_set",
                description=(
                    "Add one or more papers to the session working set. "
                    "The working set tracks papers the agent is actively exploring."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ADS bibcodes to add",
                        },
                        "source_tool": {
                            "type": "string",
                            "description": "Name of the tool that found these papers",
                        },
                        "source_context": {
                            "type": "string",
                            "default": "",
                            "description": "Query or context that produced these papers",
                        },
                        "relevance_hint": {
                            "type": "string",
                            "default": "",
                            "description": "Why these papers are relevant",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                            "description": "Optional tags for categorization",
                        },
                    },
                    "required": ["bibcodes", "source_tool"],
                },
            ),
            Tool(
                name="get_working_set",
                description=(
                    "Return the current session working set — all papers the agent "
                    "has collected for active exploration."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_session_summary",
                description=(
                    "Return summary statistics for the current session: "
                    "working set size, seen papers count."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="find_gaps",
                description=(
                    "Find papers in unexplored communities that cite papers in the "
                    "working set. Useful for discovering blind spots and cross-community "
                    "bridge papers the agent hasn't examined."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "resolution": {
                            "type": "string",
                            "enum": ["coarse", "medium", "fine"],
                            "default": "coarse",
                            "description": "Community resolution level",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                },
            ),
            Tool(
                name="clear_working_set",
                description="Clear all papers from the session working set.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_citation_context",
                description=(
                    "Get the citation context text and intent label for why a source "
                    "paper cites a target paper. Returns the surrounding text where the "
                    "citation appears, along with the classified intent (e.g., 'background', "
                    "'method', 'result'). Returns empty result for pairs without context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_bibcode": {
                            "type": "string",
                            "description": "Bibcode of the citing paper",
                        },
                        "target_bibcode": {
                            "type": "string",
                            "description": "Bibcode of the cited paper",
                        },
                    },
                    "required": ["source_bibcode", "target_bibcode"],
                },
            ),
            Tool(
                name="health_check",
                description=(
                    "Check server health: database connectivity, model cache status, "
                    "and connection pool info."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="read_paper_section",
                description=(
                    "Read a section of a paper's full-text body. "
                    "Uses section_parser to split the body into IMRaD sections. "
                    "Falls back to abstract if no body text is available. "
                    "Supports pagination via char_offset and limit."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper",
                        },
                        "section": {
                            "type": "string",
                            "default": "full",
                            "description": (
                                "Section name: 'full', 'introduction', 'methods', "
                                "'results', 'discussion', 'conclusions', etc."
                            ),
                        },
                        "char_offset": {
                            "type": "integer",
                            "default": 0,
                            "description": "Character offset for pagination",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5000,
                            "description": "Maximum characters to return",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="search_within_paper",
                description=(
                    "Search within a paper's full-text body for matching passages. "
                    "Uses PostgreSQL ts_headline to return fragments with context "
                    "around matching terms. Requires the paper to have body text."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search terms to find within the paper",
                        },
                    },
                    "required": ["bibcode", "query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        with _get_conn() as conn:
            _set_timeout(conn, name)
            result_json = _dispatch_tool(conn, name, arguments)
            return [TextContent(type="text", text=result_json)]

    return server


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _dispatch_tool(conn: psycopg.Connection, name: str, args: dict[str, Any]) -> str:
    """Route a tool call to the appropriate search function."""
    t_start = time.monotonic()
    logger.info("tool_call: %s args=%s", name, list(args.keys()))

    if name == "semantic_search":
        try:
            device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
            model, tokenizer = load_model("specter2", device=device)
            vectors = embed_batch(model, tokenizer, [args["query"]], batch_size=1)
            query_embedding = vectors[0]
        except ImportError:
            result_json = json.dumps(
                {
                    "error": "transformers/torch not installed for embedding",
                    "hint": "pip install transformers torch",
                }
            )
        else:
            filters = _parse_filters(args.get("filters"))
            limit = args.get("limit", 10)
            result = search.vector_search(
                conn, query_embedding, model_name="specter2", filters=filters, limit=limit
            )
            result_json = _result_to_json(result)

    elif name == "keyword_search":
        filters = _parse_filters(args.get("filters"))
        limit = args.get("limit", 10)
        result = search.lexical_search(conn, args["terms"], filters=filters, limit=limit)
        result_json = _result_to_json(result)

    elif name == "get_paper":
        result = search.get_paper(conn, args["bibcode"])
        result_json = _result_to_json(result)

    elif name == "get_citations":
        result = search.get_citations(conn, args["bibcode"], limit=args.get("limit", 20))
        result_json = _result_to_json(result)

    elif name == "get_references":
        result = search.get_references(conn, args["bibcode"], limit=args.get("limit", 20))
        result_json = _result_to_json(result)

    elif name == "get_author_papers":
        result = search.get_author_papers(
            conn,
            args["author_name"],
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
        )
        result_json = _result_to_json(result)

    elif name == "facet_counts":
        filters = _parse_filters(args.get("filters"))
        limit = args.get("limit", 50)
        result = search.facet_counts(conn, args["field"], filters=filters, limit=limit)
        result_json = _result_to_json(result)

    elif name == "co_citation_analysis":
        result = search.co_citation_analysis(
            conn,
            args["bibcode"],
            min_overlap=args.get("min_overlap", 2),
            limit=args.get("limit", 20),
        )
        result_json = _result_to_json(result)

    elif name == "bibliographic_coupling":
        result = search.bibliographic_coupling(
            conn,
            args["bibcode"],
            min_overlap=args.get("min_overlap", 2),
            limit=args.get("limit", 20),
        )
        result_json = _result_to_json(result)

    elif name == "citation_chain":
        max_depth = max(1, min(args.get("max_depth", 5), 5))  # clamp to [1, 5]
        result = search.citation_chain(
            conn,
            args["source_bibcode"],
            args["target_bibcode"],
            max_depth=max_depth,
        )
        result_json = _result_to_json(result)

    elif name == "temporal_evolution":
        result = search.temporal_evolution(
            conn,
            args["bibcode_or_query"],
            year_start=args.get("year_start"),
            year_end=args.get("year_end"),
        )
        result_json = _result_to_json(result)

    elif name == "get_paper_metrics":
        result = search.get_paper_metrics(conn, args["bibcode"])
        result_json = _result_to_json(result)

    elif name == "explore_community":
        result = search.explore_community(
            conn,
            args["bibcode"],
            resolution=args.get("resolution", "coarse"),
            limit=args.get("limit", 20),
        )
        result_json = _result_to_json(result)

    elif name == "concept_search":
        result = search.concept_search(
            conn,
            args["query"],
            include_subtopics=args.get("include_subtopics", True),
            limit=args.get("limit", 20),
        )
        result_json = _result_to_json(result)

    elif name == "entity_search":
        entity_type = args["entity_type"]
        entity_name = args["entity_name"]
        limit = min(args.get("limit", 20), 200)
        # Validate entity_type against known types
        _VALID_ENTITY_TYPES = {"methods", "datasets", "instruments", "materials"}
        if entity_type not in _VALID_ENTITY_TYPES:
            result_json = json.dumps(
                {
                    "error": f"Invalid entity_type '{entity_type}'. Must be one of {sorted(_VALID_ENTITY_TYPES)}"
                }
            )
            return result_json
        # Use JSONB containment (@>) which leverages the GIN index
        containment = json.dumps({entity_type: [entity_name]})
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.bibcode, e.extraction_type, e.extraction_version, e.payload,
                       p.title
                FROM extractions e
                JOIN papers p ON p.bibcode = e.bibcode
                WHERE e.payload @> %s::jsonb
                LIMIT %s
                """,
                (containment, limit),
            )
            rows = cur.fetchall()
        papers = [
            {
                "bibcode": row[0],
                "extraction_type": row[1],
                "extraction_version": row[2],
                "payload": row[3],
                "title": row[4],
            }
            for row in rows
        ]
        papers = _annotate_working_set(papers)
        result_json = json.dumps({"papers": papers, "total": len(papers)}, indent=2, default=str)

    elif name == "entity_profile":
        bibcode = args["bibcode"]
        if not bibcode or not bibcode.strip():
            return json.dumps({"error": "bibcode must be a non-empty string"})
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT extraction_type, extraction_version, payload, created_at
                FROM extractions
                WHERE bibcode = %s
                ORDER BY extraction_type, extraction_version
                """,
                (bibcode,),
            )
            rows = cur.fetchall()
        extractions = [
            {
                "extraction_type": row[0],
                "extraction_version": row[1],
                "payload": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]
        result_json = json.dumps(
            {"bibcode": bibcode, "extractions": extractions, "total": len(extractions)},
            indent=2,
            default=str,
        )

    elif name == "add_to_working_set":
        bibcodes = args["bibcodes"]
        source_tool = args["source_tool"]
        source_context = args.get("source_context", "")
        relevance_hint = args.get("relevance_hint", "")
        tags = args.get("tags", [])
        entries = []
        for bib in bibcodes:
            entry = _session_state.add_to_working_set(
                bibcode=bib,
                source_tool=source_tool,
                source_context=source_context,
                relevance_hint=relevance_hint,
                tags=tags,
            )
            entries.append(dataclasses.asdict(entry))
        result_json = json.dumps({"added": len(entries), "entries": entries}, indent=2, default=str)

    elif name == "get_working_set":
        entries = _session_state.get_working_set()
        result_json = json.dumps(
            {
                "entries": [dataclasses.asdict(e) for e in entries],
                "total": len(entries),
            },
            indent=2,
            default=str,
        )

    elif name == "get_session_summary":
        summary = _session_state.get_session_summary()
        result_json = json.dumps(summary, indent=2, default=str)

    elif name == "find_gaps":
        resolution = args.get("resolution", "coarse")
        limit = min(args.get("limit", 20), 200)
        # Hardcoded column map — never derive column names from user input
        _RESOLUTION_COLS: dict[str, str] = {
            "coarse": "community_id_coarse",
            "medium": "community_id_medium",
            "fine": "community_id_fine",
        }
        community_col = _RESOLUTION_COLS.get(resolution)
        if community_col is None:
            result_json = json.dumps(
                {
                    "error": f"Invalid resolution: {resolution}. Must be one of {sorted(_RESOLUTION_COLS)}"
                }
            )
        else:
            ws_bibcodes = [e.bibcode for e in _session_state.get_working_set()]
            # Cap bibcode array to prevent expensive ANY() scans
            ws_bibcodes = ws_bibcodes[:200]
            if not ws_bibcodes:
                result_json = json.dumps(
                    {
                        "papers": [],
                        "total": 0,
                        "message": "Working set is empty. Add papers first.",
                    },
                    indent=2,
                )
            else:
                # community_col is a Python literal from _RESOLUTION_COLS, safe for SQL
                query = f"""
                    SELECT DISTINCT p.bibcode, p.title, pm.pagerank,
                           pm.{community_col} AS community_id
                    FROM citation_edges ce
                    JOIN papers p ON p.bibcode = ce.citing
                    JOIN paper_metrics pm ON pm.bibcode = p.bibcode
                    WHERE ce.cited = ANY(%s)
                      AND pm.{community_col} IS NOT NULL
                      AND pm.{community_col} NOT IN (
                          SELECT DISTINCT pm2.{community_col}
                          FROM paper_metrics pm2
                          WHERE pm2.bibcode = ANY(%s)
                            AND pm2.{community_col} IS NOT NULL
                      )
                      AND p.bibcode <> ALL(%s)
                    ORDER BY pm.pagerank DESC NULLS LAST
                    LIMIT %s
                """
                with conn.cursor() as cur:
                    cur.execute(query, (ws_bibcodes, ws_bibcodes, ws_bibcodes, limit))
                    rows = cur.fetchall()
                papers = [
                    {
                        "bibcode": row[0],
                        "title": row[1],
                        "pagerank": row[2],
                        "community_id": row[3],
                    }
                    for row in rows
                ]
                papers = _annotate_working_set(papers)
                result_json = json.dumps(
                    {"papers": papers, "total": len(papers), "resolution": resolution},
                    indent=2,
                    default=str,
                )

    elif name == "clear_working_set":
        removed = _session_state.clear_working_set()
        result_json = json.dumps({"removed": removed}, indent=2)

    elif name == "get_citation_context":
        result = search.get_citation_context(
            conn,
            args["source_bibcode"],
            args["target_bibcode"],
        )
        result_json = _result_to_json(result)

    elif name == "read_paper_section":
        result = search.read_paper_section(
            conn,
            args["bibcode"],
            section=args.get("section", "full"),
            char_offset=args.get("char_offset", 0),
            limit=args.get("limit", 5000),
        )
        result_json = _result_to_json(result)

    elif name == "search_within_paper":
        result = search.search_within_paper(
            conn,
            args["bibcode"],
            args["query"],
        )
        result_json = _result_to_json(result)

    elif name == "health_check":
        status: dict[str, Any] = {"pool": "no_pool", "model_cached": False, "db": "unknown"}

        # Check model cache
        status["model_cached"] = len(_model_cache) > 0
        status["cached_models"] = [f"{k[0]}@{k[1]}" for k in _model_cache]

        # Check pool (inspect directly, don't call _get_pool() which would create it)
        status["pool"] = "active" if _pool is not None else "no_pool"

        # Check DB connectivity
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            status["db"] = "ok"
        except Exception:
            status["db"] = "error"

        result_json = json.dumps(status, indent=2)

    else:
        result_json = json.dumps({"error": f"Unknown tool: {name}"})

    elapsed_ms = (time.monotonic() - t_start) * 1000
    logger.info("tool_done: %s elapsed=%.1fms", name, elapsed_ms)
    return result_json


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    server = create_server()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)
    finally:
        _shutdown()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(main())
