"""MCP server exposing 7 core tools for agent navigation of the SciX corpus.

Uses the `mcp` Python SDK to register tools. Each tool is a thin wrapper
around functions in search.py. Connection pooling via psycopg.pool for
production-grade performance.

Usage:
    python -m scix.mcp_server
    # Or via MCP client configuration pointing to this module
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg

from scix.db import DEFAULT_DSN

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
    """Serialize a SearchResult to JSON with timing metadata."""
    from scix.search import SearchResult

    if isinstance(result, SearchResult):
        output: dict[str, Any] = {
            "papers": result.papers,
            "total": result.total,
            "timing_ms": result.timing_ms,
        }
        if result.metadata:
            output["metadata"] = result.metadata
        return json.dumps(output, indent=2, default=str)
    return json.dumps(result, indent=2, default=str)


def _parse_filters(filters: dict[str, Any] | None = None):
    """Parse a filter dict into a SearchFilters instance."""
    from scix.search import SearchFilters

    if not filters:
        return SearchFilters()
    return SearchFilters(
        year_min=filters.get("year_min"),
        year_max=filters.get("year_max"),
        arxiv_class=filters.get("arxiv_class"),
        doctype=filters.get("doctype"),
        first_author=filters.get("first_author"),
    )


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


def create_server():
    """Create and configure the MCP server with all 7 tools."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError(
            "mcp SDK is required for the MCP server. "
            "Install with: pip install mcp"
        )

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
                    "Uses PostgreSQL tsvector with weighted fields (title=A, abstract=B, keywords=C)."
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
                                "year", "doctype", "arxiv_class",
                                "database", "bibgroup", "property",
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
    from scix import search

    if name == "semantic_search":
        # Embed the query text to get a vector for semantic search
        try:
            from scix.embed import embed_batch, load_model

            model, tokenizer = load_model("specter2", device="cpu")
            vectors = embed_batch(model, tokenizer, [args["query"]], batch_size=1)
            query_embedding = vectors[0]
        except ImportError:
            return json.dumps({
                "error": "transformers/torch not installed for embedding",
                "hint": "pip install transformers torch",
            })

        filters = _parse_filters(args.get("filters"))
        limit = args.get("limit", 10)
        result = search.vector_search(
            conn, query_embedding, model_name="specter2", filters=filters, limit=limit
        )
        return _result_to_json(result)

    elif name == "keyword_search":
        filters = _parse_filters(args.get("filters"))
        limit = args.get("limit", 10)
        result = search.lexical_search(
            conn, args["terms"], filters=filters, limit=limit
        )
        return _result_to_json(result)

    elif name == "get_paper":
        result = search.get_paper(conn, args["bibcode"])
        return _result_to_json(result)

    elif name == "get_citations":
        result = search.get_citations(
            conn, args["bibcode"], limit=args.get("limit", 20)
        )
        return _result_to_json(result)

    elif name == "get_references":
        result = search.get_references(
            conn, args["bibcode"], limit=args.get("limit", 20)
        )
        return _result_to_json(result)

    elif name == "get_author_papers":
        result = search.get_author_papers(
            conn,
            args["author_name"],
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
        )
        return _result_to_json(result)

    elif name == "facet_counts":
        filters = _parse_filters(args.get("filters"))
        limit = args.get("limit", 50)
        result = search.facet_counts(
            conn, args["field"], filters=filters, limit=limit
        )
        return _result_to_json(result)

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
