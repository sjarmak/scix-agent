"""MCP server exposing 13 consolidated tools for agent navigation of the SciX corpus.

Uses the `mcp` Python SDK to register tools. Each tool is a thin wrapper
around functions in search.py. Connection pooling via psycopg.pool for
production-grade performance.

Consolidation (v2):
    28 original tools -> 13 agent-facing tools + deprecated aliases.
    Old tool names still work via _DEPRECATED_ALIASES but return
    ``deprecated: true`` and ``use_instead`` metadata.

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
from scix.entity_resolver import EntityResolver
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
        # max_size bumped from 10→20 to prevent pool exhaustion under
        # concurrent hybrid search load (each holds a connection for 2-3
        # sequential HNSW scans). See premortem M9.
        max_size = int(os.environ.get("SCIX_POOL_MAX", "20"))
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
    "search": float(os.environ.get("SCIX_TIMEOUT_SEARCH", "30")),
    "concept_search": float(os.environ.get("SCIX_TIMEOUT_CONCEPT", "15")),
    "get_paper": float(os.environ.get("SCIX_TIMEOUT_PAPER", "5")),
    "read_paper": float(os.environ.get("SCIX_TIMEOUT_READ_PAPER", "10")),
    "citation_graph": float(os.environ.get("SCIX_TIMEOUT_CITATIONS", "10")),
    "citation_similarity": float(os.environ.get("SCIX_TIMEOUT_COCITATION", "15")),
    "citation_chain": float(os.environ.get("SCIX_TIMEOUT_CHAIN", "20")),
    "entity": float(os.environ.get("SCIX_TIMEOUT_ENTITY", "10")),
    "entity_context": float(os.environ.get("SCIX_TIMEOUT_ENTITY_CONTEXT", "5")),
    "graph_context": float(os.environ.get("SCIX_TIMEOUT_GRAPH_CONTEXT", "10")),
    "find_gaps": float(os.environ.get("SCIX_TIMEOUT_FIND_GAPS", "15")),
    "temporal_evolution": float(os.environ.get("SCIX_TIMEOUT_TEMPORAL", "10")),
    "facet_counts": float(os.environ.get("SCIX_TIMEOUT_FACETS", "10")),
    # Legacy timeouts for deprecated aliases
    "semantic_search": float(os.environ.get("SCIX_TIMEOUT_SEMANTIC", "30")),
    "keyword_search": float(os.environ.get("SCIX_TIMEOUT_KEYWORD", "10")),
    "health_check": float(os.environ.get("SCIX_TIMEOUT_HEALTH", "3")),
}


def _set_timeout(conn: psycopg.Connection, tool_name: str) -> None:
    """Set statement_timeout for this tool's query."""
    timeout_sec = TOOL_TIMEOUTS.get(tool_name, 30)
    timeout_ms = int(timeout_sec * 1000)
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")


# ---------------------------------------------------------------------------
# HNSW index availability guard
# ---------------------------------------------------------------------------

_hnsw_index_cache: dict[str, tuple[bool, float]] = {}
_HNSW_CACHE_TTL_MISS_SEC = 30.0


def _hnsw_index_name(model_name: str) -> str:
    """Return the conventional HNSW partial-index name for a given embedding model."""
    return f"idx_embed_hnsw_{model_name}"


def _hnsw_index_exists(conn: psycopg.Connection, model_name: str) -> bool:
    """Check whether the per-model HNSW partial index on paper_embeddings exists."""
    now = time.monotonic()
    cached = _hnsw_index_cache.get(model_name)
    if cached is not None:
        exists, checked_at = cached
        if exists or (now - checked_at) < _HNSW_CACHE_TTL_MISS_SEC:
            return exists

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'paper_embeddings'
              AND indexname = %s
            """,
            (_hnsw_index_name(model_name),),
        )
        exists = cur.fetchone() is not None

    _hnsw_index_cache[model_name] = (exists, now)
    return exists


def _log_query(
    conn: psycopg.Connection,
    tool_name: str,
    params: dict[str, Any],
    latency_ms: float,
    success: bool,
    error_msg: str | None = None,
) -> None:
    """Write a row to query_log. Best-effort: failures are logged, not raised."""
    try:
        params_json = json.dumps(params, default=str)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log (tool_name, params_json, latency_ms, success, error_msg)
                VALUES (%s, %s::jsonb, %s, %s, %s)
                """,
                (tool_name, params_json, latency_ms, success, error_msg),
            )
        conn.commit()
    except Exception:
        logger.warning("Failed to log query for tool=%s", tool_name, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_json(result: Any) -> str:
    """Serialize a SearchResult to JSON with timing metadata."""
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


def _auto_track_bibcodes(result_json: str) -> None:
    """Extract bibcodes from a result and add them to the seen set."""
    try:
        data = json.loads(result_json)
        bibcodes: list[str] = []
        if isinstance(data, dict):
            papers = data.get("papers", [])
            if isinstance(papers, list):
                for p in papers:
                    if isinstance(p, dict) and "bibcode" in p:
                        bibcodes.append(p["bibcode"])
            # Single paper result
            if "bibcode" in data and not papers:
                bibcodes.append(data["bibcode"])
        if bibcodes:
            _session_state.track_seen(bibcodes)
    except (json.JSONDecodeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Deprecated aliases — old tool names map to new consolidated names
# ---------------------------------------------------------------------------

_DEPRECATED_ALIASES: dict[str, str] = {
    "semantic_search": "search",
    "keyword_search": "search",
    "get_citations": "citation_graph",
    "get_references": "citation_graph",
    "co_citation_analysis": "citation_similarity",
    "bibliographic_coupling": "citation_similarity",
    "entity_search": "entity",
    "resolve_entity": "entity",
    # entity_profile has a unique schema (raw extractions table rows). It
    # is routed to a dedicated handler via _transform_deprecated_args, but
    # use_instead points at get_paper(include_entities=true) as the modern
    # equivalent for agents migrating off the old schema.
    "entity_profile": "get_paper",
    "get_paper_metrics": "graph_context",
    "explore_community": "graph_context",
    "document_context": "get_paper",
    "get_openalex_topics": "get_paper",
    "get_author_papers": "search",
    "add_to_working_set": "get_paper",
    "get_working_set": "find_gaps",
    "get_session_summary": "find_gaps",
    "clear_working_set": "find_gaps",
    "get_citation_context": "citation_graph",
    "read_paper_section": "read_paper",
    "search_within_paper": "read_paper",
}


# ---------------------------------------------------------------------------
# Model pre-loading and lifecycle
# ---------------------------------------------------------------------------


def _init_model_impl() -> None:
    """Eagerly load INDUS model into cache at server startup."""
    try:
        device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
        load_model("indus", device=device)
        logger.info("INDUS model pre-loaded on %s", device)
    except ImportError:
        logger.warning(
            "torch/transformers not installed — semantic_search will be unavailable. "
            "Install with: pip install transformers torch"
        )
    except Exception:
        logger.exception("Failed to pre-load INDUS model")


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
# Filters schema (shared across tool definitions)
# ---------------------------------------------------------------------------

_FILTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "arxiv_class": {"type": "string"},
        "doctype": {"type": "string"},
        "first_author": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Expected consolidated tools (used by startup self-test)
# ---------------------------------------------------------------------------

EXPECTED_TOOLS: tuple[str, ...] = (
    "search",
    "concept_search",
    "get_paper",
    "read_paper",
    "citation_graph",
    "citation_similarity",
    "citation_chain",
    "entity",
    "entity_context",
    "graph_context",
    "find_gaps",
    "temporal_evolution",
    "facet_counts",
)


def startup_self_test(server: Any = None) -> dict[str, Any]:
    """Validate that list_tools() returns exactly 13 tools with valid schemas.

    Pure function — does NOT require a database connection. Inspects only
    the registered tool schemas. Runs during server initialization (see
    ``create_server``) and can also be invoked standalone from the main
    entry point.

    Args:
        server: Optional already-created MCP Server instance. If ``None``,
            a fresh server is created via ``create_server`` with the
            self-test disabled (to avoid infinite recursion).

    Returns:
        Dict with keys ``ok`` (bool), ``tool_count`` (int),
        ``tool_names`` (list[str]), and ``errors`` (list[str]).

    Raises:
        RuntimeError: If the self-test fails (wrong count, missing tool,
            invalid schema). Failures are fatal by design so the server
            never silently starts with broken tools.
    """
    import asyncio

    errors: list[str] = []
    tool_names: list[str] = []

    try:
        from mcp.types import ListToolsRequest
    except ImportError as exc:
        raise RuntimeError(f"startup_self_test: mcp SDK not available: {exc}") from exc

    if server is None:
        server = create_server(_run_self_test=False)

    try:
        handler = server.request_handlers[ListToolsRequest]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            f"startup_self_test: server has no ListToolsRequest handler: {exc}"
        ) from exc

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()

    # Real MCP server handlers wrap the ListToolsResult in a ServerResult
    # envelope (`.root.tools`); raw test fixtures may return `.tools`
    # directly. Accept both.
    tools = None
    if hasattr(result, "root") and hasattr(result.root, "tools"):
        tools = result.root.tools
    elif hasattr(result, "tools"):
        tools = result.tools
    if tools is None:
        raise RuntimeError(f"startup_self_test: unexpected list_tools result shape: {result!r}")

    tool_count = len(tools)
    if tool_count != 13:
        errors.append(f"expected exactly 13 tools, got {tool_count}")

    expected_set = set(EXPECTED_TOOLS)
    seen: set[str] = set()

    for tool in tools:
        name = getattr(tool, "name", None)
        if not name or not isinstance(name, str):
            errors.append(f"tool missing valid name: {tool!r}")
            continue
        tool_names.append(name)

        if name in seen:
            errors.append(f"duplicate tool name: {name}")
        seen.add(name)

        schema = getattr(tool, "inputSchema", None)
        if not isinstance(schema, dict):
            errors.append(f"tool {name}: inputSchema is not a dict")
            continue
        if schema.get("type") != "object":
            errors.append(
                f"tool {name}: inputSchema.type must be 'object', got " f"{schema.get('type')!r}"
            )
        if "properties" not in schema or not isinstance(schema["properties"], dict):
            errors.append(f"tool {name}: inputSchema.properties missing or not a dict")

    missing = expected_set - seen
    extra = seen - expected_set
    if missing:
        errors.append(f"missing expected tools: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected extra tools: {sorted(extra)}")

    status: dict[str, Any] = {
        "ok": not errors,
        "tool_count": tool_count,
        "tool_names": sorted(tool_names),
        "errors": errors,
    }

    if errors:
        logger.critical(
            "startup_self_test FAILED: tool_count=%d errors=%s",
            tool_count,
            errors,
        )
        raise RuntimeError(f"startup_self_test failed: {errors}")

    logger.info(
        "startup_self_test OK: %d tools registered (%s)",
        tool_count,
        ", ".join(sorted(tool_names)),
    )
    return status


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


def create_server(_run_self_test: bool = True):
    """Create and configure the MCP server with 13 consolidated tools.

    Eagerly pre-loads the INDUS model so semantic_search is fast from
    the first call.

    Args:
        _run_self_test: If True (default), run ``startup_self_test`` after
            the server is built to fail fast on broken tool schemas. Set
            to False internally by ``startup_self_test`` itself to avoid
            infinite recursion.
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
            # --- M1: Unified search ---
            Tool(
                name="search",
                description=(
                    "Search the scientific literature corpus for papers matching a "
                    "natural-language query. Returns ranked papers with titles, abstracts, "
                    "authors, years, and citation counts. Defaults to hybrid mode, the best "
                    "general-purpose search. Use concept_search instead when the query is a "
                    "formal astronomy taxonomy term (e.g., 'Exoplanets'). Use entity with "
                    "action='search' when looking up a named method, dataset, or instrument."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query or search terms",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "semantic", "keyword"],
                            "default": "hybrid",
                            "description": "Search mode: hybrid (default), semantic, or keyword",
                        },
                        "filters": _FILTERS_SCHEMA,
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            # --- concept_search (unchanged) ---
            Tool(
                name="concept_search",
                description=(
                    "Retrieve papers tagged with a formal Unified Astronomy Thesaurus (UAT) "
                    "concept. Accepts the concept label (e.g., 'Galaxies') or URI and can "
                    "expand to descendant concepts in the hierarchy. Returns papers ranked "
                    "by relevance within the concept. Use search instead when the query is "
                    "free-form natural language rather than a curated taxonomy term."
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
            # --- M6: get_paper absorbs document_context ---
            Tool(
                name="get_paper",
                description=(
                    "Fetch full metadata for a single paper by its ADS bibcode: title, "
                    "abstract, authors, affiliations, year, keywords, and citation counts. "
                    "Optionally include linked entities (methods, datasets, instruments) "
                    "detected in the paper. Use search or concept_search instead when you "
                    "do not yet know the bibcode. Use read_paper to access the full body text."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode (e.g., '2024ApJ...962L..15J')",
                        },
                        "include_entities": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include linked entities from document_context view",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            # --- S1: read_paper (combines read_paper_section + search_within_paper) ---
            Tool(
                name="read_paper",
                description=(
                    "Read or search inside one paper's full-text body. Without search_query, "
                    "returns a paginated chunk of a named section (introduction, methods, "
                    "results, discussion, conclusions, or full). With search_query, returns "
                    "highlighted passages matching terms inside that paper. Use get_paper "
                    "instead when you only need metadata and abstract, not body text."
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
                                "'results', 'discussion', 'conclusions'"
                            ),
                        },
                        "search_query": {
                            "type": "string",
                            "description": "If provided, search within the paper body instead of reading",
                        },
                        "char_offset": {
                            "type": "integer",
                            "default": 0,
                            "description": "Character offset for pagination (read mode)",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5000,
                            "description": "Max characters to return (read mode)",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            # --- M3: citation_graph (merges get_citations + get_references) ---
            Tool(
                name="citation_graph",
                description=(
                    "Walk the citation graph around a paper. direction=forward returns "
                    "papers that cite it (impact); backward returns papers it cites "
                    "(foundations); both returns each. Optionally include surrounding "
                    "citation context sentences. Use citation_similarity instead when you "
                    "want papers that are related via shared citation patterns rather than "
                    "direct citation links. Use citation_chain to trace a path between two papers."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {"type": "string", "description": "ADS bibcode"},
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward", "both"],
                            "default": "forward",
                            "description": "forward=citing papers, backward=references, both=all",
                        },
                        "include_context": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include citation context text (slower)",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            # --- M3: citation_similarity (merges co_citation + coupling) ---
            Tool(
                name="citation_similarity",
                description=(
                    "Find papers related to a seed paper through shared citation patterns. "
                    "method='co_citation' returns papers often cited together with the seed "
                    "(peer works in the same discussion). method='coupling' returns papers "
                    "that share many references with the seed (papers built on similar "
                    "foundations). Use citation_graph instead when you want direct citing "
                    "or cited papers rather than structurally similar ones."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {"type": "string", "description": "ADS bibcode"},
                        "method": {
                            "type": "string",
                            "enum": ["co_citation", "coupling"],
                            "default": "co_citation",
                            "description": "co_citation or coupling",
                        },
                        "min_overlap": {
                            "type": "integer",
                            "default": 2,
                            "description": "Minimum shared citing/referenced papers",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["bibcode"],
                },
            ),
            # --- citation_chain (unchanged) ---
            Tool(
                name="citation_chain",
                description=(
                    "Trace the shortest chain of citations from one paper to another. "
                    "Returns an ordered path of intermediate papers, or an empty path if "
                    "none exists within max_depth hops. Useful for explaining how an idea "
                    "propagated between two specific works. Use citation_graph instead when "
                    "you want the full neighborhood of citing or cited papers rather than a "
                    "single path between two endpoints."
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
            # --- M4: entity (merges entity_search + resolve_entity) ---
            Tool(
                name="entity",
                description=(
                    "Look up named scientific entities (methods, datasets, instruments, "
                    "materials). action='search' returns papers that mention the entity of "
                    "a given type. action='resolve' maps a free-text mention to canonical "
                    "entity records with aliases and identifiers. Use search instead for "
                    "free-form queries. Use entity_context once you have an entity_id and "
                    "need its full profile and relationships."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search", "resolve"],
                            "description": "search=find papers by entity, resolve=find canonical entity",
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ["methods", "datasets", "instruments", "materials"],
                            "description": "Entity type (required for action=search)",
                        },
                        "query": {
                            "type": "string",
                            "description": "Entity name to search for or resolve",
                        },
                        "discipline": {
                            "type": "string",
                            "description": "Discipline for ranking boost (resolve only)",
                        },
                        "fuzzy": {
                            "type": "boolean",
                            "default": False,
                            "description": "Enable fuzzy matching (resolve only)",
                        },
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["action", "query"],
                },
            ),
            # --- entity_context (unchanged) ---
            Tool(
                name="entity_context",
                description=(
                    "Fetch the full profile of a known entity by its entity_id: canonical "
                    "name, type, discipline, external identifiers, aliases, related "
                    "entities, and the count of papers that mention it. Use entity with "
                    "action='resolve' instead when you only have a text mention and need "
                    "to find the entity_id first. Requires a numeric entity_id as input."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "integer",
                            "description": "Entity ID (from entity search/resolve or document_context)",
                        },
                    },
                    "required": ["entity_id"],
                },
            ),
            # --- S2: graph_context (merges get_paper_metrics + explore_community) ---
            Tool(
                name="graph_context",
                description=(
                    "Get citation-graph analytics for a paper: influence scores, authority "
                    "and hub metrics, and the community it belongs to at coarse/medium/fine "
                    "resolution. Optionally also returns sibling papers in the same "
                    "community ranked by influence. Use citation_graph instead when you "
                    "want direct citing or cited papers rather than computed scores and "
                    "community membership."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode",
                        },
                        "include_community": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include sibling papers from same community",
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
            # --- M5: find_gaps (reads from implicit session state) ---
            Tool(
                name="find_gaps",
                description=(
                    "Surface papers in communities you have not yet explored that still "
                    "cite papers you already inspected via get_paper. Helps catch adjacent "
                    "literature you might be missing during a research session. Reads from "
                    "implicit session state tracked across get_paper calls. Use "
                    "citation_graph instead when you want direct citations of a single "
                    "paper rather than cross-community gap detection."
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
                        "clear_first": {
                            "type": "boolean",
                            "default": False,
                            "description": "Reset the focused set before searching",
                        },
                    },
                },
            ),
            # --- temporal_evolution (unchanged) ---
            Tool(
                name="temporal_evolution",
                description=(
                    "Show how activity around a topic or paper evolves over time. Given a "
                    "bibcode, returns citations-per-year for that paper. Given search "
                    "terms, returns publications-per-year matching that query. Useful for "
                    "tracking rising or fading topics and paper impact trajectories. Use "
                    "facet_counts instead when you want a single distribution by year "
                    "without a topic or bibcode anchor."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode_or_query": {
                            "type": "string",
                            "description": "A bibcode (citation trends) or search terms (pub volume)",
                        },
                        "year_start": {"type": "integer"},
                        "year_end": {"type": "integer"},
                    },
                    "required": ["bibcode_or_query"],
                },
            ),
            # --- facet_counts (unchanged) ---
            Tool(
                name="facet_counts",
                description=(
                    "Return a distribution of paper counts grouped by a single metadata "
                    "field: year, doctype, arxiv_class, database, bibgroup, or property. "
                    "Accepts the same filters as search to scope the distribution to a "
                    "subset. Useful for dataset overviews and filter discovery. Use "
                    "temporal_evolution instead when you need year-over-year trends tied "
                    "to a specific query or bibcode rather than a flat distribution."
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
                        "filters": _FILTERS_SCHEMA,
                        "limit": {"type": "integer", "default": 50},
                    },
                    "required": ["field"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        with _get_conn() as conn:
            # Resolve deprecated aliases for timeout lookup
            resolved_name = _DEPRECATED_ALIASES.get(name, name)
            _set_timeout(conn, resolved_name)
            t0 = time.monotonic()
            success = True
            error_msg: str | None = None
            try:
                result_json = _dispatch_tool(conn, name, arguments)
            except Exception as exc:
                success = False
                error_msg = str(exc)
                result_json = json.dumps({"error": error_msg})
                raise
            finally:
                latency_ms = (time.monotonic() - t0) * 1000
                _log_query(conn, name, arguments, latency_ms, success, error_msg)
            return [TextContent(type="text", text=result_json)]

    if _run_self_test:
        try:
            startup_self_test(server)
        except Exception:
            logger.critical("create_server: startup self-test failed — server will not start")
            raise

    return server


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _dispatch_tool(conn: psycopg.Connection, name: str, args: dict[str, Any]) -> str:
    """Route a tool call to the appropriate search function.

    Handles both new consolidated tool names and deprecated aliases.
    """
    t_start = time.monotonic()
    logger.info("tool_call: %s args=%s", name, list(args.keys()))

    # Check for deprecated alias
    deprecated = False
    original_name = name
    use_instead: str | None = None

    if name in _DEPRECATED_ALIASES:
        use_instead = _DEPRECATED_ALIASES[name]
        deprecated = True
        logger.info("deprecated_alias: %s -> %s", name, use_instead)

        # Transform args from old format to new format
        name, args = _transform_deprecated_args(original_name, use_instead, args)

    # Dispatch to the actual handler
    result_json = _dispatch_consolidated(conn, name, args)

    # Auto-track bibcodes in results
    _auto_track_bibcodes(result_json)

    # If this was a deprecated alias, wrap the result
    if deprecated and use_instead is not None:
        result_json = _wrap_deprecated(result_json, original_name, use_instead)

    elapsed_ms = (time.monotonic() - t_start) * 1000
    logger.info("tool_done: %s elapsed=%.1fms", original_name, elapsed_ms)
    return result_json


def _transform_deprecated_args(
    old_name: str, new_name: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Transform arguments from old tool format to new consolidated format."""
    new_args = dict(args)

    if old_name == "semantic_search":
        new_args["mode"] = "semantic"
        return "search", new_args

    if old_name == "keyword_search":
        new_args["query"] = new_args.pop("terms", new_args.get("query", ""))
        new_args["mode"] = "keyword"
        return "search", new_args

    if old_name == "get_citations":
        new_args["direction"] = "forward"
        return "citation_graph", new_args

    if old_name == "get_references":
        new_args["direction"] = "backward"
        return "citation_graph", new_args

    if old_name == "co_citation_analysis":
        new_args["method"] = "co_citation"
        return "citation_similarity", new_args

    if old_name == "bibliographic_coupling":
        new_args["method"] = "coupling"
        return "citation_similarity", new_args

    if old_name == "entity_search":
        new_args["action"] = "search"
        # Rename entity_name -> query
        if "entity_name" in new_args and "query" not in new_args:
            new_args["query"] = new_args.pop("entity_name")
        return "entity", new_args

    if old_name == "resolve_entity":
        new_args["action"] = "resolve"
        return "entity", new_args

    if old_name == "document_context":
        new_args["include_entities"] = True
        return "get_paper", new_args

    if old_name == "get_openalex_topics":
        new_args["include_entities"] = True
        return "get_paper", new_args

    if old_name == "get_paper_metrics":
        new_args["include_community"] = False
        return "graph_context", new_args

    if old_name == "explore_community":
        new_args["include_community"] = True
        return "graph_context", new_args

    if old_name == "get_author_papers":
        # Keep as direct handler for backward compatibility
        return "get_author_papers", new_args

    if old_name == "read_paper_section":
        return "read_paper", new_args

    if old_name == "search_within_paper":
        # Map old 'query' to 'search_query'
        if "query" in new_args:
            new_args["search_query"] = new_args.pop("query")
        return "read_paper", new_args

    if old_name == "get_citation_context":
        # Keep as direct handler — args have source_bibcode/target_bibcode
        return "get_citation_context", new_args

    # Session tools that are deprecated — handle directly in dispatch
    if old_name in {
        "add_to_working_set",
        "get_working_set",
        "get_session_summary",
        "clear_working_set",
    }:
        return old_name, new_args

    if old_name == "entity_profile":
        # entity_profile has its own dedicated handler (different schema
        # than get_paper), routed via _dispatch_consolidated.
        return "entity_profile", new_args

    return new_name, new_args


def _wrap_deprecated(result_json: str, original_name: str, use_instead: str) -> str:
    """Add deprecation metadata to a result."""
    try:
        data = json.loads(result_json)
        if isinstance(data, dict):
            data["deprecated"] = True
            data["use_instead"] = use_instead
            data["original_tool"] = original_name
            return json.dumps(data, indent=2, default=str)
    except (json.JSONDecodeError, TypeError):
        pass
    return result_json


def _dispatch_consolidated(conn: psycopg.Connection, name: str, args: dict[str, Any]) -> str:
    """Dispatch to the 13 consolidated tool handlers plus legacy session/health handlers."""

    # --- M1: Unified search ---
    if name == "search":
        return _handle_search(conn, args)

    # --- concept_search ---
    if name == "concept_search":
        result = search.concept_search(
            conn,
            args["query"],
            include_subtopics=args.get("include_subtopics", True),
            limit=args.get("limit", 20),
        )
        return _result_to_json(result)

    # --- M6: get_paper (absorbs document_context) ---
    if name == "get_paper":
        return _handle_get_paper(conn, args)

    # --- S1: read_paper ---
    if name == "read_paper":
        return _handle_read_paper(conn, args)

    # --- M3: citation_graph ---
    if name == "citation_graph":
        return _handle_citation_graph(conn, args)

    # --- M3: citation_similarity ---
    if name == "citation_similarity":
        return _handle_citation_similarity(conn, args)

    # --- citation_chain ---
    if name == "citation_chain":
        max_depth = max(1, min(args.get("max_depth", 5), 5))
        result = search.citation_chain(
            conn,
            args["source_bibcode"],
            args["target_bibcode"],
            max_depth=max_depth,
        )
        return _result_to_json(result)

    # --- M4: entity ---
    if name == "entity":
        return _handle_entity(conn, args)

    # --- entity_context ---
    if name == "entity_context":
        entity_id = args.get("entity_id")
        if entity_id is None:
            return json.dumps({"error": "entity_id is required"})
        try:
            entity_id = int(entity_id)
        except (TypeError, ValueError):
            return json.dumps({"error": "entity_id must be an integer"})
        result = search.get_entity_context(conn, entity_id)
        return _result_to_json(result)

    # --- S2: graph_context ---
    if name == "graph_context":
        return _handle_graph_context(conn, args)

    # --- M5: find_gaps ---
    if name == "find_gaps":
        return _handle_find_gaps(conn, args)

    # --- temporal_evolution ---
    if name == "temporal_evolution":
        result = search.temporal_evolution(
            conn,
            args["bibcode_or_query"],
            year_start=args.get("year_start"),
            year_end=args.get("year_end"),
        )
        return _result_to_json(result)

    # --- facet_counts ---
    if name == "facet_counts":
        filters = _parse_filters(args.get("filters"))
        limit = args.get("limit", 50)
        result = search.facet_counts(conn, args["field"], filters=filters, limit=limit)
        return _result_to_json(result)

    # --- Legacy handlers for deprecated session tools ---
    if name == "add_to_working_set":
        bibcodes = args.get("bibcodes", [])
        source_tool = args.get("source_tool", "unknown")
        entries = []
        for bib in bibcodes:
            entry = _session_state.add_to_working_set(
                bibcode=bib,
                source_tool=source_tool,
                source_context=args.get("source_context", ""),
                relevance_hint=args.get("relevance_hint", ""),
                tags=args.get("tags", []),
            )
            entries.append(dataclasses.asdict(entry))
        return json.dumps({"added": len(entries), "entries": entries}, indent=2, default=str)

    if name == "get_working_set":
        entries = _session_state.get_working_set()
        return json.dumps(
            {"entries": [dataclasses.asdict(e) for e in entries], "total": len(entries)},
            indent=2,
            default=str,
        )

    if name == "get_session_summary":
        summary = _session_state.get_session_summary()
        return json.dumps(summary, indent=2, default=str)

    if name == "clear_working_set":
        removed = _session_state.clear_working_set()
        return json.dumps({"removed": removed}, indent=2)

    if name == "get_citation_context":
        result = search.get_citation_context(
            conn,
            args["source_bibcode"],
            args["target_bibcode"],
        )
        return _result_to_json(result)

    if name == "get_author_papers":
        result = search.get_author_papers(
            conn,
            args["author_name"],
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
        )
        return _result_to_json(result)

    if name == "health_check":
        return _handle_health_check(conn)

    # Legacy entity_profile — dispatched only via deprecated alias layer,
    # not in list_tools(). Preserves original schema (raw extractions rows)
    # so existing callers don't break on the schema change to get_paper.
    if name == "entity_profile":
        return _handle_entity_profile(conn, args)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Individual tool handlers
# ---------------------------------------------------------------------------


def _handle_search(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified search: hybrid/semantic/keyword."""
    mode = args.get("mode", "hybrid")
    query = args["query"]
    filters = _parse_filters(args.get("filters"))
    limit = args.get("limit", 10)

    if mode == "keyword":
        result = search.lexical_search(conn, query, filters=filters, limit=limit)
        return _result_to_json(result)

    if mode == "semantic":
        model_name = "indus"
        if not _hnsw_index_exists(conn, model_name):
            return json.dumps(
                {
                    "error": "vector_index_unavailable",
                    "model_name": model_name,
                    "detail": (
                        f"HNSW index '{_hnsw_index_name(model_name)}' is not "
                        "available yet. Use mode='keyword' as a fallback."
                    ),
                },
                indent=2,
            )
        try:
            device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
            model, tokenizer = load_model(model_name, device=device)
            vectors = embed_batch(model, tokenizer, [query], batch_size=1)
            query_embedding = vectors[0]
        except ImportError:
            return json.dumps(
                {
                    "error": "transformers/torch not installed for embedding",
                    "hint": "pip install transformers torch",
                }
            )
        result = search.vector_search(
            conn,
            query_embedding,
            model_name=model_name,
            filters=filters,
            limit=limit,
        )
        return _result_to_json(result)

    # mode == "hybrid" (default)
    model_name = "indus"
    query_embedding = None
    if _hnsw_index_exists(conn, model_name):
        try:
            device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
            model, tokenizer = load_model(model_name, device=device)
            vectors = embed_batch(model, tokenizer, [query], batch_size=1)
            query_embedding = vectors[0]
        except ImportError:
            logger.warning("Embedding unavailable for hybrid; falling back to lexical-only")

    result = search.hybrid_search(
        conn,
        query,
        query_embedding=query_embedding,
        model_name=model_name,
        filters=filters,
        top_n=limit,
    )
    return _result_to_json(result)


def _handle_get_paper(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Get paper metadata, optionally with entities."""
    bibcode = args.get("bibcode", "")
    if not bibcode or not bibcode.strip():
        return json.dumps({"error": "bibcode must be a non-empty string"})

    include_entities = args.get("include_entities", False)

    # Auto-track as focused paper
    _session_state.track_focused(bibcode)

    if include_entities:
        result = search.get_document_context(conn, bibcode)
        return _result_to_json(result)

    result = search.get_paper(conn, bibcode)
    return _result_to_json(result)


def _handle_read_paper(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Read or search within a paper's full text."""
    bibcode = args["bibcode"]
    search_query = args.get("search_query")

    if search_query:
        result = search.search_within_paper(conn, bibcode, search_query)
        return _result_to_json(result)

    result = search.read_paper_section(
        conn,
        bibcode,
        section=args.get("section", "full"),
        char_offset=args.get("char_offset", 0),
        limit=args.get("limit", 5000),
    )
    return _result_to_json(result)


def _handle_citation_graph(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Get citations/references with direction control."""
    bibcode = args["bibcode"]
    direction = args.get("direction", "forward")
    limit = args.get("limit", 20)

    results: list[dict[str, Any]] = []

    if direction in ("forward", "both"):
        fwd = search.get_citations(conn, bibcode, limit=limit)
        if direction == "forward":
            return _result_to_json(fwd)
        results.append({"direction": "forward", "result": json.loads(_result_to_json(fwd))})

    if direction in ("backward", "both"):
        bwd = search.get_references(conn, bibcode, limit=limit)
        if direction == "backward":
            return _result_to_json(bwd)
        results.append({"direction": "backward", "result": json.loads(_result_to_json(bwd))})

    if direction == "both":
        return json.dumps({"bibcode": bibcode, "directions": results}, indent=2, default=str)

    return json.dumps({"error": f"Invalid direction: {direction}"})


def _handle_citation_similarity(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Co-citation or bibliographic coupling."""
    bibcode = args["bibcode"]
    method = args.get("method", "co_citation")
    min_overlap = args.get("min_overlap", 2)
    limit = args.get("limit", 20)

    if method == "co_citation":
        result = search.co_citation_analysis(conn, bibcode, min_overlap=min_overlap, limit=limit)
    elif method == "coupling":
        result = search.bibliographic_coupling(conn, bibcode, min_overlap=min_overlap, limit=limit)
    else:
        return json.dumps({"error": f"Invalid method: {method}. Use co_citation or coupling."})

    return _result_to_json(result)


def _handle_entity(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified entity search and resolution."""
    action = args.get("action", "search")
    query = args.get("query", "")

    if not query or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})

    if action == "resolve":
        resolver = EntityResolver(conn)
        candidates = resolver.resolve(
            query.strip(),
            discipline=args.get("discipline"),
            fuzzy=args.get("fuzzy", False),
        )
        return json.dumps(
            {
                "query": query.strip(),
                "candidates": [
                    {
                        "entity_id": c.entity_id,
                        "canonical_name": c.canonical_name,
                        "entity_type": c.entity_type,
                        "source": c.source,
                        "discipline": c.discipline,
                        "confidence": c.confidence,
                        "match_method": c.match_method,
                    }
                    for c in candidates
                ],
                "total": len(candidates),
            },
            indent=2,
            default=str,
        )

    if action == "search":
        entity_type = args.get("entity_type")
        _VALID_ENTITY_TYPES = {"methods", "datasets", "instruments", "materials"}
        if not entity_type or entity_type not in _VALID_ENTITY_TYPES:
            return json.dumps(
                {
                    "error": (
                        f"Invalid entity_type '{entity_type}'. "
                        f"Must be one of: {sorted(_VALID_ENTITY_TYPES)}"
                    )
                }
            )

        limit = min(args.get("limit", 20), 200)
        containment = json.dumps({entity_type: [query]})

        sql = """
            SELECT e.bibcode, e.extraction_type, e.extraction_version, e.payload,
                   p.title
            FROM extractions e
            JOIN papers p ON p.bibcode = e.bibcode
            WHERE e.payload @> %s::jsonb
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (containment, limit))
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
        return json.dumps({"papers": papers, "total": len(papers)}, indent=2, default=str)

    return json.dumps({"error": f"Invalid action: {action}. Use 'search' or 'resolve'."})


def _handle_graph_context(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Graph metrics and optional community exploration."""
    bibcode = args["bibcode"]
    include_community = args.get("include_community", False)

    metrics_result = search.get_paper_metrics(conn, bibcode)

    if not include_community:
        return _result_to_json(metrics_result)

    resolution = args.get("resolution", "coarse")
    limit = args.get("limit", 20)
    community_result = search.explore_community(conn, bibcode, resolution=resolution, limit=limit)

    # Merge metrics and community data
    metrics_data = json.loads(_result_to_json(metrics_result))
    community_data = json.loads(_result_to_json(community_result))
    combined = {
        "bibcode": bibcode,
        "metrics": metrics_data,
        "community": community_data,
    }
    return json.dumps(combined, indent=2, default=str)


def _handle_find_gaps(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Find gaps using implicit session state (focused papers)."""
    resolution = args.get("resolution", "coarse")
    limit = min(args.get("limit", 20), 200)
    clear_first = args.get("clear_first", False)

    if clear_first:
        _session_state.clear_focused()
        _session_state.clear_working_set()

    _RESOLUTION_COLS: dict[str, str] = {
        "coarse": "community_id_coarse",
        "medium": "community_id_medium",
        "fine": "community_id_fine",
    }
    community_col = _RESOLUTION_COLS.get(resolution)
    if community_col is None:
        return json.dumps(
            {
                "error": f"Invalid resolution: {resolution}. Must be one of {sorted(_RESOLUTION_COLS)}"
            }
        )

    # Use focused papers (from get_paper calls) as primary source,
    # fall back to working set for backward compatibility
    ws_bibcodes = _session_state.get_focused_papers()
    if not ws_bibcodes:
        ws_bibcodes = [e.bibcode for e in _session_state.get_working_set()]
    ws_bibcodes = ws_bibcodes[:200]

    if not ws_bibcodes:
        return json.dumps(
            {
                "papers": [],
                "total": 0,
                "message": "No focused papers yet. Use get_paper to inspect papers first.",
            },
            indent=2,
        )

    query = f"""
        SELECT DISTINCT p.bibcode, p.title, pm.pagerank,
               pm.{community_col} AS community_id
        FROM citation_edges ce
        JOIN papers p ON p.bibcode = ce.source_bibcode
        JOIN paper_metrics pm ON pm.bibcode = p.bibcode
        WHERE ce.target_bibcode = ANY(%s)
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
    return json.dumps(
        {"papers": papers, "total": len(papers), "resolution": resolution},
        indent=2,
        default=str,
    )


def _handle_health_check(conn: psycopg.Connection) -> str:
    """Internal health check (not in list_tools)."""
    status: dict[str, Any] = {"pool": "no_pool", "model_cached": False, "db": "unknown"}
    status["model_cached"] = len(_model_cache) > 0
    status["cached_models"] = [f"{k[0]}@{k[1]}" for k in _model_cache]
    status["pool"] = "active" if _pool is not None else "no_pool"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        status["db"] = "ok"
    except Exception:
        status["db"] = "error"
    return json.dumps(status, indent=2)


def _handle_entity_profile(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy entity_profile handler: returns raw extractions table rows.

    Preserves the pre-consolidation schema for backward compatibility with
    external callers that still reference entity_profile. New code should
    use get_paper(include_entities=true) instead.
    """
    bibcode = args.get("bibcode")
    if not bibcode:
        return json.dumps({"error": "bibcode is required"})

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
            "created_at": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]
    return json.dumps(
        {"bibcode": bibcode, "extractions": extractions, "total": len(extractions)},
        indent=2,
        default=str,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the MCP server on stdio."""
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import ServerCapabilities

    server = create_server()
    init_options = InitializationOptions(
        server_name="scix",
        server_version="0.2.0",
        capabilities=ServerCapabilities(tools={}),
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    finally:
        _shutdown()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(main())
