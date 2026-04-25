"""MCP server exposing 15 consolidated tools for agent navigation of the SciX corpus.

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
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator

import psycopg

from scix import search
from scix.db import DEFAULT_DSN
from scix.embed import _model_cache, clear_model_cache, embed_batch, load_model
from scix.entity_resolver import EntityResolver
from scix.jit.disambiguator import disambiguate_query
from scix.search import CrossEncoderReranker
from scix.session import SessionState, WorkingSetEntry

# Optional Qdrant-backed discovery tool. Feature-flagged via QDRANT_URL so the
# default production deployment (Postgres-only) is unaffected.
try:
    from scix import qdrant_tools as _qdrant_tools
except ImportError:  # pragma: no cover — qdrant-client not installed
    _qdrant_tools = None  # type: ignore[assignment]


def _qdrant_enabled() -> bool:
    return _qdrant_tools is not None and _qdrant_tools.is_enabled()

# Optional import — viz/trace_stream is only needed when the viz extras are
# installed. When absent we fall back to a no-op, and the emission hook in
# :func:`call_tool` silently skips publishing. This keeps the MCP server
# runnable in minimal deployments that don't ship FastAPI.
try:
    from scix.viz import trace_stream as _trace_stream
except ImportError:  # pragma: no cover — viz extras not installed
    _trace_stream = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Cap the number of bibcodes emitted per TraceEvent to keep event payloads
# small. SSE consumers typically only need a handful of bibcodes for linkage.
_MAX_TRACE_BIBCODES: int = 20

# ---------------------------------------------------------------------------
# Server-level session identity (stable for the lifetime of the process)
# ---------------------------------------------------------------------------

_server_session_id: str = str(uuid.uuid4())
_is_test_session: bool = bool(os.environ.get("SCIX_TEST_DSN"))

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


# Priority-ordered list of argument keys that carry the user query text.
_QUERY_ARG_KEYS: tuple[str, ...] = (
    "query",
    "bibcode",
    "author_name",
    "source_bibcode",
    "bibcode_or_query",
    "entity_name",
    "entity_id",
    "field",
    "search_query",
)


def _extract_query_text(params: dict[str, Any]) -> str | None:
    """Extract the most meaningful query string from tool arguments.

    Returns the value of the first recognised key found in *params*,
    or ``None`` if no query-like argument is present.
    """
    for key in _QUERY_ARG_KEYS:
        val = params.get(key)
        if val is not None:
            return str(val)
    return None


def _extract_result_count(result_json: str) -> int:
    """Best-effort extraction of result count from a tool's JSON output.

    Checks, in order:
      1. ``total`` (explicit count from SearchResult)
      2. ``len(papers)``
      3. ``len(results)``

    Returns 0 on parse failure or when the result represents an error.
    """
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return 0
    if not isinstance(data, dict):
        return 0
    if "error" in data:
        return 0
    if "total" in data:
        try:
            return int(data["total"])
        except (TypeError, ValueError):
            return 0
    if "papers" in data and isinstance(data["papers"], list):
        return len(data["papers"])
    if "results" in data and isinstance(data["results"], list):
        return len(data["results"])
    return 0


def _log_query(
    conn: psycopg.Connection,
    tool_name: str,
    params: dict[str, Any],
    latency_ms: float,
    success: bool,
    error_msg: str | None = None,
    *,
    result_json: str | None = None,
    session_id: str | None = None,
    is_test: bool = False,
) -> None:
    """Write a row to query_log with both legacy and migration-031 columns.

    Best-effort: failures are logged, not raised.
    """
    try:
        params_json = json.dumps(params, default=str)
        query_text = _extract_query_text(params)
        result_count = _extract_result_count(result_json) if result_json else 0
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log (
                    tool_name, params_json, latency_ms, success, error_msg,
                    tool, query, result_count, session_id, is_test
                )
                VALUES (%s, %s::jsonb, %s, %s, %s,
                        %s, %s, %s, %s, %s)
                """,
                (
                    tool_name,
                    params_json,
                    latency_ms,
                    success,
                    error_msg,
                    tool_name,
                    query_text,
                    result_count,
                    session_id,
                    is_test,
                ),
            )
        conn.commit()
    except Exception:
        logger.warning("Failed to log query for tool=%s", tool_name, exc_info=True)


def _extract_bibcodes_from_result(result_json: str | None) -> tuple[str, ...]:
    """Best-effort bibcode extraction from a tool's JSON result.

    Handles two common shapes:
      * ``{"papers": [{"bibcode": ...}, ...]}`` — multi-paper result.
      * ``{"bibcode": "..."}`` — single-paper result.

    Returns an empty tuple on any parse failure or when the result
    represents an error payload. The result is capped at
    :data:`_MAX_TRACE_BIBCODES` entries to keep emitted TraceEvents small.
    """
    if not result_json:
        return ()
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(data, dict):
        return ()

    bibcodes: list[str] = []
    papers = data.get("papers")
    if isinstance(papers, list):
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            bc = paper.get("bibcode")
            if isinstance(bc, str):
                bibcodes.append(bc)
                if len(bibcodes) >= _MAX_TRACE_BIBCODES:
                    break

    if not bibcodes:
        bc = data.get("bibcode")
        if isinstance(bc, str):
            bibcodes.append(bc)

    return tuple(bibcodes)


def _emit_trace_event(
    tool_name: str,
    latency_ms: float,
    params: dict[str, Any],
    result_json: str | None,
    success: bool,
) -> None:
    """Fire-and-forget TraceEvent emission to :mod:`scix.viz.trace_stream`.

    Called once per MCP tool dispatch (both success and failure paths).
    If :mod:`scix.viz.trace_stream` is not importable, this is a no-op.
    All exceptions are swallowed — trace emission must never break the
    tool-call hot path.
    """
    if _trace_stream is None:
        return
    try:
        bibcodes = _extract_bibcodes_from_result(result_json)
        result_summary: str | None = None
        if not success and result_json:
            try:
                parsed = json.loads(result_json)
                if isinstance(parsed, dict) and "error" in parsed:
                    result_summary = f"error: {parsed['error']}"
            except (json.JSONDecodeError, TypeError):
                result_summary = None
        event = _trace_stream.TraceEvent(
            tool_name=tool_name,
            latency_ms=latency_ms,
            params=dict(params) if params else {},
            result_summary=result_summary,
            bibcodes=bibcodes,
        )
        _trace_stream.publish(event)
    except Exception:  # pragma: no cover — defensive, emission must not raise
        logger.debug("trace emission failed for tool=%s", tool_name, exc_info=True)


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


# ---------------------------------------------------------------------------
# Coverage-bias discipline (PRD prd_full_text_applications_v2 — always-on)
# ---------------------------------------------------------------------------
#
# Every MCP response that surfaces full-text-derived signals (entity hits
# from staging.extractions, read_paper body chunks, search_within_paper
# matches) carries a top-level ``coverage_note`` string. The note tells the
# agent what fraction of the corpus has full-text coverage and points to the
# canonical analysis doc so cross-corpus comparisons are interpreted with
# the right caveat.
#
# Default policy is always-on per the PRD's Open Question default decision.

#: Repo-relative path to the coverage-bias report produced by M1.
_COVERAGE_BIAS_PATH: Path = Path(__file__).resolve().parents[2] / "results" / "full_text_coverage_bias.json"

#: Documentation path included verbatim in every coverage_note (so the link
#: survives even when the JSON file is unreadable).
_COVERAGE_DOC_PATH: str = "docs/full_text_coverage_analysis.md"


def _coverage_note_path() -> Path:
    """Return the path the coverage-bias JSON is loaded from.

    Indirected so tests can patch this single function instead of reaching
    into module globals.
    """
    return _COVERAGE_BIAS_PATH


@lru_cache(maxsize=1)
def _coverage_note() -> str:
    """Return the cached coverage-note string for the current process.

    Reads ``results/full_text_coverage_bias.json`` once and formats the
    note. If the file is missing or malformed the note still mentions the
    docs path so the agent can navigate to the explanation.
    """
    path = _coverage_note_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        # Prefer the precomputed pct field; fall back to ratio when
        # absent so older report versions still work.
        pct: float | None = None
        if isinstance(data.get("fulltext_pct"), (int, float)):
            pct = float(data["fulltext_pct"])
        else:
            ft_total = data.get("fulltext_total")
            corpus_total = data.get("corpus_total")
            if (
                isinstance(ft_total, (int, float))
                and isinstance(corpus_total, (int, float))
                and corpus_total > 0
            ):
                pct = (float(ft_total) / float(corpus_total)) * 100.0
        if pct is None:
            raise ValueError("coverage report missing fulltext_pct/fulltext_total")
        return (
            f"Coverage note: full-text coverage is {pct:.1f}% of the corpus "
            f"— see {_COVERAGE_DOC_PATH} for safe/unsafe query patterns."
        )
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as err:
        logger.warning(
            "coverage_note: could not load %s (%s); using fallback note",
            path,
            err,
        )
        return (
            "Coverage note: full-text coverage stats unavailable — "
            f"see {_COVERAGE_DOC_PATH} for safe/unsafe query patterns."
        )


def _reset_coverage_note_cache() -> None:
    """Drop the cached coverage_note string. Test-only helper."""
    _coverage_note.cache_clear()


def _inject_coverage_note(result_json: str) -> str:
    """Insert ``coverage_note`` at the top level of an existing JSON response.

    The MCP layer serialises results with ``json.dumps(..., indent=2)`` so
    we round-trip through ``json.loads`` to preserve the existing shape and
    sort order. If the response is not a JSON object (e.g. a JSON array or
    a primitive — which our handlers do not currently emit), the original
    string is returned unchanged so we never corrupt the protocol.
    """
    try:
        parsed = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return result_json
    if not isinstance(parsed, dict):
        return result_json
    parsed["coverage_note"] = _coverage_note()
    return json.dumps(parsed, indent=2, default=str)


def _parse_filters(filters: dict[str, Any] | None = None) -> search.SearchFilters:
    """Parse a filter dict into a SearchFilters instance.

    Entity filter lists are validated and size-capped at the MCP boundary —
    the SearchFilters dataclass does type validation, but the list-size cap is
    a boundary concern (blast-radius control) and lives here.
    """
    if not filters:
        return search.SearchFilters()

    entity_types = _validate_entity_list(filters.get("entity_types"), "entity_types", str)
    entity_ids = _validate_entity_list(filters.get("entity_ids"), "entity_ids", int)

    return search.SearchFilters(
        year_min=filters.get("year_min"),
        year_max=filters.get("year_max"),
        arxiv_class=filters.get("arxiv_class"),
        doctype=filters.get("doctype"),
        first_author=filters.get("first_author"),
        entity_types=entity_types,
        entity_ids=entity_ids,
    )


def _validate_entity_list(raw: Any, name: str, element_type: type) -> list[Any] | None:
    """Validate an optional entity-filter list at the MCP boundary.

    Returns the list unchanged (or None). Empty lists pass through — the
    SearchFilters dataclass normalizes them to None. Raises ValueError for
    bad types or oversized payloads so the error surfaces as a clean
    protocol-level response.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a list, got {type(raw).__name__}")
    if len(raw) > MAX_ENTITY_FILTER_ITEMS:
        raise ValueError(f"{name} has {len(raw)} items, max {MAX_ENTITY_FILTER_ITEMS}")
    for item in raw:
        # bool is a subclass of int; reject it explicitly for entity_ids so
        # an agent passing `True` does not end up querying entity_id=1.
        if element_type is int and isinstance(item, bool):
            raise ValueError(f"{name} items must be int, got bool")
        if not isinstance(item, element_type):
            raise ValueError(
                f"{name} items must be {element_type.__name__}, got {type(item).__name__}"
            )
    return raw


_MIN_YEAR = 1900
_MAX_YEAR = 2100


def _coerce_year(raw: Any, name: str) -> int | None:
    """Coerce an optional year argument from the MCP schema to a bounded int.

    MCP inputSchema validation is advisory; callers can send strings, floats,
    or out-of-range ints. Enforce the contract at the dispatch boundary so
    malformed input surfaces as a clean ValueError rather than a downstream
    psycopg type error or a multi-century SQL scan.
    """
    if raw is None:
        return None
    try:
        year = int(raw)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from err
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        raise ValueError(f"{name} must be in [{_MIN_YEAR}, {_MAX_YEAR}], got {year}")
    return year


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


# ---------------------------------------------------------------------------
# Cross-encoder reranker (lazy singleton)
# ---------------------------------------------------------------------------
#
# The default value is intentionally ``'off'``. The M1 ablation
# (commit 06a6cc3, see PRD prd_cross_encoder_reranker_local.md) showed both
# candidate cross-encoders REGRESS retrieval quality on this corpus:
#   * ms-marco-MiniLM-L-12-v2: nDCG@10 0.3255 -> 0.2802 (Δ=-0.0453, p=0.042)
#   * BAAI/bge-reranker-large: nDCG@10 0.3255 -> 0.2699 (Δ=-0.0556, p=0.026)
# Bonferroni-corrected significance threshold = 0.025 — the M4 rollout gate
# FAILS for both. Operators can still flip a non-'off' model on for
# experimentation (e.g. when a domain-tuned cross-encoder lands), but the
# production default stays off.

# Map env-var values to model_name strings consumed by CrossEncoderReranker.
_RERANK_MODEL_ALIASES: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-large": "BAAI/bge-reranker-large",
}

# Cap above which the reranker is bypassed even when use_rerank=True.
# per PRD prd_cross_encoder_reranker_local.md M3: rerank only top_k <= 20
_RERANK_TOP_K_CAP: int = 20

# Module-level cache so repeated tool calls reuse the same reranker instance.
# Construction is cheap (no weights loaded until first __call__), but caching
# avoids repeated dict lookups and keeps the lazy weight-load amortised across
# the process lifetime.
_default_reranker_cache: dict[str, CrossEncoderReranker | None] = {}


def _resolve_default_reranker_model() -> str | None:
    """Resolve ``SCIX_RERANK_DEFAULT_MODEL`` to a sentence-transformers model name.

    Returns ``None`` when the env var is unset, set to ``'off'``, or set to an
    unrecognised value (with a warning logged for the latter — consistent with
    how other env-driven config in this module degrades rather than crashes).
    """
    raw = os.environ.get("SCIX_RERANK_DEFAULT_MODEL", "off").strip().lower()
    if raw == "off":
        return None
    if raw in _RERANK_MODEL_ALIASES:
        return _RERANK_MODEL_ALIASES[raw]
    logger.warning(
        "Unknown SCIX_RERANK_DEFAULT_MODEL=%r; falling back to 'off'. "
        "Allowed values: 'off', 'minilm', 'bge-large'.",
        raw,
    )
    return None


def _get_default_reranker() -> CrossEncoderReranker | None:
    """Return the configured cross-encoder reranker, or ``None`` when disabled.

    Lazy: when ``SCIX_RERANK_DEFAULT_MODEL='off'`` (the default), no
    ``CrossEncoderReranker`` instance is constructed. When a non-'off' model is
    configured, the reranker object is built on first call and cached for the
    lifetime of the process; model weights are loaded lazily inside
    ``CrossEncoderReranker.__call__`` on first rerank.
    """
    model_name = _resolve_default_reranker_model()
    if model_name is None:
        return None
    cached = _default_reranker_cache.get(model_name)
    if cached is not None:
        return cached
    reranker = CrossEncoderReranker(model_name=model_name)
    _default_reranker_cache[model_name] = reranker
    return reranker


def _reset_default_reranker_cache() -> None:
    """Test hook: drop the cached singleton so env changes take effect."""
    _default_reranker_cache.clear()


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

# Cap list sizes at the MCP boundary to prevent pathological query construction
# (e.g. a misbehaving agent sending thousands of entity ids). This is a blast-
# radius control, not a correctness check — document_entities_canonical has
# the indexes to handle reasonable lists efficiently.
MAX_ENTITY_FILTER_ITEMS = 100

_FILTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "arxiv_class": {"type": "string"},
        "doctype": {"type": "string"},
        "first_author": {"type": "string"},
        "entity_types": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": MAX_ENTITY_FILTER_ITEMS,
            "description": (
                "Restrict results to papers linked to at least one entity of "
                "these types (e.g. 'instrument', 'mission', 'dataset'). "
                f"Empty list disables the filter. Max {MAX_ENTITY_FILTER_ITEMS} items."
            ),
        },
        "entity_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "maxItems": MAX_ENTITY_FILTER_ITEMS,
            "description": (
                "Restrict results to papers linked to at least one of these "
                "specific entity ids (from the entities table). "
                f"Empty list disables the filter. Max {MAX_ENTITY_FILTER_ITEMS} items."
            ),
        },
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
    # PRD MH-4 — Deep Search v1: provenance walk + replication enumeration
    "claim_blame",
    "find_replications",
)

# Tools that appear only when an optional backend is wired up.
_OPTIONAL_TOOLS: tuple[str, ...] = ("find_similar_by_examples",)


def _expected_tool_set() -> set[str]:
    tools = set(EXPECTED_TOOLS)
    if _qdrant_enabled():
        tools.update(_OPTIONAL_TOOLS)
    return tools


def startup_self_test(server: Any = None) -> dict[str, Any]:
    """Validate that list_tools() returns the EXPECTED_TOOLS set with valid schemas.

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

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                asyncio.run, handler(ListToolsRequest(method="tools/list"))
            ).result(timeout=10)
    else:
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
    expected_set = _expected_tool_set()
    if tool_count != len(expected_set):
        errors.append(
            f"expected exactly {len(expected_set)} tools, got {tool_count}"
        )

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

    # PRD MH-4 acceptance criterion 7: invoke claim_blame and
    # find_replications when SCIX_TEST_DSN is set so the self-test catches
    # SQL/wiring breakage end-to-end. We use defensive try/except — empty
    # results are acceptable (citation_contexts.intent may be all NULL until
    # the SciCite backfill runs), but a raised exception fails the test.
    if os.environ.get("SCIX_TEST_DSN"):
        smoke_errors = _smoke_call_new_tools()
        if smoke_errors:
            status["smoke_errors"] = smoke_errors
            logger.critical(
                "startup_self_test FAILED smoke calls: %s", smoke_errors
            )
            raise RuntimeError(f"startup_self_test smoke calls failed: {smoke_errors}")

    logger.info(
        "startup_self_test OK: %d tools registered (%s)",
        tool_count,
        ", ".join(sorted(tool_names)),
    )
    return status


def _smoke_call_new_tools() -> list[str]:
    """Invoke claim_blame and find_replications against SCIX_TEST_DSN.

    Returns a list of error strings (empty on success). Empty result sets
    are NOT errors — only raised exceptions are. This matches PRD MH-4
    acceptance criterion 7's "gracefully handle the case where
    citation_contexts.intent is all NULL" requirement.
    """
    errors: list[str] = []
    try:
        from scix.claim_blame import claim_blame
        from scix.find_replications import find_replications
    except ImportError as exc:
        return [f"import: {exc}"]

    try:
        with _get_conn() as conn:
            try:
                claim_blame("startup self-test claim", conn=conn)
            except Exception as exc:  # noqa: BLE001 — log + report
                errors.append(f"claim_blame: {exc}")
            try:
                find_replications("0000NoSuchBibcode000", conn=conn)
            except Exception as exc:  # noqa: BLE001 — log + report
                errors.append(f"find_replications: {exc}")
    except Exception as exc:  # noqa: BLE001 — pool acquire failure
        errors.append(f"pool: {exc}")
    return errors


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


def create_server(_run_self_test: bool = True):
    """Create and configure the MCP server with 15 consolidated tools.

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
        tool_list: list[Tool] = [
            # --- M1: Unified search ---
            Tool(
                name="search",
                description=(
                    "Search the scientific literature corpus (32M papers: NASA ADS "
                    "astro/planetary/earth/helio/biophysics + the arxiv mirror across "
                    "cs.*, stat.*, physics.*, etc.) for papers matching a natural-language "
                    "query. Returns ranked papers with titles, abstracts, authors, years, "
                    "and citation counts. Defaults to hybrid mode, the best general-purpose "
                    "search. For broad multi-keyword queries, pass filters.arxiv_class "
                    "(e.g. 'cs.SE') or a filters.year_min — unscoped queries run a full-text "
                    "scan over all 32M papers and may hit the statement timeout. "
                    "Use concept_search instead when the query is a "
                    "formal astronomy taxonomy term (e.g., 'Exoplanets'). Use entity with "
                    "action='search' when looking up a named method, dataset, or instrument. "
                    "Optional filters.entity_types / filters.entity_ids restrict results to "
                    "papers linked to entities of given types (e.g. ['instrument']) or to "
                    "specific entity ids resolved via the entity tool — useful for queries "
                    "like 'papers about JWST' once JWST's entity id is known."
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
                        "disambiguate": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "If true (default), check the query for ambiguous entity "
                                "mentions (e.g., 'Hubble' = mission or person). If ambiguity "
                                "is detected, the server returns disambiguation candidates "
                                "instead of search results. Set false to run the search directly."
                            ),
                        },
                        "use_rerank": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "If true (default) AND the SCIX_RERANK_DEFAULT_MODEL env var "
                                "is set to a model alias other than 'off' AND limit <= 20, "
                                "rerank the fused candidate set with a cross-encoder. The "
                                "production default for SCIX_RERANK_DEFAULT_MODEL is 'off' "
                                "because the M1 ablation showed both candidate rerankers "
                                "regress nDCG@10 on this corpus; this flag exists so an "
                                "operator can flip on a future domain-tuned reranker without "
                                "code changes. Setting use_rerank=false bypasses reranking "
                                "even when a model is configured."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
            # --- concept_search (multi-vocabulary router, dbl.7) ---
            Tool(
                name="concept_search",
                description=(
                    "Look up concepts across controlled vocabularies and return ranked "
                    "candidates tagged with their source vocabulary. Searched by default: "
                    "UAT (astronomy), OpenAlex Topics, ACM CCS (computing), MSC "
                    "(mathematics), PhySH (physics), GCMD (earth science), MeSH "
                    "(biomedical), NCBI Taxonomy (organisms), ChEBI (chemistry), Gene "
                    "Ontology (biology). Pass vocabulary=['mesh'] (or a single string) "
                    "to restrict search. Accepts a concept label (case-insensitive), an "
                    "alternate label, or a URI (concept_id or external_uri). Returns "
                    "concept hits in metadata.concepts; for UAT hits, also returns "
                    "papers tagged with the best concept (and its descendants by "
                    "default). Use search instead when the query is free-form natural "
                    "language rather than a curated taxonomy term."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concept label, alternate label, or URI",
                        },
                        "vocabulary": {
                            "description": (
                                "Restrict search to one or more vocabularies. Allowed: "
                                "'uat', 'openalex', 'acm_ccs', 'msc', 'physh', 'gcmd', "
                                "'mesh', 'ncbi_tax', 'chebi', 'gene_ontology'. "
                                "Omit to search all."
                            ),
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ],
                        },
                        "include_subtopics": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "When the best hit is a UAT concept, include papers "
                                "tagged with descendants in the hierarchy."
                            ),
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
                        "role": {
                            "type": "string",
                            "enum": [
                                "background",
                                "method",
                                "result",
                                "conclusion",
                                "other",
                            ],
                            "description": (
                                "Optional canonical role. When provided, selects "
                                "the first parsed section whose name maps to this "
                                "role (e.g. 'method' picks Methods/Observations/"
                                "Data). Takes precedence over 'section'."
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
                            "enum": [
                                "methods",
                                "datasets",
                                "instruments",
                                "materials",
                                "negative_result",
                                "quant_claim",
                            ],
                            "description": (
                                "Entity type (required for action=search). "
                                "'methods'/'datasets'/'instruments'/'materials' "
                                "search the in-line containment payload. "
                                "'negative_result' returns null-finding spans "
                                "(M3) with evidence_span. 'quant_claim' returns "
                                "extracted numeric claims (M4) with payload "
                                "{value, uncertainty, unit}; pass entity_name "
                                "to filter to one canonical quantity (e.g. 'H0')."
                            ),
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
                        "min_confidence_tier": {
                            "type": "integer",
                            "enum": [1, 2, 3],
                            "description": (
                                "Only return extractions whose confidence_tier is "
                                ">= this value (1=low, 2=medium, 3=high). Omit to "
                                "include all tiers. Applies to action='search' only."
                            ),
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Filter extractions to these provenance sources "
                                "only (e.g. ['metadata', 'ner', 'llm']). Omit to "
                                "include all sources. Applies to action='search' "
                                "only."
                            ),
                        },
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
                    "and hub metrics, and the communities it belongs to under all three "
                    "signals (citation, semantic, taxonomic) at coarse/medium/fine "
                    "resolution. The response includes a top-level `communities` block "
                    "with a sub-block per signal, each carrying `community_id`, `label`, "
                    "and `top_keywords` where available. Optionally also returns sibling "
                    "papers in the same community ranked by influence; the `signal` "
                    "parameter selects which community signal to explore for siblings. "
                    "Use citation_graph instead when you want direct citing or cited "
                    "papers rather than computed scores and community membership."
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
                        "signal": {
                            "type": "string",
                            "enum": ["citation", "semantic", "taxonomic"],
                            "default": "semantic",
                            "description": (
                                "Community signal to explore for sibling papers: "
                                "'citation' (co-citation-derived Leiden), 'semantic' "
                                "(INDUS-embedding k-means), or 'taxonomic' (arXiv-class). "
                                "Invalid values return a structured error."
                            ),
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
                    "cite papers you already inspected via get_paper. Requires a non-empty "
                    "working set — call get_paper on one or more papers first, otherwise "
                    "this returns nothing. Helps catch adjacent literature you might be "
                    "missing during a research session. Reads from implicit session state "
                    "tracked across get_paper calls. Use citation_graph instead when you "
                    "want direct citations of a single paper rather than cross-community "
                    "gap detection. The 'signal' parameter picks which community partition "
                    "to traverse: 'semantic' (default, INDUS k-means, full 32M-paper "
                    "coverage) or 'citation' (currently offline — Leiden Phase B has not "
                    "completed, so this path returns empty)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "enum": ["semantic", "citation"],
                            "default": "semantic",
                            "description": (
                                "Which community partition to traverse. "
                                "'semantic' covers all 32M papers today; "
                                "'citation' is offline until Leiden Phase B lands."
                            ),
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["coarse", "medium", "fine"],
                            "default": "medium",
                            "description": (
                                "Community resolution. 'medium' (default, "
                                "k=200) gives sharp CS-readable labels. "
                                "'coarse' (k=20) lumps many disciplines into "
                                "mega-buckets; 'fine' (k=2000) has crisper "
                                "partitions but TF-IDF labels get noisy as "
                                "buckets shrink (driven by outlier terms)."
                            ),
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
            # --- temporal_evolution ---
            Tool(
                name="temporal_evolution",
                description=(
                    "Show how activity around a topic or paper evolves over time. Given a "
                    "bibcode, returns citations-per-year for that paper. Given search "
                    "terms, returns publications-per-year plus per-year 'buckets' with "
                    "top anchor papers (ranked by PageRank) and dominant communities, so "
                    "a single call yields a usable temporal narrative instead of raw "
                    "counts. Useful for tracking rising or fading topics and paper impact "
                    "trajectories. Use facet_counts instead when you want a single "
                    "distribution by year without a topic or bibcode anchor."
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
            # --- PRD MH-4: claim_blame ---
            Tool(
                name="claim_blame",
                description=(
                    "Trace a claim back to its chronologically earliest non-retracted "
                    "origin paper by walking reverse references over all citation "
                    "contexts. Returns the origin bibcode, a Hop chain that surfaces "
                    "intent and intent_weight at every step, a confidence score in "
                    "[0,1], and a list of retraction warnings for any paper in the "
                    "lineage with a retraction event. Ranking is "
                    "(chronological_priority, intent_weight, semantic_match) with "
                    "weights {result_comparison: 1.0, method: 0.6, background: 0.3}. "
                    "Use citation_chain instead when you already know both endpoints "
                    "and just want a path; use find_replications to enumerate forward "
                    "citations to a paper."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_text": {
                            "type": "string",
                            "description": (
                                "Natural-language claim to trace, e.g. "
                                "'local H0 measurement is 73 km/s/Mpc'."
                            ),
                        },
                        "scope": {
                            "type": "object",
                            "description": (
                                "Optional ResearchScope filters (year_window, "
                                "community_ids, methodology_class, instruments, "
                                "exclude_authors, exclude_funders, min_venue_tier, "
                                "leiden_resolution). All keys optional."
                            ),
                        },
                        "candidate_limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Max claim-asserting candidates to seed.",
                        },
                        "lineage_limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Max hops to walk per candidate.",
                        },
                    },
                    "required": ["claim_text"],
                },
            ),
            # --- PRD MH-4: find_replications ---
            Tool(
                name="find_replications",
                description=(
                    "Return forward citations to a target paper, ranked by intent_weight, "
                    "with each citation annotated with citation intent, an inferred "
                    "replication relation (replicates / refutes / qualifies / partial / "
                    "unknown), and a hedge_present flag. Relation and hedge are derived "
                    "from a documented heuristic substitute for NegBERT (see module "
                    "docstring); NegBERT is the future drop-in. Use citation_graph with "
                    "direction='forward' instead when you want raw forward citations "
                    "without relation inference; use claim_blame when you want the "
                    "earliest origin of a claim rather than its replications."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_bibcode": {
                            "type": "string",
                            "description": "ADS bibcode whose forward citations to enumerate.",
                        },
                        "relation": {
                            "type": "string",
                            "enum": ["replicates", "refutes", "qualifies", "partial"],
                            "description": (
                                "Optional filter on relation_inferred. Omit to "
                                "return all relations including unknown."
                            ),
                        },
                        "scope": {
                            "type": "object",
                            "description": (
                                "Optional ResearchScope filters; year_window applies to "
                                "the citing paper's year."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 50,
                            "description": "Max citations to return.",
                        },
                    },
                    "required": ["target_bibcode"],
                },
            ),
        ]

        # Optional Qdrant-backed discovery tool — only registered when
        # QDRANT_URL is set. Lets the agent say "more like these papers, less
        # like those" with optional payload filtering. Not a replacement for
        # Postgres-backed search; an additive capability.
        if _qdrant_enabled():
            tool_list.append(Tool(
                name="find_similar_by_examples",
                description=(
                    "Return papers most similar to a set of positive example "
                    "bibcodes and least similar to negative examples, using "
                    "Qdrant's discovery API over INDUS embeddings. Supports "
                    "optional payload filters on year, doctype, arxiv_class, "
                    "and coarse citation community. Best for \"more like "
                    "these, less like those\" exploration — distinct from "
                    "`search` (query text) and `citation_similarity` "
                    "(co-citation / bibliographic coupling)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "positive_bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Papers the result should resemble.",
                        },
                        "negative_bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                            "description": "Papers the result should avoid.",
                        },
                        "limit": {"type": "integer", "default": 10},
                        "year_min": {"type": "integer"},
                        "year_max": {"type": "integer"},
                        "doctype": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "community_semantic": {"type": "integer"},
                        "arxiv_class": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["positive_bibcodes"],
                },
            ))
        return tool_list

    @server.call_tool()
    async def call_tool_handler(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        result_json = call_tool(name, arguments)
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


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Synchronously dispatch a tool by name and return its JSON result.

    Mirrors the lifecycle of the MCP request handler registered in
    :func:`create_server`: acquires a pooled connection, sets the per-tool
    statement_timeout, dispatches via :func:`_dispatch_tool`, and — in a
    ``finally`` block — records a ``query_log`` row and emits a
    :class:`scix.viz.trace_stream.TraceEvent`. Lets callers (e.g. the viz
    demo endpoint) drive the MCP tool surface in-process without going
    through the asyncio request handler, while still producing exactly one
    log row and one trace event per call.
    """
    with _get_conn() as conn:
        resolved_name = _DEPRECATED_ALIASES.get(name, name)
        _set_timeout(conn, resolved_name)
        t0 = time.monotonic()
        success = True
        error_msg: str | None = None
        result_json: str = "{}"
        try:
            result_json = _dispatch_tool(conn, name, arguments)
        except Exception as exc:
            success = False
            error_msg = str(exc)
            result_json = json.dumps({"error": error_msg})
            raise
        finally:
            latency_ms = (time.monotonic() - t0) * 1000
            _log_query(
                conn,
                name,
                arguments,
                latency_ms,
                success,
                error_msg,
                result_json=result_json,
                session_id=_server_session_id,
                is_test=_is_test_session,
            )
            _emit_trace_event(
                name,
                latency_ms,
                arguments,
                result_json,
                success,
            )
        return result_json


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
    """Dispatch to the 15 consolidated tool handlers plus legacy session/health handlers."""

    # --- optional: find_similar_by_examples (Qdrant-backed) ---
    if name == "find_similar_by_examples":
        return _handle_find_similar_by_examples(args)

    # --- M1: Unified search ---
    if name == "search":
        return _handle_search(conn, args)

    # --- concept_search (multi-vocabulary router, dbl.7) ---
    if name == "concept_search":
        result = search.concept_search(
            conn,
            args["query"],
            vocabulary=args.get("vocabulary"),
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
        year_start = _coerce_year(args.get("year_start"), "year_start")
        year_end = _coerce_year(args.get("year_end"), "year_end")
        if year_start is not None and year_end is not None and year_end < year_start:
            raise ValueError(f"year_end ({year_end}) must be >= year_start ({year_start})")
        result = search.temporal_evolution(
            conn,
            args["bibcode_or_query"],
            year_start=year_start,
            year_end=year_end,
        )
        return _result_to_json(result)

    # --- facet_counts ---
    if name == "facet_counts":
        try:
            filters = _parse_filters(args.get("filters"))
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        limit = args.get("limit", 50)
        result = search.facet_counts(conn, args["field"], filters=filters, limit=limit)
        return _result_to_json(result)

    # --- PRD MH-4: claim_blame ---
    if name == "claim_blame":
        return _handle_claim_blame(conn, args)

    # --- PRD MH-4: find_replications ---
    if name == "find_replications":
        return _handle_find_replications(conn, args)

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


def _maybe_disambiguate(conn: psycopg.Connection, query: str) -> str | None:
    """Run the query-time disambiguator and return a JSON payload iff the
    query contains at least one ambiguous mention.

    Returns ``None`` when no ambiguity is detected (the caller should then
    proceed with the normal search path). Returns a JSON string of the form
    ``{"disambiguation": [<MentionDisambiguation dicts>]}`` when at least one
    mention is flagged ``ambiguous=True``. The list contains ALL
    MentionDisambiguation results (ambiguous or not) so the caller sees the
    full extracted context.

    Disambiguator failures (DB errors, missing tables) are logged and
    treated as "no ambiguity detected" — the search path then runs normally
    rather than surfacing an opaque error at the MCP boundary.
    """
    try:
        mentions = disambiguate_query(conn, query)
    except Exception:
        logger.exception("disambiguate_query failed; continuing with search")
        return None

    if not mentions:
        return None
    if not any(m.ambiguous for m in mentions):
        return None

    payload = {
        "disambiguation": [dataclasses.asdict(m) for m in mentions],
    }
    return json.dumps(payload, indent=2, default=str)


def _handle_search(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified search: hybrid/semantic/keyword.

    When ``disambiguate`` is true (default) and the query contains at least
    one ambiguous entity mention (as determined by
    :func:`scix.jit.disambiguator.disambiguate_query`), this returns a
    ``{"disambiguation": [...]}`` JSON payload and skips the search. When
    ``disambiguate`` is false, the disambiguation check is bypassed entirely
    and the normal search path runs.
    """
    mode = args.get("mode", "hybrid")
    query = args["query"]
    disambiguate = args.get("disambiguate", True)

    if disambiguate:
        disamb_response = _maybe_disambiguate(conn, query)
        if disamb_response is not None:
            return disamb_response

    try:
        filters = _parse_filters(args.get("filters"))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
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

    # Cross-encoder rerank gating. Default is OFF: the M1 ablation
    # (commit 06a6cc3) showed both candidate cross-encoders regress nDCG@10 on
    # this corpus and fail the M4 rollout gate. The factory only constructs a
    # CrossEncoderReranker when SCIX_RERANK_DEFAULT_MODEL != 'off', so the
    # default code path never instantiates a model.
    use_rerank = bool(args.get("use_rerank", True))
    reranker: Any = None
    # per PRD prd_cross_encoder_reranker_local.md M3: rerank only top_k <= 20
    if use_rerank and limit <= _RERANK_TOP_K_CAP:
        reranker = _get_default_reranker()

    result = search.hybrid_search(
        conn,
        query,
        query_embedding=query_embedding,
        model_name=model_name,
        filters=filters,
        top_n=limit,
        reranker=reranker,
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
    """Read or search within a paper's full text.

    Both branches (read_paper_section and search_within_paper) read from
    the full-text body, so the response is annotated with the coverage
    note per the PRD's coverage-bias discipline rule.
    """
    bibcode = args["bibcode"]
    search_query = args.get("search_query")

    if search_query:
        result = search.search_within_paper(conn, bibcode, search_query)
        return _inject_coverage_note(_result_to_json(result))

    result = search.read_paper_section(
        conn,
        bibcode,
        section=args.get("section", "full"),
        char_offset=args.get("char_offset", 0),
        limit=args.get("limit", 5000),
        role=args.get("role"),
    )
    return _inject_coverage_note(_result_to_json(result))


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


#: Mapping from integer ``min_confidence_tier`` to the set of TEXT
#: ``confidence_tier`` values that satisfy the filter. The DB column is
#: constrained to ``'high' | 'medium' | 'low'`` (migration 017); callers pass
#: an integer so the public MCP contract is numerically comparable.
#:
#:     1 (low)    -> {'low', 'medium', 'high'}
#:     2 (medium) -> {'medium', 'high'}
#:     3 (high)   -> {'high'}
_TIER_MIN_TO_ALLOWED: dict[int, list[str]] = {
    1: ["low", "medium", "high"],
    2: ["medium", "high"],
    3: ["high"],
}


#: Entity types whose rows live in ``staging.extractions`` keyed on
#: ``extraction_type`` (the row IS the entity, not a containment payload).
#: For these the entity tool surfaces the raw extraction payload — the
#: shape is documented in scix.negative_results / scix.claim_extractor.
_EXTRACTION_TYPE_ENTITIES: frozenset[str] = frozenset(
    {"negative_result", "quant_claim"}
)


def _handle_entity(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified entity search and resolution."""
    action = args.get("action", "search")
    # Accept both ``query`` (current contract) and ``entity_name`` (used by
    # full-text-extraction callers, mirrors the deprecated entity_search
    # field name) so the same handler serves both call shapes.
    query = args.get("query") or args.get("entity_name") or ""
    entity_type = args.get("entity_type")

    # Only the containment-style entity types require a non-empty query.
    # Extraction-row entity types (negative_result, quant_claim) accept an
    # empty query and treat ``entity_name`` as an optional filter.
    is_extraction_row = (
        action == "search" and entity_type in _EXTRACTION_TYPE_ENTITIES
    )
    if not is_extraction_row and (not query or not query.strip()):
        return json.dumps({"error": "query must be a non-empty string"})

    if action == "resolve":
        resolver = EntityResolver(conn)
        candidates = resolver.resolve(
            query.strip(),
            discipline=args.get("discipline"),
            fuzzy=args.get("fuzzy", False),
        )
        result_json = json.dumps(
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
        return _inject_coverage_note(result_json)

    if action == "search":
        # ---- Extraction-row entity types (M3 negative_result, M4 quant_claim) ----
        if entity_type in _EXTRACTION_TYPE_ENTITIES:
            return _inject_coverage_note(
                _handle_entity_extraction_search(
                    conn,
                    extraction_type=entity_type,
                    name_filter=query.strip() if query else None,
                    limit=min(args.get("limit", 20), 200),
                )
            )

        _VALID_ENTITY_TYPES = {"methods", "datasets", "instruments", "materials"}
        if not entity_type or entity_type not in _VALID_ENTITY_TYPES:
            return json.dumps(
                {
                    "error": (
                        f"Invalid entity_type '{entity_type}'. "
                        f"Must be one of: "
                        f"{sorted(_VALID_ENTITY_TYPES | _EXTRACTION_TYPE_ENTITIES)}"
                    )
                }
            )

        limit = min(args.get("limit", 20), 200)
        containment = json.dumps({entity_type: [query]})

        # Build WHERE clauses conditionally so backward compatibility is
        # preserved: when no provenance args are supplied, the effective
        # SQL is identical to the pre-filter query.
        where_clauses: list[str] = ["e.payload @> %s::jsonb"]
        params: list[Any] = [containment]

        sources = args.get("sources")
        if sources is not None:
            if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
                return json.dumps({"error": "sources must be a list of strings"})
            where_clauses.append("e.source = ANY(%s::text[])")
            params.append(list(sources))

        min_confidence_tier = args.get("min_confidence_tier")
        if min_confidence_tier is not None:
            if (
                not isinstance(min_confidence_tier, int)
                or isinstance(min_confidence_tier, bool)
                or min_confidence_tier not in _TIER_MIN_TO_ALLOWED
            ):
                return json.dumps(
                    {
                        "error": (
                            "min_confidence_tier must be 1, 2, or 3; "
                            f"got {min_confidence_tier!r}"
                        )
                    }
                )
            where_clauses.append("e.confidence_tier = ANY(%s::text[])")
            params.append(_TIER_MIN_TO_ALLOWED[min_confidence_tier])

        # NOTE: the f-string only splices a join of whitelisted SQL
        # fragments; user-provided values are all bound via %s placeholders.
        sql = f"""
            SELECT e.bibcode, e.extraction_type, e.extraction_version, e.payload,
                   p.title
            FROM extractions e
            JOIN papers p ON p.bibcode = e.bibcode
            WHERE {" AND ".join(where_clauses)}
            LIMIT %s
        """
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
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
        return _inject_coverage_note(
            json.dumps(
                {"papers": papers, "total": len(papers)},
                indent=2,
                default=str,
            )
        )

    return json.dumps({"error": f"Invalid action: {action}. Use 'search' or 'resolve'."})


def _handle_entity_extraction_search(
    conn: psycopg.Connection,
    *,
    extraction_type: str,
    name_filter: str | None,
    limit: int,
) -> str:
    """Surface rows from ``staging.extractions`` for an extraction-row entity.

    Currently supports:

    * ``negative_result`` (M3) — rows have payload
      ``{spans: [{evidence_span, ...}], n_spans, tier_counts, ...}``.
      The handler flattens spans up so each returned row carries an
      ``evidence_span`` field (the first span on a row), preserving the
      full payload for callers that need every span.

    * ``quant_claim`` (M4) — rows have payload ``{claims: [...]}`` where
      each claim has ``{quantity, value, uncertainty, unit, ...}``. When
      ``name_filter`` is provided, claims are filtered to that canonical
      ``quantity`` value.

    The DB read is a simple ``WHERE extraction_type = %s`` scan; the
    extractions table has an index on ``(bibcode, extraction_type,
    extraction_version)`` per migration 017.
    """
    sql = """
        SELECT e.bibcode, e.extraction_type, e.extraction_version, e.payload,
               p.title
        FROM extractions e
        JOIN papers p ON p.bibcode = e.bibcode
        WHERE e.extraction_type = %s
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (extraction_type, limit))
        rows = cur.fetchall()

    papers: list[dict[str, Any]] = []
    for row in rows:
        bibcode, ext_type, ext_version, payload, title = row
        record: dict[str, Any] = {
            "bibcode": bibcode,
            "extraction_type": ext_type,
            "extraction_version": ext_version,
            "title": title,
            "payload": payload,
        }

        if extraction_type == "negative_result":
            spans = []
            if isinstance(payload, dict):
                raw_spans = payload.get("spans")
                if isinstance(raw_spans, list):
                    spans = raw_spans
            if name_filter:
                # AC: free-text query over evidence_span / match_text.
                needle = name_filter.lower()
                spans = [
                    s
                    for s in spans
                    if isinstance(s, dict)
                    and (
                        needle in str(s.get("evidence_span", "")).lower()
                        or needle in str(s.get("match_text", "")).lower()
                    )
                ]
                if not spans:
                    # No spans matched the name filter — drop this row.
                    continue
            # Surface the first span's evidence_span at the top level so
            # tests / callers don't have to dig into payload.spans[0].
            first = spans[0] if spans else None
            if isinstance(first, dict):
                record["evidence_span"] = first.get("evidence_span", "")
                record["confidence_tier"] = first.get("confidence_tier")
                record["confidence_label"] = first.get("confidence_label")
                record["section"] = first.get("section")
            else:
                record["evidence_span"] = ""
            record["spans"] = spans

        elif extraction_type == "quant_claim":
            claims = []
            if isinstance(payload, dict):
                raw_claims = payload.get("claims")
                if isinstance(raw_claims, list):
                    claims = raw_claims
            if name_filter:
                needle = name_filter.strip()
                claims = [
                    c
                    for c in claims
                    if isinstance(c, dict)
                    and str(c.get("quantity", "")) == needle
                ]
                if not claims:
                    continue
            # Promote the first claim's {value, uncertainty, unit} to the
            # top level so the response shape matches the PRD acceptance
            # contract; full claim list stays under ``claims``.
            first = claims[0] if claims else None
            if isinstance(first, dict):
                record["payload"] = {
                    "value": first.get("value"),
                    "uncertainty": first.get("uncertainty"),
                    "unit": first.get("unit"),
                    "quantity": first.get("quantity"),
                }
            record["claims"] = claims

        papers.append(record)

    papers = _annotate_working_set(papers)
    return json.dumps(
        {"papers": papers, "total": len(papers)},
        indent=2,
        default=str,
    )


def _handle_graph_context(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Graph metrics and optional community exploration.

    The metrics block always includes a ``communities`` sub-block with per-signal
    (citation / semantic / taxonomic) community memberships and labels. When
    ``include_community`` is true, an additional ``community`` block returns
    sibling papers in the community selected by ``signal`` (default ``semantic``)
    and ``resolution``. Invalid ``signal`` values propagate as a ValueError and
    are returned to the caller as a structured JSON error.
    """
    bibcode = args["bibcode"]
    include_community = args.get("include_community", False)

    metrics_result = search.get_paper_metrics(conn, bibcode)

    if not include_community:
        return _result_to_json(metrics_result)

    resolution = args.get("resolution", "coarse")
    limit = args.get("limit", 20)
    signal = args.get("signal", "semantic")
    try:
        community_result = search.explore_community(
            conn, bibcode, resolution=resolution, limit=limit, signal=signal
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

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
    """Find gaps using implicit session state (focused papers).

    The citation partition (``community_id_{coarse,medium,fine}``) is
    populated by a two-phase pipeline: Phase A marks non-giant-component
    papers with the sentinel ``-1``; Phase B overwrites giant-component
    rows with real Leiden IDs. As of 2026-04-24 Phase B has never
    completed on prod (repeated OOM during the Leiden call on a 20M-node
    induced subgraph), so all non-NULL citation rows hold the sentinel.
    The semantic partition (``community_semantic_*``) is fully populated
    and is the default.
    """
    signal = args.get("signal", "semantic")
    resolution = args.get("resolution", "medium")
    limit = min(args.get("limit", 20), 200)
    clear_first = args.get("clear_first", False)

    if clear_first:
        _session_state.clear_focused()
        _session_state.clear_working_set()

    _SIGNAL_COLUMN_PREFIX: dict[str, str] = {
        "citation": "community_id",
        "semantic": "community_semantic",
    }
    column_prefix = _SIGNAL_COLUMN_PREFIX.get(signal)
    if column_prefix is None:
        return json.dumps(
            {
                "error": (
                    f"Invalid signal: {signal}. "
                    f"Must be one of {sorted(_SIGNAL_COLUMN_PREFIX)}"
                )
            }
        )

    _VALID_RESOLUTIONS = ("coarse", "medium", "fine")
    if resolution not in _VALID_RESOLUTIONS:
        return json.dumps(
            {
                "error": (
                    f"Invalid resolution: {resolution}. "
                    f"Must be one of {sorted(_VALID_RESOLUTIONS)}"
                )
            }
        )
    community_col = f"{column_prefix}_{resolution}"

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
                "signal": signal,
                "message": "No focused papers yet. Use get_paper to inspect papers first.",
            },
            indent=2,
        )

    # For the citation signal, filter out the Phase-A sentinel (-1) which
    # marks non-giant-component papers rather than a real community.
    sentinel_filter = (
        f"AND pm.{community_col} <> -1" if signal == "citation" else ""
    )
    seed_sentinel_filter = (
        f"AND pm2.{community_col} <> -1" if signal == "citation" else ""
    )

    # LEFT JOIN communities so every result carries the community's human
    # label + top_keywords when they've been generated
    # (``scripts/generate_community_labels.py``). NULL labels drop through
    # as None — not fatal, just less legible.
    query = f"""
        SELECT DISTINCT p.bibcode, p.title, pm.pagerank,
               pm.{community_col} AS community_id,
               c.label AS community_label,
               c.top_keywords AS community_top_keywords
        FROM citation_edges ce
        JOIN papers p ON p.bibcode = ce.source_bibcode
        JOIN paper_metrics pm ON pm.bibcode = p.bibcode
        LEFT JOIN communities c
               ON c.signal = %s
              AND c.resolution = %s
              AND c.community_id = pm.{community_col}
        WHERE ce.target_bibcode = ANY(%s)
          AND pm.{community_col} IS NOT NULL
          {sentinel_filter}
          AND pm.{community_col} NOT IN (
              SELECT DISTINCT pm2.{community_col}
              FROM paper_metrics pm2
              WHERE pm2.bibcode = ANY(%s)
                AND pm2.{community_col} IS NOT NULL
                {seed_sentinel_filter}
          )
          AND p.bibcode <> ALL(%s)
        ORDER BY pm.pagerank DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(
            query,
            (signal, resolution, ws_bibcodes, ws_bibcodes, ws_bibcodes, limit),
        )
        rows = cur.fetchall()
    papers = [
        {
            "bibcode": row[0],
            "title": row[1],
            "pagerank": row[2],
            "community_id": row[3],
            "community_label": row[4],
            "community_top_keywords": row[5],
        }
        for row in rows
    ]
    papers = _annotate_working_set(papers)
    return json.dumps(
        {
            "papers": papers,
            "total": len(papers),
            "signal": signal,
            "resolution": resolution,
        },
        indent=2,
        default=str,
    )


def _handle_find_similar_by_examples(args: dict[str, Any]) -> str:
    """Dispatch for the Qdrant-backed discovery tool.

    Returns a structured error if Qdrant is not configured, so the tool can
    live in the registered tool set even in mixed deployments where the
    backend is not yet wired up. Callers should check the ``error`` field.
    """
    if not _qdrant_enabled():
        return json.dumps({
            "error": "qdrant_not_configured",
            "message": (
                "find_similar_by_examples requires the Qdrant backend "
                "(QDRANT_URL env var)."
            ),
        })

    positives = args.get("positive_bibcodes") or []
    if not positives:
        return json.dumps({"error": "positive_bibcodes is required"})

    try:
        hits = _qdrant_tools.find_similar_by_examples(
            positive_bibcodes=list(positives),
            negative_bibcodes=list(args.get("negative_bibcodes") or []) or None,
            limit=int(args.get("limit", 10)),
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
            doctype=args.get("doctype"),
            community_semantic=args.get("community_semantic"),
            arxiv_class=args.get("arxiv_class"),
        )
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("find_similar_by_examples failed")
        return json.dumps({"error": str(exc)})

    return json.dumps({
        "backend": "qdrant",
        "collection": _qdrant_tools.COLLECTION,
        "results": [dataclasses.asdict(h) for h in hits],
    })


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


def _handle_claim_blame(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the claim_blame MCP tool (PRD MH-4)."""
    from scix.claim_blame import claim_blame
    from scix.research_scope import scope_from_dict

    claim_text = args.get("claim_text")
    if not isinstance(claim_text, str) or not claim_text.strip():
        return json.dumps({"error": "claim_text must be a non-empty string"})

    scope_arg = args.get("scope")
    try:
        scope = scope_from_dict(scope_arg) if scope_arg else None
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid scope: {exc}"})

    candidate_limit = int(args.get("candidate_limit", 20))
    lineage_limit = int(args.get("lineage_limit", 10))

    result = claim_blame(
        claim_text,
        scope=scope,
        conn=conn,
        candidate_limit=candidate_limit,
        lineage_limit=lineage_limit,
    )
    return json.dumps(result, indent=2, default=str)


def _handle_find_replications(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the find_replications MCP tool (PRD MH-4)."""
    from scix.find_replications import VALID_RELATIONS, find_replications
    from scix.research_scope import scope_from_dict

    target_bibcode = args.get("target_bibcode")
    if not isinstance(target_bibcode, str) or not target_bibcode.strip():
        return json.dumps({"error": "target_bibcode must be a non-empty string"})

    relation = args.get("relation")
    if relation is not None and relation not in VALID_RELATIONS:
        return json.dumps(
            {"error": f"relation must be one of {sorted(VALID_RELATIONS)} or null"}
        )

    scope_arg = args.get("scope")
    try:
        scope = scope_from_dict(scope_arg) if scope_arg else None
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid scope: {exc}"})

    limit = int(args.get("limit", 50))

    result = find_replications(
        target_bibcode,
        relation=relation,
        scope=scope,
        conn=conn,
        limit=limit,
    )
    return json.dumps({"citations": result, "total": len(result)}, indent=2, default=str)


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
