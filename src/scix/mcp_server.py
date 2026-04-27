"""MCP server exposing 15 consolidated tools for agent navigation of the SciX corpus.

Uses the `mcp` Python SDK to register tools. Each tool is a thin wrapper
around functions in search.py. Connection pooling via psycopg.pool for
production-grade performance.

Consolidation (v3, 2026-04-25):
    Original 28 -> 13 -> 15 agent-facing tools + deprecated aliases.
    The 2026-04-25 pass merged citation_graph + citation_chain into
    citation_traverse (mode enum), retired find_similar_by_examples
    (qdrant backend out of active use), and ratified the additions of
    claim_blame, find_replications, and section_retrieval that landed
    after the original audit was written.
    Old tool names still work via ``_DEPRECATED_ALIASES`` but return
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
from typing import Any, Generator, Sequence

import psycopg

from scix import search
from scix.db import DEFAULT_DSN
from scix.embed import _model_cache, clear_model_cache, embed_batch, load_model
from scix.entity_resolver import EntityResolver
from scix.jit.disambiguator import disambiguate_query
from scix.search import CrossEncoderReranker
from scix.session import SessionState, WorkingSetEntry
from scix.synthesize import (
    DEFAULT_SECTIONS as _SYNTH_DEFAULT_SECTIONS,
    synthesize_findings as _synthesize_findings,
)

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
    "lit_review": float(os.environ.get("SCIX_TIMEOUT_LIT_REVIEW", "30")),
    "concept_search": float(os.environ.get("SCIX_TIMEOUT_CONCEPT", "15")),
    "get_paper": float(os.environ.get("SCIX_TIMEOUT_PAPER", "5")),
    "read_paper": float(os.environ.get("SCIX_TIMEOUT_READ_PAPER", "10")),
    "citation_traverse": float(os.environ.get("SCIX_TIMEOUT_TRAVERSE", "20")),
    # Legacy aliases retained so deprecated tool calls still get a sensible
    # statement_timeout before being routed through citation_traverse.
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
    # PRD nanopub-claim-extraction — paper_claims retrieval tools (mig 062).
    "read_paper_claims": float(os.environ.get("SCIX_TIMEOUT_READ_PAPER_CLAIMS", "5")),
    "find_claims": float(os.environ.get("SCIX_TIMEOUT_FIND_CLAIMS", "8")),
    # PRD MH-4 — Deep Search v1 provenance tools.
    "claim_blame": float(os.environ.get("SCIX_TIMEOUT_CLAIM_BLAME", "15")),
    "find_replications": float(os.environ.get("SCIX_TIMEOUT_FIND_REPLICATIONS", "15")),
    # Structural-citation lookup over citation_contexts.intent
    "cited_by_intent": float(os.environ.get("SCIX_TIMEOUT_CITED_BY_INTENT", "5")),
    # Terminal synthesis tool — three short SELECTs against papers,
    # citation_contexts, paper_metrics; cap matches find_gaps.
    "synthesize_findings": float(
        os.environ.get("SCIX_TIMEOUT_SYNTHESIZE_FINDINGS", "15")
    ),
}

# Tools whose backing data is missing on this deployment. Default-hidden so
# agents don't waste calls on tools that can't return real results. Override
# via SCIX_HIDDEN_TOOLS env var (comma-separated; empty string to show all).
#   * chunk_search       — Qdrant collection scix_chunks_v1 not yet populated
#   * section_retrieval  — section_embeddings table not yet populated
#   * read_paper_claims, find_claims — paper_claims table empty (no extraction
#     run yet); table itself exists per migration 062
_HIDDEN_TOOLS: frozenset[str] = frozenset(
    t.strip()
    for t in os.environ.get(
        "SCIX_HIDDEN_TOOLS",
        "chunk_search,section_retrieval,read_paper_claims,find_claims",
    ).split(",")
    if t.strip()
)


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
_COVERAGE_BIAS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "results" / "full_text_coverage_bias.json"
)

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
    # Citation tools merged into citation_traverse (2026-04-25).
    "citation_graph": "citation_traverse",
    "citation_chain": "citation_traverse",
    "get_citations": "citation_traverse",
    "get_references": "citation_traverse",
    "get_citation_context": "citation_traverse",
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
# section_retrieval tool — filters schema + RRF + snippet helpers
# ---------------------------------------------------------------------------
#
# The section_retrieval tool fuses dense HNSW search over section_embeddings
# with BM25 over papers_fulltext.sections_tsv via Reciprocal Rank Fusion.
# It uses a slimmer filter object than the search-tool _FILTERS_SCHEMA: only
# discipline, year_min, year_max, bibcode_prefix.

_SECTION_FILTERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "discipline": {
            "type": "string",
            "description": (
                "Restrict to papers whose papers.discipline equals this value "
                "(e.g. 'astrophysics')."
            ),
        },
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "bibcode_prefix": {
            "type": "string",
            "description": (
                "Restrict to bibcodes that start with this prefix "
                "(e.g. '2024ApJ' for ApJ papers from 2024)."
            ),
        },
    },
}

# Reciprocal-rank-fusion default constant (Cormack et al., 2009).
_RRF_K_DEFAULT: int = 60

# Maximum snippet length emitted by section_retrieval.
_SNIPPET_MAX_CHARS: int = 500

# nomic-embed-text-v1.5 query-time prefix. Document prefix lives in
# scix.embeddings.section_pipeline (NOMIC_DOC_PREFIX); we keep the query
# prefix local rather than reaching into the pipeline module so consumers
# that only need query encoding don't inherit the document prefix.
_NOMIC_QUERY_PREFIX: str = "search_query: "


def _truncate_snippet(text: str | None, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    """Truncate a section text to at most ``max_chars`` characters.

    Returns the empty string when ``text`` is None. Truncation is purely
    character-based — no word-boundary cleanup. The cap is a hard contract
    surfaced by the section_retrieval response schema.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _rrf_fuse(
    ranked_lists: Sequence[Sequence[Any]],
    k_rrf: int = _RRF_K_DEFAULT,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    For each candidate key ``d``, the fused score is

        score(d) = sum over lists L of 1 / (k_rrf + rank_L(d))

    where rank is 1-indexed (best = 1) and ``d`` not appearing in a list
    contributes 0 for that list. Returns a list of ``(key, score)`` sorted
    by score descending, with ties broken by the order in which the key
    was first seen across the input lists (stable).

    Inputs:
        ranked_lists: a sequence of ranked iterables, each best-first.
            Keys must be hashable.
        k_rrf: the RRF k constant; defaults to 60 per Cormack et al. 2009.

    Pure function; no DB or filesystem I/O.
    """
    if k_rrf <= 0:
        raise ValueError(f"k_rrf must be positive, got {k_rrf}")
    scores: dict[Any, float] = {}
    first_seen: dict[Any, int] = {}
    seen_counter = 0
    for ranked in ranked_lists:
        for rank_zero_based, key in enumerate(ranked):
            rank = rank_zero_based + 1  # 1-indexed
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            if key not in first_seen:
                first_seen[key] = seen_counter
                seen_counter += 1
    # Sort by score desc, then by first-seen order asc (stable tiebreak).
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )


# ---------------------------------------------------------------------------
# Expected consolidated tools (used by startup self-test)
# ---------------------------------------------------------------------------

EXPECTED_TOOLS: tuple[str, ...] = (
    "search",
    "lit_review",
    "concept_search",
    "get_paper",
    "read_paper",
    # citation_graph + citation_chain merged into citation_traverse (2026-04-25)
    "citation_traverse",
    "citation_similarity",
    "entity",
    "entity_context",
    "graph_context",
    "find_gaps",
    "temporal_evolution",
    "facet_counts",
    # PRD MH-4 — Deep Search v1: provenance walk + replication enumeration
    "claim_blame",
    "find_replications",
    # PRD section-embeddings-mcp-consolidation — section-grain hybrid retrieval
    "section_retrieval",
    # PRD nanopub-claim-extraction — paper_claims retrieval (migration 062)
    "read_paper_claims",
    "find_claims",
    # Structural-citation lookup — exploits citation_contexts.intent
    # (method / background / result_comparison) classification.
    "cited_by_intent",
    # Terminal step — bin a working set into a section outline (bead cfh9).
    "synthesize_findings",
)

# Tools that appear only when an optional backend is wired up. The
# ``chunk_search`` tool is registered iff ``_qdrant_enabled()`` (the
# scix_chunks_v1 collection lives in Qdrant; PRD chunk-embeddings-build).
_OPTIONAL_TOOLS: tuple[str, ...] = ("chunk_search",)


def _expected_tool_set() -> set[str]:
    tools = set(EXPECTED_TOOLS)
    if _OPTIONAL_TOOLS and _qdrant_enabled():
        tools.update(_OPTIONAL_TOOLS)
    # Drop tools the deployment has chosen to hide (e.g. ones whose backing
    # data isn't yet populated — see _HIDDEN_TOOLS comment).
    tools -= _HIDDEN_TOOLS
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
        errors.append(f"expected exactly {len(expected_set)} tools, got {tool_count}")

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
            logger.critical("startup_self_test FAILED smoke calls: %s", smoke_errors)
            raise RuntimeError(f"startup_self_test smoke calls failed: {smoke_errors}")

    logger.info(
        "startup_self_test OK: %d tools registered (%s)",
        tool_count,
        ", ".join(sorted(tool_names)),
    )
    return status


def _smoke_call_new_tools() -> list[str]:
    """Invoke recently added tools against SCIX_TEST_DSN to catch wiring breakage.

    Returns a list of error strings (empty on success). Empty result sets
    are NOT errors — only raised exceptions are. This matches PRD MH-4
    acceptance criterion 7's "gracefully handle the case where
    citation_contexts.intent is all NULL" requirement.

    Currently exercises:
        * claim_blame (PRD MH-4)
        * find_replications (PRD MH-4)
        * section_retrieval (PRD section-embeddings-mcp-consolidation) —
          dispatched in-process via :func:`_dispatch_tool` so we exercise
          the full MCP routing path including filter validation. Embedding
          import failures are tolerated (the section embedder requires
          ``sentence-transformers`` which is an optional install) and
          surface as a structured ``error`` payload rather than a raised
          exception.
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
            try:
                _dispatch_tool(
                    conn,
                    "section_retrieval",
                    {"query": "startup self-test", "k": 1},
                )
            except Exception as exc:  # noqa: BLE001 — log + report
                errors.append(f"section_retrieval: {exc}")
    except Exception as exc:  # noqa: BLE001 — pool acquire failure
        errors.append(f"pool: {exc}")
    return errors


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


def create_server(_run_self_test: bool = True):
    """Create and configure the MCP server with the 15 consolidated tools.

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
            # --- lit_review (composite — bead nn03) ---
            Tool(
                name="lit_review",
                description=(
                    "Open a literature-review session in one call. Composes "
                    "hybrid_search for seed papers, citation expansion (refs + "
                    "forward citations on the top-K seeds), community decomposition "
                    "(semantic-medium clusters with labels), top-venue and "
                    "year-distribution facets over the resulting working set, and "
                    "full abstracts on the highest-ranked seeds. Side effect: the "
                    "working set is populated in session state so follow-up tools "
                    "(find_gaps, etc.) operate on it without re-listing bibcodes. "
                    "Use this as the FIRST call when the user asks for a literature "
                    "review or topic survey; use plain search instead when you only "
                    "need a flat ranked list."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text research question or topic.",
                        },
                        "year_min": {
                            "type": "integer",
                            "description": "Earliest publication year to include in seed retrieval (inclusive).",
                        },
                        "year_max": {
                            "type": "integer",
                            "description": "Latest publication year to include (inclusive). Filters seeds AND citation-expanded papers.",
                        },
                        "top_seeds": {
                            "type": "integer",
                            "default": 20,
                            "description": "Hybrid-search seed count (1..100).",
                        },
                        "expansion_seeds": {
                            "type": "integer",
                            "default": 5,
                            "description": "How many top seeds to expand via refs+citations (0..top_seeds).",
                        },
                        "expand_per_seed": {
                            "type": "integer",
                            "default": 20,
                            "description": "Per-seed reference + citation fetch limit (0..50).",
                        },
                        "sample_abstracts": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of seeds to attach full abstracts to in the response (0..top_seeds).",
                        },
                        "discipline": {
                            "type": "string",
                            "description": "Optional discipline hint (currently informational, surfaced in metadata).",
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
                    "detected in the paper, each annotated with precision_estimate "
                    "(0..1) and precision_band (high/medium/low/noisy) from the dbl.3 "
                    "NER quality profile. Use search or concept_search instead when you "
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
                        "min_precision": {
                            "type": "number",
                            "description": (
                                "Optional precision_estimate floor (0..1). Drops linked "
                                "entities below the threshold; only applies when "
                                "include_entities=true. Default surfaces all entities with "
                                "precision metadata attached."
                            ),
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
            # --- citation_traverse (merges citation_graph + citation_chain, 2026-04-25) ---
            Tool(
                name="citation_traverse",
                description=(
                    "Traverse the citation graph. mode='graph' walks the neighborhood of a "
                    "single paper (citing or cited papers); mode='chain' traces the shortest "
                    "citation path between a source and a target paper. mode='graph' "
                    "(default) requires bibcode OR bibcodes=[...] for working-set "
                    "expansion (multi-paper neighborhoods returned under by_bibcode); when "
                    "neither is given, falls through to the session's focused papers. "
                    "Supports direction (forward=citing, backward=references, both=all) "
                    "and include_context. mode='chain' requires source_bibcode and "
                    "target_bibcode and accepts max_depth (working-set scoping does not "
                    "apply). Use citation_similarity instead when you want papers related "
                    "via shared citation patterns rather than direct citation links."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["graph", "chain"],
                            "default": "graph",
                            "description": (
                                "graph: walk neighbors of a single bibcode. "
                                "chain: trace shortest path between source and target."
                            ),
                        },
                        "bibcode": {
                            "type": "string",
                            "description": (
                                "ADS bibcode (used when mode='graph' and "
                                "bibcodes is not provided)"
                            ),
                        },
                        "bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of bibcodes for working-set "
                                "expansion (mode='graph' only). When provided, "
                                "the neighborhood walk runs per source bibcode "
                                "and results are returned under by_bibcode. "
                                "When omitted, falls through to the session's "
                                "focused papers (papers inspected via get_paper)."
                            ),
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward", "both"],
                            "default": "forward",
                            "description": (
                                "forward=citing papers, backward=references, both=all "
                                "(mode='graph' only)"
                            ),
                        },
                        "include_context": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Include citation context text (mode='graph' only, slower)"
                            ),
                        },
                        "source_bibcode": {
                            "type": "string",
                            "description": (
                                "Starting paper of the chain " "(required when mode='chain')"
                            ),
                        },
                        "target_bibcode": {
                            "type": "string",
                            "description": (
                                "Destination paper of the chain " "(required when mode='chain')"
                            ),
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 5,
                            "description": ("Maximum number of hops 1..5 (mode='chain' only)"),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Max neighbors to return (mode='graph' only)",
                        },
                    },
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
            # --- M4: entity (merges entity_search + resolve_entity) ---
            Tool(
                name="entity",
                description=(
                    "Look up named scientific entities across 13 vocabularies "
                    "(method, dataset, instrument, material, gene, software, "
                    "mission, organism, target, observable, chemical, "
                    "location, taxon — ~9M entities total). "
                    "action='resolve' maps a free-text mention to canonical "
                    "entity records (use this for cross-discipline lookup: "
                    "'p53', 'transformer', 'JWST'). "
                    "action='papers' returns papers tagged with an entity "
                    "via document_entities (57M paper-entity links across "
                    "16M papers); pass entity_id (from resolve) or query "
                    "(auto-resolves first candidate). "
                    "action='search' is a narrower path that searches the "
                    "older extractions table for 4 specific types "
                    "(methods/datasets/instruments/materials). "
                    "Use entity_context once you have an entity_id and "
                    "need its full profile and relationships."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search", "resolve", "papers"],
                            "description": (
                                "resolve=name→canonical entity (cross-discipline); "
                                "papers=entity→papers tagged with it; "
                                "search=narrow search of extractions table"
                            ),
                        },
                        "entity_id": {
                            "type": "integer",
                            "description": ("Entity id (from resolve). Used by action='papers'."),
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
                    "required": ["action"],
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
                    "cite papers you already inspected via get_paper. Helps catch adjacent "
                    "literature you might be missing during a research session. Two ways to "
                    "seed the working set: (a) call get_paper on one or more papers first "
                    "(implicit session state); (b) pass query='<topic>' to auto-seed via "
                    "concept_search in a single call. Use citation_graph instead when you "
                    "want direct citations of a single paper rather than cross-community "
                    "gap detection. The 'signal' parameter picks which community partition "
                    "to traverse: 'semantic' (default, INDUS k-means, full 32M-paper "
                    "coverage) or 'citation' (currently offline — Leiden Phase B has not "
                    "completed, so this path returns empty)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Optional. When the working set is empty, auto-seeds it "
                                "via concept_search(query) so the gap analysis runs in a "
                                "single call. Ignored when prior get_paper calls have "
                                "already populated the working set."
                            ),
                        },
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
                    "single bibcode, returns citations-per-year for that paper. Given "
                    "bibcodes=[...] (or with the working set populated via prior "
                    "get_paper calls), returns aggregate citations-per-year across the "
                    "set. Given search terms, returns publications-per-year plus per-year "
                    "'buckets' with top anchor papers (ranked by PageRank) and dominant "
                    "communities, so a single call yields a usable temporal narrative "
                    "instead of raw counts. Useful for tracking rising or fading topics "
                    "and paper impact trajectories. Use facet_counts instead when you "
                    "want a single distribution by year without a topic or bibcode anchor."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode_or_query": {
                            "type": "string",
                            "description": (
                                "A bibcode (citation trends) or search terms "
                                "(pub volume). Optional when bibcodes=[...] or "
                                "the session has focused papers."
                            ),
                        },
                        "bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of bibcodes for working-set "
                                "aggregate citations-per-year. When omitted, "
                                "falls through to the session's focused papers."
                            ),
                        },
                        "year_start": {"type": "integer"},
                        "year_end": {"type": "integer"},
                    },
                },
            ),
            # --- facet_counts (unchanged) ---
            Tool(
                name="facet_counts",
                description=(
                    "Return a distribution of paper counts grouped by a single metadata "
                    "field: year, doctype, arxiv_class, database, bibgroup, or property. "
                    "Accepts the same filters as search to scope the distribution to a "
                    "subset. Pass bibcodes=[...] (or rely on the session's focused "
                    "papers) to scope the distribution to a working set — useful for "
                    "characterizing the year/doctype profile of a curated paper set. "
                    "Useful for dataset overviews and filter discovery. Use "
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
                        "bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of bibcodes to scope the facet "
                                "distribution. When omitted, falls through to "
                                "the session's focused papers; when neither is "
                                "set, runs over the full corpus."
                            ),
                        },
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
            # --- Structural-citation lookup (intent-aware) ---
            Tool(
                name="cited_by_intent",
                description=(
                    "Find papers that cite a target paper for a specific reason. "
                    "Surfaces the citation_contexts.intent classification "
                    "(method / background / result_comparison) — letting agents "
                    "ask 'which papers used X as their method?' or 'which "
                    "papers compared their results to X?' — questions that "
                    "vanilla retrieval cannot answer because they require "
                    "understanding *why* one paper cites another, not just "
                    "that it does. Each result includes the source bibcode, "
                    "the intent label, and a 400-char excerpt of the citation "
                    "context. Coverage is partial (~825K contexts across 30K "
                    "source papers and 250K cited papers); papers without "
                    "context coverage return empty cleanly. Use "
                    "citation_traverse instead when you want raw forward "
                    "citations regardless of intent."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_bibcode": {
                            "type": "string",
                            "description": ("ADS bibcode whose incoming citations to filter."),
                        },
                        "intent": {
                            "type": "string",
                            "enum": ["method", "background", "result_comparison"],
                            "description": (
                                "Citation intent. 'method' = papers that used "
                                "this work's method; 'background' = papers "
                                "citing this as background context "
                                "(introductions, motivation); "
                                "'result_comparison' = papers comparing "
                                "their results to this work's. Omit for "
                                "any-intent."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Max papers to return (1..200).",
                        },
                    },
                    "required": ["target_bibcode"],
                },
            ),
            # --- Terminal synthesis (bead cfh9) ---
            Tool(
                name="synthesize_findings",
                description=(
                    "Bin a working set of papers into a section outline ready "
                    "for the agent to write up. Pure mechanical aggregation: "
                    "each paper is assigned to a section first by modal "
                    "citation_contexts.intent (method->methods, "
                    "background->background, result_comparison->results), then "
                    "by community membership (papers in the working set's "
                    "modal community fall through to 'background', papers in "
                    "minority communities to 'open_questions'). Returns a "
                    "deterministic structure with cited_papers per section "
                    "(bibcode, title, year, abstract_snippet, role), a "
                    "theme_summary built from each section's most-common "
                    "community labels (no LLM), unattributed_bibcodes for "
                    "papers with no signal, and a coverage block reporting "
                    "how many bibcodes had each kind of signal. Falls "
                    "through to the session's focused papers when "
                    "working_set_bibcodes is omitted (mirrors find_gaps). "
                    "Use after lit_review or a retrieval+characterization "
                    "loop when you want a scaffold to write the synthesis."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "working_set_bibcodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Bibcodes to synthesise. Omit to fall "
                                "through to session-focused papers (capped "
                                "at 200)."
                            ),
                        },
                        "sections": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": list(_SYNTH_DEFAULT_SECTIONS),
                            "description": (
                                "Section names to emit, in order. Default: "
                                "background, methods, results, "
                                "open_questions. Custom names that don't "
                                "appear in the intent map only receive "
                                "papers via the community fall-through."
                            ),
                        },
                        "max_papers_per_section": {
                            "type": "integer",
                            "default": 8,
                            "description": (
                                "Cap on per-section cited_papers length. "
                                "Sorted by year desc, bibcode asc for "
                                "deterministic output."
                            ),
                        },
                    },
                },
            ),
            # --- PRD section-embeddings-mcp-consolidation: section_retrieval ---
            Tool(
                name="section_retrieval",
                description=(
                    "Retrieve passages at section granularity. Encodes the query "
                    "with a local nomic-embed-text-v1.5 model and runs a hybrid "
                    "search that fuses (a) dense HNSW search over "
                    "section_embeddings (halfvec(1024)) and (b) BM25 over the "
                    "papers_fulltext.sections tsvector via Reciprocal Rank Fusion "
                    "(k=60). Returns a ranked list of section hits each carrying "
                    "bibcode, section_heading, a snippet (<=500 chars of the "
                    "section text), the fused score, and a canonical_url for "
                    "attribution. Use search instead when you need paper-level "
                    "results rather than section-level passages; use read_paper "
                    "to read a specific section once you have the bibcode."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language passage query.",
                        },
                        "k": {
                            "type": "integer",
                            "default": 10,
                            "description": "Max number of section hits to return.",
                        },
                        "filters": _SECTION_FILTERS_SCHEMA,
                    },
                    "required": ["query"],
                },
            ),
            # --- PRD nanopub-claim-extraction: read_paper_claims ---
            Tool(
                name="read_paper_claims",
                description=(
                    "Return claims extracted from a single paper, ordered by "
                    "their position in the source text "
                    "(section_index, paragraph_index, char_span_start). Each "
                    "claim carries its provenance contract (bibcode + section/"
                    "paragraph/char span), the verbatim claim_text, a "
                    "claim_type tag (factual / methodological / comparative / "
                    "speculative / cited_from_other), an optional "
                    "subject-predicate-object decomposition, and an optional "
                    "extractor confidence. Use find_claims instead when you "
                    "want to search across all papers by claim text."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {
                            "type": "string",
                            "description": "ADS bibcode of the paper.",
                        },
                        "claim_type": {
                            "type": "string",
                            "enum": [
                                "factual",
                                "methodological",
                                "comparative",
                                "speculative",
                                "cited_from_other",
                            ],
                            "description": (
                                "Optional filter on claim_type. Omit to " "return all types."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 50,
                            "description": "Max claims to return (1..500).",
                        },
                    },
                    "required": ["bibcode"],
                },
            ),
            # --- PRD nanopub-claim-extraction: find_claims ---
            Tool(
                name="find_claims",
                description=(
                    "Search across all extracted paper_claims by natural-"
                    "language query, ranked by ts_rank against the GIN "
                    "to_tsvector('english', claim_text) index. The query is "
                    "treated as a phrase via plainto_tsquery (no operators "
                    "required). Optional entity_id filters to claims linked "
                    "to that entity as either subject or object. Optional "
                    "claim_type narrows by category. Use read_paper_claims "
                    "instead when you already know the bibcode and want all "
                    "claims from one paper."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural-language query over claim text. "
                                "Treated as a phrase by plainto_tsquery."
                            ),
                        },
                        "claim_type": {
                            "type": "string",
                            "enum": [
                                "factual",
                                "methodological",
                                "comparative",
                                "speculative",
                                "cited_from_other",
                            ],
                            "description": (
                                "Optional filter on claim_type. Omit to " "return all types."
                            ),
                        },
                        "entity_id": {
                            "type": "integer",
                            "description": (
                                "Optional entity id; restricts results to "
                                "claims where linked_entity_subject_id OR "
                                "linked_entity_object_id matches."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 25,
                            "description": "Max claims to return (1..500).",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

        # find_similar_by_examples retired 2026-04-25 — Qdrant backend
        # (gated by QDRANT_URL) is not in active use. The dispatch layer
        # raises a structured "tool_removed" error if any caller still
        # invokes the name.

        # --- PRD chunk-embeddings-build: chunk_search (optional, Qdrant) ---
        # Gated on QDRANT_URL via _qdrant_enabled(); only registered when the
        # scix_chunks_v1 collection is reachable. When QDRANT_URL is unset the
        # tool is hidden so list_tools advertises only what the deployment can
        # actually serve.
        if _qdrant_enabled():
            tool_list.append(
                Tool(
                    name="chunk_search",
                    description=(
                        "Chunk-grain semantic search over the scix_chunks_v1 "
                        "Qdrant collection. Encodes the query with INDUS "
                        "(768-dim, mean-pooled) and runs an ANN query against "
                        "section-aware sliding-window chunks of paper bodies. "
                        "Best for method/dataset queries and other narrow, "
                        "passage-level questions where the relevant text is a "
                        "few sentences inside a section. Use search "
                        "(abstract-level) when broader paper-level relevance "
                        "is enough; use section_retrieval when section-level "
                        "(rather than chunk-level) granularity is preferred. "
                        "Filters cover year window, arxiv_class, "
                        "community_id_med (medium-resolution Leiden "
                        "community), section_heading (canonical norm), and "
                        "bibcode (restrict to specific papers)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Natural-language passage query. "
                                    "Encoded with INDUS (mean pooling, "
                                    "768-dim) before the ANN call."
                                ),
                            },
                            "filters": {
                                "type": "object",
                                "description": (
                                    "Optional payload filters; all keys "
                                    "AND-combined. Omit any to disable."
                                ),
                                "properties": {
                                    "year_min": {
                                        "type": "integer",
                                        "description": "Earliest paper year (inclusive).",
                                    },
                                    "year_max": {
                                        "type": "integer",
                                        "description": "Latest paper year (inclusive).",
                                    },
                                    "arxiv_class": {
                                        "description": (
                                            "arXiv class filter; accepts a "
                                            "string or a list of strings "
                                            "(e.g. 'astro-ph.EP' or "
                                            "['astro-ph.EP', 'astro-ph.GA'])."
                                        ),
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        ],
                                    },
                                    "community_id_med": {
                                        "description": (
                                            "Medium-resolution Leiden "
                                            "community id; integer or list "
                                            "of integers."
                                        ),
                                        "oneOf": [
                                            {"type": "integer"},
                                            {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            },
                                        ],
                                    },
                                    "section_heading": {
                                        "description": (
                                            "Canonical normalized section "
                                            "heading(s) (e.g. 'methods', "
                                            "'results'). Pass-through to the "
                                            "section_heading_norm payload "
                                            "filter."
                                        ),
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        ],
                                    },
                                    "bibcode": {
                                        "description": (
                                            "Restrict to specific paper "
                                            "bibcode(s); string or list."
                                        ),
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        ],
                                    },
                                },
                            },
                            "limit": {
                                "type": "integer",
                                "default": 20,
                                "description": (
                                    "Max chunk hits to return; clamped to " "[1, 100]."
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                )
            )

        if _HIDDEN_TOOLS:
            tool_list = [t for t in tool_list if t.name not in _HIDDEN_TOOLS]
        return tool_list

    @server.call_tool()
    async def call_tool_handler(name: str, arguments: dict[str, Any]) -> list[TextContent]:
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

    if old_name == "citation_graph":
        new_args["mode"] = "graph"
        return "citation_traverse", new_args

    if old_name == "citation_chain":
        new_args["mode"] = "chain"
        return "citation_traverse", new_args

    if old_name == "get_citations":
        new_args["mode"] = "graph"
        new_args["direction"] = "forward"
        return "citation_traverse", new_args

    if old_name == "get_references":
        new_args["mode"] = "graph"
        new_args["direction"] = "backward"
        return "citation_traverse", new_args

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
        # Keep as direct handler — args have source_bibcode/target_bibcode.
        # Routed to the legacy get_citation_context handler via the dispatch
        # layer; the deprecation envelope still surfaces use_instead =
        # citation_traverse so agents migrate.
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
    """Dispatch to the 15 consolidated tool handlers plus legacy/health handlers."""

    # --- find_similar_by_examples retired 2026-04-25 ---
    if name == "find_similar_by_examples":
        return json.dumps(
            {
                "error": "tool_removed",
                "removed_in": "2026-04-25",
                "message": (
                    "find_similar_by_examples was retired in 2026-04-25 because the "
                    "Qdrant backend is not in active use. There is no replacement; "
                    "use search with semantic mode and entity filters, or "
                    "citation_similarity with method='coupling', for the closest "
                    "behaviour."
                ),
            }
        )

    # --- M1: Unified search ---
    if name == "search":
        return _handle_search(conn, args)

    # --- lit_review (composite — bead nn03) ---
    if name == "lit_review":
        return _handle_lit_review(conn, args)

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

    # --- citation_traverse (merges citation_graph + citation_chain) ---
    if name == "citation_traverse":
        return _handle_citation_traverse(conn, args)

    # --- legacy direct dispatch for callers that bypass _DEPRECATED_ALIASES.
    # Most callers route via the alias layer (which rewrites the name to
    # citation_traverse before dispatch arrives here), but a few in-process
    # call sites — and the dispatched legacy handler in
    # _transform_deprecated_args for get_citation_context — still arrive
    # with the legacy names. We forward them here so behaviour is preserved.
    if name == "citation_graph":
        return _handle_citation_graph(conn, args)
    if name == "citation_chain":
        max_depth = max(1, min(args.get("max_depth", 5), 5))
        result = search.citation_chain(
            conn,
            args["source_bibcode"],
            args["target_bibcode"],
            max_depth=max_depth,
        )
        return _result_to_json(result)

    # --- M3: citation_similarity ---
    if name == "citation_similarity":
        return _handle_citation_similarity(conn, args)

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
        return _handle_temporal_evolution(conn, args)

    # --- facet_counts ---
    if name == "facet_counts":
        return _handle_facet_counts(conn, args)

    # --- PRD MH-4: claim_blame ---
    if name == "claim_blame":
        return _handle_claim_blame(conn, args)

    # --- PRD MH-4: find_replications ---
    if name == "find_replications":
        return _handle_find_replications(conn, args)

    # --- Structural-citation lookup (intent-aware) ---
    if name == "cited_by_intent":
        return _handle_cited_by_intent(conn, args)

    # --- Terminal synthesis (bead cfh9) ---
    if name == "synthesize_findings":
        return _handle_synthesize_findings(conn, args)

    # --- PRD section-embeddings-mcp-consolidation: section_retrieval ---
    if name == "section_retrieval":
        return _handle_section_retrieval(conn, args)

    # --- PRD nanopub-claim-extraction: read_paper_claims ---
    if name == "read_paper_claims":
        return _handle_read_paper_claims(conn, args)

    # --- PRD nanopub-claim-extraction: find_claims ---
    if name == "find_claims":
        return _handle_find_claims(conn, args)

    # --- PRD chunk-embeddings-build: chunk_search (Qdrant-gated) ---
    if name == "chunk_search":
        return _handle_chunk_search(conn, args)

    # --- Legacy handlers for deprecated session tools ---
    if name == "add_to_working_set":
        bibcodes = args.get("bibcodes", [])
        source_tool = args.get("source_tool", "unknown")
        added = _session_state.add_bibcodes_to_working_set(
            bibcodes,
            source_tool=source_tool,
            source_context=args.get("source_context", ""),
            relevance_hint=args.get("relevance_hint", ""),
            tags=args.get("tags", []),
        )
        # Return the post-cap entries that match the bibcodes we added so
        # callers can confirm what's in the working set.
        seen = set(bibcodes)
        entries = [
            dataclasses.asdict(e) for e in _session_state.get_working_set() if e.bibcode in seen
        ]
        return json.dumps({"added": added, "entries": entries}, indent=2, default=str)

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


def _handle_lit_review(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Composite tool: open a literature-review session in one call.

    Wraps ``scix.search.lit_review`` and threads the session-state
    singleton through so the working set is populated for follow-up
    tool calls. See bead ``scix_experiments-nn03``.
    """
    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})

    def _coerce_int(name: str, default: int | None) -> int | None:
        v = args.get(name, default)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    result = search.lit_review(
        conn,
        query,
        year_min=_coerce_int("year_min", None),
        year_max=_coerce_int("year_max", None),
        top_seeds=_coerce_int("top_seeds", 20) or 20,
        expand_per_seed=_coerce_int("expand_per_seed", 20) or 20,
        expansion_seeds=_coerce_int("expansion_seeds", 5) or 5,
        sample_abstracts=_coerce_int("sample_abstracts", 5) or 5,
        discipline=args.get("discipline"),
        session_state=_session_state,
    )
    return _result_to_json(result)


def _attach_precision_to_linked_entities(
    conn: psycopg.Connection,
    paper: dict[str, Any],
    *,
    min_precision: float | None = None,
) -> None:
    """Annotate ``paper['linked_entities']`` with precision_estimate metadata.

    Mutates the paper dict in place. Each linked entity gains
    ``precision_estimate`` (rounded float) and ``precision_band`` derived
    from ``scix.extract.ner_quality_profile.precision_estimate``.

    The matview row carries entity_id, name, type, link_type, confidence —
    but not source or evidence. We pull (source, evidence) per entity_id
    from ``document_entities + entities`` for this bibcode so the profile
    lookup has the full ``(entity_type, source, agreement, year)`` tuple.

    When the same entity has multiple link rows for a bibcode, agreement
    is reduced positively: True if any True, else False if any False, else
    None — matches the union-positive semantics already used in entity tool.

    When ``min_precision`` is set, drops linked entities whose final
    estimate is below the threshold. Entities for which the profile lookup
    failed (no precision_estimate attached) are kept regardless so
    filtering never accidentally hides entire entities due to a profile
    miss.
    """
    linked = paper.get("linked_entities")
    if not linked or not isinstance(linked, list):
        return
    bibcode = paper.get("bibcode")
    if not bibcode:
        return

    # Per-bibcode pull of source + agreement keyed by entity_id. Single
    # query — same join the entity tool uses, just constrained to one
    # paper.
    sources: dict[int, str] = {}
    agreements: dict[int, bool | None] = {}
    sql = """
        SELECT de.entity_id, e.source, de.evidence
        FROM document_entities de
        JOIN entities e ON e.id = de.entity_id
        WHERE de.bibcode = %s
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (bibcode,))
            rows = cur.fetchall()
    except Exception:
        # Best-effort: lookup failures must not break get_paper for callers
        # that don't need precision metadata.
        return

    _MISSING = object()
    agreements_raw: dict[int, Any] = {}
    for entity_id, source, evidence in rows:
        if entity_id is None:
            continue
        sources.setdefault(entity_id, source or "")
        a: bool | None = None
        if isinstance(evidence, dict):
            a_raw = evidence.get("agreement")
            if isinstance(a_raw, bool):
                a = a_raw
        prev = agreements_raw.get(entity_id, _MISSING)
        if prev is _MISSING:
            agreements_raw[entity_id] = a
        elif prev is True or a is True:
            agreements_raw[entity_id] = True
        elif prev is False or a is False:
            agreements_raw[entity_id] = False
    for k, v in agreements_raw.items():
        agreements[k] = v if isinstance(v, bool) else None

    year = paper.get("year")
    year_int = year if isinstance(year, int) else None

    from scix.extract.ner_quality_profile import (
        precision_band,
        precision_estimate,
    )

    # eq95: drop denylisted (name, type) pairs from linked_entities so
    # generic-word noise ('data'/'dataset', 'method'/'method', etc.) never
    # surfaces alongside real entities. Applied before precision estimate
    # so the SQL noise lookup we just did isn't wasted on rows we'd drop
    # anyway — but the lookup is keyed by entity_id, not canonical_name,
    # so the cost is one batch regardless.
    from scix.extract.ner_denylist import is_denylisted as _is_denylisted

    enriched: list[Any] = []
    for ent in linked:
        if not isinstance(ent, dict):
            enriched.append(ent)
            continue
        if _is_denylisted(ent.get("name"), ent.get("type")):
            continue
        eid = ent.get("entity_id")
        etype = ent.get("type") or ""
        src = sources.get(eid, "") if isinstance(eid, int) else ""
        agr = agreements.get(eid) if isinstance(eid, int) else None
        try:
            pe = precision_estimate(
                entity_type=etype,
                source=src,
                agreement=agr,
                year=year_int,
            )
            ent["precision_estimate"] = round(pe, 2)
            ent["precision_band"] = precision_band(pe)
        except Exception:
            # Quality profile is best-effort — never break the row on a
            # lookup failure.
            pass
        if (
            min_precision is not None
            and isinstance(ent.get("precision_estimate"), (int, float))
            and ent["precision_estimate"] < min_precision
        ):
            continue
        enriched.append(ent)

    paper["linked_entities"] = enriched


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
        min_prec_raw = args.get("min_precision")
        min_precision: float | None
        if isinstance(min_prec_raw, bool) or min_prec_raw is None:
            min_precision = None
        elif isinstance(min_prec_raw, (int, float)):
            min_precision = float(min_prec_raw)
        else:
            min_precision = None
        for paper in result.papers:
            if isinstance(paper, dict):
                _attach_precision_to_linked_entities(conn, paper, min_precision=min_precision)
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


def _resolve_working_set_bibcodes(args: dict[str, Any]) -> list[str]:
    """Return the bibcodes to scope a tool call to.

    Resolution order:
        1. ``args["bibcodes"]`` if non-empty (explicit caller intent).
        2. ``_session_state.get_focused_papers()`` (papers inspected via
           ``get_paper`` during the session).
        3. ``_session_state.get_working_set()`` as a final fallback for
           backward compatibility.

    Returns an empty list when no working-set source is available — callers
    decide whether that's an error (e.g. ``temporal_evolution`` with no
    ``bibcode_or_query``) or a no-op signaling full-corpus behaviour
    (e.g. ``facet_counts``).

    This is the canonical pattern referenced by the bead-3uvn working_set
    abstraction; ``find_gaps`` uses an inline equivalent.
    """
    explicit = args.get("bibcodes")
    if isinstance(explicit, list) and explicit:
        return [str(b) for b in explicit]

    focused = _session_state.get_focused_papers()
    if focused:
        return list(focused)

    return [e.bibcode for e in _session_state.get_working_set()]


def _handle_facet_counts(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Facet counts with optional working-set scoping.

    When ``bibcodes`` is omitted, falls through to the session's focused
    papers (see ``_resolve_working_set_bibcodes``). When neither is set,
    runs the unscoped corpus-wide facet — preserves the legacy contract.
    """
    try:
        filters = _parse_filters(args.get("filters"))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    limit = args.get("limit", 50)
    bibcodes = _resolve_working_set_bibcodes(args) or None
    try:
        result = search.facet_counts(
            conn,
            args["field"],
            filters=filters,
            limit=limit,
            bibcodes=bibcodes,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return _result_to_json(result)


def _handle_temporal_evolution(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Temporal evolution with optional working-set scoping.

    Resolution order for the bibcode set:
        1. ``args["bibcodes"]`` if non-empty.
        2. session focused papers (multi-paper aggregate citations mode).
        3. ``args["bibcode_or_query"]`` (legacy single-paper / query path).

    Returns a clean JSON error when none of the three sources is provided.
    """
    year_start = _coerce_year(args.get("year_start"), "year_start")
    year_end = _coerce_year(args.get("year_end"), "year_end")
    if year_start is not None and year_end is not None and year_end < year_start:
        raise ValueError(f"year_end ({year_end}) must be >= year_start ({year_start})")

    bibcodes = _resolve_working_set_bibcodes(args)
    bibcode_or_query = args.get("bibcode_or_query")

    if not bibcodes and not bibcode_or_query:
        return json.dumps(
            {
                "error": (
                    "temporal_evolution requires either bibcode_or_query, an "
                    "explicit bibcodes=[...] list, or a non-empty working set "
                    "(call get_paper on one or more papers first)."
                )
            }
        )

    # When working-set bibcodes are present, drive temporal_evolution from
    # the bibcode list and ignore bibcode_or_query (the per-paper / query
    # path is mutually exclusive with the multi-paper aggregate path).
    try:
        result = search.temporal_evolution(
            conn,
            None if bibcodes else bibcode_or_query,
            year_start=year_start,
            year_end=year_end,
            bibcodes=bibcodes or None,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return _result_to_json(result)


def _handle_citation_traverse(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified citation graph traversal.

    Dispatches by mode:
        * mode='graph' (default) — neighborhood walk. Accepts either a
          single ``bibcode`` (legacy) or a multi-paper ``bibcodes=[...]``
          list (working-set expansion). When neither is given, falls
          through to the session's focused papers. Multi-bibcode results
          are returned under ``by_bibcode`` keyed by source bibcode.
        * mode='chain' — shortest-path search, requires source_bibcode and
          target_bibcode and accepts max_depth (clamped to 1..5). Forwards
          to scix.search.citation_chain. Single-source-target by definition;
          working-set scoping does not apply.

    Returns a structured JSON ``error`` payload for invalid mode or
    missing required fields rather than raising — keeps the MCP boundary
    contract consistent with the rest of the dispatch layer.
    """
    mode = args.get("mode", "graph")

    if mode == "graph":
        # Single-bibcode (legacy) takes precedence over working-set mode:
        # an explicit single bibcode means the agent wants that paper
        # specifically. Multi-bibcode mode is engaged either when the
        # caller passes bibcodes=[...] explicitly or, with no single
        # bibcode given, the session has focused papers to fall through to.
        single_bibcode = args.get("bibcode")
        if single_bibcode:
            return _handle_citation_graph(conn, args)
        ws_bibcodes = _resolve_working_set_bibcodes(args)
        if ws_bibcodes:
            return _handle_citation_traverse_multi(conn, ws_bibcodes, args)
        return json.dumps(
            {
                "error": (
                    "bibcode is required when mode='graph' (or pass "
                    "bibcodes=[...] / focus papers via get_paper for "
                    "working-set mode)."
                )
            }
        )

    if mode == "chain":
        source = args.get("source_bibcode")
        target = args.get("target_bibcode")
        if not source or not target:
            return json.dumps(
                {"error": ("source_bibcode and target_bibcode are required " "when mode='chain'")}
            )
        max_depth = max(1, min(args.get("max_depth", 5), 5))
        result = search.citation_chain(
            conn,
            source,
            target,
            max_depth=max_depth,
        )
        return _result_to_json(result)

    return json.dumps(
        {
            "error": f"Invalid mode: {mode!r}. Use 'graph' or 'chain'.",
        }
    )


def _handle_citation_traverse_multi(
    conn: psycopg.Connection,
    bibcodes: list[str],
    args: dict[str, Any],
) -> str:
    """Walk the citation neighborhood of multiple bibcodes.

    Iterates ``_handle_citation_graph`` per source bibcode and aggregates
    the results into a ``by_bibcode`` mapping. The per-bibcode ``limit`` is
    preserved unchanged (so an agent passing ``limit=20`` gets up to 20
    neighbors per source paper). Bibcodes that error out (missing paper,
    DB error) are surfaced as ``{"error": "..."}`` entries rather than
    aborting the whole call — keeps multi-paper exploration robust.
    """
    per_bibcode: dict[str, Any] = {}
    for bib in bibcodes:
        per_args = dict(args)
        per_args["bibcode"] = bib
        per_args.pop("bibcodes", None)
        try:
            single_json = _handle_citation_graph(conn, per_args)
            per_bibcode[bib] = json.loads(single_json)
        except Exception as exc:  # pragma: no cover — surfaces tool-level error
            per_bibcode[bib] = {"error": str(exc)}

    return json.dumps(
        {
            "mode": "graph",
            "scope": "working_set",
            "bibcodes": list(bibcodes),
            "by_bibcode": per_bibcode,
        },
        indent=2,
        default=str,
    )


def _enrich_citations_with_intent(
    conn: psycopg.Connection,
    *,
    target_bibcode: str,
    source_bibcodes: list[str],
    direction: str,
) -> dict[str, str]:
    """Return {source_bibcode: intent} for any covered citation contexts.

    Covers forward direction (sources that cite target) and backward
    (target cites these references — passed in as ``source_bibcodes`` with
    direction='backward'). Citation_contexts is keyed
    (source_bibcode, target_bibcode) so we swap the WHERE column based
    on direction. Returns empty dict if nothing covered (~99.7% of edges
    are not in citation_contexts per bead 79n).
    """
    if not source_bibcodes:
        return {}
    if direction == "forward":
        sql = (
            "SELECT source_bibcode, intent FROM citation_contexts "
            "WHERE target_bibcode = %s AND source_bibcode = ANY(%s) "
            "AND intent IS NOT NULL"
        )
        params: tuple = (target_bibcode, list(source_bibcodes))
    else:  # backward
        sql = (
            "SELECT target_bibcode AS bib, intent FROM citation_contexts "
            "WHERE source_bibcode = %s AND target_bibcode = ANY(%s) "
            "AND intent IS NOT NULL"
        )
        params = (target_bibcode, list(source_bibcodes))
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return {row[0]: row[1] for row in cur.fetchall()}


def _annotate_papers_with_intent(
    papers: list[dict[str, Any]], intent_by_bibcode: dict[str, str]
) -> list[dict[str, Any]]:
    """Add 'intent' field to each paper dict if covered by citation_contexts."""
    for p in papers:
        bib = p.get("bibcode")
        if bib and bib in intent_by_bibcode:
            p["intent"] = intent_by_bibcode[bib]
    return papers


def _handle_citation_graph(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Get citations/references with direction control.

    Each returned edge is annotated with 'intent' (method / background /
    result_comparison) when the citation appears in citation_contexts —
    surfacing the structural-citation signal for ~0.27% of edges that
    have context coverage.
    """
    bibcode = args["bibcode"]
    direction = args.get("direction", "forward")
    limit = args.get("limit", 20)

    def annotate(result_json_str: str, dir_: str) -> str:
        try:
            payload = json.loads(result_json_str)
        except (ValueError, TypeError):
            return result_json_str
        papers = payload.get("papers") or []
        if not papers:
            return result_json_str
        sources = [p["bibcode"] for p in papers if p.get("bibcode")]
        intents = _enrich_citations_with_intent(
            conn,
            target_bibcode=bibcode,
            source_bibcodes=sources,
            direction=dir_,
        )
        if intents:
            _annotate_papers_with_intent(papers, intents)
        return json.dumps(payload, indent=2, default=str)

    results: list[dict[str, Any]] = []

    if direction in ("forward", "both"):
        fwd = search.get_citations(conn, bibcode, limit=limit)
        fwd_json = annotate(_result_to_json(fwd), "forward")
        if direction == "forward":
            return fwd_json
        results.append({"direction": "forward", "result": json.loads(fwd_json)})

    if direction in ("backward", "both"):
        bwd = search.get_references(conn, bibcode, limit=limit)
        bwd_json = annotate(_result_to_json(bwd), "backward")
        if direction == "backward":
            return bwd_json
        results.append({"direction": "backward", "result": json.loads(bwd_json)})

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
_EXTRACTION_TYPE_ENTITIES: frozenset[str] = frozenset({"negative_result", "quant_claim"})


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
    # action='papers' accepts entity_id directly, no query needed when given.
    is_extraction_row = action == "search" and entity_type in _EXTRACTION_TYPE_ENTITIES
    is_papers_with_id = action == "papers" and args.get("entity_id") is not None
    if not is_extraction_row and not is_papers_with_id and (not query or not query.strip()):
        return json.dumps({"error": "query must be a non-empty string"})

    if action == "resolve":
        resolver = EntityResolver(conn)
        candidates = resolver.resolve(
            query.strip(),
            discipline=args.get("discipline"),
            fuzzy=args.get("fuzzy", False),
        )
        # eq95: drop denylisted (canonical_name, entity_type) pairs so
        # noisy generic-word entities ('data'/'dataset', 'method'/'method',
        # etc.) don't surface as resolution candidates. Caller can still
        # query a denylisted entity by passing its entity_id directly.
        from scix.extract.ner_denylist import is_denylisted as _is_denylisted

        candidates = [c for c in candidates if not _is_denylisted(c.canonical_name, c.entity_type)]
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

    if action == "papers":
        # Surface document_entities (57.7M rows linking papers to harvested
        # entities across all 13 types — gene, software, mission, organism,
        # target, observable, chemical, location, taxon, plus the original
        # methods/datasets/instruments/materials). This is the dbl-epic
        # payoff: every entity in the resolver maps to its tagged papers.
        entity_id = args.get("entity_id")
        if entity_id is None:
            # Fall back to resolving the query first if no entity_id given
            if not query.strip():
                return json.dumps(
                    {"error": "entity_id or query must be provided for action='papers'"}
                )
            resolver = EntityResolver(conn)
            cands = resolver.resolve(query.strip(), fuzzy=False)
            # eq95: skip past denylisted candidates rather than auto-picking
            # one — protects callers who passed a query that resolves to a
            # noisy generic-word entity. Callers who explicitly want a
            # denylisted entity can pass entity_id directly.
            from scix.extract.ner_denylist import is_denylisted as _is_denylisted

            cands = [c for c in cands if not _is_denylisted(c.canonical_name, c.entity_type)]
            if not cands:
                return json.dumps({"query": query, "entity_id": None, "papers": [], "total": 0})
            entity_id = cands[0].entity_id

        try:
            entity_id = int(entity_id)
        except (TypeError, ValueError):
            return json.dumps({"error": "entity_id must be an integer"})

        limit = min(args.get("limit", 20), 200)
        # Pull entity metadata (entity_type, source) and per-link
        # provenance (match_method, evidence with optional 'agreement'
        # flag from the classifier post-pass) so we can attach a
        # precision_estimate per result row — making the dbl.3 D3
        # quality_profile visible at the agent surface.
        sql = """
            SELECT de.bibcode, de.link_type, de.confidence, de.match_method,
                   de.evidence,
                   e.canonical_name AS entity_name,
                   e.entity_type    AS entity_type,
                   e.source         AS entity_source,
                   p.title, p.year, p.authors[1] AS first_author, p.citation_count
            FROM document_entities de
            JOIN entities e ON e.id = de.entity_id
            LEFT JOIN papers p ON p.bibcode = de.bibcode
            WHERE de.entity_id = %s
            ORDER BY p.citation_count DESC NULLS LAST, de.bibcode ASC
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (entity_id, limit))
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
        papers = [dict(zip(cols, r)) for r in rows]

        # Attach precision_estimate + precision_band per row.
        # Source: dbl.3 quality_profile from src/scix/extract/ner_quality_profile.py.
        # Per-row inputs: entity_type (from entities), source (from entities,
        # 'gliner' triggers the empirical precision lookup, anything else
        # falls to LEXICAL_PRECISION_DEFAULT), agreement (from
        # document_entities.evidence->>'agreement' when the classifier
        # post-pass has run), year (from papers).
        from scix.extract.ner_quality_profile import (
            precision_band,
            precision_estimate,
        )

        entity_type_val: str | None = None
        for p in papers:
            ev = p.get("evidence") or {}
            agreement_raw = ev.get("agreement") if isinstance(ev, dict) else None
            agreement: bool | None
            if isinstance(agreement_raw, bool):
                agreement = agreement_raw
            else:
                agreement = None
            etype = p.get("entity_type") or ""
            esrc = p.get("entity_source") or ""
            year = p.get("year")
            year_int = int(year) if isinstance(year, int) else None
            try:
                pe = precision_estimate(
                    entity_type=etype,
                    source=esrc,
                    agreement=agreement,
                    year=year_int,
                )
                p["precision_estimate"] = round(pe, 2)
                p["precision_band"] = precision_band(pe)
            except Exception:
                # Quality profile is best-effort; never break the response
                # on a profile lookup failure.
                pass
            if entity_type_val is None:
                entity_type_val = etype

        return json.dumps(
            {
                "entity_id": entity_id,
                "entity_type": entity_type_val,
                "papers": papers,
                "total": len(papers),
            },
            indent=2,
            default=str,
        )

    return json.dumps({"error": f"Invalid action: {action}. Use 'search', 'resolve', or 'papers'."})


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
                    if isinstance(c, dict) and str(c.get("quantity", "")) == needle
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

    Canonical reference for the working-set fall-through pattern (bead
    scix_experiments-3uvn): when the caller doesn't pass an explicit list
    of bibcodes, the tool consults ``_session_state.get_focused_papers()``
    (papers inspected via ``get_paper``) and falls back to the broader
    working set. The same shape is used by ``facet_counts``,
    ``temporal_evolution``, and ``citation_traverse`` (mode='graph') —
    see ``_resolve_working_set_bibcodes``.

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
                    f"Invalid signal: {signal}. " f"Must be one of {sorted(_SIGNAL_COLUMN_PREFIX)}"
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
    # fall back to working set for backward compatibility.
    ws_bibcodes = _session_state.get_focused_papers()
    if not ws_bibcodes:
        ws_bibcodes = [e.bibcode for e in _session_state.get_working_set()]
    ws_bibcodes = ws_bibcodes[:200]

    # When no working set is populated and the caller passed a query, seed
    # the working set on-the-fly via concept_search so single-call agents
    # can run gap analysis in one shot. Pure convenience — same downstream
    # logic, just bootstrapped.
    seed_query = args.get("query")
    auto_seeded = False
    if not ws_bibcodes and isinstance(seed_query, str) and seed_query.strip():
        try:
            from scix.search import concept_search as _concept_search

            seed_result = _concept_search(
                conn, seed_query.strip(), limit=20, include_subtopics=False
            )
            ws_bibcodes = [
                p["bibcode"]
                for p in (seed_result.papers or [])
                if isinstance(p, dict) and p.get("bibcode")
            ]
            auto_seeded = bool(ws_bibcodes)
        except Exception:
            # Best-effort: fall through to the no-papers branch below.
            ws_bibcodes = []

    if not ws_bibcodes:
        return json.dumps(
            {
                "papers": [],
                "total": 0,
                "signal": signal,
                "message": (
                    "No focused papers and no query provided. "
                    "Use get_paper(bibcode) to inspect papers first, or "
                    "pass query='<topic>' to auto-seed via concept_search."
                ),
            },
            indent=2,
        )

    # For the citation signal, filter out the Phase-A sentinel (-1) which
    # marks non-giant-component papers rather than a real community.
    sentinel_filter = f"AND pm.{community_col} <> -1" if signal == "citation" else ""
    seed_sentinel_filter = f"AND pm2.{community_col} <> -1" if signal == "citation" else ""

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


# ---------------------------------------------------------------------------
# chunk_search handler (PRD chunk-embeddings-build)
# ---------------------------------------------------------------------------

# Lazy module-level cache for the INDUS embedder used by chunk_search. Loaded
# on first dispatch so server startup stays fast and Qdrant-disabled
# deployments never pay the model load cost.
_indus_embedder: tuple[Any, Any] | None = None


def _get_indus_embedder() -> tuple[Any, Any]:
    """Return a cached (model, tokenizer) pair for the INDUS encoder.

    Reuses :func:`scix.embed.load_model`, which has its own
    :data:`scix.embed._model_cache`, so even repeated process restarts only
    pay the disk read cost. CPU-only by default; chunk_search is interactive
    and the per-query cost is dominated by the Qdrant round-trip, not the
    encoder.
    """
    global _indus_embedder
    if _indus_embedder is None:
        # Re-import locally so tests that monkeypatch ``scix.embed.load_model``
        # see the patched function rather than the binding captured at import.
        from scix import embed as _embed

        _indus_embedder = _embed.load_model("indus", device="cpu")
    return _indus_embedder


def _normalize_str_list(value: Any) -> list[str] | None:
    """Coerce a str-or-list filter value into a list[str] (or None)."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else None
    if isinstance(value, (list, tuple)):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or None
    raise ValueError(f"expected string or list of strings, got {type(value).__name__}")


def _normalize_int_list(value: Any) -> list[int] | None:
    """Coerce an int-or-list filter value into a list[int] (or None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int but not meaningful as a community id.
        raise ValueError("expected integer or list of integers, got bool")
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for v in value:
            if isinstance(v, bool):
                raise ValueError("community_id_med list contains a bool")
            try:
                out.append(int(v))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"could not coerce {v!r} to int") from exc
        return out or None
    raise ValueError(f"expected integer or list of integers, got {type(value).__name__}")


def _handle_chunk_search(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the ``chunk_search`` MCP tool.

    Returns a JSON string with shape::

        {
            "matches": [
                {
                    "bibcode": str,
                    "chunk_id": int,
                    "section_heading": str | null,
                    "score": float,
                    "snippet": str | null,
                },
                ...
            ],
            "total": int,
            "filter_summary": {...},
        }

    If Qdrant is not configured (``QDRANT_URL`` unset or qdrant-client not
    installed), returns ``{"error": "qdrant_disabled", ...}`` so callers can
    detect the gate without an exception.
    """
    if not _qdrant_enabled():
        return json.dumps(
            {
                "error": "qdrant_disabled",
                "message": (
                    "chunk_search requires the Qdrant backend "
                    "(set QDRANT_URL and install qdrant-client)."
                ),
            },
            indent=2,
        )

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"}, indent=2)
    query = query.strip()

    # --- limit clamp ---
    raw_limit = args.get("limit", 20)
    try:
        limit = int(raw_limit) if raw_limit is not None else 20
    except (TypeError, ValueError):
        return json.dumps(
            {"error": f"limit must be an integer, got {raw_limit!r}"},
            indent=2,
        )
    if limit < 1:
        limit = 1
    elif limit > 100:
        limit = 100

    # --- filter parsing ---
    filters_raw = args.get("filters") or {}
    if not isinstance(filters_raw, dict):
        return json.dumps(
            {"error": f"filters must be an object, got {type(filters_raw).__name__}"},
            indent=2,
        )

    try:
        year_min = filters_raw.get("year_min")
        year_max = filters_raw.get("year_max")
        if year_min is not None:
            year_min = int(year_min)
        if year_max is not None:
            year_max = int(year_max)
        arxiv_class = _normalize_str_list(filters_raw.get("arxiv_class"))
        community_id_med = _normalize_int_list(filters_raw.get("community_id_med"))
        section_heading = _normalize_str_list(filters_raw.get("section_heading"))
        bibcode = _normalize_str_list(filters_raw.get("bibcode"))
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid filters: {exc}"}, indent=2)

    # --- encode query via INDUS (mean pooling, 768-dim) ---
    try:
        # Re-import lazily so tests that monkeypatch ``scix.embed.embed_batch``
        # after import time see the patched function.
        from scix import embed as _embed

        model, tokenizer = _get_indus_embedder()
        vectors = _embed.embed_batch(model, tokenizer, [query], batch_size=1, pooling="mean")
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("chunk_search: INDUS encode failed")
        return json.dumps({"error": f"encode_failed: {exc}"}, indent=2)
    if not vectors:
        return json.dumps({"error": "encode_failed: no vector returned"}, indent=2)
    vector = vectors[0]

    # --- ANN call + snippet hydration ---
    try:
        hits = _qdrant_tools.chunk_search_by_text(
            vector,
            year_min=year_min,
            year_max=year_max,
            arxiv_class=arxiv_class,
            community_id_med=community_id_med,
            section_heading_norm=section_heading,
            bibcode=bibcode,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("chunk_search: Qdrant query failed")
        return json.dumps({"error": f"qdrant_failed: {exc}"}, indent=2)

    try:
        hits = _qdrant_tools.fetch_chunk_snippets(conn, hits)
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("chunk_search: snippet fetch failed; returning hits without snippets")
        # Non-fatal — keep hits with snippet=None rather than dropping the call.
        # We still surface the failure as a warning field so the caller can
        # decide whether to retry.
        snippet_warning: str | None = f"snippet_fetch_failed: {exc}"
    else:
        snippet_warning = None

    matches = [
        {
            "bibcode": h.bibcode,
            "chunk_id": h.chunk_id,
            "section_heading": h.section_heading or h.section_heading_norm,
            "score": h.score,
            "snippet": h.snippet,
        }
        for h in hits
    ]

    filter_summary: dict[str, Any] = {"limit": limit}
    if year_min is not None:
        filter_summary["year_min"] = year_min
    if year_max is not None:
        filter_summary["year_max"] = year_max
    if arxiv_class is not None:
        filter_summary["arxiv_class"] = arxiv_class
    if community_id_med is not None:
        filter_summary["community_id_med"] = community_id_med
    if section_heading is not None:
        filter_summary["section_heading"] = section_heading
    if bibcode is not None:
        filter_summary["bibcode"] = bibcode

    payload: dict[str, Any] = {
        "matches": matches,
        "total": len(matches),
        "filter_summary": filter_summary,
    }
    if snippet_warning is not None:
        payload["warning"] = snippet_warning
    return json.dumps(payload, indent=2, default=str)


def _handle_find_similar_by_examples(args: dict[str, Any]) -> str:
    """Dispatch for the Qdrant-backed discovery tool.

    Returns a structured error if Qdrant is not configured, so the tool can
    live in the registered tool set even in mixed deployments where the
    backend is not yet wired up. Callers should check the ``error`` field.
    """
    if not _qdrant_enabled():
        return json.dumps(
            {
                "error": "qdrant_not_configured",
                "message": (
                    "find_similar_by_examples requires the Qdrant backend " "(QDRANT_URL env var)."
                ),
            }
        )

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

    return json.dumps(
        {
            "backend": "qdrant",
            "collection": _qdrant_tools.COLLECTION,
            "results": [dataclasses.asdict(h) for h in hits],
        }
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
        return json.dumps({"error": f"relation must be one of {sorted(VALID_RELATIONS)} or null"})

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


# ---------------------------------------------------------------------------
# cited_by_intent handler — exploits citation_contexts.intent classification
# ---------------------------------------------------------------------------

_VALID_CITATION_INTENTS: frozenset[str] = frozenset({"method", "background", "result_comparison"})


def _handle_cited_by_intent(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Find papers that cite ``target_bibcode`` for a specific reason.

    Surfaces the structural-citation signal in ``citation_contexts.intent``:
    each row carries an intent label (method / background / result_comparison)
    classified from the citation context text. This lets agents answer
    questions like 'which papers used X as their method?' or 'which papers
    compared their results to X?' — questions that vanilla retrieval cannot
    answer because they require knowing *why* one paper cites another, not
    just that it does.

    Coverage: ~825K citation contexts across ~30K source papers and ~250K
    cited papers. For papers not covered, returns empty cleanly.
    """
    target_bibcode = args.get("target_bibcode")
    if not isinstance(target_bibcode, str) or not target_bibcode.strip():
        return json.dumps({"error": "target_bibcode must be a non-empty string"})

    intent = args.get("intent")
    if intent is not None and intent not in _VALID_CITATION_INTENTS:
        return json.dumps(
            {
                "error": (
                    f"intent must be one of {sorted(_VALID_CITATION_INTENTS)} " f"or null (any)"
                )
            }
        )

    limit = min(int(args.get("limit", 20)), 200)

    sql = """
        SELECT
            cc.source_bibcode,
            cc.intent,
            substr(cc.context_text, 1, 400) AS context_excerpt,
            p.title,
            p.year,
            p.authors[1] AS first_author,
            p.citation_count
        FROM citation_contexts cc
        LEFT JOIN papers p ON cc.source_bibcode = p.bibcode
        WHERE cc.target_bibcode = %s
          AND ( %s::text IS NULL OR cc.intent = %s::text )
        ORDER BY p.citation_count DESC NULLS LAST, cc.id ASC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (target_bibcode.strip(), intent, intent, limit))
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    papers = [dict(zip(cols, r)) for r in rows]
    return json.dumps(
        {
            "target_bibcode": target_bibcode,
            "intent": intent,
            "papers": papers,
            "total": len(papers),
        },
        indent=2,
        default=str,
    )


# ---------------------------------------------------------------------------
# synthesize_findings handler (bead cfh9)
# ---------------------------------------------------------------------------


def _handle_synthesize_findings(
    conn: psycopg.Connection, args: dict[str, Any]
) -> str:
    """Bin a working set of papers into a section outline.

    Mirrors the ``find_gaps`` fall-through pattern: explicit
    ``working_set_bibcodes`` win; otherwise read from the session's
    focused papers; otherwise from the working set. Pure mechanism —
    the actual synthesis logic lives in :mod:`scix.synthesize` and is
    LLM-free per ZFC.
    """
    raw_bibcodes = args.get("working_set_bibcodes")
    if raw_bibcodes is None:
        # Session fall-through (matches find_gaps).
        bibcodes: list[str] = _session_state.get_focused_papers()
        if not bibcodes:
            bibcodes = [e.bibcode for e in _session_state.get_working_set()]
    elif isinstance(raw_bibcodes, list):
        bibcodes = [b for b in raw_bibcodes if isinstance(b, str)]
    else:
        return json.dumps(
            {"error": "working_set_bibcodes must be a list of strings"},
        )

    sections = args.get("sections")
    if sections is not None and not isinstance(sections, list):
        return json.dumps({"error": "sections must be a list of strings"})

    raw_cap = args.get("max_papers_per_section", 8)
    try:
        max_papers = int(raw_cap)
    except (TypeError, ValueError):
        return json.dumps(
            {"error": "max_papers_per_section must be an integer"},
        )
    # Hard cap to keep payload sizes sane; matches find_gaps' 200 cap.
    max_papers = max(0, min(max_papers, 200))

    result = _synthesize_findings(
        conn,
        working_set_bibcodes=bibcodes,
        sections=sections,
        max_papers_per_section=max_papers,
    )
    return json.dumps(result.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# section_retrieval handler
# ---------------------------------------------------------------------------


def _encode_section_query(query: str, dimensions: int = 1024) -> list[float]:
    """Encode a query string with the local nomic-embed-text-v1.5 model.

    Reuses :func:`scix.embeddings.section_pipeline._load_model` (lazy) and
    :func:`scix.embeddings.section_pipeline.encode_batch` so this module
    inherits the same model loader the indexing pipeline uses. The query
    is prefixed with ``"search_query: "`` per the nomic model card.

    Returns a 1024-dim Python list[float]. Raises ImportError if
    sentence_transformers is not installed (caller is expected to wrap and
    return a structured MCP error).
    """
    from scix.embeddings.section_pipeline import (  # local import — lazy
        DEFAULT_MODEL,
        _load_model,
        encode_batch,
    )

    model = _load_model(DEFAULT_MODEL)
    prefixed = _NOMIC_QUERY_PREFIX + (query or "")
    vectors = encode_batch(model, [prefixed], dimensions=dimensions)
    if not vectors:
        raise RuntimeError("section query encoder returned no vectors")
    return vectors[0]


def _section_filter_clauses(
    filters: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Build SQL fragments + parameter list for the section_retrieval filters.

    Returns a tuple of ``(extra_sql, params)`` where ``extra_sql`` is a
    string of zero or more ``AND <clause>`` fragments referring to columns
    on the ``papers`` row aliased as ``p``. Params are bound positionally.

    Filter contract (matches _SECTION_FILTERS_SCHEMA):
        - discipline   -> p.discipline = %s
        - year_min     -> p.year >= %s
        - year_max     -> p.year <= %s
        - bibcode_prefix -> p.bibcode LIKE %s   (caller-supplied trailing % logic)
    """
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    discipline = filters.get("discipline")
    if discipline is not None:
        clauses.append("AND p.discipline = %s")
        params.append(str(discipline))
    year_min = _coerce_year(filters.get("year_min"), "year_min")
    if year_min is not None:
        clauses.append("AND p.year >= %s")
        params.append(year_min)
    year_max = _coerce_year(filters.get("year_max"), "year_max")
    if year_max is not None:
        clauses.append("AND p.year <= %s")
        params.append(year_max)
    bibcode_prefix = filters.get("bibcode_prefix")
    if bibcode_prefix is not None:
        clauses.append("AND p.bibcode LIKE %s")
        params.append(f"{bibcode_prefix}%")
    return (" " + " ".join(clauses)) if clauses else "", params


def _section_dense_retrieve(
    conn: psycopg.Connection,
    query_vector: Sequence[float],
    filter_sql: str,
    filter_params: list[Any],
    fanout: int,
) -> list[tuple[str, int, float]]:
    """Run the dense leg of section retrieval inside an explicit transaction.

    Sets ``hnsw.iterative_scan = 'relaxed'`` and ``hnsw.ef_search = 100``
    via ``SET LOCAL`` so they roll back on transaction end and don't leak
    to other pool consumers.

    Returns a list of ``(bibcode, section_index, distance)`` tuples ordered
    by distance ascending (best first).
    """
    if fanout <= 0:
        return []
    vector_literal = "[" + ",".join(repr(float(v)) for v in query_vector) + "]"
    sql = f"""
        SELECT se.bibcode, se.section_index,
               (se.embedding <=> %s::halfvec) AS distance
        FROM section_embeddings se
        JOIN papers p ON p.bibcode = se.bibcode
        WHERE TRUE
        {filter_sql}
        ORDER BY se.embedding <=> %s::halfvec
        LIMIT %s
    """
    params = [vector_literal, *filter_params, vector_literal, fanout]
    rows: list[tuple[str, int, float]] = []
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed'")
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            cur.execute(sql, params)
            for row in cur.fetchall():
                rows.append((row[0], int(row[1]), float(row[2])))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
    return rows


def _section_bm25_retrieve(
    conn: psycopg.Connection,
    query: str,
    filter_sql: str,
    filter_params: list[Any],
    fanout: int,
) -> list[tuple[str, int, float]]:
    """Run the BM25 leg over papers_fulltext.sections_tsv.

    The tsvector index ranks the *paper*; we then unnest each matching
    paper's sections and emit one row per section whose body text matches
    the query terms via plainto_tsquery, scored by ts_rank on that section
    text. Returns ``(bibcode, section_index, ts_rank)`` tuples sorted by
    rank descending (higher = better).
    """
    if fanout <= 0:
        return []
    sql = f"""
        WITH matching_papers AS (
            SELECT pf.bibcode, pf.sections,
                   ts_rank(pf.sections_tsv, plainto_tsquery('english', %s)) AS paper_rank
            FROM papers_fulltext pf
            JOIN papers p ON p.bibcode = pf.bibcode
            WHERE pf.sections_tsv @@ plainto_tsquery('english', %s)
            {filter_sql}
            ORDER BY paper_rank DESC
            LIMIT %s
        ),
        per_section AS (
            SELECT mp.bibcode,
                   (sec.ord - 1)::int AS section_index,
                   ts_rank(
                       to_tsvector('english',
                           coalesce(sec.value->>'heading', '') || ' ' ||
                           coalesce(sec.value->>'text', '')
                       ),
                       plainto_tsquery('english', %s)
                   ) AS section_rank
            FROM matching_papers mp,
                 jsonb_array_elements(mp.sections) WITH ORDINALITY AS sec(value, ord)
            WHERE to_tsvector('english',
                      coalesce(sec.value->>'heading', '') || ' ' ||
                      coalesce(sec.value->>'text', '')
                  ) @@ plainto_tsquery('english', %s)
        )
        SELECT bibcode, section_index, section_rank
        FROM per_section
        ORDER BY section_rank DESC, bibcode, section_index
        LIMIT %s
    """
    params = [
        query,  # paper_rank ts_rank
        query,  # paper match
        *filter_params,
        fanout,  # paper LIMIT
        query,  # section_rank ts_rank
        query,  # section match
        fanout,  # final LIMIT
    ]
    rows: list[tuple[str, int, float]] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            rows.append((row[0], int(row[1]), float(row[2])))
    return rows


def _hydrate_section_payload(
    conn: psycopg.Connection,
    keys: Sequence[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Fetch heading + text for each (bibcode, section_index) key.

    Reads ``papers_fulltext.sections`` JSONB once per bibcode and indexes
    into the requested section. Returns a dict keyed by (bibcode, idx)
    whose values carry ``section_heading`` and ``snippet`` (truncated).
    """
    if not keys:
        return {}
    bibcodes = sorted({k[0] for k in keys})
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    sections_by_bibcode: dict[str, list[Any]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, sections FROM papers_fulltext WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        for bibcode, sections in cur.fetchall():
            if isinstance(sections, (str, bytes)):
                try:
                    sections = json.loads(sections)
                except (TypeError, ValueError, json.JSONDecodeError):
                    sections = []
            if isinstance(sections, list):
                sections_by_bibcode[bibcode] = sections
    for bibcode, idx in keys:
        sections = sections_by_bibcode.get(bibcode) or []
        section: dict[str, Any] | None = None
        if 0 <= idx < len(sections) and isinstance(sections[idx], dict):
            section = sections[idx]
        heading = (section.get("heading") if section else None) or ""
        text = (section.get("text") if section else None) or ""
        payloads[(bibcode, idx)] = {
            "section_heading": heading,
            "snippet": _truncate_snippet(text, _SNIPPET_MAX_CHARS),
        }
    return payloads


def _hydrate_canonical_urls(
    conn: psycopg.Connection,
    bibcodes: Sequence[str],
) -> dict[str, str | None]:
    """Map each bibcode to a canonical_url.

    Uses the first identifier in ``papers.identifier`` matching the arXiv
    pattern (mirrors :func:`scix.search._lookup_arxiv_id`) and feeds it to
    :func:`scix.sources.ar5iv._build_canonical_url`. bibcodes without an
    arXiv identifier map to None.
    """
    if not bibcodes:
        return {}
    from scix.sources.ar5iv import _ARXIV_ID_RE, _build_canonical_url

    out: dict[str, str | None] = {bibcode: None for bibcode in bibcodes}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, identifier FROM papers WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
        for bibcode, identifiers in cur.fetchall():
            if not identifiers:
                continue
            for ident in identifiers:
                if isinstance(ident, str) and _ARXIV_ID_RE.match(ident):
                    out[bibcode] = _build_canonical_url(ident)
                    break
    return out


def _handle_section_retrieval(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for ``section_retrieval``.

    Encodes the query with the local nomic model, runs dense HNSW + BM25
    retrieval in parallel (sequentially in code, structurally independent),
    fuses ranks via Reciprocal Rank Fusion (k=60), hydrates section text +
    canonical_url, and returns the top ``k`` items.
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})

    try:
        k = int(args.get("k", 10))
    except (TypeError, ValueError):
        return json.dumps({"error": "k must be an integer"})
    if k <= 0:
        return json.dumps({"error": "k must be positive"})
    # Cap fanout to keep blast radius bounded; matches the convention used
    # elsewhere in this module (find_gaps caps at 200).
    k = min(k, 200)

    try:
        filter_sql, filter_params = _section_filter_clauses(args.get("filters"))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    # Encode the query with the local nomic model.
    try:
        query_vector = _encode_section_query(query)
    except ImportError:
        return json.dumps(
            {
                "error": "embedding_dependency_missing",
                "hint": (
                    "section_retrieval requires sentence-transformers. "
                    "Install with: pip install -e .[search]"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("section_retrieval encode failed")
        return json.dumps({"error": f"encode_failed: {exc}"})

    fanout = max(50, k * 10)

    # Dense leg — explicit txn so SET LOCAL settings apply.
    try:
        dense_rows = _section_dense_retrieve(conn, query_vector, filter_sql, filter_params, fanout)
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("section_retrieval dense leg failed")
        return json.dumps({"error": f"dense_retrieve_failed: {exc}"})

    # BM25 leg.
    try:
        bm25_rows = _section_bm25_retrieve(conn, query, filter_sql, filter_params, fanout)
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("section_retrieval bm25 leg failed")
        return json.dumps({"error": f"bm25_retrieve_failed: {exc}"})

    dense_keys: list[tuple[str, int]] = [(b, i) for (b, i, _d) in dense_rows]
    bm25_keys: list[tuple[str, int]] = [(b, i) for (b, i, _r) in bm25_rows]

    fused = _rrf_fuse([dense_keys, bm25_keys], k_rrf=_RRF_K_DEFAULT)
    top_keys = [key for (key, _score) in fused[:k]]

    payloads = _hydrate_section_payload(conn, top_keys)
    bibcodes = sorted({k_[0] for k_ in top_keys})
    canonical_urls = _hydrate_canonical_urls(conn, bibcodes)

    score_by_key = {key: score for (key, score) in fused}
    results: list[dict[str, Any]] = []
    for key in top_keys:
        bibcode, idx = key
        payload = payloads.get(key) or {}
        results.append(
            {
                "bibcode": bibcode,
                "section_heading": payload.get("section_heading", ""),
                "snippet": payload.get("snippet", ""),
                "score": float(score_by_key.get(key, 0.0)),
                "canonical_url": canonical_urls.get(bibcode),
            }
        )
    return json.dumps(
        {"results": results, "total": len(results)},
        indent=2,
        default=str,
    )


# ---------------------------------------------------------------------------
# paper_claims retrieval handlers (PRD nanopub-claim-extraction)
# ---------------------------------------------------------------------------


def _handle_read_paper_claims(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the ``read_paper_claims`` MCP tool.

    Thin wrapper over :func:`scix.claims.retrieval.read_paper_claims` that
    surfaces structured-error JSON for invalid inputs (matches the
    convention used by other handlers in this module).
    """
    from scix.claims.retrieval import read_paper_claims

    bibcode = args.get("bibcode")
    if not isinstance(bibcode, str) or not bibcode.strip():
        return json.dumps({"error": "bibcode must be a non-empty string"})

    claim_type = args.get("claim_type")
    if claim_type is not None and not isinstance(claim_type, str):
        return json.dumps({"error": "claim_type must be a string or omitted"})

    limit = args.get("limit", 50)

    try:
        rows = read_paper_claims(
            conn,
            bibcode=bibcode,
            claim_type=claim_type,
            limit=limit,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {
            "bibcode": bibcode,
            "claim_type": claim_type,
            "claims": rows,
            "total": len(rows),
        },
        indent=2,
        default=str,
    )


def _handle_find_claims(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the ``find_claims`` MCP tool.

    Thin wrapper over :func:`scix.claims.retrieval.find_claims`. Coerces
    optional ``entity_id`` to int and surfaces structured-error JSON for
    invalid inputs.
    """
    from scix.claims.retrieval import find_claims

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})

    claim_type = args.get("claim_type")
    if claim_type is not None and not isinstance(claim_type, str):
        return json.dumps({"error": "claim_type must be a string or omitted"})

    entity_id = args.get("entity_id")
    if entity_id is not None:
        try:
            entity_id = int(entity_id)
        except (TypeError, ValueError):
            return json.dumps({"error": "entity_id must be an integer or omitted"})

    limit = args.get("limit", 25)

    try:
        rows = find_claims(
            conn,
            query=query,
            claim_type=claim_type,
            entity_id=entity_id,
            limit=limit,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {
            "query": query,
            "claim_type": claim_type,
            "entity_id": entity_id,
            "claims": rows,
            "total": len(rows),
        },
        indent=2,
        default=str,
    )


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
