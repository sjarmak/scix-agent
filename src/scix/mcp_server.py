"""MCP server exposing the consolidated tool surface for agent navigation of the SciX corpus.

Uses the `mcp` Python SDK to register tools. Each tool is a thin wrapper
around functions in search.py. Connection pooling via psycopg.pool for
production-grade performance.

The current tool registry is enumerated in ``EXPECTED_TOOLS``; the
agent-visible subset is gated by ``_HIDDEN_TOOLS`` (overridable via
``SCIX_HIDDEN_TOOLS``). The premortem tool-count cap (≤ 15 visible)
is tracked in ``docs/mcp_tool_audit_2026-04.md`` and policed by ADRs.

Consolidation (v3, 2026-04-25):
    Original 28 → 13 → ~15 agent-facing tools + deprecated aliases.
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
from typing import Any, Callable, Generator, Mapping, Protocol, Sequence

import psycopg

from scix import search
from scix.db import DEFAULT_DSN
from scix.embed import (
    _model_cache,
    clear_model_cache,
    embed_batch,  # noqa: F401  patch surface for scix.mcp_handlers.search
    load_model,
)
from scix.entity_resolver import EntityResolver  # noqa: F401  patch surface for handlers
from scix.jit.disambiguator import disambiguate_query
from scix.mcp_errors import ErrorCode
from scix.mcp_tool_specs import (
    _CHUNK_SEARCH_SPEC,
    _FILTERS_SCHEMA,  # noqa: F401  re-exported for callers/tests
    _SECTION_FILTERS_SCHEMA,  # noqa: F401  re-exported for callers/tests
    _SIGNAL_USED_DESCRIPTION,  # noqa: F401  re-exported for callers/tests
    _TOOL_SPECS,
    MAX_ENTITY_FILTER_ITEMS,
)
from scix.search import CrossEncoderReranker
from scix.session import SessionState
from scix.synthesize import (
    MAX_WORKING_SET_BIBCODES,
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

# Session-level statement_timeout applied to every MCP connection (bead
# scix_experiments-82j0, third postgres OOM 2026-06-12). Sent in the libpq
# startup packet so it applies before the first statement, with no
# per-connection round-trip. The per-tool SET LOCAL timeouts tighten this
# inside each tool transaction; this default is the safety net for everything
# that runs outside one (startup smoke calls, query_log writes, helpers
# invoked before _set_timeout). Deliberately NOT an ALTER ROLE migration: the
# MCP server shares its postgres role with multi-hour batch work (index
# builds, backfills) that a role-level timeout would break.
_SESSION_STATEMENT_TIMEOUT_MS = int(
    float(os.environ.get("SCIX_SESSION_STATEMENT_TIMEOUT", "120")) * 1000
)
if _SESSION_STATEMENT_TIMEOUT_MS <= 0:
    raise ValueError(
        "SCIX_SESSION_STATEMENT_TIMEOUT must be > 0 seconds; got "
        f"{os.environ.get('SCIX_SESSION_STATEMENT_TIMEOUT')!r}"
    )
_CONN_OPTIONS = f"-c statement_timeout={_SESSION_STATEMENT_TIMEOUT_MS}"


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
            kwargs={"options": _CONN_OPTIONS},
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
        conn = psycopg.connect(os.environ.get("SCIX_DSN", DEFAULT_DSN), options=_CONN_OPTIONS)
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
    # forward_citations (bead 9afa): merges cited_by_intent + find_replications
    # behind an ``annotate`` enum. Cap matches the slower (relation) leg.
    "forward_citations": float(os.environ.get("SCIX_TIMEOUT_FORWARD_CITATIONS", "15")),
    # Structural-citation lookup over citation_contexts.intent — retained as a
    # deprecated alias of forward_citations(annotate='intent').
    "cited_by_intent": float(os.environ.get("SCIX_TIMEOUT_CITED_BY_INTENT", "5")),
    # Claim/finding extraction surface (bead c996) — split from entity tool's
    # entity_type enum under bead mh14. Default-hidden today (extractions table
    # has 0 rows for negative_result/quant_claim on prod); explicit timeout so
    # operators can tune without a code change once the table is populated.
    "claim_search": float(os.environ.get("SCIX_TIMEOUT_CLAIM_SEARCH", "10")),
    # Terminal synthesis tool — three short SELECTs against papers,
    # citation_contexts, paper_metrics; cap matches find_gaps.
    "synthesize_findings": float(os.environ.get("SCIX_TIMEOUT_SYNTHESIZE_FINDINGS", "15")),
}

# Result-limit convention (bead 9afa / docs/mcp_tool_audit_2026-06.md §5):
# tools default to DEFAULT_RESULT_LIMIT results and cap at MAX_WORKING_SET_BIBCODES
# (=200) unless a tool documents a justified exception. Documented exceptions:
# ``search`` (default 10 — established page size, eval-baseline-pinned),
# ``facet_counts`` (default 50 — distribution buckets), and ``lit_review``'s
# composite sub-counts (top_seeds / expansion_seeds — domain-specific knobs).
DEFAULT_RESULT_LIMIT = 20

# Tools whose backing data is missing on this deployment. Default-hidden so
# agents don't waste calls on tools that can't return real results. Override
# via SCIX_HIDDEN_TOOLS env var (comma-separated; empty string to show all).
#   * chunk_search       — Qdrant collection scix_chunks_v1 not yet populated
#   * section_retrieval  — section_embeddings table not yet populated
#   * read_paper_claims, find_claims — paper_claims table empty (no extraction
#     run yet); table itself exists per migration 062
#   * claim_search       — extractions table has 0 rows for negative_result /
#     quant_claim on prod (bead c996); unhide once an M3/M4 extraction run
#     populates them. Tool is registered + tested unconditionally; only the
#     tools/list visibility is gated.
# The default-configuration hidden set. Kept as a named constant (not inlined
# into the env lookup) so the import-time visible-surface guard near
# ``EXPECTED_TOOLS`` can compute the as-shipped visible count independently of
# any ``SCIX_HIDDEN_TOOLS`` override — operators who unhide tools for testing
# must not trip the cap assert.
_DEFAULT_HIDDEN_TOOLS_STR = (
    "chunk_search,section_retrieval,read_paper_claims,find_claims,claim_search"
)

_HIDDEN_TOOLS: frozenset[str] = frozenset(
    t.strip()
    for t in os.environ.get(
        "SCIX_HIDDEN_TOOLS",
        _DEFAULT_HIDDEN_TOOLS_STR,
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


def _vector_index_names(model_name: str) -> tuple[str, ...]:
    """Return the candidate ANN partial-index names for a given embedding model.

    The dense lane is served by whichever ANN index exists on the per-model
    partial: the legacy pgvector HNSW index (``idx_embed_hnsw_<model>``) or the
    pgvectorscale StreamingDiskANN index (``idx_embed_diskann_<model>``). Both
    are built over the same ``(embedding)::vector(768)`` expression, so the
    dense query in ``search.py`` is index-agnostic — only this existence gate
    needs to know about both.
    """
    return (f"idx_embed_hnsw_{model_name}", f"idx_embed_diskann_{model_name}")


def _hnsw_index_exists(conn: psycopg.Connection, model_name: str) -> bool:
    """Check whether a per-model ANN partial index (HNSW or DiskANN) exists.

    Name retained for caller compatibility; it now gates on either ANN index
    so the dense lane re-enables automatically once the DiskANN index is built.

    Qdrant short-circuit (bead jg4a): when the model has a Qdrant collection
    and QDRANT_URL is set, the dense lane is available regardless of whether
    the legacy paper_embeddings pg index exists (it was dropped in ADR-013).
    """
    if search._qdrant_dense_gated(model_name):
        return True

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
              AND indexname = ANY(%s)
            """,
            (list(_vector_index_names(model_name)),),
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


def _detect_unscoped_broad_block(result_json: str | None) -> bool:
    """Return True when ``result_json`` carries the unscoped-broad-block marker.

    The ``search`` tool's unscoped-broad-query guard emits a structured
    response with ``{"unscoped_broad_blocked": true, "error_code":
    "unscoped_broad_query", "error": "<human-readable message>", ...}``.
    ``_log_query`` lifts this marker into ``query_log.error_msg`` so
    operators can track block rate via a single SELECT — see bead
    ``scix_experiments-uerc`` (telemetry contract) and
    ``scix_experiments-x5jg`` (error_code envelope convention).

    Detection keys on the ``unscoped_broad_blocked`` flag, NOT on the
    ``error`` or ``error_code`` field, so the lift mechanism is stable
    even if the human/machine error fields change.
    """
    if not result_json:
        return False
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and data.get("unscoped_broad_blocked") is True


# Single source of truth for the query_log INSERT column order.
# Tests use this tuple to map captured params to named fields, so adding
# a column here automatically updates every downstream assertion that
# indexes by name (see _CaptureCursor in tests/test_mcp_search_unscoped_guard.py).
_LOG_QUERY_COLS: tuple[str, ...] = (
    "tool_name",
    "params_json",
    "latency_ms",
    "success",
    "error_msg",
    "tool",
    "query",
    "result_count",
    "session_id",
    "is_test",
)

# Per-column placeholder casts — only params_json needs the explicit jsonb cast.
_LOG_QUERY_PLACEHOLDERS: tuple[str, ...] = tuple(
    "%s::jsonb" if c == "params_json" else "%s" for c in _LOG_QUERY_COLS
)

_LOG_QUERY_INSERT_SQL: str = (
    f"INSERT INTO query_log ({', '.join(_LOG_QUERY_COLS)}) "
    f"VALUES ({', '.join(_LOG_QUERY_PLACEHOLDERS)})"
)


class _LogQueryCursor(Protocol):
    """Narrow cursor Protocol — the slice :func:`_log_query` uses.

    Concrete ``psycopg.Cursor[Row]`` satisfies this structurally; test
    fakes only need to expose ``execute`` plus the context-manager dunders.
    """

    def execute(self, sql: str, params: Any = ..., /) -> Any: ...

    def __enter__(self) -> "_LogQueryCursor": ...

    def __exit__(self, *args: Any) -> Any: ...


class _LogQueryConnection(Protocol):
    """Narrow connection Protocol — the slice :func:`_log_query` uses.

    Documents the actual contract :func:`_log_query` has on its connection
    arg so test fakes can declare structural compatibility instead of
    needing a ``# type: ignore[arg-type]`` at every call site. The
    ``info`` attribute is read defensively via ``getattr`` so it is not
    declared here.
    """

    def cursor(self) -> _LogQueryCursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _cap_params_lists(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``params`` with oversized list values truncated.

    Callers like ``temporal_evolution`` / ``find_gaps`` / ``synthesize_findings``
    may pass hundreds of bibcodes, but the tools themselves cap working sets at
    :data:`MAX_WORKING_SET_BIBCODES`, so storing the full list in
    ``query_log.params_json`` records inputs the tool never used and bloats
    telemetry over time (bead ``scix_experiments-pbh8``). Cap any list-typed
    value at the canonical bound. A new dict is returned so the caller's live
    arguments dict is never mutated.
    """
    if not any(isinstance(v, list) and len(v) > MAX_WORKING_SET_BIBCODES for v in params.values()):
        return params
    return {
        k: (v[:MAX_WORKING_SET_BIBCODES] if isinstance(v, list) else v) for k, v in params.items()
    }


def _log_query(
    conn: _LogQueryConnection,
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

    Lifts the ``unscoped_broad_blocked`` marker from ``result_json`` into
    ``error_msg`` (when no real error_msg is set) so operators can track
    the unscoped-broad-query block rate without a JSONB scan over result
    payloads — see bead ``scix_experiments-uerc``.
    """
    try:
        params_json = json.dumps(_cap_params_lists(params), default=str)
        query_text = _extract_query_text(params)
        result_count = _extract_result_count(result_json) if result_json else 0
        if error_msg is None and _detect_unscoped_broad_block(result_json):
            error_msg = _UNSCOPED_BROAD_TAG
        # If the tool dispatch left the connection in INERROR (a swallowed
        # statement_timeout, or a propagated QueryCanceled that exited the
        # try block before commit), the INSERT below would itself raise
        # InFailedSqlTransaction and the row would be lost. Roll back first
        # so this best-effort log write actually lands in the table. See
        # bead wzqz. Guarded with getattr because some test fixtures pass
        # a fake connection object without ``info``.
        info = getattr(conn, "info", None)
        if info is not None and info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
            conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                _LOG_QUERY_INSERT_SQL,
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

    Handles three common shapes:
      * ``{"papers": [{"bibcode": ...}, ...]}`` — multi-paper result.
      * ``{"bibcode": "..."}`` — single-paper result.
      * ``{"metadata": {"working_set_bibcodes": [...]}}`` — composite tools
        like ``lit_review`` whose seeds list (papers) is short but whose
        full working set lives under metadata. Working-set bibcodes are
        appended after seeds, dedup'd, so the trace reflects every paper
        the tool actually touched.

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
    seen: set[str] = set()

    def _push(bc: str) -> bool:
        """Append if new and under cap. Return False once cap is hit."""
        if bc in seen:
            return True
        bibcodes.append(bc)
        seen.add(bc)
        return len(bibcodes) < _MAX_TRACE_BIBCODES

    papers = data.get("papers")
    if isinstance(papers, list):
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            bc = paper.get("bibcode")
            if isinstance(bc, str):
                if not _push(bc):
                    break

    if len(bibcodes) < _MAX_TRACE_BIBCODES:
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            ws = metadata.get("working_set_bibcodes")
            if isinstance(ws, list):
                for bc in ws:
                    if isinstance(bc, str):
                        if not _push(bc):
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
# Unscoped-broad-query guard for the `search` tool (bead scix_experiments-uerc)
# ---------------------------------------------------------------------------
#
# The `search` tool's description warns that unscoped queries run a full-text
# scan over all 32M papers and may hit the statement timeout. Agents that don't
# read the description hit the timeout and get a generic DB error. This guard
# intercepts unscoped + broad queries before they reach Postgres and returns
# a structured error with actionable hints.
#
# Heuristic: a query is "broad" when it has >= 3 tokens OR length >= 30 chars.
# A request is "scoped" when filters has at least one non-None, non-empty value
# in {year_min, year_max, arxiv_class, doctype, first_author, entity_types,
# entity_ids}. Empty lists count as no filter (matches SearchFilters
# normalization in scix.search).

_UNSCOPED_BROAD_MIN_TOKENS: int = 3
_UNSCOPED_BROAD_MIN_CHARS: int = 30

# Stable telemetry tag — surfaced in result_json AND lifted into query_log.error_msg
# by _log_query so operators can track unscoped-broad block rate with a single
# SELECT count(*) FROM query_log WHERE error_msg = 'unscoped_broad_query'.
# Sourced from the closed error-code catalog so the telemetry tag and the
# response ``error_code`` cannot drift (bead scix_experiments-ir2h).
_UNSCOPED_BROAD_TAG: str = ErrorCode.UNSCOPED_BROAD_QUERY


def _filters_are_scoped(filters: dict[str, Any] | None) -> bool:
    """Return True when ``filters`` constrains the candidate set.

    A filter is "scoping" when at least one of these fields has a non-None,
    non-empty value: year_min, year_max, arxiv_class, doctype, first_author,
    entity_types, entity_ids. ``filters=None``, ``filters={}``, and
    ``filters={'year_min': None, 'entity_ids': []}`` all count as unscoped.
    """
    if not filters:
        return False
    for field in (
        "year_min",
        "year_max",
        "arxiv_class",
        "doctype",
        "first_author",
        "entity_types",
        "entity_ids",
    ):
        value = filters.get(field)
        if value is None:
            continue
        # Empty list/string normalizes to "no filter".
        if isinstance(value, (list, str)) and len(value) == 0:
            continue
        # Non-positive year bounds don't constrain anything in practice.
        if field in ("year_min", "year_max") and isinstance(value, int) and value <= 0:
            continue
        return True
    return False


def _is_unscoped_broad_query(
    query: str,
    filters: dict[str, Any] | None,
    *,
    bypass: bool = False,
) -> bool:
    """Return True when the query should be blocked by the unscoped-broad guard.

    The guard only fires when ALL of the following hold:
      * ``bypass`` is False (escape hatch for tests / power users).
      * The request has no scoping filters (see ``_filters_are_scoped``).
      * The query is broad: >= 3 tokens OR length >= 30 chars (after strip).

    Empty / whitespace-only queries are not blocked here — the schema
    enforces presence of ``query`` and the existing search code handles
    blank input.
    """
    if bypass:
        return False
    if _filters_are_scoped(filters):
        return False
    stripped = (query or "").strip()
    if not stripped:
        return False
    token_count = len(stripped.split())
    if token_count >= _UNSCOPED_BROAD_MIN_TOKENS:
        return True
    if len(stripped) >= _UNSCOPED_BROAD_MIN_CHARS:
        return True
    return False


def _unscoped_broad_response(query: str) -> str:
    """Build the structured unscoped-broad-query error payload.

    The response carries the stable ``unscoped_broad_blocked: true`` flag so
    ``_log_query`` can lift it into ``query_log.error_msg`` for telemetry.

    Per bead ``scix_experiments-x5jg`` the stable machine identifier lives
    in ``error_code``; ``error`` is a human-readable message. Telemetry
    detection (``_detect_unscoped_broad_block``) keys on the
    ``unscoped_broad_blocked`` flag, not on either ``error`` field.
    """
    payload = {
        "error": (
            "Unscoped broad query rejected — "
            "would scan all 32M papers and likely hit statement timeout."
        ),
        "error_code": _UNSCOPED_BROAD_TAG,
        "hint": (
            "Unscoped broad queries scan all 32M papers and frequently hit the "
            "statement timeout. Add filters.arxiv_class (e.g. 'astro-ph') or "
            "filters.year_min to scope the search."
        ),
        "query": query,
        "suggestions": {
            "filters.year_min": 2020,
        },
        "bypass": (
            "Pass bypass_unscoped_guard=true to run the unscoped query anyway "
            "(may hit statement timeout)."
        ),
        "unscoped_broad_blocked": True,
    }
    return json.dumps(payload, indent=2, default=str)


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
    # entity_context folded into entity(action='profile') (bead 9afa, 2026-06).
    "entity_context": "entity",
    # cited_by_intent + find_replications merged into forward_citations
    # (bead 9afa, 2026-06) behind annotate='intent' | 'relation'.
    "cited_by_intent": "forward_citations",
    "find_replications": "forward_citations",
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
# The domain-tuned carve-out has since been tested too (bead 4skc; see
# results/retrieval_eval_50q_rerank_indus.md): nasa-impact/nasa-smd-ibm-ranker
# also REGRESSES on the re-baselined Qdrant pool (nDCG@10 0.2242 -> 0.1843,
# Δ=-0.0400, p=0.074) despite passing its own home benchmark — NO-GO. Operators
# can still flip a non-'off' model on for experimentation, but the production
# default stays off; no evaluated reranker has cleared the rollout gate.

# Map env-var values to model_name strings consumed by CrossEncoderReranker.
_RERANK_MODEL_ALIASES: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-large": "BAAI/bge-reranker-large",
    "indus-ranker": "nasa-impact/nasa-smd-ibm-ranker",
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
        "Allowed values: 'off', 'minilm', 'bge-large', 'indus-ranker'.",
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
# ---------------------------------------------------------------------------
# section_retrieval tool — filters schema + RRF + snippet helpers
# ---------------------------------------------------------------------------
#
# The section_retrieval tool fuses dense HNSW search over section_embeddings
# with BM25 over papers_fulltext.sections_tsv via Reciprocal Rank Fusion.
# It uses a slimmer filter object than the search-tool _FILTERS_SCHEMA: only
# year_min, year_max, bibcode_prefix.
#
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
    # entity_context folded into entity(action='profile') (bead 9afa, 2026-06);
    # the old name stays callable as a deprecated alias.
    "graph_context",
    "find_gaps",
    "temporal_evolution",
    "facet_counts",
    # PRD MH-4 — Deep Search v1: provenance walk back to a claim's origin.
    "claim_blame",
    # forward_citations (bead 9afa, 2026-06): merges cited_by_intent +
    # find_replications behind an ``annotate`` enum (intent | relation). Both
    # legacy names stay callable as deprecated aliases.
    "forward_citations",
    # PRD section-embeddings-mcp-consolidation — section-grain hybrid retrieval
    "section_retrieval",
    # PRD nanopub-claim-extraction — paper_claims retrieval (migration 062)
    "read_paper_claims",
    "find_claims",
    # Claim/finding extraction surface (bead c996) — split out from the
    # entity tool's entity_type enum under bead mh14. Default-hidden
    # because the extractions table has 0 rows for negative_result /
    # quant_claim on prod.
    "claim_search",
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


# ---------------------------------------------------------------------------
# Visible-surface cap guard (ADR-pinned; bead 9afa / xjqi)
# ---------------------------------------------------------------------------
#
# The premortem tool-count concern caps the *agent-visible* surface at 15 to
# protect tool-selection accuracy (see docs/mcp_tool_audit_2026-06.md). This
# import-time assert pins the as-shipped default surface so a future addition
# that pushes it past 15 fails loudly at import rather than silently drifting
# (the 15→17 drift that motivated bead xjqi went unnoticed for two months).
#
# It evaluates the DEFAULT hidden set, not the live ``_HIDDEN_TOOLS`` — an
# operator who sets ``SCIX_HIDDEN_TOOLS=`` to unhide tools for testing must not
# trip the guard. ``chunk_search`` is optional (Qdrant-only) and also hidden by
# default, so it never counts toward the default surface.
VISIBLE_TOOL_CAP = 15

_DEFAULT_HIDDEN_TOOLS = frozenset(
    t.strip() for t in _DEFAULT_HIDDEN_TOOLS_STR.split(",") if t.strip()
)
_DEFAULT_VISIBLE_TOOLS = set(EXPECTED_TOOLS) - _DEFAULT_HIDDEN_TOOLS
if len(_DEFAULT_VISIBLE_TOOLS) > VISIBLE_TOOL_CAP:
    raise RuntimeError(
        f"MCP visible tool surface ({len(_DEFAULT_VISIBLE_TOOLS)}) exceeds the "
        f"ADR-pinned cap of {VISIBLE_TOOL_CAP}: {sorted(_DEFAULT_VISIBLE_TOOLS)}. "
        f"Consolidate (merge real overlaps) or raise the cap via ADR — see "
        f"docs/mcp_tool_audit_2026-06.md §6."
    )


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
                f"tool {name}: inputSchema.type must be 'object', got {schema.get('type')!r}"
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


# Static MCP tool surface (bead oahz): the agent-visible tool schemas, hoisted
# out of list_tools() so the function is a thin assembler and the schemas are
# inspectable module-level data. Stored as plain dicts (not Tool objects) so
# importing this module never requires the optional `mcp` package; list_tools
# builds Tool(**spec) at call time. chunk_search is Qdrant-gated and kept
# separate so it is only advertised when _qdrant_enabled().

def create_server(_run_self_test: bool = True, _preload_model: bool = True):
    """Create and configure the MCP server with the consolidated tool surface.

    Eagerly pre-loads the INDUS model so semantic_search is fast from
    the first call.

    Args:
        _run_self_test: If True (default), run ``startup_self_test`` after
            the server is built to fail fast on broken tool schemas. Set
            to False internally by ``startup_self_test`` itself to avoid
            infinite recursion.
        _preload_model: If True (default), eagerly load the INDUS model. Set
            to False when the server is built only to read the static tool
            surface (e.g. the contract generator / conformance suite in
            :mod:`scix.mcp_contract`), which never touches the model.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError("mcp SDK is required for the MCP server. Install with: pip install mcp")

    if _preload_model:
        _init_model_impl()

    server = Server("scix")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tool_list: list[Tool] = [Tool(**spec) for spec in _TOOL_SPECS]
        if _qdrant_enabled():
            tool_list.append(Tool(**_CHUNK_SEARCH_SPEC))
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
            result_json = json.dumps({"error": error_msg, "error_code": ErrorCode.INTERNAL_ERROR})
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


def _kw_terms_to_query(a: dict[str, Any]) -> None:
    """keyword_search(terms) -> search(query): move ``terms`` into ``query``,
    falling back to any existing ``query`` (then ``""``) when ``terms`` is absent."""
    a["query"] = a.pop("terms", a.get("query", ""))


def _entity_name_to_query(a: dict[str, Any]) -> None:
    """entity_search(entity_name) -> entity(query): rename only when ``query`` is
    not already set, so an explicit ``query`` wins."""
    if "entity_name" in a and "query" not in a:
        a["query"] = a.pop("entity_name")


def _query_to_search_query(a: dict[str, Any]) -> None:
    """search_within_paper(query) -> read_paper(search_query)."""
    if "query" in a:
        a["search_query"] = a.pop("query")


@dataclasses.dataclass(frozen=True)
class _AliasTransform:
    """How to rewrite one deprecated tool call to its consolidated form.

    ``target`` is the consolidated tool name to dispatch. ``set_args`` are keys
    force-set on the args (mode/action/method/direction/annotate flags).
    ``arg_fn`` is an in-place massage for the few aliases that rename keys
    (only three need it); ``None`` for the purely declarative majority.
    """

    target: str
    set_args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    arg_fn: Callable[[dict[str, Any]], None] | None = None


# Single source of truth for deprecated-alias rewriting, replacing the former
# 24-branch if-chain that had to be kept in sync with ``_DEPRECATED_ALIASES`` by
# hand. An entry whose ``target`` equals its key is a pure legacy passthrough
# (the consolidated dispatcher routes it to a dedicated handler).
_ALIAS_TRANSFORMS: dict[str, _AliasTransform] = {
    "semantic_search": _AliasTransform("search", {"mode": "semantic"}),
    "keyword_search": _AliasTransform("search", {"mode": "keyword"}, _kw_terms_to_query),
    "citation_graph": _AliasTransform("citation_traverse", {"mode": "graph"}),
    "citation_chain": _AliasTransform("citation_traverse", {"mode": "chain"}),
    "get_citations": _AliasTransform(
        "citation_traverse", {"mode": "graph", "direction": "forward"}
    ),
    "get_references": _AliasTransform(
        "citation_traverse", {"mode": "graph", "direction": "backward"}
    ),
    "co_citation_analysis": _AliasTransform("citation_similarity", {"method": "co_citation"}),
    "bibliographic_coupling": _AliasTransform("citation_similarity", {"method": "coupling"}),
    "entity_search": _AliasTransform("entity", {"action": "search"}, _entity_name_to_query),
    "resolve_entity": _AliasTransform("entity", {"action": "resolve"}),
    "entity_context": _AliasTransform("entity", {"action": "profile"}),
    "cited_by_intent": _AliasTransform("forward_citations", {"annotate": "intent"}),
    "find_replications": _AliasTransform("forward_citations", {"annotate": "relation"}),
    "document_context": _AliasTransform("get_paper", {"include_entities": True}),
    "get_openalex_topics": _AliasTransform("get_paper", {"include_entities": True}),
    "get_paper_metrics": _AliasTransform("graph_context", {"include_community": False}),
    "explore_community": _AliasTransform("graph_context", {"include_community": True}),
    "get_author_papers": _AliasTransform("get_author_papers"),
    "read_paper_section": _AliasTransform("read_paper"),
    "search_within_paper": _AliasTransform("read_paper", arg_fn=_query_to_search_query),
    "get_citation_context": _AliasTransform("get_citation_context"),
    "add_to_working_set": _AliasTransform("add_to_working_set"),
    "get_working_set": _AliasTransform("get_working_set"),
    "get_session_summary": _AliasTransform("get_session_summary"),
    "clear_working_set": _AliasTransform("clear_working_set"),
    "entity_profile": _AliasTransform("entity_profile"),
}


def _transform_deprecated_args(
    old_name: str, new_name: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Transform arguments from old tool format to new consolidated format."""
    new_args = dict(args)
    spec = _ALIAS_TRANSFORMS.get(old_name)
    if spec is None:
        return new_name, new_args
    if spec.arg_fn is not None:
        spec.arg_fn(new_args)
    new_args.update(spec.set_args)
    return spec.target, new_args


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
    """Dispatch to the consolidated tool handlers plus legacy/health handlers.

    Looks ``name`` up in :data:`_HANDLER_REGISTRY` (the single source of truth
    for tool -> handler routing) and invokes it; an unknown name returns the
    structured ``unknown_tool`` error. Legacy/deprecated names reach here only
    after :func:`_transform_deprecated_args` has rewritten the agent-visible
    aliases, but several legacy names (citation_graph/_chain, the session tools,
    entity_profile, ...) are also routed directly for in-process callers that
    bypass the alias layer — so the registry carries them too.
    """
    handler = _handler_registry().get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}", "error_code": ErrorCode.UNKNOWN_TOOL})
    return handler(conn, args)


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
























#: Entity types that used to be smuggled into the entity tool under
#: ``entity_type`` for action='search'. They are claim/finding extraction
#: payload kinds (M3 negative_result, M4 quant_claim), not entities, and
#: were removed from the entity tool's schema in 2026-04 under bead
#: ``scix_experiments-mh14``. The handler rejects them with a structured
#: error so a client that rediscovers the old contract gets a clear,
#: actionable response instead of falling through to a different code path.
#:
#: The dedicated home for these extractions is the ``claim_search`` MCP
#: tool added under bead ``scix_experiments-c996``; the rejection error
#: points callers there. ``_handle_entity_extraction_search`` below is
#: the shared implementation called by ``_handle_claim_search``.
#:
#: Canonical list lives at ``_CLAIM_SEARCH_ACTIONS`` below — the
#: ``claim_search`` tool accepts exactly these as ``action`` values. The
#: entity tool reuses that frozenset for its rejection guard so the two
#: surfaces stay in sync if a third extraction kind is added.








# ---------------------------------------------------------------------------
# chunk_search handler (PRD chunk-embeddings-build)
# ---------------------------------------------------------------------------

# Lazy module-level cache for the INDUS embedder used by chunk_search. Loaded
# on first dispatch so server startup stays fast and Qdrant-disabled
# deployments never pay the model load cost.
_indus_embedder: tuple[Any, Any] | None = None






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






# ---------------------------------------------------------------------------
# cited_by_intent handler — exploits citation_contexts.intent classification
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# synthesize_findings handler (bead cfh9)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# section_retrieval handler
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# Tool -> handler routing. Handlers live in the scix.mcp_handlers subpackage
# (bead scix_experiments-pebe). The registry and the handful of handler names a
# few callers import from this module are resolved LAZILY: those handler modules
# import back from here, so importing them eagerly at module load would create a
# load-order cycle (importing a handler module first would deadlock). Building on
# first use keeps ``import scix.mcp_server`` and ``import scix.mcp_handlers.*``
# order-independent. ``_handle_health_check`` stays local (it reads the live
# ``_pool``).
# ---------------------------------------------------------------------------
_HANDLER_REGISTRY_CACHE: dict[str, Callable[[psycopg.Connection, dict[str, Any]], str]] | None = None

#: Handler-region names a few callers import via ``from scix.mcp_server import``.
#: Served through PEP 562 ``__getattr__`` for the same anti-cycle reason.
_LAZY_REEXPORTS: dict[str, tuple[str, str]] = {
    "_handle_cited_by_intent": ("citation", "_handle_cited_by_intent"),
    "_handle_entity": ("entity", "_handle_entity"),
    "_handle_find_claims": ("synthesis", "_handle_find_claims"),
    "_handle_graph_context": ("entity", "_handle_graph_context"),
    "_handle_read_paper_claims": ("synthesis", "_handle_read_paper_claims"),
    "_section_filter_clauses": ("sections", "_section_filter_clauses"),
    "_encode_section_query": ("sections", "_encode_section_query"),
}


def _build_handler_registry() -> dict[str, Callable[[psycopg.Connection, dict[str, Any]], str]]:
    """Import the handler subpackage and assemble the tool -> handler map."""
    from scix.mcp_handlers import citation, entity, paper, search, sections, synthesis

    return {
        "find_similar_by_examples": search._handle_removed_find_similar,
        "search": search._handle_search,
        "lit_review": search._handle_lit_review,
        "concept_search": search._handle_concept_search,
        "get_paper": paper._handle_get_paper,
        "read_paper": paper._handle_read_paper,
        "citation_traverse": citation._handle_citation_traverse,
        "citation_graph": citation._handle_citation_graph,
        "citation_chain": citation._handle_citation_chain,
        "citation_similarity": citation._handle_citation_similarity,
        "entity": entity._handle_entity,
        "entity_context": entity._handle_entity_context,
        "graph_context": entity._handle_graph_context,
        "find_gaps": entity._handle_find_gaps,
        "temporal_evolution": search._handle_temporal_evolution,
        "facet_counts": search._handle_facet_counts,
        "claim_blame": citation._handle_claim_blame,
        "forward_citations": citation._handle_forward_citations,
        "find_replications": citation._handle_find_replications,
        "cited_by_intent": citation._handle_cited_by_intent,
        "claim_search": entity._handle_claim_search,
        "synthesize_findings": synthesis._handle_synthesize_findings,
        "section_retrieval": sections._handle_section_retrieval,
        "read_paper_claims": synthesis._handle_read_paper_claims,
        "find_claims": synthesis._handle_find_claims,
        "chunk_search": search._handle_chunk_search,
        "add_to_working_set": paper._handle_add_to_working_set,
        "get_working_set": paper._handle_get_working_set,
        "get_session_summary": paper._handle_get_session_summary,
        "clear_working_set": paper._handle_clear_working_set,
        "get_citation_context": paper._handle_get_citation_context,
        "get_author_papers": paper._handle_get_author_papers,
        "health_check": lambda conn, args: _handle_health_check(conn),
        "entity_profile": entity._handle_entity_profile,
    }


def _handler_registry() -> dict[str, Callable[[psycopg.Connection, dict[str, Any]], str]]:
    """Return the cached tool -> handler registry, building it on first call."""
    global _HANDLER_REGISTRY_CACHE
    if _HANDLER_REGISTRY_CACHE is None:
        _HANDLER_REGISTRY_CACHE = _build_handler_registry()
    return _HANDLER_REGISTRY_CACHE


def __getattr__(name: str) -> Any:  # PEP 562 — lazy module attributes
    if name == "_HANDLER_REGISTRY":
        return _handler_registry()
    target = _LAZY_REEXPORTS.get(name)
    if target is not None:
        import importlib

        module = importlib.import_module(f"scix.mcp_handlers.{target[0]}")
        return getattr(module, target[1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
