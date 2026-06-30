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
    Old tool names still work via ``_ALIAS_TRANSFORMS`` but return
    ``deprecated: true`` and ``use_instead`` metadata.

File size (bead scix_experiments-2qx3): the pure, stateless helper layer was
extracted to :mod:`scix.mcp_runtime`. What remains here is server wiring +
test-patch surface — both of which the PL ruling pinned to this module: the
dispatch table (plus alias transforms and ``_maybe_disambiguate``),
``startup_self_test``, ``create_server``/``call_tool``, the ``mcp_runtime``
re-export block, ``_handle_health_check``, and the stateful singletons tests
patch on the ``mcp_server`` namespace (trace publisher, reranker, coverage
note). That floor exceeds the 800 coding-style default; it also exceeds the
earlier PL/2qx3 relaxed-cap estimate of ≤1100, because that estimate was
measured against a work-in-progress intermediate that had (incorrectly) moved
the patched singletons out, breaking the test-patch surface. Restoring that
surface — the bead's hard invariant — raises the irreducible floor.
800-cap relaxed to 1327 here per PL/2qx3 ratification 2026-06-19; reducing
further requires breaking the test-patch invariant, which is rejected.

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
from typing import Any, Callable, Generator, Mapping

import psycopg

from scix import search  # noqa: F401  patch surface: tests patch scix.mcp_server.search.*
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

# Re-export the runtime-helper layer (bead scix_experiments-2qx3). These helpers
# live in :mod:`scix.mcp_runtime` (which depends on neither this module nor the
# handler subpackage); they are re-bound here so the historical
# ``scix.mcp_server.<helper>`` import/patch surface keeps working and so the
# wiring functions below (``call_tool``/``_dispatch_tool``) can call them by
# bare name. Only PURE, stateless helpers live in mcp_runtime; the trace
# publisher (``_trace_stream``/``_emit_trace_event``) and the reranker singleton
# (``CrossEncoderReranker``/``_get_default_reranker``) stay in THIS module —
# they are stateful server-process state (like ``_pool``/``_model_cache``) and
# tests patch ``_trace_stream`` / ``CrossEncoderReranker`` on the ``mcp_server``
# namespace, so their consumers must read those names from here at call time.
from scix.mcp_runtime import (  # noqa: F401  re-export: historical patch/import surface
    _LOG_QUERY_COLS,
    _MAX_TRACE_BIBCODES,
    _NOMIC_QUERY_PREFIX,
    _RERANK_MODEL_ALIASES,
    _RERANK_TOP_K_CAP,
    _RRF_K_DEFAULT,
    _SNIPPET_MAX_CHARS,
    DEFAULT_RESULT_LIMIT,
    _annotate_working_set,
    _auto_track_bibcodes,
    _cap_params_lists,
    _coerce_year,
    _detect_unscoped_broad_block,
    _extract_bibcodes_from_result,
    _extract_query_text,
    _extract_result_count,
    _filters_are_scoped,
    _hnsw_index_cache,
    _hnsw_index_exists,
    _is_unscoped_broad_query,
    _log_query,
    _LogQueryConnection,
    _LogQueryCursor,
    _parse_filters,
    _resolve_default_reranker_model,
    _result_to_json,
    _rrf_fuse,
    _session_state,
    _truncate_snippet,
    _unscoped_broad_response,
    _validate_entity_list,
    _vector_index_names,
)
from scix.mcp_tool_specs import (
    _CHUNK_SEARCH_SPEC,
    _FILTERS_SCHEMA,  # noqa: F401  re-exported for callers/tests
    _SECTION_FILTERS_SCHEMA,  # noqa: F401  re-exported for callers/tests
    _SIGNAL_USED_DESCRIPTION,  # noqa: F401  re-exported for callers/tests
    _TOOL_SPECS,
    MAX_ENTITY_FILTER_ITEMS,  # noqa: F401  re-exported for callers/tests
)
from scix.search import CrossEncoderReranker
from scix.synthesize import (  # noqa: F401  re-export: historical patch/import surface
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


logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Trace-event publisher (optional, fire-and-forget)
# ---------------------------------------------------------------------------
#
# Lives here rather than in mcp_runtime because ``_trace_stream`` is patched by
# tests on the ``mcp_server`` namespace; ``_emit_trace_event`` must read it from
# this module at call time for that patch to take effect. The bibcode-extraction
# helper it calls is a pure function and stays in mcp_runtime (re-exported above).
#
# Optional import — viz/trace_stream is only needed when the viz extras are
# installed. When absent we fall back to a no-op, and the emission hook silently
# skips publishing.
try:
    from scix.viz import trace_stream as _trace_stream
except ImportError:  # pragma: no cover — viz extras not installed
    _trace_stream = None  # type: ignore[assignment]


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
# Cross-encoder reranker singleton (lazy, process-lifetime)
# ---------------------------------------------------------------------------
#
# Lives here rather than in mcp_runtime because the construction reads the
# ``CrossEncoderReranker`` class from this module's namespace, which tests patch
# to inject a stub (no model weights downloaded). The env→model-name resolution
# is pure config and stays in mcp_runtime (``_resolve_default_reranker_model``,
# re-exported above). Default is OFF: see that helper for the ablation evidence.
#
# Cache so repeated tool calls reuse the same instance. Construction is cheap
# (no weights loaded until first __call__); caching amortises the lazy
# weight-load across the process lifetime.
_default_reranker_cache: dict[str, CrossEncoderReranker | None] = {}


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
# Lives here rather than in mcp_runtime because the note is lru_cached
# process state and tests patch ``_coverage_note_path`` on the ``mcp_server``
# namespace; ``_coverage_note`` must read that name from this module at call
# time for the patch to take effect (bead 2qx3). Default policy is always-on
# per the PRD's Open Question default decision.

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


# Deprecated-alias routing (old tool name -> consolidated target + modern-tool
# guidance) lives in the single ``_ALIAS_TRANSFORMS`` table further down, next
# to the rewriter that consumes it.


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
        spec = _ALIAS_TRANSFORMS.get(name)
        resolved_name = spec.guidance if spec is not None else name
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

    spec = _ALIAS_TRANSFORMS.get(name)
    if spec is not None:
        use_instead = spec.guidance
        deprecated = True
        logger.info("deprecated_alias: %s -> %s", name, use_instead)

        # Transform args from old format to new format
        name, args = _transform_deprecated_args(original_name, args)

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

    ``use_instead`` is the modern tool name advertised to agents in the
    deprecation envelope. It defaults to ``target`` and only diverges for the
    seven self-targeting passthroughs that still dispatch to their own
    dedicated handler but point migrating agents at a different modern tool
    (e.g. ``get_citation_context`` dispatches to itself but advertises
    ``citation_traverse``). Folding it in here retires the parallel
    ``_DEPRECATED_ALIASES`` map that had to be hand-synced with this table.
    """

    target: str
    set_args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    arg_fn: Callable[[dict[str, Any]], None] | None = None
    use_instead: str | None = None

    @property
    def guidance(self) -> str:
        """Modern tool name to advertise; the dispatch target unless overridden."""
        return self.use_instead or self.target


# Single source of truth for deprecated-alias routing: target, forced args, any
# key rename, AND the modern-tool guidance advertised to agents (``use_instead``,
# defaulting to ``target``). Replaces both the former 24-branch if-chain and the
# parallel ``_DEPRECATED_ALIASES`` map, which had to be hand-synced with this
# table. An entry whose ``target`` equals its key is a pure legacy passthrough
# (the consolidated dispatcher routes it to a dedicated handler); those carry an
# explicit ``use_instead`` pointing at the modern equivalent.
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
    "get_author_papers": _AliasTransform("get_author_papers", use_instead="search"),
    "read_paper_section": _AliasTransform("read_paper"),
    "search_within_paper": _AliasTransform("read_paper", arg_fn=_query_to_search_query),
    "get_citation_context": _AliasTransform("get_citation_context", use_instead="citation_traverse"),
    "add_to_working_set": _AliasTransform("add_to_working_set", use_instead="get_paper"),
    "get_working_set": _AliasTransform("get_working_set", use_instead="find_gaps"),
    "get_session_summary": _AliasTransform("get_session_summary", use_instead="find_gaps"),
    "clear_working_set": _AliasTransform("clear_working_set", use_instead="find_gaps"),
    "entity_profile": _AliasTransform("entity_profile", use_instead="get_paper"),
}


def _transform_deprecated_args(
    old_name: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Rewrite a deprecated alias call to ``(target, new_args)``.

    Copies ``args`` (callers reuse the original for logging), applies the
    alias's optional key-rename ``arg_fn`` and forced ``set_args``, and returns
    the consolidated target name. An unknown ``old_name`` (no transform entry)
    is the identity transform — ``(old_name, copy-of-args)`` — though in
    practice :func:`_dispatch_tool` only calls this for known aliases.
    """
    new_args = dict(args)
    spec = _ALIAS_TRANSFORMS.get(old_name)
    if spec is None:
        return old_name, new_args
    if spec.arg_fn is not None:
        spec.arg_fn(new_args)
    new_args.update(spec.set_args)
    return spec.target, new_args


def _wrap_deprecated(result_json: str, original_name: str, use_instead: str) -> str:
    """Add deprecation metadata to a result."""
    try:
        data = json.loads(result_json)
        if isinstance(data, dict):
            wrapped = {
                **data,
                "deprecated": True,
                "use_instead": use_instead,
                "original_tool": original_name,
            }
            return json.dumps(wrapped, indent=2, default=str)
    except (json.JSONDecodeError, TypeError):
        pass
    return result_json


def _dispatch_consolidated(conn: psycopg.Connection, name: str, args: dict[str, Any]) -> str:
    """Dispatch to the consolidated tool handlers plus legacy/health handlers.

    Looks ``name`` up in :data:`_HANDLER_REGISTRY` (the single source of truth
    for tool -> handler routing) and invokes it; an unknown name returns the
    structured ``unknown_tool`` error. Deprecated aliases reach here only after
    :func:`_transform_deprecated_args` has rewritten them to their consolidated
    target, so the registry holds only reachable names: the consolidated tools,
    the self-targeting legacy passthroughs (the session tools, get_author_papers,
    get_citation_context, entity_profile — whose ``_AliasTransform.target`` is
    the name itself), and the hard-removed ``find_similar_by_examples`` stub.
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


# Lazy module-level cache for the INDUS embedder used by the chunk_search
# handler (scix.mcp_handlers.search, reached via _srv._indus_embedder). Loaded
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
    "_handle_search": ("search", "_handle_search"),
    "_section_filter_clauses": ("sections", "_section_filter_clauses"),
    "_encode_section_query": ("sections", "_encode_section_query"),
}


def _build_handler_registry() -> dict[str, Callable[[psycopg.Connection, dict[str, Any]], str]]:
    """Import the handler subpackage and assemble the tool -> handler map."""
    # Aliased: module-level ``search`` is ``scix.search`` (the patch surface
    # tests reach via ``scix.mcp_server.search``); this is the handler module.
    from scix.mcp_handlers import (
        citation,
        claim,
        entity,
        paper,
        sections,
        synthesis,
    )
    from scix.mcp_handlers import search as search_handlers

    return {
        "find_similar_by_examples": search_handlers._handle_removed_find_similar,
        "search": search_handlers._handle_search,
        "lit_review": search_handlers._handle_lit_review,
        "concept_search": search_handlers._handle_concept_search,
        "get_paper": paper._handle_get_paper,
        "read_paper": paper._handle_read_paper,
        "citation_traverse": citation._handle_citation_traverse,
        "citation_similarity": citation._handle_citation_similarity,
        "entity": entity._handle_entity,
        "graph_context": entity._handle_graph_context,
        "find_gaps": entity._handle_find_gaps,
        "temporal_evolution": search_handlers._handle_temporal_evolution,
        "facet_counts": search_handlers._handle_facet_counts,
        "claim_blame": citation._handle_claim_blame,
        "forward_citations": citation._handle_forward_citations,
        "claim_search": claim._handle_claim_search,
        "synthesize_findings": synthesis._handle_synthesize_findings,
        "section_retrieval": sections._handle_section_retrieval,
        "read_paper_claims": synthesis._handle_read_paper_claims,
        "find_claims": synthesis._handle_find_claims,
        "chunk_search": search_handlers._handle_chunk_search,
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
