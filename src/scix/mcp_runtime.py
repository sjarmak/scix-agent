"""Neutral runtime-helper layer for the SciX MCP server (bead scix_experiments-2qx3).

Extracted from :mod:`scix.mcp_server` so that module stays thin wiring (pool +
dispatch + create_server/call_tool/startup_self_test/main + health check) while
the shared, non-wiring helpers live here: logging/trace, result serialization,
filter parsing, coverage notes, RRF/snippet, the HNSW availability guard, the
cross-encoder reranker, year/entity validation, the unscoped-broad guard, and
session-state helpers.

This module depends on neither :mod:`scix.mcp_server` nor
:mod:`scix.mcp_handlers`: both import FROM it. ``scix.mcp_server`` re-exports
the names below so the historical ``scix.mcp_server.<helper>`` import/patch
surface keeps working; the handler subpackage imports the pure helpers from
here directly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Protocol, Sequence

import psycopg

from scix import search
from scix.mcp_errors import ErrorCode
from scix.mcp_tool_specs import MAX_ENTITY_FILTER_ITEMS
from scix.session import SessionState
from scix.synthesize import MAX_WORKING_SET_BIBCODES

# Handlers and this module log under the historical "scix.mcp_server" channel
# so operator log filters keep working after the extraction.
logger = logging.getLogger("scix.mcp_server")

# Result-limit convention (bead 9afa / docs/mcp_tool_audit_2026-06.md §5):
# tools default to DEFAULT_RESULT_LIMIT results and cap at MAX_WORKING_SET_BIBCODES
# (=200) unless a tool documents a justified exception. Documented exceptions:
# ``search`` (default 10 — established page size, eval-baseline-pinned),
# ``facet_counts`` (default 50 — distribution buckets), and ``lit_review``'s
# composite sub-counts (top_seeds / expansion_seeds — domain-specific knobs).
DEFAULT_RESULT_LIMIT = 20


# Cap the number of bibcodes emitted per TraceEvent to keep event payloads
# small. SSE consumers typically only need a handful of bibcodes for linkage.
_MAX_TRACE_BIBCODES: int = 20



# ---------------------------------------------------------------------------
# HNSW index availability guard
# ---------------------------------------------------------------------------

_hnsw_index_cache: dict[str, tuple[bool, float]] = {}
_HNSW_CACHE_TTL_MISS_SEC = 30.0


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
# Cross-encoder reranker — model-name resolution only
# ---------------------------------------------------------------------------
#
# The lazy reranker *singleton* (cache + ``_get_default_reranker`` +
# ``_reset_default_reranker_cache``) lives in :mod:`scix.mcp_server`, not here:
# it is stateful server-process state, and tests patch ``CrossEncoderReranker``
# on the ``mcp_server`` namespace, so the construction site must read the class
# from there at call time. This module keeps only the pure, stateless config
# resolution below.
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
