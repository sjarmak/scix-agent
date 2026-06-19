"""Search & discovery MCP handlers (search, lit_review, facet/temporal, concept, chunk, community)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg

from scix import mcp_server as _srv
from scix import search
from scix.mcp_errors import ErrorCode
from scix.mcp_handlers._common import (
    _resolve_working_set_bibcodes,
)
from scix.mcp_server import (
    _RERANK_TOP_K_CAP,
    _coerce_year,
    _get_default_reranker,
    _parse_filters,
    _qdrant_tools,
    _result_to_json,
    _session_state,
    _unscoped_broad_response,
    _vector_index_names,
)
from scix.synthesize import MAX_WORKING_SET_BIBCODES

logger = logging.getLogger("scix.mcp_server")


def _community_expand_no_seed_response(
    *,
    query: str,
    reason: str,
    candidates: list[dict[str, Any]] | None = None,
) -> str:
    """Build the structured ``community_expand_no_seed`` envelope.

    Mirrors the error-envelope conventions from beads ``uerc`` and ``x5jg``:
    machine-readable ``error_code`` plus a human-readable ``error`` and a
    ``hint`` pointing at the recovery path. When ``candidates`` is provided,
    the agent can pick one and re-issue with ``filters.entity_ids=[<id>]``.
    """
    payload: dict[str, Any] = {
        "error": (
            "community_expand requires a single seed entity. "
            f"Could not derive one from the request: {reason}."
        ),
        "error_code": ErrorCode.COMMUNITY_EXPAND_NO_SEED,
        "hint": (
            "Either pass exactly one filters.entity_ids=[<id>] or use the "
            "entity tool with action='resolve' to disambiguate the query "
            "into a single entity_id, then retry community_expand."
        ),
        "query": query,
    }
    if candidates is not None:
        payload["candidates"] = candidates
    return json.dumps(payload, indent=2, default=str)

def _resolve_community_expand_seed(
    conn: psycopg.Connection,
    query: str,
    filters: search.SearchFilters,
) -> tuple[int | None, str | None]:
    """Resolve a seed entity_id for the community-expand lane.

    Decision tree (per PRD §4):
        1. Explicit ``filters.entity_ids`` with exactly one id → use it.
        2. Free-text resolution to a single unambiguous entity → use it.
        3. Anything else (zero matches, multiple matches, multi-id explicit
           filter) → return ``(None, json_envelope)`` so the caller can
           short-circuit with the structured error.

    Returns
    -------
    (seed_entity_id, error_response_json)
        Exactly one of the two is non-None.
    """
    if filters.entity_ids is not None:
        if len(filters.entity_ids) == 1:
            return int(filters.entity_ids[0]), None
        return None, _community_expand_no_seed_response(
            query=query,
            reason=(f"filters.entity_ids must contain exactly 1 id, got {len(filters.entity_ids)}"),
        )

    resolver = _srv.EntityResolver(conn)
    try:
        candidates = resolver.resolve(query.strip())
    except Exception:
        logger.exception("community_expand: EntityResolver.resolve failed")
        return None, _community_expand_no_seed_response(
            query=query,
            reason="entity resolution failed",
        )

    if not candidates:
        return None, _community_expand_no_seed_response(
            query=query,
            reason="no entity matches the query",
        )

    if len(candidates) > 1:
        candidate_dicts = [
            {
                "entity_id": c.entity_id,
                "canonical_name": c.canonical_name,
                "entity_type": c.entity_type,
                "source": getattr(c, "source", None),
                "confidence": getattr(c, "confidence", None),
            }
            for c in candidates[:10]
        ]
        return None, _community_expand_no_seed_response(
            query=query,
            reason=(
                f"query resolved to {len(candidates)} entities — pick one "
                f"via filters.entity_ids and retry"
            ),
            candidates=candidate_dicts,
        )

    return int(candidates[0].entity_id), None

def _handle_community_expand(
    conn: psycopg.Connection,
    query: str,
    filters: search.SearchFilters,
    limit: int,
) -> str:
    """Run the community-expand lane and serialise the response.

    Lifts ``metadata.error_code='seed_too_broad'`` from the underlying
    function up to a top-level structured-error response so agents can
    branch on ``error_code`` without inspecting nested metadata.
    """
    seed_id, error_response = _resolve_community_expand_seed(conn, query, filters)
    if error_response is not None:
        return error_response
    assert seed_id is not None  # for type checker

    result = search.community_expand_search(
        conn,
        seed_id,
        top_k=limit,
        filters=filters,
    )

    # Lift seed_too_broad from metadata into a top-level envelope. The MCP
    # contract is that error_code lives at the response root (see x5jg).
    if result.metadata.get("error_code") == ErrorCode.SEED_TOO_BROAD:
        payload = {
            "error_code": ErrorCode.SEED_TOO_BROAD,
            "error": result.metadata.get("error", "seed entity is too broad"),
            "hint": result.metadata.get(
                "hint",
                "Narrow to a more specific entity.",
            ),
            "seed_entity_id": result.metadata.get("seed_entity_id"),
            "seed_paper_count": result.metadata.get("seed_paper_count"),
            "query": query,
            "timing_ms": result.timing_ms,
        }
        return json.dumps(payload, indent=2, default=str)

    return _result_to_json(result)

def _handle_search(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified search: hybrid/semantic/keyword.

    When ``disambiguate`` is true (default) and the query contains at least
    one ambiguous entity mention (as determined by
    :func:`scix.jit.disambiguator.disambiguate_query`), this returns a
    ``{"disambiguation": [...]}`` JSON payload and skips the search. When
    ``disambiguate`` is false, the disambiguation check is bypassed entirely
    and the normal search path runs.

    Unscoped + broad queries (no filters AND >=3 tokens or >=30 chars) are
    blocked by an early guard that returns a structured ``unscoped_broad_query``
    error rather than letting them fall into the 32M-paper full-text scan.
    Pass ``bypass_unscoped_guard=true`` to skip the guard. See bead
    ``scix_experiments-uerc``.
    """
    mode = args.get("mode", "hybrid")
    query = args["query"]
    disambiguate = args.get("disambiguate", True)
    bypass_guard = bool(args.get("bypass_unscoped_guard", False))

    # Unscoped-broad-query guard (bead uerc) — fires before disambiguation
    # and before any DB / embedding work so blocked queries cost ~0ms.
    if _srv._is_unscoped_broad_query(query, args.get("filters"), bypass=bypass_guard):
        return _unscoped_broad_response(query)

    if disambiguate:
        disamb_response = _srv._maybe_disambiguate(conn, query)
        if disamb_response is not None:
            return disamb_response

    try:
        filters = _parse_filters(args.get("filters"))
    except ValueError as exc:
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_FILTERS})
    limit = args.get("limit", 10)

    # Community-expansion lane (bead xz4.1.40). Off by default; gated behind
    # the explicit ``community_expand`` flag like alias_expansion / ontology
    # parser. Replaces hybrid retrieval rather than RRF-fusing — see PRD §4.
    if bool(args.get("community_expand", False)):
        return _handle_community_expand(conn, query, filters, limit)

    if mode == "keyword":
        result = search.lexical_search(conn, query, filters=filters, limit=limit)
        return _result_to_json(result)

    if mode == "semantic":
        model_name = "indus"
        if not _srv._hnsw_index_exists(conn, model_name):
            return json.dumps(
                {
                    "error": "vector_index_unavailable",
                    "error_code": ErrorCode.VECTOR_INDEX_UNAVAILABLE,
                    "model_name": model_name,
                    "detail": (
                        f"No ANN index ({' or '.join(_vector_index_names(model_name))}) "
                        "is available yet. Use mode='keyword' as a fallback."
                    ),
                },
                indent=2,
            )
        try:
            device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
            model, tokenizer = _srv.load_model(model_name, device=device)
            vectors = _srv.embed_batch(model, tokenizer, [query], batch_size=1)
            query_embedding = vectors[0]
        except ImportError:
            return json.dumps(
                {
                    "error": "transformers/torch not installed for embedding",
                    "error_code": ErrorCode.DEPENDENCY_MISSING,
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
    if _srv._hnsw_index_exists(conn, model_name):
        try:
            device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
            model, tokenizer = _srv.load_model(model_name, device=device)
            vectors = _srv.embed_batch(model, tokenizer, [query], batch_size=1)
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
        return json.dumps(
            {
                "error": "query must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

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

def _handle_facet_counts(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Facet counts with optional working-set scoping.

    When ``bibcodes`` is omitted, falls through to the session's focused
    papers (see ``_resolve_working_set_bibcodes``). When neither is set,
    runs the unscoped corpus-wide facet — preserves the legacy contract.
    """
    try:
        filters = _parse_filters(args.get("filters"))
    except ValueError as exc:
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_FILTERS})
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
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_PARAM_VALUE})
    return _result_to_json(result)

def _handle_temporal_evolution(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Temporal evolution with optional working-set scoping.

    Resolution order for the bibcode set:
        1. ``args["bibcodes"]`` if non-empty.
        2. session focused papers (multi-paper aggregate citations mode).
        3. ``args["query"]`` (single-paper / query path; the legacy param name
           ``bibcode_or_query`` is still accepted as a synonym — bead 9afa).

    Returns a clean JSON error when none of the three sources is provided.
    """
    year_start = _coerce_year(args.get("year_start"), "year_start")
    year_end = _coerce_year(args.get("year_end"), "year_end")
    if year_start is not None and year_end is not None and year_end < year_start:
        raise ValueError(f"year_end ({year_end}) must be >= year_start ({year_start})")

    # Cap to match synthesize.MAX_WORKING_SET_BIBCODES — focused_papers
    # FIFO at 500 (bead u0j1) and explicit args["bibcodes"] is unbounded,
    # so without this cap the ANY(%s) array could grow past the ceiling.
    bibcodes = _resolve_working_set_bibcodes(args)[:MAX_WORKING_SET_BIBCODES]
    # Canonical key is ``query`` (bead 9afa); accept the legacy
    # ``bibcode_or_query`` name as a synonym so existing callers don't break.
    bibcode_or_query = args.get("query") or args.get("bibcode_or_query")

    if not bibcodes and not bibcode_or_query:
        return json.dumps(
            {
                "error": (
                    "temporal_evolution requires either query (bibcode or search "
                    "terms), an explicit bibcodes=[...] list, or a non-empty "
                    "working set (call get_paper on one or more papers first)."
                ),
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
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
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_PARAM_VALUE})
    return _result_to_json(result)

def _get_indus_embedder() -> tuple[Any, Any]:
    """Return a cached (model, tokenizer) pair for the INDUS encoder.

    Reuses :func:`scix.embed.load_model`, which has its own
    :data:`scix.embed._model_cache`, so even repeated process restarts only
    pay the disk read cost. CPU-only by default; chunk_search is interactive
    and the per-query cost is dominated by the Qdrant round-trip, not the
    encoder.
    """
    # The cache lives on the mcp_server module (``_srv._indus_embedder``) so it
    # stays a single process-wide slot that callers/tests reset via
    # ``mcp_server._indus_embedder = None``.
    if _srv._indus_embedder is None:
        # Re-import locally so tests that monkeypatch ``scix.embed.load_model``
        # see the patched function rather than the binding captured at import.
        from scix import embed as _embed

        _srv._indus_embedder = _embed.load_model("indus", device="cpu")
    return _srv._indus_embedder

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
    if not _srv._qdrant_enabled():
        return json.dumps(
            {
                "error": "qdrant_disabled",
                "error_code": ErrorCode.QDRANT_DISABLED,
                "message": (
                    "chunk_search requires the Qdrant backend "
                    "(set QDRANT_URL and install qdrant-client)."
                ),
            },
            indent=2,
        )

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return json.dumps(
            {
                "error": "query must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            },
            indent=2,
        )
    query = query.strip()

    # --- limit clamp ---
    raw_limit = args.get("limit", 20)
    try:
        limit = int(raw_limit) if raw_limit is not None else 20
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": f"limit must be an integer, got {raw_limit!r}",
                "error_code": ErrorCode.INVALID_LIMIT,
            },
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
            {
                "error": f"filters must be an object, got {type(filters_raw).__name__}",
                "error_code": ErrorCode.INVALID_FILTERS,
            },
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
        return json.dumps(
            {"error": f"invalid filters: {exc}", "error_code": ErrorCode.INVALID_FILTERS},
            indent=2,
        )

    # --- encode query via INDUS (mean pooling, 768-dim) ---
    try:
        # Re-import lazily so tests that monkeypatch ``scix.embed.embed_batch``
        # after import time see the patched function.
        from scix import embed as _embed

        model, tokenizer = _get_indus_embedder()
        vectors = _embed.embed_batch(model, tokenizer, [query], batch_size=1, pooling="mean")
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("chunk_search: INDUS encode failed")
        return json.dumps(
            {"error": f"encode_failed: {exc}", "error_code": ErrorCode.ENCODE_FAILED},
            indent=2,
        )
    if not vectors:
        return json.dumps(
            {
                "error": "encode_failed: no vector returned",
                "error_code": ErrorCode.ENCODE_FAILED,
            },
            indent=2,
        )
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
        return json.dumps(
            {"error": f"qdrant_failed: {exc}", "error_code": ErrorCode.QDRANT_FAILED},
            indent=2,
        )

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

def _handle_removed_find_similar(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """find_similar_by_examples was retired 2026-04-25 (Qdrant backend unused)."""
    return json.dumps(
        {
            "error": "tool_removed",
            "error_code": ErrorCode.TOOL_REMOVED,
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

def _handle_concept_search(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """concept_search — multi-vocabulary taxonomy router (dbl.7)."""
    result = search.concept_search(
        conn,
        args["query"],
        vocabulary=args.get("vocabulary"),
        include_subtopics=args.get("include_subtopics", True),
        limit=args.get("limit", 20),
        fallback=args.get("fallback", True),
    )
    return _result_to_json(result)
