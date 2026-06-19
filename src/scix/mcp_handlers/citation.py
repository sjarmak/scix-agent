"""Citation-graph MCP handlers (traverse, graph, similarity, forward, replications, intent, blame)."""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

from scix import search
from scix.mcp_errors import ErrorCode
from scix.mcp_handlers._common import (
    _missing_required_params_error,
    _resolve_working_set_bibcodes,
)
from scix.mcp_server import (
    DEFAULT_RESULT_LIMIT,
    _annotate_working_set,
    _result_to_json,
)
from scix.synthesize import MAX_WORKING_SET_BIBCODES

logger = logging.getLogger("scix.mcp_server")


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
    contract consistent with the rest of the dispatch layer. Validation
    runs BEFORE any DB access (bead scix_experiments-zjt9): the payload
    carries ``error_code='missing_required_params'`` plus ``mode`` /
    ``required`` / ``got`` so agents can correct without a probe round-trip.
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
        # Nothing to traverse from — neither single bibcode, explicit
        # bibcodes=[...], nor a focused session working set. Return the
        # structured validation envelope before touching the DB.
        return _missing_required_params_error(
            mode="graph",
            required=["bibcode"],
            got=[],
            message=(
                "bibcode is required when mode='graph' (or pass "
                "bibcodes=[...] / focus papers via get_paper for "
                "working-set mode)."
            ),
        )

    if mode == "chain":
        source = args.get("source_bibcode")
        target = args.get("target_bibcode")
        if not source or not target:
            got = [
                name
                for name, value in (
                    ("source_bibcode", source),
                    ("target_bibcode", target),
                )
                if value
            ]
            return _missing_required_params_error(
                mode="chain",
                required=["source_bibcode", "target_bibcode"],
                got=got,
                message=("source_bibcode and target_bibcode are required when mode='chain'"),
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
            "error_code": ErrorCode.INVALID_MODE,
        }
    )

def _handle_citation_traverse_multi(
    conn: psycopg.Connection,
    bibcodes: list[str],
    args: dict[str, Any],
) -> str:
    """Walk the citation neighborhood of multiple bibcodes.

    Fetches all requested neighborhoods with batched queries (one per direction
    against ``citation_edges`` plus one per direction against
    ``citation_contexts`` for intent) instead of looping a per-bibcode
    ``_handle_citation_graph`` call. This bounds DB round-trips at a small
    constant regardless of working-set size — the previous per-bibcode loop
    could fire up to ``len(bibcodes)`` sequential queries (FIFO cap 200) and
    reliably timed out on large working sets (bead scix_experiments-sd71). The
    per-bibcode ``limit`` is preserved (each source paper gets up to ``limit``
    neighbors), as is the ``by_bibcode`` output shape.
    """
    direction = args.get("direction", "forward")
    limit = args.get("limit", 20)

    if direction not in ("forward", "backward", "both"):
        err = {
            "error": f"Invalid direction: {direction}. Use 'forward', 'backward', or 'both'.",
            "error_code": ErrorCode.INVALID_DIRECTION,
        }
        return json.dumps(
            {
                "mode": "graph",
                "scope": "working_set",
                "bibcodes": list(bibcodes),
                "by_bibcode": {bib: err for bib in bibcodes},
            },
            indent=2,
            default=str,
        )

    fwd_papers: dict[str, list[dict[str, Any]]] = {}
    bwd_papers: dict[str, list[dict[str, Any]]] = {}
    fwd_intents: dict[str, dict[str, str]] = {}
    bwd_intents: dict[str, dict[str, str]] = {}

    if direction in ("forward", "both"):
        fwd_papers = search.get_citations_batch(conn, list(bibcodes), limit=limit)
        fwd_intents = _enrich_citations_with_intent_batch(
            conn, neighbors_by_bibcode=fwd_papers, direction="forward"
        )
    if direction in ("backward", "both"):
        bwd_papers = search.get_references_batch(conn, list(bibcodes), limit=limit)
        bwd_intents = _enrich_citations_with_intent_batch(
            conn, neighbors_by_bibcode=bwd_papers, direction="backward"
        )

    per_bibcode: dict[str, Any] = {}
    for bib in bibcodes:
        if direction == "forward":
            per_bibcode[bib] = _build_traverse_direction(
                fwd_papers.get(bib, []), fwd_intents.get(bib)
            )
        elif direction == "backward":
            per_bibcode[bib] = _build_traverse_direction(
                bwd_papers.get(bib, []), bwd_intents.get(bib)
            )
        else:  # both
            per_bibcode[bib] = {
                "bibcode": bib,
                "directions": [
                    {
                        "direction": "forward",
                        "result": _build_traverse_direction(
                            fwd_papers.get(bib, []), fwd_intents.get(bib)
                        ),
                    },
                    {
                        "direction": "backward",
                        "result": _build_traverse_direction(
                            bwd_papers.get(bib, []), bwd_intents.get(bib)
                        ),
                    },
                ],
            }

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

def _build_traverse_direction(
    papers: list[dict[str, Any]], intents: dict[str, str] | None
) -> dict[str, Any]:
    """Build a single-direction payload matching ``_handle_citation_graph``.

    Applies working-set annotation and (where covered) citation intent, so the
    batched working-set path returns the same per-paper shape as the
    single-bibcode path.
    """
    annotated = _annotate_working_set(papers)
    if intents:
        _annotate_papers_with_intent(annotated, intents)
    return {"papers": annotated, "total": len(annotated), "timing_ms": {}}

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

def _enrich_citations_with_intent_batch(
    conn: psycopg.Connection,
    *,
    neighbors_by_bibcode: dict[str, list[dict[str, Any]]],
    direction: str,
) -> dict[str, dict[str, str]]:
    """Batched form of ``_enrich_citations_with_intent`` for many traversed papers.

    Returns ``{traversed_bibcode: {neighbor_bibcode: intent}}`` from a single
    ``citation_contexts`` query keyed on every traversed bibcode that has
    neighbors. Because citation_contexts coverage is sparse (~0.27% of edges,
    bead 79n), the result set stays small even when fetching all covered
    contexts for the queried papers; ``_annotate_papers_with_intent`` only
    applies the entries whose neighbor bibcode is actually in the result list.
    """
    traversed = [bib for bib, papers in neighbors_by_bibcode.items() if papers]
    if not traversed:
        return {}
    # forward: traversed paper is the cited target; neighbors are citing sources.
    # backward: traversed paper is the citing source; neighbors are cited targets.
    if direction == "forward":
        sql = (
            "SELECT target_bibcode, source_bibcode, intent FROM citation_contexts "
            "WHERE target_bibcode = ANY(%s) AND intent IS NOT NULL"
        )
    else:
        sql = (
            "SELECT source_bibcode, target_bibcode, intent FROM citation_contexts "
            "WHERE source_bibcode = ANY(%s) AND intent IS NOT NULL"
        )
    out: dict[str, dict[str, str]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (traversed,))
        for traversed_bib, neighbor_bib, intent in cur.fetchall():
            out.setdefault(traversed_bib, {})[neighbor_bib] = intent
    return out

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

    return json.dumps(
        {
            "error": f"Invalid direction: {direction}. Use 'forward', 'backward', or 'both'.",
            "error_code": ErrorCode.INVALID_DIRECTION,
        }
    )

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
        return json.dumps(
            {
                "error": f"Invalid method: {method}. Use co_citation or coupling.",
                "error_code": ErrorCode.INVALID_METHOD,
            }
        )

    return _result_to_json(result)

def _handle_claim_blame(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the claim_blame MCP tool (PRD MH-4)."""
    from scix.claim_blame import claim_blame
    from scix.research_scope import scope_from_dict

    claim_text = args.get("claim_text")
    if not isinstance(claim_text, str) or not claim_text.strip():
        return json.dumps(
            {
                "error": "claim_text must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    scope_arg = args.get("scope")
    try:
        scope = scope_from_dict(scope_arg) if scope_arg else None
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid scope: {exc}", "error_code": ErrorCode.INVALID_SCOPE})

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

#: Annotation axes accepted by the ``forward_citations`` MCP tool (bead 9afa).
_FORWARD_CITATION_ANNOTATIONS: frozenset[str] = frozenset({"intent", "relation"})

def _handle_forward_citations(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Enumerate forward citations to a paper, annotated by intent or relation.

    Merges the former ``cited_by_intent`` and ``find_replications`` tools
    (bead 9afa). The ``annotate`` enum selects the annotation axis and the
    response shape is delegated verbatim to the original handler so the two
    deprecated aliases return byte-for-byte what they used to:

        * annotate='intent'   -> _handle_cited_by_intent  (intent label +
          400-char excerpt per citing paper; ``intent`` filter applies)
        * annotate='relation' -> _handle_find_replications (inferred
          replication relation + hedge flag; ``relation`` / ``scope`` apply)

    Anchor param is ``bibcode`` (consistent with ``citation_traverse``);
    ``target_bibcode`` is accepted as a synonym so the alias layer and any
    direct callers of the old tools keep working. ``limit`` follows the
    DEFAULT_RESULT_LIMIT / MAX_WORKING_SET_BIBCODES convention (audit §5).
    """
    annotate = args.get("annotate", "intent")
    if annotate not in _FORWARD_CITATION_ANNOTATIONS:
        return json.dumps(
            {
                "error": (
                    f"annotate must be one of {sorted(_FORWARD_CITATION_ANNOTATIONS)}; "
                    f"got {annotate!r}"
                ),
                "error_code": ErrorCode.INVALID_PARAM_VALUE,
            }
        )

    # Normalize the anchor to the key the delegate handlers read.
    delegated = dict(args)
    anchor = args.get("bibcode") or args.get("target_bibcode")
    if anchor is not None:
        delegated["target_bibcode"] = anchor

    # Apply the shared limit convention (default 20, cap 200) before delegating
    # so both legs share one policy regardless of their historical defaults.
    raw_limit = args.get("limit", DEFAULT_RESULT_LIMIT)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": f"limit must be an integer, got {raw_limit!r}",
                "error_code": ErrorCode.INVALID_LIMIT,
            }
        )
    if limit < 1:
        limit = DEFAULT_RESULT_LIMIT
    delegated["limit"] = min(limit, MAX_WORKING_SET_BIBCODES)

    if annotate == "relation":
        return _handle_find_replications(conn, delegated)
    return _handle_cited_by_intent(conn, delegated)

def _handle_find_replications(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the find_replications MCP tool (PRD MH-4)."""
    from scix.find_replications import VALID_RELATIONS, find_replications
    from scix.research_scope import scope_from_dict

    target_bibcode = args.get("target_bibcode")
    if not isinstance(target_bibcode, str) or not target_bibcode.strip():
        return json.dumps(
            {
                "error": "target_bibcode must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    relation = args.get("relation")
    if relation is not None and relation not in VALID_RELATIONS:
        return json.dumps(
            {
                "error": f"relation must be one of {sorted(VALID_RELATIONS)} or null",
                "error_code": ErrorCode.INVALID_PARAM_VALUE,
            }
        )

    scope_arg = args.get("scope")
    try:
        scope = scope_from_dict(scope_arg) if scope_arg else None
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid scope: {exc}", "error_code": ErrorCode.INVALID_SCOPE})

    limit = int(args.get("limit", 50))

    result = find_replications(
        target_bibcode,
        relation=relation,
        scope=scope,
        conn=conn,
        limit=limit,
    )
    # find_replications now returns {"citations": [...], "coverage": {...}}.
    # Add the (cheap) total convenience key for backward compatibility with
    # earlier agents that read ``total``.
    response = {
        "citations": result["citations"],
        "total": len(result["citations"]),
        "coverage": result["coverage"],
    }
    return json.dumps(response, indent=2, default=str)

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
    cited papers. For papers not covered, returns empty cleanly. The
    ``coverage`` block on every response (mirroring ``claim_blame`` /
    ``find_replications``) lets agents distinguish 'no events' (target is
    in citation_contexts but has no incoming intent-classified citations)
    from 'no coverage' (target is not in citation_contexts at all).

    Results are deduplicated to one row per ``source_bibcode``: a single
    citing paper may have multiple matching contexts (e.g. references the
    target several times), and surfacing them as separate rows wastes the
    ``limit`` budget. ``n_contexts`` reports the per-source count so
    agents can see context density without the bloat.
    """
    from scix.citation_contexts_coverage import compute_coverage

    target_bibcode = args.get("target_bibcode")
    if not isinstance(target_bibcode, str) or not target_bibcode.strip():
        return json.dumps(
            {
                "error": "target_bibcode must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    intent = args.get("intent")
    if intent is not None and intent not in _VALID_CITATION_INTENTS:
        return json.dumps(
            {
                "error": (f"intent must be one of {sorted(_VALID_CITATION_INTENTS)} or null (any)"),
                "error_code": ErrorCode.INVALID_PARAM_VALUE,
            }
        )

    limit = min(int(args.get("limit", 20)), MAX_WORKING_SET_BIBCODES)
    target = target_bibcode.strip()

    # Window-function dedup: one row per source_bibcode, keeping the
    # earliest-id matching context as the excerpt and counting all matching
    # contexts in n_contexts. Outer ORDER BY prioritises high-citation citing
    # papers; ASC tiebreaker on source_bibcode for deterministic output.
    sql = """
        WITH ranked AS (
            SELECT
                cc.source_bibcode,
                cc.intent,
                substr(cc.context_text, 1, 400) AS context_excerpt,
                p.title,
                p.year,
                p.authors[1] AS first_author,
                p.citation_count,
                COUNT(*) OVER (PARTITION BY cc.source_bibcode) AS n_contexts,
                ROW_NUMBER() OVER (
                    PARTITION BY cc.source_bibcode
                    ORDER BY cc.id ASC
                ) AS rn
            FROM citation_contexts cc
            LEFT JOIN papers p ON cc.source_bibcode = p.bibcode
            WHERE cc.target_bibcode = %s
              AND ( %s::text IS NULL OR cc.intent = %s::text )
        )
        SELECT
            source_bibcode, intent, context_excerpt,
            title, year, first_author, citation_count, n_contexts
        FROM ranked
        WHERE rn = 1
        ORDER BY citation_count DESC NULLS LAST, source_bibcode ASC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (target, intent, intent, limit))
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    papers = [dict(zip(cols, r)) for r in rows]
    coverage = compute_coverage(conn, [target])
    return json.dumps(
        {
            "target_bibcode": target_bibcode,
            "intent": intent,
            "papers": papers,
            "total": len(papers),
            "coverage": coverage,
        },
        indent=2,
        default=str,
    )

def _handle_citation_chain(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy citation_chain direct dispatch (agent surface: citation_traverse)."""
    max_depth = max(1, min(args.get("max_depth", 5), 5))
    result = search.citation_chain(
        conn,
        args["source_bibcode"],
        args["target_bibcode"],
        max_depth=max_depth,
    )
    return _result_to_json(result)
