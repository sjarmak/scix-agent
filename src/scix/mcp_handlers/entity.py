"""Entity & claim MCP handlers (entity, claim_search, extraction, graph_context, find_gaps, profiles)."""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

from scix import mcp_server as _srv
from scix import search
from scix.mcp_errors import ErrorCode
from scix.mcp_server import (
    _annotate_working_set,
    _inject_coverage_note,
    _result_to_json,
    _session_state,
)
from scix.synthesize import MAX_WORKING_SET_BIBCODES

logger = logging.getLogger("scix.mcp_server")


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

#: The four entity-containment types that the entity tool's
#: ``action='search'`` path still supports. These map to
#: ``staging.extractions`` rows whose ``payload`` is a JSONB object keyed by
#: type name ({"methods": ["JWST", ...]}).
_VALID_ENTITY_TYPES: frozenset[str] = frozenset({"methods", "datasets", "instruments", "materials"})

def _handle_entity(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Unified entity search and resolution."""
    action = args.get("action", "search")
    # Accept both ``query`` (current contract) and ``entity_name`` (used by
    # full-text-extraction callers, mirrors the deprecated entity_search
    # field name) so the same handler serves both call shapes.
    query = args.get("query") or args.get("entity_name") or ""
    entity_type = args.get("entity_type")

    # mh14: reject the legacy extraction-row kinds at the front door so the
    # error is the same regardless of action and the caller can recover in
    # one turn. These used to be valid entity_type values but they're
    # claim/finding extractions, not entities — a follow-up bead (c996)
    # will surface them under a dedicated tool.
    if entity_type in _CLAIM_SEARCH_ACTIONS:
        return json.dumps(
            {
                "error": (
                    f"entity_type='{entity_type}' is no longer accepted by the "
                    f"entity tool — it is a claim/finding extraction, not an "
                    f"entity. Valid entity_type values are: "
                    f"{sorted(_VALID_ENTITY_TYPES)}. Use "
                    f"claim_search(action='{entity_type}') instead "
                    f"(bead scix_experiments-c996); see "
                    f"docs/mcp_tool_audit_2026-04.md for the rationale."
                ),
                "error_code": ErrorCode.ENTITY_LEGACY_EXTRACTION_TYPE,
            }
        )

    # action='profile' (folded-in entity_context, bead 9afa): full profile of
    # one entity by numeric entity_id. Handled before the query gate because
    # it takes an id, not text.
    if action == "profile":
        entity_id = args.get("entity_id")
        if entity_id is None:
            return json.dumps(
                {
                    "error": "entity_id is required for action='profile'",
                    "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
                }
            )
        try:
            entity_id = int(entity_id)
        except (TypeError, ValueError):
            return json.dumps(
                {
                    "error": "entity_id must be an integer",
                    "error_code": ErrorCode.INVALID_PARAM_TYPE,
                }
            )
        result = search.get_entity_context(conn, entity_id)
        return _result_to_json(result)

    # action='papers' accepts entity_id directly, no query needed when given.
    is_papers_with_id = action == "papers" and args.get("entity_id") is not None
    if not is_papers_with_id and (not query or not query.strip()):
        return json.dumps(
            {
                "error": "query must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    if action == "resolve":
        resolver = _srv.EntityResolver(conn)
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
        # mh14: legacy extraction-row kinds (negative_result, quant_claim)
        # are filtered out earlier by the front-door check; this branch
        # only sees real containment-payload entity types.
        if not entity_type or entity_type not in _VALID_ENTITY_TYPES:
            return json.dumps(
                {
                    "error": (
                        f"Invalid entity_type '{entity_type}'. "
                        f"Must be one of: {sorted(_VALID_ENTITY_TYPES)}"
                    ),
                    "error_code": ErrorCode.INVALID_PARAM_VALUE,
                }
            )

        limit = min(args.get("limit", 20), MAX_WORKING_SET_BIBCODES)
        containment = json.dumps({entity_type: [query]})

        # Build WHERE clauses conditionally so backward compatibility is
        # preserved: when no provenance args are supplied, the effective
        # SQL is identical to the pre-filter query.
        where_clauses: list[str] = ["e.payload @> %s::jsonb"]
        params: list[Any] = [containment]

        sources = args.get("sources")
        if sources is not None:
            if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
                return json.dumps(
                    {
                        "error": "sources must be a list of strings",
                        "error_code": ErrorCode.INVALID_PARAM_TYPE,
                    }
                )
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
                            f"min_confidence_tier must be 1, 2, or 3; got {min_confidence_tier!r}"
                        ),
                        "error_code": ErrorCode.INVALID_PARAM_VALUE,
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
                    {
                        "error": "entity_id or query must be provided for action='papers'",
                        "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
                    }
                )
            resolver = _srv.EntityResolver(conn)
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
            return json.dumps(
                {
                    "error": "entity_id must be an integer",
                    "error_code": ErrorCode.INVALID_PARAM_TYPE,
                }
            )

        limit = min(args.get("limit", 20), MAX_WORKING_SET_BIBCODES)
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

    return json.dumps(
        {
            "error": f"Invalid action: {action}. Use 'search', 'resolve', 'papers', or 'profile'.",
            "error_code": ErrorCode.INVALID_ACTION,
        }
    )

#: Action values accepted by the ``claim_search`` MCP tool. Mirrors the
#: ``staging.extractions.extraction_type`` values that ``_handle_entity_
#: extraction_search`` knows how to flatten. Adding a new action here
#: requires teaching the helper the new payload shape.
_CLAIM_SEARCH_ACTIONS: frozenset[str] = frozenset({"negative_result", "quant_claim"})

def _handle_claim_search(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch ``claim_search`` to the shared extraction-search helper.

    Validates the public input contract (action enum, limit cap), then
    delegates to :func:`_handle_entity_extraction_search` which owns the
    SQL and the per-extraction-type payload flattening. Coverage note is
    injected at the top level for parity with the entity tool surface
    (both read from the ``extractions`` table).
    """
    action = args.get("action")
    if not isinstance(action, str) or action not in _CLAIM_SEARCH_ACTIONS:
        return json.dumps(
            {
                "error": (
                    f"Invalid action: {action!r}. Must be one of {sorted(_CLAIM_SEARCH_ACTIONS)}."
                ),
                "error_code": ErrorCode.INVALID_ACTION,
            }
        )

    query = args.get("query")
    name_filter = query.strip() if isinstance(query, str) and query.strip() else None

    raw_limit = args.get("limit", 20)
    try:
        limit = min(int(raw_limit), MAX_WORKING_SET_BIBCODES)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": f"limit must be an integer, got {raw_limit!r}",
                "error_code": ErrorCode.INVALID_LIMIT,
            }
        )
    if limit < 1:
        limit = 20

    result_json = _handle_entity_extraction_search(
        conn,
        extraction_type=action,
        name_filter=name_filter,
        limit=limit,
    )
    return _inject_coverage_note(result_json)

def _handle_entity_extraction_search(
    conn: psycopg.Connection,
    *,
    extraction_type: str,
    name_filter: str | None,
    limit: int,
) -> str:
    """Surface rows from ``staging.extractions`` for a claim/finding-extraction kind.

    .. note::

       The ``entity`` MCP tool's ``entity_type`` enum no longer routes
       through this helper — under bead ``scix_experiments-mh14`` the
       legacy ``negative_result`` / ``quant_claim`` values were rejected
       at the front door because they are claim/finding extractions, not
       entities. The dedicated home is the ``claim_search`` tool added
       under bead ``scix_experiments-c996``, which calls this helper
       directly via :func:`_handle_claim_search`.

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

    # Default 'medium' matches the schema default and find_gaps so back-to-back
    # agent calls land on the same community partition (bead unmm).
    resolution = args.get("resolution", "medium")
    limit = args.get("limit", 20)
    signal = args.get("signal", "semantic")
    try:
        community_result = search.explore_community(
            conn, bibcode, resolution=resolution, limit=limit, signal=signal
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_PARAM_VALUE})

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
    limit = min(args.get("limit", 20), MAX_WORKING_SET_BIBCODES)
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
                    f"Invalid signal: {signal}. Must be one of {sorted(_SIGNAL_COLUMN_PREFIX)}"
                ),
                "error_code": ErrorCode.INVALID_PARAM_VALUE,
            }
        )

    _VALID_RESOLUTIONS = ("coarse", "medium", "fine")
    if resolution not in _VALID_RESOLUTIONS:
        return json.dumps(
            {
                "error": (
                    f"Invalid resolution: {resolution}. Must be one of {sorted(_VALID_RESOLUTIONS)}"
                ),
                "error_code": ErrorCode.INVALID_PARAM_VALUE,
            }
        )
    community_col = f"{column_prefix}_{resolution}"

    # Use focused papers (from get_paper calls) as primary source,
    # fall back to working set for backward compatibility.
    ws_bibcodes = _session_state.get_focused_papers()
    if not ws_bibcodes:
        ws_bibcodes = [e.bibcode for e in _session_state.get_working_set()]
    ws_bibcodes = ws_bibcodes[:MAX_WORKING_SET_BIBCODES]

    # When no working set is populated and the caller passed a query, seed
    # the working set on-the-fly via concept_search so single-call agents
    # can run gap analysis in one shot. Pure convenience — same downstream
    # logic, just bootstrapped.
    seed_query = args.get("query")
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
        except Exception:
            # Best-effort: fall through to the no-papers branch below.
            logger.debug(
                "find_gaps auto-seed via concept_search failed for query=%r",
                seed_query,
                exc_info=True,
            )
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

def _handle_entity_profile(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy entity_profile handler: returns raw extractions table rows.

    Preserves the pre-consolidation schema for backward compatibility with
    external callers that still reference entity_profile. New code should
    use get_paper(include_entities=true) instead.
    """
    bibcode = args.get("bibcode")
    if not bibcode:
        return json.dumps(
            {
                "error": "bibcode is required",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

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

def _handle_entity_context(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy entity_context direct dispatch (agent surface: entity action=profile)."""
    entity_id = args.get("entity_id")
    if entity_id is None:
        return json.dumps(
            {
                "error": "entity_id is required",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )
    try:
        entity_id = int(entity_id)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": "entity_id must be an integer",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            }
        )
    result = search.get_entity_context(conn, entity_id)
    return _result_to_json(result)
