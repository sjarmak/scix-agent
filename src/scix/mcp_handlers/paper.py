"""Paper-retrieval and session/working-set MCP handlers."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

import psycopg

from scix import search
from scix.mcp_errors import ErrorCode
from scix.mcp_runtime import (
    _result_to_json,
    _session_state,
)
from scix.mcp_server import _inject_coverage_note

logger = logging.getLogger("scix.mcp_server")


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

    # eq95: drop denylisted (name, type) pairs from linked_entities so
    # generic-word noise ('data'/'dataset', 'method'/'method', etc.) never
    # surfaces alongside real entities. Applied before precision estimate
    # so the SQL noise lookup we just did isn't wasted on rows we'd drop
    # anyway — but the lookup is keyed by entity_id, not canonical_name,
    # so the cost is one batch regardless.
    from scix.extract.ner_denylist import is_denylisted as _is_denylisted
    from scix.extract.ner_quality_profile import (
        precision_band,
        precision_estimate,
    )

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
        return json.dumps(
            {
                "error": "bibcode must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

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

def _handle_add_to_working_set(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy session tool — add bibcodes to the working set."""
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
    entries = [dataclasses.asdict(e) for e in _session_state.get_working_set() if e.bibcode in seen]
    return json.dumps({"added": added, "entries": entries}, indent=2, default=str)

def _handle_get_working_set(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy session tool — read the current working set."""
    entries = _session_state.get_working_set()
    return json.dumps(
        {"entries": [dataclasses.asdict(e) for e in entries], "total": len(entries)},
        indent=2,
        default=str,
    )

def _handle_get_session_summary(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy session tool — summarize the working set."""
    summary = _session_state.get_session_summary()
    return json.dumps(summary, indent=2, default=str)

def _handle_clear_working_set(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy session tool — clear the working set."""
    removed = _session_state.clear_working_set()
    return json.dumps({"removed": removed}, indent=2)

def _handle_get_citation_context(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy get_citation_context direct dispatch (source/target bibcode pair)."""
    result = search.get_citation_context(
        conn,
        args["source_bibcode"],
        args["target_bibcode"],
    )
    return _result_to_json(result)

def _handle_get_author_papers(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Legacy get_author_papers direct dispatch."""
    result = search.get_author_papers(
        conn,
        args["author_name"],
        year_min=args.get("year_min"),
        year_max=args.get("year_max"),
    )
    return _result_to_json(result)
