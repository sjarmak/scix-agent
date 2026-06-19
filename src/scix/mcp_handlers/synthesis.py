"""Synthesis & claim-reading MCP handlers (synthesize_findings, read_paper_claims, find_claims)."""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

from scix.mcp_errors import ErrorCode
from scix.mcp_handlers._common import (
    _session_fallthrough_bibcodes,
)
from scix.mcp_server import (
    DEFAULT_RESULT_LIMIT,
)
from scix.synthesize import MAX_WORKING_SET_BIBCODES
from scix.synthesize import synthesize_findings as _synthesize_findings

logger = logging.getLogger("scix.mcp_server")


def _handle_synthesize_findings(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Bin a working set of papers into a section outline.

    Mirrors the ``find_gaps`` fall-through pattern: explicit
    ``working_set_bibcodes`` win; otherwise read from the session's
    focused papers; otherwise from the working set. Pure mechanism —
    the actual synthesis logic lives in :mod:`scix.synthesize` and is
    LLM-free per ZFC.
    """
    raw_bibcodes = args.get("working_set_bibcodes")
    if raw_bibcodes is None:
        # Session fall-through — same path as 3uvn / find_gaps.
        bibcodes: list[str] = _session_fallthrough_bibcodes()
    elif isinstance(raw_bibcodes, list):
        bibcodes = [b for b in raw_bibcodes if isinstance(b, str)]
    else:
        return json.dumps(
            {
                "error": "working_set_bibcodes must be a list of strings",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            },
        )

    sections = args.get("sections")
    if sections is not None and not isinstance(sections, list):
        return json.dumps(
            {
                "error": "sections must be a list of strings",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            }
        )

    raw_cap = args.get("max_papers_per_section", 8)
    try:
        max_papers = int(raw_cap)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": "max_papers_per_section must be an integer",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            },
        )
    # Hard cap to keep payload sizes sane; matches find_gaps' cap.
    max_papers = max(0, min(max_papers, MAX_WORKING_SET_BIBCODES))

    raw_overrides = args.get("section_overrides")
    if raw_overrides is not None and not isinstance(raw_overrides, dict):
        return json.dumps(
            {
                "error": "section_overrides must be an object {bibcode: section_name}",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            },
        )

    # Bead tq0t: two boolean opt-ins for additive grounding fields. Both
    # default False to preserve the default wire format. Validate as
    # actual bools rather than truthy-coercing — the MCP SDK delivers
    # JSON-parsed values, but a hand-crafted client could send a string
    # like "false" which `bool()` would silently flip to True.
    raw_full = args.get("include_full_abstracts", False)
    if not isinstance(raw_full, bool):
        return json.dumps(
            {
                "error": "include_full_abstracts must be a boolean",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            },
            indent=2,
        )
    raw_ctx = args.get("include_citation_contexts", False)
    if not isinstance(raw_ctx, bool):
        return json.dumps(
            {
                "error": "include_citation_contexts must be a boolean",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            },
            indent=2,
        )
    include_full_abstracts: bool = raw_full
    include_citation_contexts: bool = raw_ctx

    result = _synthesize_findings(
        conn,
        working_set_bibcodes=bibcodes,
        sections=sections,
        max_papers_per_section=max_papers,
        section_overrides=raw_overrides,
        include_full_abstracts=include_full_abstracts,
        include_citation_contexts=include_citation_contexts,
    )
    return json.dumps(result.to_dict(), indent=2, default=str)

def _handle_read_paper_claims(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for the ``read_paper_claims`` MCP tool.

    Thin wrapper over :func:`scix.claims.retrieval.read_paper_claims` that
    surfaces structured-error JSON for invalid inputs (matches the
    convention used by other handlers in this module).
    """
    from scix.claims.retrieval import read_paper_claims

    bibcode = args.get("bibcode")
    if not isinstance(bibcode, str) or not bibcode.strip():
        return json.dumps(
            {
                "error": "bibcode must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    claim_type = args.get("claim_type")
    if claim_type is not None and not isinstance(claim_type, str):
        return json.dumps(
            {
                "error": "claim_type must be a string or omitted",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            }
        )

    limit = args.get("limit", DEFAULT_RESULT_LIMIT)

    try:
        rows = read_paper_claims(
            conn,
            bibcode=bibcode,
            claim_type=claim_type,
            limit=limit,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_PARAM_VALUE})

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
        return json.dumps(
            {
                "error": "query must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    claim_type = args.get("claim_type")
    if claim_type is not None and not isinstance(claim_type, str):
        return json.dumps(
            {
                "error": "claim_type must be a string or omitted",
                "error_code": ErrorCode.INVALID_PARAM_TYPE,
            }
        )

    entity_id = args.get("entity_id")
    if entity_id is not None:
        try:
            entity_id = int(entity_id)
        except (TypeError, ValueError):
            return json.dumps(
                {
                    "error": "entity_id must be an integer or omitted",
                    "error_code": ErrorCode.INVALID_PARAM_TYPE,
                }
            )

    limit = args.get("limit", DEFAULT_RESULT_LIMIT)

    try:
        rows = find_claims(
            conn,
            query=query,
            claim_type=claim_type,
            entity_id=entity_id,
            limit=limit,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_PARAM_VALUE})

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
