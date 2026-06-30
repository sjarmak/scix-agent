"""Claim/finding-extraction MCP handler (``claim_search`` tool).

Split out of :mod:`scix.mcp_handlers.entity` (bead scix_experiments-2qx3) so
the entity module stays under the file-size cap. The ``claim_search`` tool and
the entity tool both read from ``staging.extractions``, but they answer
different questions — entities (methods/datasets/instruments/materials) vs
claim/finding extractions (negative_result/quant_claim) — so they split on
their reason to change, not on line count.

:mod:`scix.mcp_handlers.entity` imports :data:`_CLAIM_SEARCH_ACTIONS` from here
to reject the legacy claim/finding ``entity_type`` values at the entity front
door; there is no back-import, so the two modules are acyclic.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

from scix.mcp_errors import ErrorCode
from scix.mcp_runtime import _annotate_working_set
from scix.mcp_server import _inject_coverage_note
from scix.synthesize import MAX_WORKING_SET_BIBCODES

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
