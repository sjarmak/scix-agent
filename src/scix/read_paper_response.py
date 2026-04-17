"""Pure response-envelope builder for the ``read_paper`` MCP tool (schema v2).

This module is intentionally IO-free: no database, no HTTP, no model calls.
It takes the already-resolved ``sibling_result`` (from
:func:`scix.search.read_fulltext_with_sibling_fallback`), the suppress-set
(from :func:`scix.publisher_suppress.load_suppress_list`), and any existing
v1 fields, and assembles the ``schema_version=2`` response envelope
specified in ``docs/mcp_tool_contracts.md``.

All licensing and suppression invariants from ADR-006 (including the
2026-04-17 Addendum on cross-bibcode LaTeX propagation) are enforced here.
"""

from __future__ import annotations

from typing import Any

from scix.search import LATEX_DERIVED_SOURCES

__all__ = ["build_read_paper_response", "MAX_SNIPPET_CHARS"]


# Per ADR-006 Addendum (2026-04-17): cross-bibcode LaTeX emissions are capped
# at this many characters. Same-bibcode LaTeX is emitted without truncation
# because the snippet budget is enforced at the ingest/read-through layer
# (scix.sources.licensing.enforce_snippet_budget). This constant guards the
# cross-bibcode propagation case specifically.
MAX_SNIPPET_CHARS: int = 500

_SCHEMA_VERSION: int = 2


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` capped at ``limit`` characters; no ellipsis added here.

    The ADR-006 enforce_snippet_budget helper handles ellipsis in the
    ingest lane; this builder applies a hard post-assembly cap for the
    cross-bibcode LaTeX propagation case (Addendum rule a).
    """
    if len(text) <= limit:
        return text
    return text[:limit]


def _resolve_publisher(
    v1_base_fields: dict[str, Any] | None,
    sibling_result: dict[str, Any],
) -> str | None:
    """Return the publisher string to check against the suppress set.

    Preference order (per task spec rule f):
      1. ``v1_base_fields["publisher"]`` if present.
      2. ``sibling_result["row"]["publisher"]`` if a row is present.
    """
    if v1_base_fields is not None:
        pub = v1_base_fields.get("publisher")
        if isinstance(pub, str) and pub.strip():
            return pub
    row = sibling_result.get("row")
    if isinstance(row, dict):
        pub = row.get("publisher")
        if isinstance(pub, str) and pub.strip():
            return pub
    return None


def _is_suppressed(publisher: str | None, suppress_set: frozenset[str]) -> bool:
    """Case-insensitive containment check. Mirrors publisher_suppress.is_suppressed
    but avoids the cross-module import so this builder stays dependency-lean.
    """
    if publisher is None:
        return False
    return publisher.strip().lower() in suppress_set


def build_read_paper_response(
    requested_bibcode: str,
    sibling_result: dict[str, Any],
    suppress_set: frozenset[str],
    abstract_text: str | None,
    v1_base_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``schema_version=2`` ``read_paper`` response envelope.

    Pure function. See ``docs/mcp_tool_contracts.md`` for the authoritative
    shape, and ``docs/ADR/006_arxiv_licensing.md`` (Addendum) for the
    cross-bibcode LaTeX propagation rules.

    Args:
        requested_bibcode: The bibcode the caller asked for.
        sibling_result: Output of
            :func:`scix.search.read_fulltext_with_sibling_fallback`.
        suppress_set: Output of
            :func:`scix.publisher_suppress.load_suppress_list`.
        abstract_text: Abstract body to emit when no full-text is served.
            Ignored when the response includes body text.
        v1_base_fields: Pre-existing v1-shaped fields (e.g., bibcode,
            title, abstract) to preserve in the v2 response. Additive-only:
            every key present here appears in the result unless suppression
            explicitly strips it (body/sections/source_version).

    Returns:
        A dict with ``schema_version=2`` and ``source_bibcode`` always set,
        plus whichever optional v2 fields apply to the scenario.
    """
    # Rule (a): start with a v2 envelope seeded from v1 base fields.
    response: dict[str, Any] = dict(v1_base_fields) if v1_base_fields else {}
    response["schema_version"] = _SCHEMA_VERSION
    response["suppressed_by_publisher"] = False
    response["source_bibcode"] = requested_bibcode  # default; overridden by (c)

    hit = bool(sibling_result.get("hit"))
    sibling = sibling_result.get("sibling")
    row = sibling_result.get("row") if isinstance(sibling_result.get("row"), dict) else None

    if hit and sibling is None:
        # Rule (b): direct hit.
        if row is not None:
            if "source" in row:
                response["source"] = row["source"]
            if "body" in row and row["body"] is not None:
                response["body"] = row["body"]
            if "sections" in row and row["sections"] is not None:
                response["sections"] = row["sections"]
            if "source_version" in row and row["source_version"] is not None:
                response["source_version"] = row["source_version"]
            if "canonical_url" in row and row["canonical_url"] is not None:
                response["canonical_url"] = row["canonical_url"]
        response["source_bibcode"] = requested_bibcode

    elif hit and sibling_result.get("served_from_sibling_bibcode"):
        # Rule (c): sibling hit (LaTeX-derived row propagated across bibcodes).
        served_from = sibling_result["served_from_sibling_bibcode"]
        response["served_from_sibling_bibcode"] = served_from
        response["source_bibcode"] = served_from
        canonical = sibling_result.get("canonical_url")
        if canonical is not None:
            response["canonical_url"] = canonical
        if row is not None:
            if "source" in row:
                response["source"] = row["source"]
            if "body" in row and row["body"] is not None:
                response["body"] = row["body"]
            if "sections" in row and row["sections"] is not None:
                response["sections"] = row["sections"]
            if "source_version" in row and row["source_version"] is not None:
                response["source_version"] = row["source_version"]

    elif (not hit) and sibling_result.get("miss_with_hint"):
        # Rule (d): miss-with-hint. No body/sections.
        response["source"] = "abstract"
        response["fulltext_available_under_sibling"] = sibling_result[
            "fulltext_available_under_sibling"
        ]
        response["hint"] = sibling_result.get("hint", "")
        if abstract_text is not None and "abstract" not in response:
            response["abstract"] = abstract_text
        response.pop("body", None)
        response.pop("sections", None)

    else:
        # Rule (e): pure miss.
        response["source"] = "abstract"
        if abstract_text is not None and "abstract" not in response:
            response["abstract"] = abstract_text

    # Rule (f): publisher suppression override. Overrides (b)/(c)/(d)/(e).
    publisher = _resolve_publisher(v1_base_fields, sibling_result)
    if _is_suppressed(publisher, suppress_set):
        response["suppressed_by_publisher"] = True
        response["source"] = "abstract"
        response.pop("body", None)
        response.pop("sections", None)
        response.pop("source_version", None)

    # Rule (g): cross-bibcode LaTeX snippet truncation. Same-bibcode LaTeX
    # passes through unchanged (rule h falls out of this guard).
    current_source = response.get("source")
    if (
        current_source in LATEX_DERIVED_SOURCES
        and requested_bibcode != response.get("source_bibcode")
    ):
        if isinstance(response.get("body"), str) and len(response["body"]) > MAX_SNIPPET_CHARS:
            response["body"] = _truncate(response["body"], MAX_SNIPPET_CHARS)
        if isinstance(response.get("snippet"), str) and len(response["snippet"]) > MAX_SNIPPET_CHARS:
            response["snippet"] = _truncate(response["snippet"], MAX_SNIPPET_CHARS)

    return response
