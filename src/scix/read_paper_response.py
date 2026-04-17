"""Pure response-envelope builder for the ``read_paper`` MCP tool (schema v2).

IO-free: no database, no HTTP, no model calls. Takes the already-resolved
``sibling_result`` (from :func:`scix.search.read_fulltext_with_sibling_fallback`),
the suppress-set (from :func:`scix.publisher_suppress.load_suppress_list`), and
any existing v1 fields, and assembles the ``schema_version=2`` response
envelope specified in ``docs/mcp_tool_contracts.md``.

Licensing and suppression invariants from ADR-006 (including the 2026-04-17
Addendum on cross-bibcode LaTeX propagation) are enforced here.
"""

from __future__ import annotations

from typing import Any

from scix.publisher_suppress import is_suppressed
from scix.sources.ar5iv import LATEX_DERIVED_SOURCES
from scix.sources.licensing import DEFAULT_SNIPPET_BUDGET, enforce_snippet_budget

__all__ = ["build_read_paper_response", "MAX_SNIPPET_CHARS"]

# Re-exported alias for the cross-bibcode LaTeX snippet cap. The canonical
# value lives in scix.sources.licensing (ADR-006).
MAX_SNIPPET_CHARS: int = DEFAULT_SNIPPET_BUDGET

_SCHEMA_VERSION: int = 2

_ROW_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "source",
    "body",
    "sections",
    "source_version",
)


def _copy_row_fields(response: dict[str, Any], row: dict[str, Any]) -> None:
    for key in _ROW_PASSTHROUGH_KEYS:
        value = row.get(key)
        if value is not None or key == "source":
            if key in row:
                response[key] = value


def _resolve_publisher(
    v1_base_fields: dict[str, Any] | None,
    sibling_result: dict[str, Any],
) -> str | None:
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


def _apply_cross_bibcode_snippet_cap(
    response: dict[str, Any], requested_bibcode: str
) -> None:
    if response.get("source") not in LATEX_DERIVED_SOURCES:
        return
    if requested_bibcode == response.get("source_bibcode"):
        return

    canonical = response.get("canonical_url") or f"https://arxiv.org/abs/{response.get('source_bibcode', '')}"
    for field in ("body", "snippet"):
        text = response.get(field)
        if isinstance(text, str) and len(text) > MAX_SNIPPET_CHARS:
            payload = enforce_snippet_budget(text, canonical, budget=MAX_SNIPPET_CHARS)
            response[field] = payload.snippet


def build_read_paper_response(
    requested_bibcode: str,
    sibling_result: dict[str, Any],
    suppress_set: frozenset[str],
    abstract_text: str | None,
    v1_base_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``schema_version=2`` ``read_paper`` response envelope.

    See ``docs/mcp_tool_contracts.md`` for the authoritative shape and
    ``docs/ADR/006_arxiv_licensing.md`` (Addendum) for the cross-bibcode
    LaTeX propagation rules.

    Args:
        requested_bibcode: The bibcode the caller asked for.
        sibling_result: Output of
            :func:`scix.search.read_fulltext_with_sibling_fallback`.
        suppress_set: Output of
            :func:`scix.publisher_suppress.load_suppress_list`.
        abstract_text: Abstract body to emit when no full-text is served.
        v1_base_fields: Pre-existing v1-shaped fields to preserve in the v2
            response. Additive-only: every key present here appears in the
            result unless suppression explicitly strips it.

    Returns:
        A dict with ``schema_version=2`` and ``source_bibcode`` always set,
        plus whichever optional v2 fields apply to the scenario.
    """
    response: dict[str, Any] = dict(v1_base_fields) if v1_base_fields else {}
    response["schema_version"] = _SCHEMA_VERSION
    response["suppressed_by_publisher"] = False
    response["source_bibcode"] = requested_bibcode

    hit = bool(sibling_result.get("hit"))
    sibling = sibling_result.get("sibling")
    raw_row = sibling_result.get("row")
    row = raw_row if isinstance(raw_row, dict) else None

    if hit and sibling is None:
        if row is not None:
            _copy_row_fields(response, row)
            if row.get("canonical_url") is not None:
                response["canonical_url"] = row["canonical_url"]

    elif hit and sibling_result.get("served_from_sibling_bibcode"):
        served_from = sibling_result["served_from_sibling_bibcode"]
        response["served_from_sibling_bibcode"] = served_from
        response["source_bibcode"] = served_from
        canonical = sibling_result.get("canonical_url")
        if canonical is not None:
            response["canonical_url"] = canonical
        if row is not None:
            _copy_row_fields(response, row)

    elif (not hit) and sibling_result.get("miss_with_hint"):
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
        response["source"] = "abstract"
        if abstract_text is not None and "abstract" not in response:
            response["abstract"] = abstract_text

    publisher = _resolve_publisher(v1_base_fields, sibling_result)
    if is_suppressed(publisher, suppress_set):
        response["suppressed_by_publisher"] = True
        response["source"] = "abstract"
        response.pop("body", None)
        response.pop("sections", None)
        response.pop("source_version", None)

    _apply_cross_bibcode_snippet_cap(response, requested_bibcode)

    return response
