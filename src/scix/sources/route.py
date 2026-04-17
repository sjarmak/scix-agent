"""Full-text request routing decision tree.

Given a :class:`RouteInput` describing what we already know about a paper
(existing fulltext row, sibling rows, ADS body availability, R2-eligibility
metadata), :func:`route_fulltext_request` returns a :class:`RouteDecision`
naming the tier that should service the request.

Rule order (first match wins) — see the full-text 100% coverage PRD §Design:

1. ``has_fulltext_row=True``
   → ``serve_existing`` (idempotency short-circuit)
2. ``sibling_row_source ∈ {ar5iv, arxiv_local}``
   → ``serve_sibling`` (LaTeX-derived siblings only)
3. ``has_ads_body=True``
   → ``tier1_ads_body`` (ADS body ALWAYS wins over Tier 3)
4. ``doctype ∈ {article, eprint, review}``
   AND ``doi is not None``
   AND ``openalex_has_pdf_url=True``
   → ``tier3_docling``
5. else → ``abstract_only``

Non-LaTeX sibling sources (``s2orc``, ``ads_body``, ``docling``) do NOT trigger
``serve_sibling`` — they fall through to the subsequent rules. A row with an
ADS body is ALWAYS routed to Tier 1, never Tier 3.

The module is pure logic: no IO, no DB, no logging. It is trivially testable
and safe to call from any context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Tier = Literal[
    "serve_existing",
    "serve_sibling",
    "tier1_ads_body",
    "tier3_docling",
    "abstract_only",
]


@dataclass(frozen=True)
class RouteInput:
    """Inputs required to route a full-text request.

    All fields reflect metadata known at routing time; the routing function
    performs no IO. ``sibling_row_source`` is the ``source`` value of an
    existing ``papers_fulltext`` row for a sibling identifier (e.g. an arXiv
    preprint row linked to a published bibcode) or ``None`` if no sibling row
    exists.
    """

    bibcode: str
    has_fulltext_row: bool
    sibling_row_source: str | None
    has_ads_body: bool
    doctype: str | None
    doi: str | None
    openalex_has_pdf_url: bool


@dataclass(frozen=True)
class RouteDecision:
    """Result of routing: which tier should handle the request, and why."""

    tier: Tier
    reason: str
    source_hint: str | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LaTeX-derived sibling sources eligible for serve_sibling (ADR-006).
# Non-LaTeX siblings (s2orc, ads_body, docling) fall through to subsequent
# rules rather than being served as siblings.
LATEX_SIBLING_SOURCES: frozenset[str] = frozenset({"ar5iv", "arxiv_local"})

# Doctypes eligible for Tier 3 (Docling on OpenAlex PDFs).
R2_ELIGIBLE_DOCTYPES: frozenset[str] = frozenset({"article", "eprint", "review"})


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_fulltext_request(inp: RouteInput) -> RouteDecision:
    """Route a full-text request to the correct tier.

    See module docstring for the full rule table. Rules are evaluated in
    order and the first match wins.
    """
    # Rule 1: existing row short-circuits everything.
    if inp.has_fulltext_row:
        return RouteDecision(
            tier="serve_existing",
            reason="existing papers_fulltext row present",
            source_hint=None,
        )

    # Rule 2: LaTeX-derived sibling row.
    if inp.sibling_row_source in LATEX_SIBLING_SOURCES:
        return RouteDecision(
            tier="serve_sibling",
            reason=f"LaTeX-derived sibling row present (source={inp.sibling_row_source})",
            source_hint=inp.sibling_row_source,
        )

    # Rule 3: ADS body available — ALWAYS beats Tier 3.
    if inp.has_ads_body:
        return RouteDecision(
            tier="tier1_ads_body",
            reason="ADS body available; ADS body always wins over Tier 3",
            source_hint="ads_body",
        )

    # Rule 4: R2-eligible (doctype, DOI, OpenAlex PDF URL).
    if (
        inp.doctype in R2_ELIGIBLE_DOCTYPES
        and inp.doi is not None
        and inp.openalex_has_pdf_url
    ):
        return RouteDecision(
            tier="tier3_docling",
            reason="R2-eligible: doctype/doi/openalex_pdf_url all satisfied",
            source_hint="openalex",
        )

    # Rule 5: fall-through.
    return RouteDecision(
        tier="abstract_only",
        reason="no full-text source available; serving abstract only",
        source_hint=None,
    )
