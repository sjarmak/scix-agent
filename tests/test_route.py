"""Tests for scix.sources.route.

Covers every branch of the 5-rule decision tree plus the four explicit
non-negotiables from the work-unit spec:

- AC5: has_ads_body=True + R2-eligible metadata routes to tier1_ads_body,
       NOT tier3_docling.
- AC6: sibling_row_source='s2orc' does NOT trigger serve_sibling.
- AC7: no-doi + no-body + no-arxiv + no-existing row routes to abstract_only.
- AC8: has_fulltext_row=True short-circuits to serve_existing regardless of
       other fields.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest

from scix.sources.route import (
    LATEX_SIBLING_SOURCES,
    R2_ELIGIBLE_DOCTYPES,
    RouteDecision,
    RouteInput,
    Tier,
    route_fulltext_request,
)


def _mk(**overrides: Any) -> RouteInput:
    """Build a RouteInput with sensible defaults, overridden per-test."""
    defaults: dict[str, Any] = {
        "bibcode": "2023ApJ...100..123X",
        "has_fulltext_row": False,
        "sibling_row_source": None,
        "has_ads_body": False,
        "doctype": "article",
        "doi": None,
        "openalex_has_pdf_url": False,
    }
    defaults.update(overrides)
    return RouteInput(**defaults)


# ---------------------------------------------------------------------------
# Rule 1: serve_existing short-circuit
# ---------------------------------------------------------------------------


def test_existing_row_routes_to_serve_existing() -> None:
    decision = route_fulltext_request(_mk(has_fulltext_row=True))
    assert decision.tier == "serve_existing"
    assert decision.source_hint is None


def test_existing_row_short_circuits_all_other_flags() -> None:
    """AC8: has_fulltext_row=True wins regardless of other fields."""
    decision = route_fulltext_request(
        _mk(
            has_fulltext_row=True,
            sibling_row_source="ar5iv",
            has_ads_body=True,
            doctype="article",
            doi="10.1000/foo",
            openalex_has_pdf_url=True,
        )
    )
    assert decision.tier == "serve_existing"


# ---------------------------------------------------------------------------
# Rule 2: serve_sibling (LaTeX-derived only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["ar5iv", "arxiv_local"])
def test_latex_sibling_routes_to_serve_sibling(source: str) -> None:
    decision = route_fulltext_request(_mk(sibling_row_source=source))
    assert decision.tier == "serve_sibling"
    assert decision.source_hint == source


def test_s2orc_sibling_does_not_trigger_serve_sibling() -> None:
    """AC6: non-LaTeX sibling (s2orc) falls through to abstract_only."""
    decision = route_fulltext_request(_mk(sibling_row_source="s2orc"))
    assert decision.tier != "serve_sibling"
    assert decision.tier == "abstract_only"


def test_s2orc_sibling_plus_ads_body_routes_to_tier1() -> None:
    """Non-LaTeX sibling falls through; ADS body then wins."""
    decision = route_fulltext_request(
        _mk(sibling_row_source="s2orc", has_ads_body=True)
    )
    assert decision.tier == "tier1_ads_body"


@pytest.mark.parametrize("source", ["ads_body", "docling"])
def test_other_nonlatex_siblings_fall_through(source: str) -> None:
    decision = route_fulltext_request(_mk(sibling_row_source=source))
    assert decision.tier != "serve_sibling"


# ---------------------------------------------------------------------------
# Rule 3: tier1_ads_body (ADS body NEVER goes to Tier 3)
# ---------------------------------------------------------------------------


def test_ads_body_with_r2_eligible_routes_to_tier1_not_tier3() -> None:
    """AC5: has_ads_body=True ALWAYS wins over Tier 3, even when R2-eligible."""
    decision = route_fulltext_request(
        _mk(
            has_ads_body=True,
            doctype="article",
            doi="10.1000/foo",
            openalex_has_pdf_url=True,
        )
    )
    assert decision.tier == "tier1_ads_body"
    assert decision.tier != "tier3_docling"
    assert decision.source_hint == "ads_body"


# ---------------------------------------------------------------------------
# Rule 4: tier3_docling (R2 eligibility)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doctype", ["article", "eprint", "review"])
def test_r2_eligible_routes_to_tier3(doctype: str) -> None:
    decision = route_fulltext_request(
        _mk(doctype=doctype, doi="10.1000/foo", openalex_has_pdf_url=True)
    )
    assert decision.tier == "tier3_docling"
    assert decision.source_hint == "openalex"


def test_r2_ineligible_doctype_routes_to_abstract_only() -> None:
    decision = route_fulltext_request(
        _mk(doctype="abstract", doi="10.1000/foo", openalex_has_pdf_url=True)
    )
    assert decision.tier == "abstract_only"


def test_r2_missing_doi_routes_to_abstract_only() -> None:
    decision = route_fulltext_request(
        _mk(doctype="article", doi=None, openalex_has_pdf_url=True)
    )
    assert decision.tier == "abstract_only"


def test_r2_missing_pdf_url_routes_to_abstract_only() -> None:
    decision = route_fulltext_request(
        _mk(doctype="article", doi="10.1000/foo", openalex_has_pdf_url=False)
    )
    assert decision.tier == "abstract_only"


# ---------------------------------------------------------------------------
# Rule 5: abstract_only fall-through
# ---------------------------------------------------------------------------


def test_nothing_available_routes_to_abstract_only() -> None:
    """AC7: no-doi + no-body + no-arxiv + no-existing → abstract_only."""
    decision = route_fulltext_request(
        _mk(
            has_fulltext_row=False,
            sibling_row_source=None,
            has_ads_body=False,
            doctype="article",
            doi=None,
            openalex_has_pdf_url=False,
        )
    )
    assert decision.tier == "abstract_only"
    assert decision.source_hint is None


# ---------------------------------------------------------------------------
# Type / invariant checks
# ---------------------------------------------------------------------------


def test_tier_literal_set_is_exactly_the_five_values() -> None:
    """AC3: the Tier Literal must enumerate exactly these five values."""
    assert set(get_args(Tier)) == {
        "serve_existing",
        "serve_sibling",
        "tier1_ads_body",
        "tier3_docling",
        "abstract_only",
    }


def test_route_input_and_decision_are_frozen() -> None:
    inp = _mk()
    with pytest.raises(Exception):
        inp.has_fulltext_row = True  # type: ignore[misc]
    dec = RouteDecision(tier="abstract_only", reason="x", source_hint=None)
    with pytest.raises(Exception):
        dec.tier = "serve_existing"  # type: ignore[misc]


def test_constants_are_frozensets_with_expected_members() -> None:
    assert isinstance(LATEX_SIBLING_SOURCES, frozenset)
    assert LATEX_SIBLING_SOURCES == {"ar5iv", "arxiv_local"}
    assert isinstance(R2_ELIGIBLE_DOCTYPES, frozenset)
    assert R2_ELIGIBLE_DOCTYPES == {"article", "eprint", "review"}
