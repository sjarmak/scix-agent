"""Tests for ``build_read_paper_response`` (PRD R3/R6/R16).

The builder is pure: it assembles the schema_version=2 ``read_paper``
response envelope from an already-resolved ``sibling_result`` plus an
in-memory suppress-set. No DB, no HTTP, no model calls.

Reference: ``docs/mcp_tool_contracts.md`` (the four canonical JSON
examples) and ``docs/ADR/006_arxiv_licensing.md`` Addendum (cross-bibcode
LaTeX propagation).
"""

from __future__ import annotations

import pytest

from scix.read_paper_response import MAX_SNIPPET_CHARS, build_read_paper_response
from scix.search import LATEX_DERIVED_SOURCES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REQUESTED = "2024ApJ...961...42S"
SIBLING = "2023arXiv231012345S"
ARXIV_URL = "https://arxiv.org/abs/2310.12345"
DOI_URL = "https://doi.org/10.3847/1538-4357/ad1234"


def _v1_base(**overrides) -> dict:
    base = {
        "bibcode": REQUESTED,
        "title": "Example Paper",
        "abstract": "We present a worked example ...",
    }
    base.update(overrides)
    return base


def _direct_hit_result(**row_overrides) -> dict:
    row = {
        "source": "publisher_body",
        "body": "Section 1. Introduction ...",
        "source_version": "published-v1",
        "canonical_url": DOI_URL,
    }
    row.update(row_overrides)
    return {"hit": True, "row": row, "sibling": None}


def _sibling_hit_result(body: str = "Section 1. Introduction ...", source: str = "ar5iv") -> dict:
    return {
        "hit": True,
        "row": {
            "source": source,
            "body": body,
            "source_version": "arxiv-v2",
        },
        "sibling": SIBLING,
        "served_from_sibling_bibcode": SIBLING,
        "canonical_url": ARXIV_URL,
    }


def _miss_with_hint_result() -> dict:
    return {
        "hit": False,
        "miss_with_hint": True,
        "fulltext_available_under_sibling": SIBLING,
        "hint": f"call read_paper(bibcode={SIBLING})",
    }


def _pure_miss_result() -> dict:
    return {"hit": False, "miss_with_hint": False}


EMPTY_SUPPRESS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Invariants (schema_version, source_bibcode, backward compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_result",
    [
        _direct_hit_result,
        _sibling_hit_result,
        _miss_with_hint_result,
        _pure_miss_result,
    ],
)
def test_every_response_has_schema_version_2(make_result):
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=make_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text="abs",
        v1_base_fields=_v1_base(),
    )
    assert resp["schema_version"] == 2


@pytest.mark.parametrize(
    "make_result",
    [
        _direct_hit_result,
        _sibling_hit_result,
        _miss_with_hint_result,
        _pure_miss_result,
    ],
)
def test_every_response_has_source_bibcode(make_result):
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=make_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text="abs",
        v1_base_fields=_v1_base(),
    )
    assert "source_bibcode" in resp
    assert isinstance(resp["source_bibcode"], str) and resp["source_bibcode"]


def test_direct_hit_preserves_all_v1_base_fields():
    v1 = _v1_base(extra_v1_field="keep-me", nested={"ok": True})
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_direct_hit_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text="abs",
        v1_base_fields=v1,
    )
    # Every v1 key must round-trip unchanged (additive-only contract).
    for key, value in v1.items():
        assert resp[key] == value


# ---------------------------------------------------------------------------
# Scenario 1: direct hit
# ---------------------------------------------------------------------------


def test_direct_hit_source_bibcode_equals_requested():
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_direct_hit_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text=None,
        v1_base_fields=_v1_base(),
    )
    assert resp["source_bibcode"] == REQUESTED
    assert "served_from_sibling_bibcode" not in resp
    assert resp["body"] == "Section 1. Introduction ..."
    assert resp["source"] == "publisher_body"
    assert resp["source_version"] == "published-v1"
    assert resp["suppressed_by_publisher"] is False


# ---------------------------------------------------------------------------
# Scenario 2: sibling hit
# ---------------------------------------------------------------------------


def test_sibling_hit_includes_served_from_sibling_and_canonical_url():
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_sibling_hit_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text=None,
        v1_base_fields=_v1_base(),
    )
    assert resp["served_from_sibling_bibcode"] == SIBLING
    assert resp["canonical_url"] == ARXIV_URL
    assert resp["source_bibcode"] == SIBLING
    assert resp["source"] == "ar5iv"
    assert resp["source_version"] == "arxiv-v2"


# ---------------------------------------------------------------------------
# Scenario 3: miss-with-hint
# ---------------------------------------------------------------------------


def test_miss_with_hint_has_hint_fields_and_no_body():
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_miss_with_hint_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text="We present ...",
        v1_base_fields=_v1_base(),
    )
    assert resp["fulltext_available_under_sibling"] == SIBLING
    assert resp["hint"] == f"call read_paper(bibcode={SIBLING})"
    assert resp["source"] == "abstract"
    assert resp["source_bibcode"] == REQUESTED
    assert "body" not in resp
    assert "sections" not in resp


# ---------------------------------------------------------------------------
# Scenario 4: pure miss
# ---------------------------------------------------------------------------


def test_pure_miss_is_abstract_only():
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_pure_miss_result(),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text="We present ...",
        v1_base_fields=_v1_base(),
    )
    assert resp["source"] == "abstract"
    assert resp["source_bibcode"] == REQUESTED
    assert "body" not in resp
    assert "fulltext_available_under_sibling" not in resp


# ---------------------------------------------------------------------------
# Rule (f): publisher suppression override
# ---------------------------------------------------------------------------


def test_suppressed_publisher_forces_abstract_strips_body_and_sets_flag():
    v1 = _v1_base(publisher="Elsevier")
    suppress = frozenset({"elsevier"})
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_direct_hit_result(),
        suppress_set=suppress,
        abstract_text=None,
        v1_base_fields=v1,
    )
    assert resp["suppressed_by_publisher"] is True
    assert resp["source"] == "abstract"
    assert "body" not in resp
    assert "sections" not in resp
    assert "source_version" not in resp
    # v1 base-field (publisher) still round-trips.
    assert resp["publisher"] == "Elsevier"


def test_suppression_overrides_sibling_hit():
    # Publisher lives on the sibling row, not v1_base_fields.
    sibling_res = _sibling_hit_result()
    sibling_res["row"]["publisher"] = "ElsevierBV"
    suppress = frozenset({"elsevierbv"})
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=sibling_res,
        suppress_set=suppress,
        abstract_text=None,
        v1_base_fields=_v1_base(),
    )
    assert resp["suppressed_by_publisher"] is True
    assert resp["source"] == "abstract"
    assert "body" not in resp


# ---------------------------------------------------------------------------
# Rule (g)/(h): cross-bibcode LaTeX truncation
# ---------------------------------------------------------------------------


def test_cross_bibcode_latex_truncated_to_500_chars():
    assert MAX_SNIPPET_CHARS == 500
    big_body = "x" * 2000
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=_sibling_hit_result(body=big_body, source="ar5iv"),
        suppress_set=EMPTY_SUPPRESS,
        abstract_text=None,
        v1_base_fields=_v1_base(),
    )
    assert resp["source"] in LATEX_DERIVED_SOURCES
    assert resp["source_bibcode"] != REQUESTED
    assert len(resp["body"]) <= MAX_SNIPPET_CHARS
    assert len(resp["body"]) == MAX_SNIPPET_CHARS


def test_same_bibcode_latex_not_truncated():
    # Direct hit whose row is already LaTeX-derived (unusual but valid):
    # source_bibcode == requested_bibcode, so the cross-bibcode truncation
    # guard must NOT fire.
    big_body = "y" * 2000
    direct = {
        "hit": True,
        "row": {
            "source": "ar5iv",
            "body": big_body,
            "source_version": "arxiv-v2",
            "canonical_url": ARXIV_URL,
        },
        "sibling": None,
    }
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=direct,
        suppress_set=EMPTY_SUPPRESS,
        abstract_text=None,
        v1_base_fields=_v1_base(),
    )
    assert resp["source"] == "ar5iv"
    assert resp["source_bibcode"] == REQUESTED
    assert resp["body"] == big_body
    assert len(resp["body"]) == 2000


def test_cross_bibcode_non_latex_not_truncated():
    # A publisher_body sibling-hit body should NOT be truncated — the
    # <=500-char rule only applies to LaTeX-derived sources.
    big_body = "z" * 1200
    sres = {
        "hit": True,
        "row": {
            "source": "publisher_body",
            "body": big_body,
            "source_version": "published-v1",
        },
        "sibling": SIBLING,
        "served_from_sibling_bibcode": SIBLING,
        "canonical_url": DOI_URL,
    }
    resp = build_read_paper_response(
        requested_bibcode=REQUESTED,
        sibling_result=sres,
        suppress_set=EMPTY_SUPPRESS,
        abstract_text=None,
        v1_base_fields=_v1_base(),
    )
    assert resp["body"] == big_body  # untouched; not LaTeX-derived
