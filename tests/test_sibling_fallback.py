"""Tests for ``read_fulltext_with_sibling_fallback`` (PRD R3).

The function is pure: all IO is injected via ``fetch_row``,
``fetch_aliases``, and ``fetch_canonical_url``. These tests use
dict-backed closures and lambdas; no DB or HTTP is touched.
"""

from __future__ import annotations

import pytest

from scix.search import (
    LATEX_DERIVED_SOURCES,
    read_fulltext_with_sibling_fallback,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_fetchers(
    rows: dict[str, dict],
    aliases: dict[str, list[str]],
    urls: dict[str, str] | None = None,
):
    """Build the three injected callables from dict-backed tables.

    Also returns a ``calls`` dict so tests can assert which fetchers
    were invoked and how often.
    """
    calls: dict[str, list[str]] = {
        "fetch_row": [],
        "fetch_aliases": [],
        "fetch_canonical_url": [],
    }

    def fetch_row(bibcode: str):
        calls["fetch_row"].append(bibcode)
        return rows.get(bibcode)

    def fetch_aliases(bibcode: str):
        calls["fetch_aliases"].append(bibcode)
        return list(aliases.get(bibcode, []))

    def fetch_canonical_url(bibcode: str):
        calls["fetch_canonical_url"].append(bibcode)
        if urls is None:
            return f"https://example.test/{bibcode}"
        return urls.get(bibcode)

    return fetch_row, fetch_aliases, fetch_canonical_url, calls


# ---------------------------------------------------------------------------
# Constant sanity
# ---------------------------------------------------------------------------


def test_latex_derived_sources_constant_is_exactly_the_expected_frozenset():
    """AC #3: module defines the required constant."""
    assert LATEX_DERIVED_SOURCES == frozenset({"ar5iv", "arxiv_local"})
    assert isinstance(LATEX_DERIVED_SOURCES, frozenset)


# ---------------------------------------------------------------------------
# Rule 1: direct hit
# ---------------------------------------------------------------------------


def test_direct_hit_returns_row_without_touching_aliases():
    row = {"bibcode": "2020ApJ...900..100A", "source": "s2orc", "body": "..."}
    rows = {"2020ApJ...900..100A": row}
    fetch_row, fetch_aliases, fetch_canonical_url, calls = _make_fetchers(
        rows=rows, aliases={}
    )

    result = read_fulltext_with_sibling_fallback(
        "2020ApJ...900..100A", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result == {"hit": True, "row": row, "sibling": None}
    assert calls["fetch_row"] == ["2020ApJ...900..100A"]
    # Direct hit must short-circuit — no alias / URL lookups.
    assert calls["fetch_aliases"] == []
    assert calls["fetch_canonical_url"] == []


# ---------------------------------------------------------------------------
# Rule 3: LaTeX-derived sibling hit
# ---------------------------------------------------------------------------


def test_ar5iv_sibling_hit_returns_row_with_served_from_and_canonical_url():
    ar5iv_row = {
        "bibcode": "arXiv:2001.01234",
        "source": "ar5iv",
        "body": "LaTeX body",
    }
    rows = {"arXiv:2001.01234": ar5iv_row}  # primary bibcode absent
    aliases = {"2020ApJ...900..100A": ["arXiv:2001.01234"]}
    urls = {"arXiv:2001.01234": "https://ar5iv.example/arXiv:2001.01234"}
    fetch_row, fetch_aliases, fetch_canonical_url, calls = _make_fetchers(
        rows=rows, aliases=aliases, urls=urls
    )

    result = read_fulltext_with_sibling_fallback(
        "2020ApJ...900..100A", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result["hit"] is True
    assert result["row"] is ar5iv_row
    assert result["sibling"] == "arXiv:2001.01234"
    assert result["served_from_sibling_bibcode"] == "arXiv:2001.01234"
    assert result["canonical_url"] == "https://ar5iv.example/arXiv:2001.01234"
    assert calls["fetch_canonical_url"] == ["arXiv:2001.01234"]


def test_arxiv_local_sibling_hit_returns_row_with_served_from_and_canonical_url():
    local_row = {
        "bibcode": "arXiv:2001.01234",
        "source": "arxiv_local",
        "body": "local LaTeX body",
    }
    rows = {"arXiv:2001.01234": local_row}
    aliases = {"2020ApJ...900..100A": ["arXiv:2001.01234"]}
    urls = {"arXiv:2001.01234": "https://local.example/arXiv:2001.01234"}
    fetch_row, fetch_aliases, fetch_canonical_url, _calls = _make_fetchers(
        rows=rows, aliases=aliases, urls=urls
    )

    result = read_fulltext_with_sibling_fallback(
        "2020ApJ...900..100A", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result == {
        "hit": True,
        "row": local_row,
        "sibling": "arXiv:2001.01234",
        "served_from_sibling_bibcode": "arXiv:2001.01234",
        "canonical_url": "https://local.example/arXiv:2001.01234",
    }


# ---------------------------------------------------------------------------
# Rule 4: non-LaTeX sibling → miss-with-hint, no row propagation
# ---------------------------------------------------------------------------


def test_s2orc_sibling_returns_miss_with_hint_and_does_not_include_row():
    s2orc_row = {
        "bibcode": "SIBLING_BIB",
        "source": "s2orc",
        "body": "...",
    }
    rows = {"SIBLING_BIB": s2orc_row}
    aliases = {"REQUESTED_BIB": ["SIBLING_BIB"]}
    fetch_row, fetch_aliases, fetch_canonical_url, calls = _make_fetchers(
        rows=rows, aliases=aliases
    )

    result = read_fulltext_with_sibling_fallback(
        "REQUESTED_BIB", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result == {
        "hit": False,
        "miss_with_hint": True,
        "fulltext_available_under_sibling": "SIBLING_BIB",
        "hint": "call read_paper(bibcode=SIBLING_BIB)",
    }
    # Critical: the sibling's row MUST NOT leak into the response.
    assert "row" not in result
    # Canonical URL is a LaTeX-path concern only; don't fetch it on miss.
    assert calls["fetch_canonical_url"] == []


@pytest.mark.parametrize("non_latex_source", ["s2orc", "ads_body", "docling"])
def test_non_latex_sibling_never_propagates_row(non_latex_source: str):
    """AC #6: non-LaTeX sources never cause row propagation."""
    sibling_row = {
        "bibcode": "SIB",
        "source": non_latex_source,
        "body": "...",
    }
    rows = {"SIB": sibling_row}
    aliases = {"REQ": ["SIB"]}
    fetch_row, fetch_aliases, fetch_canonical_url, _calls = _make_fetchers(
        rows=rows, aliases=aliases
    )

    result = read_fulltext_with_sibling_fallback(
        "REQ", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result["hit"] is False
    assert result["miss_with_hint"] is True
    assert result["fulltext_available_under_sibling"] == "SIB"
    assert "row" not in result
    assert "served_from_sibling_bibcode" not in result
    assert "canonical_url" not in result


# ---------------------------------------------------------------------------
# Cycle guard: fetch_aliases returns the requested bibcode
# ---------------------------------------------------------------------------


def test_fetch_aliases_returning_self_does_not_cause_recursion_or_cycle():
    """AC #4: if aliases include the requested bibcode itself, we must not
    call fetch_row on it twice and must not recurse/cycle.
    """
    rows: dict[str, dict] = {}  # nothing exists
    aliases = {"REQ": ["REQ", "REQ", "SIB_NONE"]}
    fetch_row, fetch_aliases, fetch_canonical_url, calls = _make_fetchers(
        rows=rows, aliases=aliases
    )

    result = read_fulltext_with_sibling_fallback(
        "REQ", fetch_row, fetch_aliases, fetch_canonical_url
    )

    # Complete miss (no sibling had data either).
    assert result == {"hit": False, "miss_with_hint": False}
    # fetch_row called on REQ exactly once (rule 1), and on SIB_NONE once.
    # Critically, REQ must NOT appear twice — cycle guard working.
    assert calls["fetch_row"].count("REQ") == 1
    assert "SIB_NONE" in calls["fetch_row"]


def test_duplicate_siblings_are_deduplicated():
    rows: dict[str, dict] = {}
    aliases = {"REQ": ["SIB", "SIB", "SIB"]}
    fetch_row, fetch_aliases, fetch_canonical_url, calls = _make_fetchers(
        rows=rows, aliases=aliases
    )

    read_fulltext_with_sibling_fallback(
        "REQ", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert calls["fetch_row"].count("SIB") == 1


# ---------------------------------------------------------------------------
# Rule 5: complete miss
# ---------------------------------------------------------------------------


def test_complete_miss_returns_hit_false_miss_with_hint_false():
    fetch_row, fetch_aliases, fetch_canonical_url, _calls = _make_fetchers(
        rows={}, aliases={}
    )

    result = read_fulltext_with_sibling_fallback(
        "REQ", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result == {"hit": False, "miss_with_hint": False}


# ---------------------------------------------------------------------------
# LaTeX wins over non-LaTeX when both are present
# ---------------------------------------------------------------------------


def test_latex_sibling_wins_over_non_latex_sibling():
    ar5iv_row = {"bibcode": "LATEX_SIB", "source": "ar5iv", "body": "tex"}
    s2orc_row = {"bibcode": "S2_SIB", "source": "s2orc", "body": "pdf"}
    rows = {"LATEX_SIB": ar5iv_row, "S2_SIB": s2orc_row}
    # Put the non-LaTeX sibling first to verify ordering doesn't beat
    # the LaTeX-preference rule.
    aliases = {"REQ": ["S2_SIB", "LATEX_SIB"]}
    urls = {"LATEX_SIB": "https://ar5iv.example/LATEX_SIB"}
    fetch_row, fetch_aliases, fetch_canonical_url, _calls = _make_fetchers(
        rows=rows, aliases=aliases, urls=urls
    )

    result = read_fulltext_with_sibling_fallback(
        "REQ", fetch_row, fetch_aliases, fetch_canonical_url
    )

    assert result["hit"] is True
    assert result["row"] is ar5iv_row
    assert result["served_from_sibling_bibcode"] == "LATEX_SIB"
    assert result["canonical_url"] == "https://ar5iv.example/LATEX_SIB"
