"""Tests for scix.jit.disambiguator — query-time entity disambiguation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest

from scix.jit.disambiguator import (
    EntityCandidate,
    MentionDisambiguation,
    disambiguate_query,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_conn(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a mock psycopg.Connection whose cursor.fetchall() returns ``rows``.

    The mock is shaped to satisfy psycopg's ``with conn.cursor(row_factory=...)``
    context-manager idiom used in the module under test.
    """
    conn = MagicMock(spec=psycopg.Connection)
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # Expose the inner cursor for assertion access.
    conn._fake_cursor = cursor  # type: ignore[attr-defined]
    return conn


def _row(
    *,
    matched_ngram: str,
    entity_id: int,
    canonical_name: str,
    entity_type: str,
    paper_count: int,
) -> dict[str, Any]:
    return {
        "matched_ngram": matched_ngram,
        "entity_id": entity_id,
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "paper_count": paper_count,
    }


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


class TestDataclassInvariants:
    def test_entity_candidate_frozen(self) -> None:
        c = EntityCandidate(
            entity_id=1,
            entity_type="mission",
            display="Hubble Space Telescope",
            score=1.0,
            paper_count=45000,
        )
        with pytest.raises(AttributeError):
            c.score = 0.5  # type: ignore[misc]

    def test_mention_disambiguation_frozen(self) -> None:
        m = MentionDisambiguation(
            mention="JWST",
            ambiguous=False,
            candidates=(),
            default_type=None,
        )
        with pytest.raises(AttributeError):
            m.ambiguous = True  # type: ignore[misc]

    def test_candidates_field_is_tuple(self) -> None:
        m = MentionDisambiguation(
            mention="x",
            ambiguous=False,
            candidates=(),
            default_type=None,
        )
        assert isinstance(m.candidates, tuple)


# ---------------------------------------------------------------------------
# Acceptance criterion 6(a) — unambiguous mention
# ---------------------------------------------------------------------------


class TestUnambiguousMention:
    def test_jwst_nirspec_query_all_unambiguous(self) -> None:
        # Only one candidate per mention — both JWST and NIRSpec are unique.
        rows = [
            _row(
                matched_ngram="jwst",
                entity_id=1,
                canonical_name="James Webb Space Telescope",
                entity_type="mission",
                paper_count=12000,
            ),
            _row(
                matched_ngram="nirspec",
                entity_id=2,
                canonical_name="NIRSpec",
                entity_type="instrument",
                paper_count=3400,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "JWST NIRSpec spectroscopy")

        assert len(results) == 2
        for r in results:
            assert r.ambiguous is False
            assert len(r.candidates) == 1
            assert r.default_type == r.candidates[0].entity_type

    def test_single_candidate_score_is_one(self) -> None:
        rows = [
            _row(
                matched_ngram="jwst",
                entity_id=1,
                canonical_name="James Webb Space Telescope",
                entity_type="mission",
                paper_count=12000,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "JWST")

        assert len(results) == 1
        assert results[0].candidates[0].score == 1.0
        assert results[0].candidates[0].paper_count == 12000


# ---------------------------------------------------------------------------
# Acceptance criterion 6(b) — ambiguous mention across types
# ---------------------------------------------------------------------------


class TestAmbiguousMentionAcrossTypes:
    def test_hubble_query_ambiguous_with_two_types(self) -> None:
        # Mimic PRD example: "Hubble" maps to both a mission and a person,
        # each with > 10 papers and different entity_type.
        rows = [
            _row(
                matched_ngram="hubble",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
            _row(
                matched_ngram="hubble",
                entity_id=102,
                canonical_name="Edwin Hubble",
                entity_type="person",
                paper_count=120,
            ),
            # M31 resolves unambiguously.
            _row(
                matched_ngram="m31",
                entity_id=200,
                canonical_name="Messier 31",
                entity_type="galaxy",
                paper_count=8800,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "Hubble observations of M31")

        assert len(results) == 2
        hubble = next(r for r in results if r.mention.lower() == "hubble")
        m31 = next(r for r in results if r.mention.lower() == "m31")

        assert hubble.ambiguous is True
        assert len(hubble.candidates) == 2
        # Sorted by paper_count DESC.
        assert hubble.candidates[0].paper_count == 45000
        assert hubble.candidates[0].entity_type == "mission"
        assert hubble.candidates[1].paper_count == 120
        assert hubble.candidates[1].entity_type == "person"
        # Scores normalized to the top candidate.
        assert hubble.candidates[0].score == 1.0
        assert hubble.candidates[1].score == pytest.approx(120 / 45000)
        # default_type follows the top candidate.
        assert hubble.default_type == "mission"

        assert m31.ambiguous is False

    def test_mention_preserves_original_casing(self) -> None:
        rows = [
            _row(
                matched_ngram="hubble",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "Hubble images")
        assert results[0].mention == "Hubble"


# ---------------------------------------------------------------------------
# Acceptance criterion 6(c) — below-threshold candidate excluded from
# the ambiguity verdict (but still returned in candidates)
# ---------------------------------------------------------------------------


class TestBelowThresholdCandidate:
    def test_one_candidate_below_threshold_not_ambiguous(self) -> None:
        # Two candidates, different types — but the "person" is below threshold.
        rows = [
            _row(
                matched_ngram="hubble",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
            _row(
                matched_ngram="hubble",
                entity_id=102,
                canonical_name="Edwin Hubble",
                entity_type="person",
                paper_count=5,  # below default min_paper_count=10
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "Hubble observations")
        assert len(results) == 1
        hubble = results[0]

        # Both candidates returned, sorted.
        assert len(hubble.candidates) == 2
        # But NOT flagged as ambiguous because only one candidate is above
        # the threshold.
        assert hubble.ambiguous is False

    def test_configurable_min_paper_count(self) -> None:
        rows = [
            _row(
                matched_ngram="hubble",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
            _row(
                matched_ngram="hubble",
                entity_id=102,
                canonical_name="Edwin Hubble",
                entity_type="person",
                paper_count=50,
            ),
        ]
        # With min_paper_count=10 → ambiguous (both above threshold).
        conn1 = _make_conn(rows)
        results1 = disambiguate_query(conn1, "Hubble", min_paper_count=10)
        assert results1[0].ambiguous is True

        # With min_paper_count=100 → not ambiguous (person below threshold).
        conn2 = _make_conn(rows)
        results2 = disambiguate_query(conn2, "Hubble", min_paper_count=100)
        assert results2[0].ambiguous is False

    def test_same_type_collision_not_ambiguous(self) -> None:
        # Two candidates both above threshold but SAME entity_type → not ambiguous.
        rows = [
            _row(
                matched_ngram="vega",
                entity_id=301,
                canonical_name="Vega (alpha Lyr)",
                entity_type="star",
                paper_count=5000,
            ),
            _row(
                matched_ngram="vega",
                entity_id=302,
                canonical_name="Vega in Centaurus",
                entity_type="star",
                paper_count=800,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "Vega spectroscopy")
        assert len(results) == 1
        assert results[0].ambiguous is False


# ---------------------------------------------------------------------------
# Acceptance criterion 6(d) — case-insensitive alias match
# ---------------------------------------------------------------------------


class TestCaseInsensitiveMatch:
    def test_query_lowercased_before_sql_lookup(self) -> None:
        rows = [
            _row(
                matched_ngram="hst",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
        ]
        conn = _make_conn(rows)

        # Query in UPPER case; mention preserves case; DB receives lowercase ngrams.
        disambiguate_query(conn, "HST ACS imaging")

        # The execute call should have been made with lowercased ngrams.
        executed = conn._fake_cursor.execute.call_args
        params = executed[0][1]  # second positional arg is the params dict
        ngrams = params["ngrams"]
        assert "hst" in ngrams
        assert "acs" in ngrams
        assert "hst imaging" not in ngrams  # "HST ACS imaging" tokens stay in order
        # None of the ngrams carry uppercase letters.
        assert all(g == g.lower() for g in ngrams)

    def test_sql_uses_lower_on_both_sides(self) -> None:
        """SQL must apply lower() on canonical_name and alias for index hits."""
        rows: list[dict[str, Any]] = []
        conn = _make_conn(rows)

        disambiguate_query(conn, "Hubble")
        executed = conn._fake_cursor.execute.call_args
        sql = executed[0][0]
        assert "lower(e.canonical_name)" in sql
        assert "lower(ea.alias)" in sql


# ---------------------------------------------------------------------------
# Acceptance criterion 6(e) — empty query
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    def test_empty_string_returns_empty_list(self) -> None:
        conn = _make_conn([])
        assert disambiguate_query(conn, "") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        conn = _make_conn([])
        assert disambiguate_query(conn, "   \t\n  ") == []

    def test_empty_query_does_not_touch_db(self) -> None:
        conn = _make_conn([])
        disambiguate_query(conn, "")
        assert conn.cursor.called is False

    def test_punctuation_only_returns_empty_list(self) -> None:
        conn = _make_conn([])
        # _TOKEN_RE finds no alphanumeric tokens.
        assert disambiguate_query(conn, "!?.,;") == []


# ---------------------------------------------------------------------------
# Additional coverage: candidate dedup, ngram ordering, no-hit mentions
# ---------------------------------------------------------------------------


class TestExtras:
    def test_deduplicates_same_entity_matched_via_alias_and_canonical(self) -> None:
        # Same entity shows up twice because query text matched both its
        # canonical name and an alias in the same ngram window.
        rows = [
            _row(
                matched_ngram="hubble space telescope",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
            _row(
                matched_ngram="hubble space telescope",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "Hubble Space Telescope observations")
        # Find the 3-gram mention (preserves original casing).
        hit = next(r for r in results if r.mention.lower() == "hubble space telescope")
        assert len(hit.candidates) == 1
        assert hit.candidates[0].entity_id == 101

    def test_mentions_ordered_by_first_appearance(self) -> None:
        rows = [
            _row(
                matched_ngram="jwst",
                entity_id=1,
                canonical_name="James Webb Space Telescope",
                entity_type="mission",
                paper_count=12000,
            ),
            _row(
                matched_ngram="nirspec",
                entity_id=2,
                canonical_name="NIRSpec",
                entity_type="instrument",
                paper_count=3400,
            ),
        ]
        conn = _make_conn(rows)

        results = disambiguate_query(conn, "NIRSpec data from JWST")
        assert [r.mention.lower() for r in results] == ["nirspec", "jwst"]

    def test_query_with_no_matches_returns_empty_list(self) -> None:
        # Mentions extracted but DB returns zero rows.
        conn = _make_conn([])
        assert disambiguate_query(conn, "unrelated free text here") == []

    def test_mention_deduped_by_lowercase(self) -> None:
        rows = [
            _row(
                matched_ngram="hubble",
                entity_id=101,
                canonical_name="Hubble Space Telescope",
                entity_type="mission",
                paper_count=45000,
            ),
        ]
        conn = _make_conn(rows)

        # "Hubble" appears twice in different casings; only one mention emitted.
        results = disambiguate_query(conn, "Hubble hubble HUBBLE")
        assert len(results) == 1
        # First-seen surface preserved.
        assert results[0].mention == "Hubble"

    def test_all_zero_paper_counts_score_zero(self) -> None:
        rows = [
            _row(
                matched_ngram="foo",
                entity_id=1,
                canonical_name="Foo",
                entity_type="concept",
                paper_count=0,
            ),
            _row(
                matched_ngram="foo",
                entity_id=2,
                canonical_name="Foo Alternative",
                entity_type="concept",
                paper_count=0,
            ),
        ]
        conn = _make_conn(rows)
        results = disambiguate_query(conn, "foo")
        assert len(results) == 1
        assert all(c.score == 0.0 for c in results[0].candidates)
        assert results[0].ambiguous is False  # neither above threshold
