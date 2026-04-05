"""Tests for entity resolver module.

Uses mock psycopg connections to avoid requiring a live PostgreSQL database.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scix.entity_resolver import EntityCandidate, EntityResolver, _deduplicate

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_conn(query_results: dict[str, list[dict]] | None = None) -> MagicMock:
    """Create a mock psycopg.Connection that returns predefined rows.

    query_results maps a SQL fragment (substring) to the list of dicts
    that fetchall() should return for queries containing that fragment.
    """
    if query_results is None:
        query_results = {}

    conn = MagicMock()
    cursor = MagicMock()

    def _execute(sql: str, params: dict | None = None) -> None:
        cursor._last_sql = sql
        cursor._last_params = params

    def _fetchall() -> list[dict]:
        sql = getattr(cursor, "_last_sql", "")
        for fragment, rows in query_results.items():
            if fragment in sql:
                return rows
        return []

    cursor.execute = _execute
    cursor.fetchall = _fetchall
    cursor.__enter__ = lambda self: self
    cursor.__exit__ = lambda self, *args: None

    conn.cursor = MagicMock(return_value=cursor)
    return conn


def _entity_row(
    *,
    id: int = 1,
    canonical_name: str = "Test Entity",
    entity_type: str = "target",
    source: str = "test",
    discipline: str | None = None,
    sim: float | None = None,
) -> dict:
    """Build a row dict matching the SELECT shape used by EntityResolver."""
    row = {
        "id": id,
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "source": source,
        "discipline": discipline,
    }
    if sim is not None:
        row["sim"] = sim
    return row


# ---------------------------------------------------------------------------
# EntityCandidate tests
# ---------------------------------------------------------------------------


class TestEntityCandidate:
    """EntityCandidate is a frozen dataclass with all required fields."""

    def test_fields(self) -> None:
        c = EntityCandidate(
            entity_id=1,
            canonical_name="Bennu",
            entity_type="target",
            source="physh",
            discipline="planetary",
            confidence=1.0,
            match_method="exact_canonical",
        )
        assert c.entity_id == 1
        assert c.canonical_name == "Bennu"
        assert c.entity_type == "target"
        assert c.source == "physh"
        assert c.discipline == "planetary"
        assert c.confidence == 1.0
        assert c.match_method == "exact_canonical"

    def test_frozen(self) -> None:
        c = EntityCandidate(
            entity_id=1,
            canonical_name="Bennu",
            entity_type="target",
            source="physh",
            discipline=None,
            confidence=1.0,
            match_method="exact_canonical",
        )
        with pytest.raises(AttributeError):
            c.confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Exact canonical match
# ---------------------------------------------------------------------------


class TestExactCanonicalMatch:
    """resolve() returns confidence 1.0 for exact canonical name matches."""

    def test_single_exact_match(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(
                        id=10,
                        canonical_name="Bennu",
                        entity_type="target",
                        source="physh",
                        discipline="planetary",
                    )
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Bennu")
        assert len(results) >= 1
        assert results[0].confidence == 1.0
        assert results[0].match_method == "exact_canonical"
        assert results[0].canonical_name == "Bennu"
        assert results[0].entity_id == 10

    def test_multiple_exact_matches(self) -> None:
        """Multiple entities with same canonical name (different type/source)."""
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(
                        id=1, canonical_name="Mars", source="physh", discipline="planetary"
                    ),
                    _entity_row(
                        id=2, canonical_name="Mars", source="wikidata", discipline="planetary"
                    ),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Mars")
        assert len(results) == 2
        assert all(r.confidence == 1.0 for r in results)
        assert all(r.match_method == "exact_canonical" for r in results)


# ---------------------------------------------------------------------------
# Alias match
# ---------------------------------------------------------------------------


class TestAliasMatch:
    """resolve() returns confidence 0.9 for alias matches."""

    def test_alias_match(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],  # no canonical match (FROM entities without JOIN)
                "FROM entity_aliases": [
                    _entity_row(
                        id=10,
                        canonical_name="(101955) Bennu",
                        entity_type="target",
                        source="physh",
                        discipline="planetary",
                    )
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("1999 RQ36")
        assert len(results) >= 1
        assert results[0].confidence == 0.9
        assert results[0].match_method == "alias"
        assert results[0].canonical_name == "(101955) Bennu"

    def test_mahli_alias(self) -> None:
        """MAHLI resolves to Mars Hand Lens Imager via alias."""
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [
                    _entity_row(
                        id=42,
                        canonical_name="Mars Hand Lens Imager",
                        entity_type="instrument",
                        source="physh",
                    )
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("MAHLI")
        assert len(results) >= 1
        assert results[0].canonical_name == "Mars Hand Lens Imager"
        assert results[0].match_method == "alias"


# ---------------------------------------------------------------------------
# Identifier match
# ---------------------------------------------------------------------------


class TestIdentifierMatch:
    """resolve() returns confidence 0.85 for identifier matches."""

    def test_identifier_match(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(id=10, canonical_name="Bennu"),
                ],
                "FROM entity_identifiers": [
                    _entity_row(id=10, canonical_name="Bennu"),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Q1234567")
        # Should have canonical match (1.0) and identifier match (0.85)
        # After dedup, keep the higher confidence one
        assert len(results) >= 1

    def test_identifier_only(self) -> None:
        """Identifier match when no canonical or alias match."""
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [],
                "FROM entity_identifiers": [
                    _entity_row(
                        id=99,
                        canonical_name="Chandra X-ray Observatory",
                        entity_type="mission",
                        source="wikidata",
                    )
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Q219615")
        assert len(results) == 1
        assert results[0].confidence == 0.85
        assert results[0].match_method == "identifier"


# ---------------------------------------------------------------------------
# Fuzzy match
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    """resolve() with fuzzy=True falls back to pg_trgm similarity."""

    def test_fuzzy_fallback(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [],
                "FROM entity_identifiers": [],
                "similarity": [
                    _entity_row(
                        id=10,
                        canonical_name="Bennu",
                        discipline="planetary",
                        sim=0.65,
                    ),
                    _entity_row(
                        id=20,
                        canonical_name="Benny",
                        discipline=None,
                        sim=0.45,
                    ),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Benu", fuzzy=True, fuzzy_threshold=0.3)
        assert len(results) == 2
        assert all(r.match_method == "fuzzy" for r in results)
        # Sorted by confidence DESC
        assert results[0].confidence == 0.65
        assert results[1].confidence == 0.45

    def test_fuzzy_not_used_when_exact_match_exists(self) -> None:
        """Fuzzy is skipped when earlier stages found results."""
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(id=10, canonical_name="Bennu"),
                ],
                "similarity": [
                    _entity_row(id=20, canonical_name="Benny", sim=0.5),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Bennu", fuzzy=True)
        # Should only have the exact match, fuzzy not triggered
        assert all(r.match_method == "exact_canonical" for r in results)

    def test_fuzzy_disabled_by_default(self) -> None:
        """Without fuzzy=True, no fuzzy results even if no other matches."""
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [],
                "FROM entity_identifiers": [],
                "similarity": [
                    _entity_row(id=10, canonical_name="Bennu", sim=0.65),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Benu")
        assert results == []


# ---------------------------------------------------------------------------
# Discipline ranking
# ---------------------------------------------------------------------------


class TestDisciplineRanking:
    """Discipline parameter boosts matching candidates by 0.05."""

    def test_discipline_boost(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(id=1, canonical_name="Mars", discipline="planetary"),
                    _entity_row(id=2, canonical_name="Mars", discipline="mythology"),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Mars", discipline="planetary")
        assert len(results) == 2
        # planetary Mars should be first (1.0 + 0.05 capped at 1.0 vs 1.0)
        # Both start at 1.0 but planetary gets boost (capped at 1.0)
        # Actually both are 1.0 for exact match; the planetary one stays 1.0
        # The mythology one stays 1.0 too. Let's verify the boost is applied.
        planetary = [r for r in results if r.discipline == "planetary"][0]
        mythology = [r for r in results if r.discipline == "mythology"][0]
        # With exact match (1.0), boost caps at 1.0
        assert planetary.confidence == 1.0
        assert mythology.confidence == 1.0

    def test_discipline_boost_on_alias_match(self) -> None:
        """Discipline boost is more visible on alias matches (0.9 -> 0.95)."""
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [
                    _entity_row(id=1, canonical_name="Mars Rover", discipline="planetary"),
                    _entity_row(id=2, canonical_name="Mars Bar", discipline="food"),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Mars Buggy", discipline="planetary")
        planetary = [r for r in results if r.discipline == "planetary"][0]
        food = [r for r in results if r.discipline == "food"][0]
        assert planetary.confidence == pytest.approx(0.95)  # 0.9 + 0.05
        assert food.confidence == 0.9
        # Planetary should sort first
        assert results[0].discipline == "planetary"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Deduplication keeps highest confidence per entity_id."""

    def test_dedup_keeps_highest_confidence(self) -> None:
        candidates = [
            EntityCandidate(
                entity_id=1,
                canonical_name="Bennu",
                entity_type="target",
                source="physh",
                discipline=None,
                confidence=0.85,
                match_method="identifier",
            ),
            EntityCandidate(
                entity_id=1,
                canonical_name="Bennu",
                entity_type="target",
                source="physh",
                discipline=None,
                confidence=1.0,
                match_method="exact_canonical",
            ),
        ]
        result = _deduplicate(candidates)
        assert len(result) == 1
        assert result[0].confidence == 1.0
        assert result[0].match_method == "exact_canonical"

    def test_dedup_preserves_different_entities(self) -> None:
        candidates = [
            EntityCandidate(
                entity_id=1,
                canonical_name="Mars",
                entity_type="target",
                source="a",
                discipline=None,
                confidence=1.0,
                match_method="exact_canonical",
            ),
            EntityCandidate(
                entity_id=2,
                canonical_name="Mars",
                entity_type="target",
                source="b",
                discipline=None,
                confidence=1.0,
                match_method="exact_canonical",
            ),
        ]
        result = _deduplicate(candidates)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Never LIMIT 1
# ---------------------------------------------------------------------------


class TestNeverLimit1:
    """Results are never limited to a single candidate."""

    def test_multiple_candidates_returned(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(id=1, canonical_name="Mars", source="a"),
                    _entity_row(id=2, canonical_name="Mars", source="b"),
                    _entity_row(id=3, canonical_name="Mars", source="c"),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("Mars")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    """Queries use lower() for case-insensitive matching."""

    def test_resolve_case_insensitive(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(id=1, canonical_name="Bennu"),
                ],
            }
        )
        resolver = EntityResolver(conn)
        # The mock doesn't actually do case comparison, but we verify
        # the resolver passes the mention through and SQL uses lower()
        results = resolver.resolve("bennu")
        assert len(results) >= 1
        assert results[0].canonical_name == "Bennu"


# ---------------------------------------------------------------------------
# Batch resolution
# ---------------------------------------------------------------------------


class TestResolveBatch:
    """resolve_batch returns dict mapping each mention to its candidates."""

    def test_batch_returns_dict(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [
                    _entity_row(id=1, canonical_name="Bennu"),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve_batch(["Bennu", "Mars"])
        assert isinstance(results, dict)
        assert "Bennu" in results
        assert "Mars" in results
        assert isinstance(results["Bennu"], list)
        assert isinstance(results["Mars"], list)

    def test_batch_empty_list(self) -> None:
        conn = _make_mock_conn()
        resolver = EntityResolver(conn)
        results = resolver.resolve_batch([])
        assert results == {}

    def test_batch_passes_parameters(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [],
                "FROM entity_identifiers": [],
                "similarity": [
                    _entity_row(id=1, canonical_name="Bennu", sim=0.6),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve_batch(
            ["Benu"], discipline="planetary", fuzzy=True, fuzzy_threshold=0.4
        )
        assert "Benu" in results
        # Should have fuzzy results since no exact/alias/identifier match
        assert len(results["Benu"]) >= 1


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    """Results are sorted by confidence DESC, then canonical_name ASC."""

    def test_sort_order(self) -> None:
        conn = _make_mock_conn(
            {
                "lower(canonical_name) = lower": [],
                "FROM entity_aliases": [],
                "FROM entity_identifiers": [],
                "similarity": [
                    _entity_row(id=1, canonical_name="Zebra", sim=0.5),
                    _entity_row(id=2, canonical_name="Alpha", sim=0.5),
                    _entity_row(id=3, canonical_name="Beta", sim=0.8),
                ],
            }
        )
        resolver = EntityResolver(conn)
        results = resolver.resolve("test", fuzzy=True)
        assert len(results) == 3
        # Highest confidence first
        assert results[0].canonical_name == "Beta"
        assert results[0].confidence == 0.8
        # Same confidence: alphabetical
        assert results[1].canonical_name == "Alpha"
        assert results[2].canonical_name == "Zebra"


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestNoMatch:
    """resolve() returns empty list when nothing matches."""

    def test_no_results(self) -> None:
        conn = _make_mock_conn()
        resolver = EntityResolver(conn)
        results = resolver.resolve("NonexistentEntity12345")
        assert results == []
