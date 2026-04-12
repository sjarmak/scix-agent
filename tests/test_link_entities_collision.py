"""Regression tests for the collision-aware caching resolver.

Covers the u02-collision-bug-fix: the caching resolver in
`src/scix/link_entities.py` used to silently drop every collision at the
cache layer via `if key and key not in cache`, letting the first ontology
own every ambiguous canonical name. The fix stores all candidates per
lowercase key as a list and exposes them via `resolve_all()` so an
ambiguous name like "Mercury" (SSODNet planet row vs. GCMD chemical
element row) surfaces both entities to the disambiguator.

Uses the same in-memory MagicMock connection pattern as
tests/test_link_entities.py — no database writes, safe under any
SCIX_TEST_DSN (including unset).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from scix.link_entities import EntityResolver, ResolverMatch


def _make_conn_with_entities(
    entities: list[tuple[int, str]],
    aliases: list[tuple[int, str]],
) -> MagicMock:
    """Build a mock psycopg connection whose cursor returns the given rows.

    First ``fetchall`` call yields the entities table, second yields the
    entity_aliases table — matching the order in
    ``_CachingResolver._build_cache``.
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.side_effect = [entities, aliases]
    return conn


class TestCollisionAwareResolver:
    def test_ambiguous_canonical_name_returns_all_candidates(self) -> None:
        """Two entities sharing the same lowercase canonical name must both surface.

        Mirrors the real-world SSODNet "Mercury" (planet) vs. GCMD
        "Mercury" (chemical element) collision that motivated the fix.
        """
        conn = _make_conn_with_entities(
            entities=[
                (101, "Mercury"),  # SSODNet planet
                (202, "mercury"),  # GCMD chemical element (different case)
            ],
            aliases=[],
        )
        resolver = EntityResolver(conn)

        candidates = resolver.resolve_all("mercury")

        assert len(candidates) >= 2, (
            f"Expected >=2 candidates for ambiguous 'mercury'; "
            f"got {len(candidates)}: {candidates}"
        )
        entity_ids = {c.entity_id for c in candidates}
        assert entity_ids == {101, 202}
        for candidate in candidates:
            assert isinstance(candidate, ResolverMatch)
            assert candidate.match_method == "canonical_exact"
            assert candidate.confidence == 1.0

    def test_canonical_and_alias_collision_both_surface(self) -> None:
        """A canonical name colliding with an alias of a different entity must not drop either."""
        conn = _make_conn_with_entities(
            entities=[(1, "ALMA")],
            aliases=[(2, "alma")],
        )
        resolver = EntityResolver(conn)

        candidates = resolver.resolve_all("alma")

        assert len(candidates) >= 2
        entity_ids = {c.entity_id for c in candidates}
        assert entity_ids == {1, 2}
        # Canonical must be ordered before alias so resolve() preserves
        # the historical canonical-priority contract.
        assert candidates[0].entity_id == 1
        assert candidates[0].match_method == "canonical_exact"
        assert candidates[1].entity_id == 2
        assert candidates[1].match_method == "alias_exact"

    def test_resolve_backward_compat_picks_first_candidate(self) -> None:
        """resolve() still returns a single match for legacy callers (canonical wins)."""
        conn = _make_conn_with_entities(
            entities=[
                (101, "Mercury"),
                (202, "Mercury"),
            ],
            aliases=[],
        )
        resolver = EntityResolver(conn)

        match = resolver.resolve("mercury")

        assert match is not None
        assert match.entity_id == 101  # First-loaded canonical wins
        assert match.match_method == "canonical_exact"

    def test_resolve_all_empty_for_unknown_mention(self) -> None:
        conn = _make_conn_with_entities(entities=[], aliases=[])
        resolver = EntityResolver(conn)
        assert resolver.resolve_all("nothing-here") == []

    def test_resolve_all_empty_for_blank_mention(self) -> None:
        conn = _make_conn_with_entities(entities=[(1, "ALMA")], aliases=[])
        resolver = EntityResolver(conn)
        assert resolver.resolve_all("   ") == []
