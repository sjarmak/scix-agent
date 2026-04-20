"""Tests for hybrid_search alias-expansion + ontology-parser flags.

The unit tests mock all DB-touching helpers (lexical_search, vector_search,
_resolve_entity_ids_for_properties) so they run without a database. The
integration test gates on SCIX_TEST_DSN — see CLAUDE.md "Database Safety".
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from scix.aho_corasick import EntityRow
from scix.alias_expansion import (
    build_alias_automaton_from_rows,
    clear_automaton_cache,
)
from scix.search import (
    SearchFilters,
    SearchResult,
    _merge_filters,
    hybrid_search,
)

# ---------------------------------------------------------------------------
# _merge_filters
# ---------------------------------------------------------------------------


class TestMergeFilters:
    def test_none_base_yields_extras(self) -> None:
        merged = _merge_filters(
            None,
            extra_entity_types=("instrument",),
            extra_entity_ids=(7, 9),
        )
        assert merged.entity_types == ("instrument",)
        assert merged.entity_ids == (7, 9)

    def test_unions_with_existing_filter(self) -> None:
        base = SearchFilters(
            year_min=2020,
            entity_types=("mission",),
            entity_ids=(1, 2),
        )
        merged = _merge_filters(
            base,
            extra_entity_types=("instrument", "mission"),
            extra_entity_ids=(2, 3),
        )
        # Order-preserving union, deduped.
        assert merged.entity_types == ("mission", "instrument")
        assert merged.entity_ids == (1, 2, 3)
        assert merged.year_min == 2020

    def test_no_extras_returns_equivalent(self) -> None:
        base = SearchFilters(year_max=2024, entity_types=("asteroid",))
        merged = _merge_filters(base)
        assert merged == base


# ---------------------------------------------------------------------------
# Ontology parser wiring
# ---------------------------------------------------------------------------


class TestOntologyParserWiring:
    def _fake_lex(self, papers: list[dict]) -> SearchResult:
        return SearchResult(papers=papers, total=len(papers), timing_ms={"lexical_ms": 0.1})

    def test_lifts_entity_types_into_filters(self) -> None:
        """entity_types from the parser should flow through to lexical_search filters."""
        captured_filters: dict[str, SearchFilters | None] = {}

        def fake_lex(conn, q, *, filters=None, limit=20, **kw):
            captured_filters.setdefault("first", filters)
            return SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.1})

        with (
            patch("scix.search.lexical_search", side_effect=fake_lex),
            patch("scix.search._model_has_embeddings", return_value=False),
            patch(
                "scix.search._resolve_entity_ids_for_properties",
                return_value=(),
            ),
        ):
            hybrid_search(
                MagicMock(),
                "infrared instruments on space telescopes",
                enable_ontology_parser=True,
                include_body=False,
            )

        f = captured_filters["first"]
        assert f is not None
        assert f.entity_types is not None
        # parser yields 'instrument' and 'telescope' from the query
        assert "instrument" in f.entity_types
        assert "telescope" in f.entity_types

    def test_resolves_properties_filter_to_entity_ids(self) -> None:
        """A KNOWN_MISSIONS+entity_type query should resolve to entity_ids via DB lookup."""
        captured: dict[str, SearchFilters | None] = {}

        def fake_lex(conn, q, *, filters=None, limit=20, **kw):
            captured.setdefault("first", filters)
            return SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.1})

        with (
            patch("scix.search.lexical_search", side_effect=fake_lex),
            patch("scix.search._model_has_embeddings", return_value=False),
            patch(
                "scix.search._resolve_entity_ids_for_properties",
                return_value=(101, 202),
            ) as mock_resolve,
        ):
            result = hybrid_search(
                MagicMock(),
                "JWST instruments",
                enable_ontology_parser=True,
                include_body=False,
            )

        mock_resolve.assert_called_once()
        f = captured["first"]
        assert f is not None
        assert f.entity_ids == (101, 202)
        assert "instrument" in (f.entity_types or ())
        assert result.metadata["ontology_clauses"] >= 2  # entity_type + properties clauses
        assert result.metadata["ontology_entity_ids"] == 2

    def test_no_clauses_skips_filter_mutation(self) -> None:
        """A query with no ontology hooks should leave filters untouched."""
        captured: dict[str, SearchFilters | None] = {}

        def fake_lex(conn, q, *, filters=None, limit=20, **kw):
            captured.setdefault("first", filters)
            return SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.1})

        with (
            patch("scix.search.lexical_search", side_effect=fake_lex),
            patch("scix.search._model_has_embeddings", return_value=False),
        ):
            hybrid_search(
                MagicMock(),
                "dark matter haloes",
                enable_ontology_parser=True,
                include_body=False,
            )

        f = captured["first"]
        assert f is None or (f.entity_types is None and f.entity_ids is None)


# ---------------------------------------------------------------------------
# Alias expansion wiring
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_alias_cache() -> None:
    clear_automaton_cache()
    yield
    clear_automaton_cache()


def _build_test_automaton():
    rows = [
        EntityRow(
            entity_id=1,
            surface="JWST",
            canonical_name="James Webb Space Telescope",
            ambiguity_class="unambiguous",
            is_alias=True,
        ),
        EntityRow(
            entity_id=1,
            surface="James Webb Space Telescope",
            canonical_name="James Webb Space Telescope",
            ambiguity_class="unambiguous",
            is_alias=False,
        ),
        EntityRow(
            entity_id=2,
            surface="MIRI",
            canonical_name="Mid-Infrared Instrument",
            ambiguity_class="unambiguous",
            is_alias=True,
        ),
        EntityRow(
            entity_id=2,
            surface="Mid-Infrared Instrument",
            canonical_name="Mid-Infrared Instrument",
            ambiguity_class="unambiguous",
            is_alias=False,
        ),
    ]
    return build_alias_automaton_from_rows(
        rows,
        entity_type_by_id={1: "telescope", 2: "instrument"},
    )


class TestAliasExpansionWiring:
    def test_extra_lexical_lanes_added(self) -> None:
        """Each matched entity contributes one extra lexical_search call (capped)."""
        automaton = _build_test_automaton()

        lane_queries: list[str] = []

        def fake_lex(conn, q, *, filters=None, limit=20, **kw):
            lane_queries.append(q)
            return SearchResult(
                papers=[{"bibcode": f"b/{q[:8]}", "score": 1.0}],
                total=1,
                timing_ms={"lexical_ms": 0.1},
            )

        with (
            patch("scix.search.lexical_search", side_effect=fake_lex),
            patch("scix.search._model_has_embeddings", return_value=False),
        ):
            result = hybrid_search(
                MagicMock(),
                "JWST MIRI imaging",
                enable_alias_expansion=True,
                alias_automaton=automaton,
                include_body=False,
            )

        # First call is the original query; subsequent calls are the canonical
        # lanes for matched entities.
        assert lane_queries[0] == "JWST MIRI imaging"
        canonical_lanes = lane_queries[1:]
        assert "James Webb Space Telescope" in canonical_lanes
        assert "Mid-Infrared Instrument" in canonical_lanes
        assert result.metadata["alias_matches"] == 2
        assert result.metadata["alias_lanes"] == 2

    def test_lane_cap_respected(self) -> None:
        """No more than _MAX_ALIAS_LEXICAL_LANES extra calls regardless of matches."""
        from scix.search import _MAX_ALIAS_LEXICAL_LANES

        # Build an automaton with more entities than the cap.
        rows = []
        type_map: dict[int, str] = {}
        for eid in range(1, _MAX_ALIAS_LEXICAL_LANES + 3):
            short = f"E{eid}"
            canonical = f"Entity Number {eid}"
            rows.append(
                EntityRow(
                    entity_id=eid,
                    surface=short,
                    canonical_name=canonical,
                    ambiguity_class="unambiguous",
                    is_alias=True,
                )
            )
            rows.append(
                EntityRow(
                    entity_id=eid,
                    surface=canonical,
                    canonical_name=canonical,
                    ambiguity_class="unambiguous",
                    is_alias=False,
                )
            )
            type_map[eid] = "thing"
        automaton = build_alias_automaton_from_rows(rows, entity_type_by_id=type_map)

        query = " ".join(f"E{eid}" for eid in range(1, _MAX_ALIAS_LEXICAL_LANES + 3))

        lane_queries: list[str] = []

        def fake_lex(conn, q, *, filters=None, limit=20, **kw):
            lane_queries.append(q)
            return SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.1})

        with (
            patch("scix.search.lexical_search", side_effect=fake_lex),
            patch("scix.search._model_has_embeddings", return_value=False),
        ):
            hybrid_search(
                MagicMock(),
                query,
                enable_alias_expansion=True,
                alias_automaton=automaton,
                include_body=False,
            )

        # Original + at most _MAX_ALIAS_LEXICAL_LANES extras
        assert len(lane_queries) == 1 + _MAX_ALIAS_LEXICAL_LANES

    def test_no_matches_no_extra_lanes(self) -> None:
        automaton = _build_test_automaton()

        lane_queries: list[str] = []

        def fake_lex(conn, q, *, filters=None, limit=20, **kw):
            lane_queries.append(q)
            return SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.1})

        with (
            patch("scix.search.lexical_search", side_effect=fake_lex),
            patch("scix.search._model_has_embeddings", return_value=False),
        ):
            result = hybrid_search(
                MagicMock(),
                "dark matter haloes",
                enable_alias_expansion=True,
                alias_automaton=automaton,
                include_body=False,
            )

        assert lane_queries == ["dark matter haloes"]
        assert result.metadata["alias_lanes"] == 0


# ---------------------------------------------------------------------------
# Default-off contract
# ---------------------------------------------------------------------------


class TestFlagsDefaultOff:
    def test_neither_flag_calls_extra_helpers(self) -> None:
        """With both flags False (default), the new code paths must be silent."""
        with (
            patch("scix.search.lexical_search") as mock_lex,
            patch("scix.search._model_has_embeddings", return_value=False),
            patch("scix.search._resolve_entity_ids_for_properties") as mock_resolve,
        ):
            mock_lex.return_value = SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.1})
            result = hybrid_search(MagicMock(), "any query", include_body=False)

        mock_resolve.assert_not_called()
        # exactly one lexical call (the original query); no alias lanes
        assert mock_lex.call_count == 1
        assert "ontology_clauses" not in result.metadata
        assert "alias_lanes" not in result.metadata


# ---------------------------------------------------------------------------
# Integration test (real DB, gated on SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


class _RollbackFixture(Exception):
    """Internal marker — raised inside the integration test's transaction so
    psycopg rolls back the seeded fixture rows."""


@pytest.mark.integration
def test_hybrid_search_with_both_flags_against_scix_test() -> None:
    """End-to-end: insert a couple of entities + papers, run hybrid_search with
    both flags on, verify results return without error and the metadata
    surfaces the new counters.
    """
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        pytest.skip("SCIX_TEST_DSN not set")

    import psycopg

    from scix.alias_expansion import build_alias_automaton_from_rows

    captured: dict[str, object] = {}

    with psycopg.connect(dsn) as conn:
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO entities (canonical_name, entity_type, source, properties) "
                        "VALUES (%s, %s, %s, %s::jsonb) RETURNING id",
                        (
                            "James Webb Space Telescope",
                            "telescope",
                            "test_fixture_xz4_1",
                            '{"mission": "JWST"}',
                        ),
                    )
                    jwst_id = cur.fetchone()[0]
                    cur.execute(
                        "INSERT INTO entities (canonical_name, entity_type, source, properties) "
                        "VALUES (%s, %s, %s, %s::jsonb) RETURNING id",
                        (
                            "Mid-Infrared Instrument",
                            "instrument",
                            "test_fixture_xz4_1",
                            '{"mission": "JWST"}',
                        ),
                    )
                    miri_id = cur.fetchone()[0]
                    cur.executemany(
                        "INSERT INTO entity_aliases (entity_id, alias, alias_source) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        [
                            (jwst_id, "JWST", "test_fixture_xz4_1"),
                            (jwst_id, "Webb", "test_fixture_xz4_1"),
                            (miri_id, "MIRI", "test_fixture_xz4_1"),
                        ],
                    )

                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, canonical_name, entity_type FROM entities " "WHERE source = %s",
                        ("test_fixture_xz4_1",),
                    )
                    ents = cur.fetchall()
                    cur.execute(
                        "SELECT entity_id, alias FROM entity_aliases " "WHERE entity_id = ANY(%s)",
                        ([e[0] for e in ents],),
                    )
                    aliases = cur.fetchall()

                type_map = {e[0]: e[2] for e in ents}
                rows = []
                for eid, canonical, _etype in ents:
                    rows.append(
                        EntityRow(
                            entity_id=eid,
                            surface=canonical,
                            canonical_name=canonical,
                            ambiguity_class="unambiguous",
                            is_alias=False,
                        )
                    )
                for eid, alias in aliases:
                    canonical = next(c for i, c, _ in ents if i == eid)
                    rows.append(
                        EntityRow(
                            entity_id=eid,
                            surface=alias,
                            canonical_name=canonical,
                            ambiguity_class="unambiguous",
                            is_alias=True,
                        )
                    )
                automaton = build_alias_automaton_from_rows(rows, entity_type_by_id=type_map)

                result = hybrid_search(
                    conn,
                    "JWST MIRI instruments",
                    enable_alias_expansion=True,
                    enable_ontology_parser=True,
                    alias_automaton=automaton,
                    include_body=False,
                    top_n=5,
                )
                captured["result"] = result
                # Force the transaction to roll back; psycopg treats any
                # exception inside `conn.transaction()` as a rollback signal.
                raise _RollbackFixture()
        except _RollbackFixture:
            pass

    result = captured["result"]
    # The query carries:
    #   - 2 alias surface matches (JWST + MIRI)
    #   - 1 entity-type clause ("instruments" -> 'instrument')
    #   - 1 properties-filter clause (JWST + entity-type co-occurrence
    #     -> {"mission": "JWST"}) which resolves to entity_ids on the fixture.
    assert isinstance(result, SearchResult)
    assert result.metadata.get("alias_matches", 0) >= 2
    assert result.metadata.get("ontology_clauses", 0) >= 2
    assert result.metadata.get("ontology_entity_ids", 0) >= 1
