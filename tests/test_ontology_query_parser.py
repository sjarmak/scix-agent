"""Tests for the ontology-aware query parser (xz4.1.25)."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scix.ontology_query_parser import (
    ASTEROID_TAXONOMY_LETTERS,
    ENTITY_TYPE_TERMS,
    KNOWN_MISSIONS,
    OntologyClause,
    default_vocabulary,
    parse_query,
)

# ---------------------------------------------------------------------------
# Behaviour against the bead's worked examples.
# ---------------------------------------------------------------------------


class TestParseQueryWorkedExamples:
    def test_jwst_instruments_lifts_entity_type_and_mission(self) -> None:
        parsed = parse_query("JWST instruments")

        assert "instrument" in parsed.entity_types
        assert {"mission": "JWST"} in parsed.properties_filters
        assert parsed.original_query == "JWST instruments"
        assert parsed.residual_query == "JWST instruments"

    def test_flagship_nasa_missions_yields_mission_only(self) -> None:
        parsed = parse_query("flagship NASA missions to the outer solar system")

        assert "mission" in parsed.entity_types
        # NASA is not a mission; KNOWN_MISSIONS holds only specific missions.
        assert parsed.properties_filters == ()

    def test_m_type_asteroid_lifts_taxonomy_letter(self) -> None:
        parsed = parse_query("M-type asteroid metallic composition from near-infrared spectroscopy")

        assert "asteroid" in parsed.entity_types
        assert {"taxonomy": "M"} in parsed.properties_filters

    def test_multiple_entity_types_in_one_query(self) -> None:
        parsed = parse_query("infrared instruments on space telescopes for exoplanet detection")

        assert "instrument" in parsed.entity_types
        assert "telescope" in parsed.entity_types
        assert "exoplanet" in parsed.entity_types

    def test_no_ontology_terms_returns_empty(self) -> None:
        parsed = parse_query("dark matter")

        assert parsed.clauses == ()
        assert parsed.entity_types == ()
        assert parsed.properties_filters == ()
        assert parsed.residual_query == "dark matter"

    def test_empty_query(self) -> None:
        parsed = parse_query("")

        assert parsed.original_query == ""
        assert parsed.residual_query == ""
        assert parsed.clauses == ()
        assert parsed.entity_types == ()
        assert parsed.properties_filters == ()


# ---------------------------------------------------------------------------
# residual_query parity (filters AUGMENT, never strip).
# ---------------------------------------------------------------------------


class TestResidualQueryParity:
    @pytest.mark.parametrize(
        "query",
        [
            "",
            "dark matter",
            "JWST instruments",
            "M-type asteroid metallic composition",
            "flagship NASA missions to the outer solar system",
            "infrared instruments on space telescopes for exoplanet detection",
            "Cassini mission to Saturn",
            "Voyager spacecraft outer planets",
        ],
    )
    def test_residual_equals_original(self, query: str) -> None:
        parsed = parse_query(query)
        assert parsed.residual_query == parsed.original_query == query


# ---------------------------------------------------------------------------
# Clause-level structure.
# ---------------------------------------------------------------------------


class TestClauseStructure:
    def test_entity_type_clause_has_token_span(self) -> None:
        parsed = parse_query("JWST instruments")
        entity_clauses = [c for c in parsed.clauses if c.properties_filter is None]
        assert len(entity_clauses) == 1
        clause = entity_clauses[0]
        assert clause.entity_type == "instrument"
        assert clause.surface == "instruments"
        start, end = clause.span
        assert "JWST instruments"[start:end] == "instruments"

    def test_mission_clause_has_canonical_form_and_span(self) -> None:
        parsed = parse_query("jwst instruments")
        mission_clauses = [c for c in parsed.clauses if c.properties_filter is not None]
        assert len(mission_clauses) == 1
        clause = mission_clauses[0]
        assert clause.properties_filter == {"mission": "JWST"}
        assert clause.surface == "jwst"
        start, end = clause.span
        assert "jwst instruments"[start:end] == "jwst"

    def test_asteroid_taxonomy_clause(self) -> None:
        parsed = parse_query("S-type asteroid silicate")
        tax_clauses = [c for c in parsed.clauses if c.properties_filter == {"taxonomy": "S"}]
        assert len(tax_clauses) == 1
        clause = tax_clauses[0]
        assert clause.entity_type == "asteroid"
        start, end = clause.span
        assert "S-type asteroid silicate"[start:end] == "S-type"

    def test_mission_without_entity_type_is_not_lifted(self) -> None:
        # No entity-type token in this query -> mission clause not emitted.
        parsed = parse_query("JWST captured a beautiful image")
        assert parsed.entity_types == ()
        assert parsed.properties_filters == ()
        assert parsed.clauses == ()

    def test_unknown_taxonomy_letter_ignored(self) -> None:
        # 'Z' is not in ASTEROID_TAXONOMY_LETTERS allowlist.
        parsed = parse_query("Z-type asteroid")
        assert {"taxonomy": "Z"} not in parsed.properties_filters

    def test_dedupe_repeated_entity_type(self) -> None:
        parsed = parse_query("instrument and instruments are instrument synonyms")
        instrument_clauses = [
            c
            for c in parsed.clauses
            if c.entity_type == "instrument" and c.properties_filter is None
        ]
        assert len(instrument_clauses) == 1


# ---------------------------------------------------------------------------
# Immutability and frozen dataclass semantics.
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_parsed_query_is_frozen(self) -> None:
        parsed = parse_query("JWST instruments")
        with pytest.raises(dataclasses.FrozenInstanceError):
            parsed.original_query = "mutated"  # type: ignore[misc]

    def test_ontology_clause_is_frozen(self) -> None:
        clause = OntologyClause(
            entity_type="instrument",
            properties_filter=None,
            surface="instruments",
            span=(0, 11),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            clause.entity_type = "mission"  # type: ignore[misc]

    def test_default_vocabulary_returns_constants(self) -> None:
        vocab = default_vocabulary()
        assert vocab["entity_type_terms"] is ENTITY_TYPE_TERMS
        assert vocab["known_missions"] is KNOWN_MISSIONS
        assert vocab["asteroid_taxonomy_letters"] is ASTEROID_TAXONOMY_LETTERS


# ---------------------------------------------------------------------------
# Determinism: same input -> equal output.
# ---------------------------------------------------------------------------


class TestDeterminism:
    @given(
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),
                whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po"),
            ),
            min_size=0,
            max_size=120,
        )
    )
    def test_parse_query_is_deterministic(self, query: str) -> None:
        first = parse_query(query)
        second = parse_query(query)
        assert first == second

    def test_parsed_query_equality_uses_value_semantics(self) -> None:
        a = parse_query("JWST instruments")
        b = parse_query("JWST instruments")
        assert a == b


# ---------------------------------------------------------------------------
# Type-safety and input validation.
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_string_query_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            parse_query(123)  # type: ignore[arg-type]

    def test_custom_vocabulary_overrides_default(self) -> None:
        custom = {
            "entity_type_terms": {"galaxies": "galaxy", "galaxy": "galaxy"},
            "known_missions": frozenset({"SDSS"}),
            "asteroid_taxonomy_letters": frozenset({"M"}),
        }
        parsed = parse_query("SDSS galaxies in the local universe", vocabulary=custom)
        assert "galaxy" in parsed.entity_types
        assert {"mission": "SDSS"} in parsed.properties_filters

    def test_custom_vocabulary_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            parse_query("anything", vocabulary={"entity_type_terms": ["bad"]})
