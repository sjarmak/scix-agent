"""Tests for query-time alias expansion (xz4.1.24).

Two layers:

* Pure-library tests of :mod:`scix.alias_expansion` — no DB required.
  They build small in-memory automatons and exercise homograph handling,
  case-insensitive matching, dedup, alias caps, and empty-query
  handling.
* DB integration tests (marked ``@pytest.mark.integration``) that seed
  two entities into ``SCIX_TEST_DSN`` inside a savepoint, build the
  automaton from the DB, and assert end-to-end behavior.
"""

from __future__ import annotations

import os
from typing import Iterator

import psycopg
import pytest

from scix.aho_corasick import EntityRow
from scix.alias_expansion import (
    AliasAutomaton,
    AliasExpansion,
    ExpansionResult,
    build_alias_automaton,
    build_alias_automaton_from_rows,
    clear_automaton_cache,
    expand_query,
)

# ---------------------------------------------------------------------------
# Fixtures (in-memory)
# ---------------------------------------------------------------------------


def _hubble_rows() -> list[EntityRow]:
    return [
        EntityRow(
            entity_id=101,
            surface="Hubble Space Telescope",
            canonical_name="Hubble Space Telescope",
            ambiguity_class="unique",
            is_alias=False,
        ),
        EntityRow(
            entity_id=101,
            surface="HST",
            canonical_name="Hubble Space Telescope",
            ambiguity_class="unique",
            is_alias=True,
        ),
        EntityRow(
            entity_id=101,
            surface="Hubble",
            canonical_name="Hubble Space Telescope",
            ambiguity_class="unique",
            is_alias=True,
        ),
    ]


def _psyche_homograph_rows() -> list[EntityRow]:
    """Two entities both carrying the alias 'Psyche'."""
    asteroid = [
        EntityRow(
            entity_id=201,
            surface="16 Psyche",
            canonical_name="16 Psyche",
            ambiguity_class="unique",
            is_alias=False,
        ),
        EntityRow(
            entity_id=201,
            surface="Psyche",
            canonical_name="16 Psyche",
            ambiguity_class="unique",
            is_alias=True,
        ),
        EntityRow(
            entity_id=201,
            surface="Psyche asteroid",
            canonical_name="16 Psyche",
            ambiguity_class="unique",
            is_alias=True,
        ),
    ]
    spacecraft = [
        EntityRow(
            entity_id=202,
            surface="Psyche Mission",
            canonical_name="Psyche Mission",
            ambiguity_class="unique",
            is_alias=False,
        ),
        EntityRow(
            entity_id=202,
            surface="Psyche",
            canonical_name="Psyche Mission",
            ambiguity_class="unique",
            is_alias=True,
        ),
    ]
    return asteroid + spacecraft


def _hubble_automaton() -> AliasAutomaton:
    return build_alias_automaton_from_rows(
        _hubble_rows(),
        entity_type_by_id={101: "telescope"},
    )


def _psyche_automaton() -> AliasAutomaton:
    return build_alias_automaton_from_rows(
        _psyche_homograph_rows(),
        entity_type_by_id={201: "asteroid", 202: "spacecraft"},
    )


# ---------------------------------------------------------------------------
# Unit tests — basic expansion
# ---------------------------------------------------------------------------


class TestBasicExpansion:
    def test_expands_acronym_to_canonical_and_aliases(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "HST observations of cool brown dwarfs", automaton=bundle)
        assert isinstance(result, ExpansionResult)
        assert result.entity_ids == (101,)
        assert "Hubble Space Telescope" in result.expanded_terms
        assert "Hubble" in result.expanded_terms
        # The matched surface itself is not duplicated into expanded_terms
        # via the alias path — it appears in matched_surface and may also
        # not be in `aliases` because we strip the matched_surface out.
        assert all(isinstance(t, str) for t in result.expanded_terms)

    def test_match_records_span_and_surface(self) -> None:
        bundle = _hubble_automaton()
        query = "HST observations of cool brown dwarfs"
        result = expand_query(None, query, automaton=bundle)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert isinstance(m, AliasExpansion)
        assert m.entity_id == 101
        assert m.canonical_name == "Hubble Space Telescope"
        assert m.entity_type == "telescope"
        assert m.matched_surface == "HST"
        assert query[m.span[0] : m.span[1]] == "HST"

    def test_aliases_exclude_matched_surface_and_canonical(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "HST imaging", automaton=bundle)
        m = result.matches[0]
        assert "HST" not in m.aliases
        assert "Hubble Space Telescope" not in m.aliases
        assert "Hubble" in m.aliases


# ---------------------------------------------------------------------------
# Unit tests — disambiguation
# ---------------------------------------------------------------------------


class TestHomographDisambiguation:
    def test_short_homograph_alone_is_dropped(self) -> None:
        """'Psyche' alone is a homograph — should drop both entities."""
        bundle = _psyche_automaton()
        result = expand_query(None, "Psyche surface mineralogy", automaton=bundle)
        assert (
            result.entity_ids == ()
        ), f"Expected no matches for ambiguous bare 'Psyche', got {result.entity_ids}"

    def test_long_form_co_present_resolves_homograph(self) -> None:
        """When 'Psyche asteroid' (long-form alias of 201) is present, only
        the asteroid entity fires for the bare 'Psyche' surface."""
        bundle = _psyche_automaton()
        query = "Psyche asteroid composition Psyche imaging"
        result = expand_query(None, query, automaton=bundle)
        assert 201 in result.entity_ids
        assert 202 not in result.entity_ids

    def test_unambiguous_short_surface_fires(self) -> None:
        """HST is short but unambiguous in the loaded automaton — must fire
        despite the long-form gate (the relaxation rule)."""
        bundle = _hubble_automaton()
        result = expand_query(None, "HST imaging", automaton=bundle)
        assert result.entity_ids == (101,)

    def test_disambiguator_off_fires_every_match(self) -> None:
        bundle = _psyche_automaton()
        result = expand_query(
            None,
            "Psyche surface mineralogy",
            automaton=bundle,
            require_long_form_disambiguator=False,
        )
        assert set(result.entity_ids) == {201, 202}


# ---------------------------------------------------------------------------
# Unit tests — case insensitivity, dedup, caps, empty
# ---------------------------------------------------------------------------


class TestCaseAndDedup:
    def test_lowercase_query_matches_mixed_case_surface(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "hubble space telescope deep field", automaton=bundle)
        assert result.entity_ids == (101,)

    def test_uppercase_query_matches_mixed_case_surface(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "HUBBLE deep field", automaton=bundle)
        assert result.entity_ids == (101,)

    def test_dedup_when_canonical_and_alias_both_hit(self) -> None:
        """'HST' (alias) and 'Hubble Space Telescope' (canonical) both
        appear in the query — entity_ids and expanded_terms must dedup."""
        bundle = _hubble_automaton()
        query = "HST and the Hubble Space Telescope archive"
        result = expand_query(None, query, automaton=bundle)
        assert result.entity_ids == (101,)
        # canonical name appears once in expanded_terms
        assert result.expanded_terms.count("Hubble Space Telescope") == 1
        # both spans recorded as separate matches
        spans = [m.span for m in result.matches]
        assert len(spans) == len(set(spans))


class TestAliasCap:
    def test_max_aliases_per_entity_respected(self) -> None:
        rows = [
            EntityRow(
                entity_id=900,
                surface="Big Entity",
                canonical_name="Big Entity",
                ambiguity_class="unique",
                is_alias=False,
            )
        ]
        for i in range(20):
            rows.append(
                EntityRow(
                    entity_id=900,
                    surface=f"alias_{i:02d}",
                    canonical_name="Big Entity",
                    ambiguity_class="unique",
                    is_alias=True,
                )
            )
        bundle = build_alias_automaton_from_rows(rows, entity_type_by_id={900: "thing"})
        result = expand_query(
            None,
            "alias_03 study",
            automaton=bundle,
            max_aliases_per_entity=5,
        )
        assert len(result.matches) == 1
        assert len(result.matches[0].aliases) == 5


class TestEmptyAndBoundaries:
    def test_empty_query_returns_empty_result(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "", automaton=bundle)
        assert result == ExpansionResult(
            original_query="",
            matches=(),
            expanded_terms=(),
            entity_ids=(),
        )

    def test_empty_query_preserves_original(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "", automaton=bundle)
        assert result.original_query == ""

    def test_no_match_returns_empty_with_query(self) -> None:
        bundle = _hubble_automaton()
        result = expand_query(None, "completely unrelated text", automaton=bundle)
        assert result.original_query == "completely unrelated text"
        assert result.matches == ()
        assert result.entity_ids == ()
        assert result.expanded_terms == ()

    def test_word_boundary_avoids_substring_hit(self) -> None:
        """'HST' must not match inside 'GHOST'."""
        bundle = _hubble_automaton()
        result = expand_query(None, "GHOST imaging survey", automaton=bundle)
        assert result.entity_ids == ()


# ---------------------------------------------------------------------------
# Integration tests — require SCIX_TEST_DSN
# ---------------------------------------------------------------------------


def _test_dsn_or_skip() -> str:
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        pytest.skip("SCIX_TEST_DSN not set — skipping DB integration test")
    return dsn


@pytest.fixture()
def db_conn() -> Iterator[psycopg.Connection]:
    """Yield a connection to ``SCIX_TEST_DSN`` and roll back at teardown.

    Skips the test if ``SCIX_TEST_DSN`` is unset or unreachable.
    """
    dsn = _test_dsn_or_skip()
    try:
        conn = psycopg.connect(dsn)
    except psycopg.OperationalError as exc:
        pytest.skip(f"SCIX_TEST_DSN unreachable: {exc}")
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
        clear_automaton_cache()


@pytest.mark.integration
class TestExpandQueryIntegration:
    def test_jwst_miri_round_trip(self, db_conn: psycopg.Connection) -> None:
        """Insert two entities + aliases, build automaton from DB, expand."""
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source)
                VALUES (%s, %s, %s) RETURNING id
                """,
                ("James Webb Space Telescope", "mission", "test_alias_expansion"),
            )
            jwst_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source)
                VALUES (%s, %s, %s) RETURNING id
                """,
                ("Mid-Infrared Instrument", "instrument", "test_alias_expansion"),
            )
            miri_id = cur.fetchone()[0]
            cur.executemany(
                """
                INSERT INTO entity_aliases (entity_id, alias, alias_source)
                VALUES (%s, %s, %s)
                """,
                [
                    (jwst_id, "JWST", "test_alias_expansion"),
                    (jwst_id, "Webb", "test_alias_expansion"),
                    (miri_id, "MIRI", "test_alias_expansion"),
                ],
            )

        # Force a rebuild — clear cache because the data changed mid-txn.
        clear_automaton_cache()
        bundle = build_alias_automaton(db_conn)
        result = expand_query(db_conn, "JWST MIRI imaging", automaton=bundle)

        assert jwst_id in result.entity_ids
        assert miri_id in result.entity_ids
        assert "James Webb Space Telescope" in result.expanded_terms
        assert "Mid-Infrared Instrument" in result.expanded_terms
