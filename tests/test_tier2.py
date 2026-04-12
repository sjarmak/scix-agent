"""Tests for u09 Tier-2 Aho-Corasick abstract linker.

Two layers:

* Pure-library tests of :mod:`scix.aho_corasick` — no DB required. They
  exercise ambiguity-aware firing (the HST / Hubble Space Telescope
  acceptance criterion), boundary-safe matching, and automaton
  picklability.
* DB integration tests of ``scripts/link_tier2.py`` — require
  ``SCIX_TEST_DSN`` pointing at a non-production database. They seed
  small fixtures (papers + entities + curated_entity_core) and assert
  end-to-end behavior including the per-entity linkage cap and
  ``link_policy='llm_only'`` demotion.
"""

from __future__ import annotations

import pathlib
import pickle
import sys
from typing import Iterator

import psycopg
import pytest

from scix.aho_corasick import (
    EntityRow,
    LinkCandidate,
    build_automaton,
    link_abstract,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import link_tier2  # noqa: E402

from tests.helpers import get_test_dsn  # noqa: E402

# ---------------------------------------------------------------------------
# Pure library tests (no DB)
# ---------------------------------------------------------------------------


def _hst_rows() -> list[EntityRow]:
    """Fixture: one homograph entity (HST) with a long-form alias."""
    return [
        EntityRow(
            entity_id=101,
            surface="HST",
            canonical_name="Hubble Space Telescope",
            ambiguity_class="homograph",
            is_alias=True,
        ),
        EntityRow(
            entity_id=101,
            surface="Hubble Space Telescope",
            canonical_name="Hubble Space Telescope",
            ambiguity_class="homograph",
            is_alias=False,
        ),
    ]


def _jwst_rows() -> list[EntityRow]:
    return [
        EntityRow(
            entity_id=202,
            surface="JWST",
            canonical_name="James Webb Space Telescope",
            ambiguity_class="unique",
            is_alias=True,
        ),
        EntityRow(
            entity_id=202,
            surface="James Webb Space Telescope",
            canonical_name="James Webb Space Telescope",
            ambiguity_class="unique",
            is_alias=False,
        ),
    ]


class TestAhoCorasickAmbiguityGate:
    def test_homograph_alone_does_not_fire(self) -> None:
        automaton = build_automaton(_hst_rows())
        abstract = "We observed the field with HST over three epochs."
        out = link_abstract(abstract, automaton)
        assert out == [], f"HST alone should not fire, got {out}"

    def test_homograph_with_long_form_fires(self) -> None:
        automaton = build_automaton(_hst_rows())
        abstract = (
            "We observed the field with HST. " "The Hubble Space Telescope provided deep imaging."
        )
        out = link_abstract(abstract, automaton)
        entity_ids = {c.entity_id for c in out}
        assert 101 in entity_ids, f"expected entity 101, got {out}"

    def test_homograph_long_form_only_fires(self) -> None:
        # Long-form alone (no short form) should still fire — the
        # long-form is itself the disambiguator.
        automaton = build_automaton(_hst_rows())
        abstract = "We used the Hubble Space Telescope for UV imaging."
        out = link_abstract(abstract, automaton)
        assert any(c.entity_id == 101 for c in out)

    def test_unique_fires_unconditionally(self) -> None:
        automaton = build_automaton(_jwst_rows())
        abstract = "JWST spectra show water vapor."
        out = link_abstract(abstract, automaton)
        assert any(c.entity_id == 202 for c in out)

    def test_disambiguator_override(self) -> None:
        automaton = build_automaton(_hst_rows())
        abstract = "HST is a common acronym with no long-form in context."

        def always_yes(entity_id: int, surface: str, abstract: str) -> bool:
            return True

        out_yes = link_abstract(abstract, automaton, disambiguator=always_yes)
        assert any(c.entity_id == 101 for c in out_yes)

        def always_no(entity_id: int, surface: str, abstract: str) -> bool:
            return False

        out_no = link_abstract(abstract, automaton, disambiguator=always_no)
        assert not any(c.entity_id == 101 for c in out_no)

    def test_broken_disambiguator_fails_closed(self) -> None:
        automaton = build_automaton(_hst_rows())
        abstract = "HST is a common acronym."

        def broken(entity_id: int, surface: str, abstract: str) -> bool:
            raise RuntimeError("boom")

        out = link_abstract(abstract, automaton, disambiguator=broken)
        assert out == []


class TestAhoCorasickBoundary:
    def test_substring_match_rejected(self) -> None:
        rows = [
            EntityRow(
                entity_id=1,
                surface="ACT",
                canonical_name="Atacama Cosmology Telescope",
                ambiguity_class="unique",
                is_alias=True,
            ),
        ]
        automaton = build_automaton(rows)
        # "ACTION" contains "ACT" as a prefix; we require word boundaries.
        out = link_abstract("The ACTION was fast.", automaton)
        assert out == [], f"substring match should be rejected, got {out}"

    def test_whole_word_match_accepted(self) -> None:
        rows = [
            EntityRow(
                entity_id=1,
                surface="ACT",
                canonical_name="Atacama Cosmology Telescope",
                ambiguity_class="unique",
                is_alias=True,
            ),
        ]
        automaton = build_automaton(rows)
        out = link_abstract("The ACT collaboration released data.", automaton)
        assert len(out) == 1
        assert out[0].entity_id == 1


class TestAhoCorasickPicklable:
    def test_automaton_roundtrip(self) -> None:
        automaton = build_automaton(_jwst_rows())
        blob = pickle.dumps(automaton)
        loaded = pickle.loads(blob)
        out = link_abstract("JWST is amazing", loaded)
        assert any(c.entity_id == 202 for c in out)


# ---------------------------------------------------------------------------
# Integration fixture against scix_test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip("SCIX_TEST_DSN must be set to a non-production DSN for tier-2 tests")
    return test_dsn


_SEED_PAPERS: list[tuple[str, str]] = [
    (
        "test_u09_0001",
        "We used the Hubble Space Telescope to image a distant quasar. HST data revealed a jet.",
    ),
    (
        "test_u09_0002",
        "HST is a common acronym in other fields but not here.",  # homograph alone
    ),
    (
        "test_u09_0003",
        "JWST observations confirmed the presence of water vapor in the atmosphere.",
    ),
    (
        "test_u09_0004",
        "The James Webb Space Telescope provided deep NIR imaging of the galaxy.",
    ),
    (
        "test_u09_0005",
        "ALMA interferometry revealed dust continuum emission at millimeter wavelengths.",
    ),
    (
        "test_u09_0006",
        "We present a multiwavelength survey with no specific instrument mentioned.",
    ),
]

_SEED_ENTITIES: list[tuple[str, str, str]] = [
    # (canonical_name, source, ambiguity_class)
    ("Hubble Space Telescope", "unit_test_u09", "homograph"),
    ("James Webb Space Telescope", "unit_test_u09", "unique"),
    ("ALMA Observatory", "unit_test_u09", "domain_safe"),
]

_SEED_ALIASES: list[tuple[str, str]] = [
    ("Hubble Space Telescope", "HST"),
    ("James Webb Space Telescope", "JWST"),
    ("ALMA Observatory", "ALMA"),
]


def _cleanup(conn: psycopg.Connection) -> None:
    bibcodes = [p[0] for p in _SEED_PAPERS]
    canonicals = [e[0] for e in _SEED_ENTITIES]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM curated_entity_core "
            " WHERE entity_id IN ("
            "     SELECT id FROM entities "
            "      WHERE canonical_name = ANY(%s) AND source = 'unit_test_u09'"
            " )",
            (canonicals,),
        )
        cur.execute(
            "DELETE FROM entities " " WHERE canonical_name = ANY(%s) AND source = 'unit_test_u09'",
            (canonicals,),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
    conn.commit()


def _seed(conn: psycopg.Connection) -> dict[str, int]:
    _cleanup(conn)
    name_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        for bibcode, abstract in _SEED_PAPERS:
            cur.execute(
                "INSERT INTO papers (bibcode, abstract) VALUES (%s, %s)",
                (bibcode, abstract),
            )
        for canonical, source, ambiguity in _SEED_ENTITIES:
            cur.execute(
                "INSERT INTO entities (canonical_name, entity_type, source, ambiguity_class) "
                "VALUES (%s, %s, %s, %s::entity_ambiguity_class) RETURNING id",
                (canonical, "test_type", source, ambiguity),
            )
            row = cur.fetchone()
            assert row is not None
            name_to_id[canonical] = int(row[0])

        for canonical, alias in _SEED_ALIASES:
            cur.execute(
                "INSERT INTO entity_aliases (entity_id, alias, alias_source) VALUES (%s, %s, %s)",
                (name_to_id[canonical], alias, "test_seed_u09"),
            )

        for canonical in name_to_id:
            cur.execute(
                "INSERT INTO curated_entity_core (entity_id, query_hits_14d) VALUES (%s, %s) "
                "ON CONFLICT (entity_id) DO NOTHING",
                (name_to_id[canonical], 1),
            )
    conn.commit()
    return name_to_id


@pytest.fixture()
def seeded_conn(dsn: str) -> Iterator[tuple[psycopg.Connection, dict[str, int]]]:
    conn = psycopg.connect(dsn)
    try:
        ids = _seed(conn)
        yield conn, ids
    finally:
        try:
            _cleanup(conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestLinkTier2EndToEnd:
    def test_fetch_entity_rows_returns_curated_surfaces(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_conn
        rows = link_tier2.fetch_entity_rows(conn)
        # We should see canonical + alias rows for all 3 seeded entities.
        surfaces = {(r.entity_id, r.surface, r.is_alias) for r in rows}
        hst_id = ids["Hubble Space Telescope"]
        jwst_id = ids["James Webb Space Telescope"]
        alma_id = ids["ALMA Observatory"]
        assert (hst_id, "Hubble Space Telescope", False) in surfaces
        assert (hst_id, "HST", True) in surfaces
        assert (jwst_id, "JWST", True) in surfaces
        assert (alma_id, "ALMA", True) in surfaces

    def test_run_writes_tier2_rows_honoring_ambiguity(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_conn
        stats = link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09_",
            max_per_entity=1_000,
        )
        assert stats.papers_scanned == len(_SEED_PAPERS)
        assert stats.rows_inserted >= 3

        hst_id = ids["Hubble Space Telescope"]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, entity_id, tier, link_type, match_method "
                "  FROM document_entities "
                " WHERE bibcode LIKE 'test_u09_%' AND tier = 2"
            )
            rows = cur.fetchall()

        assert rows, "expected at least one tier-2 row"
        for _bibcode, _entity_id, tier, link_type, method in rows:
            assert tier == 2
            assert link_type == link_tier2.TIER2_LINK_TYPE
            assert method == link_tier2.TIER2_MATCH_METHOD

        # HST must only appear for paper 0001 (co-present with "Hubble
        # Space Telescope"), NOT for paper 0002 (HST alone).
        hst_bibcodes = {b for b, e, *_ in rows if e == hst_id}
        assert "test_u09_0001" in hst_bibcodes
        assert "test_u09_0002" not in hst_bibcodes

    def test_per_entity_cap_demotes_link_policy(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_conn
        # JWST matches on papers 0003 AND 0004. Force cap=1 to trigger
        # demotion on the second match.
        stats = link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09_",
            max_per_entity=1,
        )
        assert stats.entities_demoted >= 1

        jwst_id = ids["James Webb Space Telescope"]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT link_policy::text FROM entities WHERE id = %s",
                (jwst_id,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "llm_only", f"JWST should be demoted, got {row[0]}"

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_entities " " WHERE entity_id = %s AND tier = 2",
                (jwst_id,),
            )
            n = cur.fetchone()[0]
        assert n == 1, f"cap=1 should cap JWST at exactly 1 tier-2 row, got {n}"

    def test_dry_run_rolls_back(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_conn
        link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09_",
            max_per_entity=1_000,
            dry_run=True,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_entities "
                " WHERE bibcode LIKE 'test_u09_%' AND tier = 2"
            )
            n = cur.fetchone()[0]
        assert n == 0
