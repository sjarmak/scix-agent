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
import tempfile
from typing import Iterator
from unittest import mock

import psycopg
import pytest

from scix.aho_corasick import (
    EntityRow,
    build_automaton,
    link_abstract,
    link_text,
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


class TestLinkTextAlias:
    """``link_text`` is the canonical name; ``link_abstract`` stays as a
    back-compat alias because section_linker.py and other callers still
    import the old name. Both must do exactly the same thing."""

    def test_link_text_matches_link_abstract(self) -> None:
        automaton = build_automaton(_jwst_rows())
        text = "JWST observations confirmed water vapor."
        assert link_text(text, automaton) == link_abstract(text, automaton)

    def test_link_text_handles_empty(self) -> None:
        automaton = build_automaton(_jwst_rows())
        assert link_text("", automaton) == []
        assert link_text(None, automaton) == []  # type: ignore[arg-type]

    def test_link_text_is_text_agnostic(self) -> None:
        # Same automaton, body-shaped input. The function must not care
        # whether the text came from an abstract or a paper body.
        automaton = build_automaton(_jwst_rows())
        body = (
            "1. Introduction\n\nWe describe observations.\n\n"
            "2. Methods\n\nJWST data were reduced with the standard pipeline."
        )
        out = link_text(body, automaton)
        assert any(c.entity_id == 202 for c in out)


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


# ---------------------------------------------------------------------------
# Summary report tests
# ---------------------------------------------------------------------------


class TestTier2Summary:
    def test_write_summary_creates_file(self) -> None:
        stats = link_tier2.Tier2Stats(
            papers_scanned=100,
            candidates_generated=50,
            rows_inserted=42,
            entities_demoted=2,
            entities_with_links=10,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = pathlib.Path(tmpdir) / "tier2_summary.md"
            link_tier2.write_tier2_summary(stats, outpath, wall_seconds=123.4)
            assert outpath.exists()
            content = outpath.read_text()
            assert "100" in content  # papers_scanned
            assert "42" in content  # rows_inserted
            assert "2" in content  # entities_demoted
            assert "10" in content  # entities_with_links
            assert "2m 3s" in content  # wall time formatting

    def test_write_summary_contains_sections(self) -> None:
        stats = link_tier2.Tier2Stats(
            papers_scanned=23_000_000,
            candidates_generated=500_000,
            rows_inserted=450_000,
            entities_demoted=0,
            entities_with_links=500,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = pathlib.Path(tmpdir) / "tier2_summary.md"
            link_tier2.write_tier2_summary(stats, outpath, wall_seconds=3661.0)
            content = outpath.read_text()
            assert "# Tier 2 Aho-Corasick Linker Summary" in content
            assert "Papers scanned" in content
            assert "Rows inserted" in content
            assert "Entities demoted" in content
            assert "1h 1m 1s" in content

    def test_write_summary_dry_run_label(self) -> None:
        stats = link_tier2.Tier2Stats(
            papers_scanned=10,
            candidates_generated=5,
            rows_inserted=3,
            entities_demoted=0,
            entities_with_links=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = pathlib.Path(tmpdir) / "tier2_summary.md"
            link_tier2.write_tier2_summary(stats, outpath, wall_seconds=1.0, dry_run=True)
            content = outpath.read_text()
            assert "DRY RUN" in content


# ---------------------------------------------------------------------------
# Production guard tests (no DB required)
# ---------------------------------------------------------------------------


class TestProductionGuard:
    """Verify --allow-prod guard on link_tier2.main()."""

    def test_refuses_production_dsn_without_allow_prod(self) -> None:
        """main() must exit non-zero when DSN is production and --allow-prod is absent."""
        rc = link_tier2.main(["--db-url", "dbname=scix"])
        assert rc == 2

    def test_accepts_non_production_dsn_without_flag(self) -> None:
        """main() should not reject a non-production DSN even without --allow-prod.

        We pass --dry-run and --bibcode-prefix to avoid any real DB work;
        the guard check happens before the connection attempt, so a
        connection error is expected if SCIX_TEST_DSN is unset. We only
        need to verify the guard does NOT reject.
        """
        # Use a clearly non-production DSN. Connection will fail but the
        # guard should pass (rc != 2).
        try:
            rc = link_tier2.main(["--db-url", "dbname=scix_test", "--dry-run"])
        except psycopg.OperationalError:
            # Connection failure is fine — means we passed the guard.
            return
        # If we somehow connected (scix_test running), any rc != 2 is fine.
        assert rc != 2

    def test_allows_production_dsn_with_flag(self) -> None:
        """main() should pass the guard when --allow-prod is given, even for
        a production DSN. We mock get_connection so no real DB call is made."""
        mock_conn = mock.MagicMock()
        mock_conn.cursor.return_value.__enter__ = mock.MagicMock(
            return_value=mock.MagicMock(fetchall=mock.MagicMock(return_value=[]))
        )
        mock_conn.cursor.return_value.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("link_tier2.get_connection", return_value=mock_conn):
            rc = link_tier2.main(
                [
                    "--db-url",
                    "dbname=scix",
                    "--allow-prod",
                    "--dry-run",
                    "--bibcode-prefix",
                    "NONEXISTENT_",
                ]
            )
        assert rc != 2


# ---------------------------------------------------------------------------
# Body-source linking — dbl.19
# ---------------------------------------------------------------------------

# Body fixtures share the same entity pool as _SEED_ENTITIES so we can
# reuse _seed/_cleanup. The bodies deliberately mention entities that do
# NOT appear in the abstract for the same paper, mirroring real corpora
# where instruments / pipelines surface in methods, not abstracts.

_SEED_BODY_PAPERS: list[tuple[str, str, str, list[str], list[str]]] = [
    # (bibcode, abstract, body, property[], arxiv_class[])
    (
        "test_u09b_0001",
        # Abstract mentions JWST so the abstract-pass also produces a
        # tier-2 row on this paper (used by the separability test).
        "We present a multiwavelength survey including JWST imaging.",
        (
            "1. Introduction\n\n"
            "We present a multiwavelength survey.\n\n"
            "2. Methods\n\n"
            "ALMA interferometry provided dust continuum maps. "
            "Hubble Space Telescope imaging supplied UV photometry."
        ),
        ["OPENACCESS"],
        [],
    ),
    (
        "test_u09b_0002",
        "We describe a quasar field.",
        # Body mentions HST alone — homograph, must NOT fire (no long-form
        # co-presence in the body itself).
        "Methods\n\nWe used HST as a generic acronym in this body.",
        ["OPENACCESS"],
        [],
    ),
    (
        "test_u09b_0003",
        "Closed-access paper abstract — no instruments.",
        # Body mentions JWST but paper is NOT OA and NOT a preprint — the
        # OA gate must filter it out by default.
        "Methods\n\nJWST spectra confirmed the detection.",
        [],  # no OPENACCESS
        [],  # no arxiv_class
    ),
    (
        "test_u09b_0004",
        "Preprint paper abstract.",
        # Body mentions ALMA; not OA but IS a preprint, so it passes the
        # OA gate.
        "Methods\n\nALMA observations covered band 6.",
        [],
        ["astro-ph.GA"],
    ),
]


def _seed_bodies(conn: psycopg.Connection) -> dict[str, int]:
    """Seed papers WITH bodies plus the same 3 entities as the abstract
    fixture. Reuses _cleanup() against the abstract bibcodes too so the
    fixture is order-independent.
    """
    bibcodes = [p[0] for p in _SEED_BODY_PAPERS]
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

    name_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        for bibcode, abstract, body, prop, arxiv_class in _SEED_BODY_PAPERS:
            cur.execute(
                "INSERT INTO papers (bibcode, abstract, body, property, arxiv_class) "
                "VALUES (%s, %s, %s, %s, %s)",
                (bibcode, abstract, body, prop, arxiv_class),
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
                (name_to_id[canonical], alias, "test_seed_u09b"),
            )
        for canonical in name_to_id:
            cur.execute(
                "INSERT INTO curated_entity_core (entity_id, query_hits_14d) VALUES (%s, %s) "
                "ON CONFLICT (entity_id) DO NOTHING",
                (name_to_id[canonical], 1),
            )
    conn.commit()
    return name_to_id


def _cleanup_bodies(conn: psycopg.Connection) -> None:
    bibcodes = [p[0] for p in _SEED_BODY_PAPERS]
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


@pytest.fixture()
def seeded_body_conn(dsn: str) -> Iterator[tuple[psycopg.Connection, dict[str, int]]]:
    conn = psycopg.connect(dsn)
    try:
        ids = _seed_bodies(conn)
        yield conn, ids
    finally:
        try:
            _cleanup_bodies(conn)
        finally:
            conn.close()


class TestIterPaperBatchesBody:
    """``iter_paper_batches`` must support a body source AND honor the
    OA gate by default."""

    def test_body_source_yields_body_text(
        self, seeded_body_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_body_conn
        batches = list(
            link_tier2.iter_paper_batches(
                conn,
                bibcode_prefix="test_u09b_",
                text_source=link_tier2.TEXT_SOURCE_BODY,
            )
        )
        flat = [pair for batch in batches for pair in batch]
        # OA gate default: closed-access paper test_u09b_0003 is excluded.
        bibcodes = {bc for bc, _ in flat}
        assert "test_u09b_0001" in bibcodes
        assert "test_u09b_0002" in bibcodes
        assert (
            "test_u09b_0003" not in bibcodes
        ), "closed-access paper must be filtered by OA gate by default"
        assert "test_u09b_0004" in bibcodes  # preprint
        # The yielded text is the body, not the abstract.
        for bibcode, text in flat:
            if bibcode == "test_u09b_0001":
                assert "Methods" in text
                assert "ALMA" in text

    def test_body_source_with_include_closed(
        self, seeded_body_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_body_conn
        batches = list(
            link_tier2.iter_paper_batches(
                conn,
                bibcode_prefix="test_u09b_",
                text_source=link_tier2.TEXT_SOURCE_BODY,
                include_closed=True,
            )
        )
        bibcodes = {bc for batch in batches for bc, _ in batch}
        # All four papers when the OA gate is bypassed.
        assert "test_u09b_0003" in bibcodes

    def test_abstract_source_unchanged(
        self, seeded_body_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_body_conn
        # Default text_source remains "abstract" — back-compat.
        batches = list(
            link_tier2.iter_paper_batches(
                conn,
                bibcode_prefix="test_u09b_",
            )
        )
        # Abstract path doesn't apply the OA gate (abstracts are
        # universally indexable).
        bibcodes = {bc for batch in batches for bc, _ in batch}
        assert bibcodes == {p[0] for p in _SEED_BODY_PAPERS}


class TestRunTier2BodyMode:
    def test_body_mode_writes_tier3_rows(
        self, seeded_body_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_body_conn
        stats = link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09b_",
            max_per_entity=1_000,
            text_source=link_tier2.TEXT_SOURCE_BODY,
        )
        # 0001 hits 2 entities (HST via long-form + ALMA);
        # 0002 hits 0 (HST homograph alone, no long-form);
        # 0003 is filtered by OA gate;
        # 0004 hits 1 entity (ALMA).
        assert stats.papers_scanned == 3
        assert stats.rows_inserted >= 3

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, entity_id, tier, link_type, match_method, evidence "
                "  FROM document_entities "
                " WHERE bibcode LIKE 'test_u09b_%'"
            )
            rows = cur.fetchall()

        assert rows, "expected tier-3 body rows"
        for _bibcode, _entity_id, tier, link_type, method, evidence in rows:
            assert tier == link_tier2.TIER3_TIER == 3
            assert link_type == link_tier2.TIER3_LINK_TYPE == "body_match"
            assert method == link_tier2.TIER3_MATCH_METHOD
            # evidence.source must record where the match came from so
            # downstream readers can split abstract vs body coverage.
            assert evidence["source"] == "body"

    def test_body_mode_oa_gate_drops_closed(
        self, seeded_body_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_body_conn
        link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09b_",
            max_per_entity=1_000,
            text_source=link_tier2.TEXT_SOURCE_BODY,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_entities WHERE bibcode = 'test_u09b_0003'")
            n = cur.fetchone()[0]
        assert n == 0, "closed-access paper must not be linked at body source"

    def test_body_mode_separable_from_abstract(
        self, seeded_body_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        # Run abstract pass + body pass; both must coexist on the same
        # paper because tier differs.
        conn, ids = seeded_body_conn
        link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09b_",
            max_per_entity=1_000,
            text_source=link_tier2.TEXT_SOURCE_ABSTRACT,
        )
        link_tier2.run_tier2_link(
            conn,
            workers=1,
            bibcode_prefix="test_u09b_",
            max_per_entity=1_000,
            text_source=link_tier2.TEXT_SOURCE_BODY,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tier, count(*) FROM document_entities "
                " WHERE bibcode LIKE 'test_u09b_%' GROUP BY tier ORDER BY tier"
            )
            tier_counts = dict(cur.fetchall())
        assert tier_counts.get(link_tier2.TIER2_TIER, 0) >= 1
        assert tier_counts.get(link_tier2.TIER3_TIER, 0) >= 1


class TestEvidenceSourceField:
    def test_abstract_evidence_source_field(self) -> None:
        # Pure unit test — no DB required.
        from scix.aho_corasick import LinkCandidate

        cand = LinkCandidate(
            entity_id=1,
            canonical_name="X",
            matched_surface="x",
            start=0,
            end=1,
            confidence=0.85,
            ambiguity_class="unique",
        )
        ev_abs = link_tier2._evidence_json(cand, source=link_tier2.TEXT_SOURCE_ABSTRACT)
        ev_body = link_tier2._evidence_json(cand, source=link_tier2.TEXT_SOURCE_BODY)
        import json

        assert json.loads(ev_abs)["source"] == "abstract"
        assert json.loads(ev_body)["source"] == "body"


class TestBodyModeCli:
    def test_text_source_body_argument_parses(self) -> None:
        parser = link_tier2._build_parser()
        args = parser.parse_args(
            [
                "--text-source",
                "body",
                "--bibcode-prefix",
                "test_",
            ]
        )
        assert args.text_source == "body"

    def test_text_source_default_abstract(self) -> None:
        parser = link_tier2._build_parser()
        args = parser.parse_args([])
        assert args.text_source == link_tier2.TEXT_SOURCE_ABSTRACT
