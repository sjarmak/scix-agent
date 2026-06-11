"""Tests for :mod:`scix.section_linker` and the section_entities writer.

Three layers:

* ``parse_sections`` — JSONB normalization edge cases (no DB).
* ``link_paper_sections`` — section-grain matching with the real
  Aho-Corasick automaton (no DB).
* End-to-end CLI smoke test — seeds papers + papers_fulltext + entities
  in scix_test and exercises ``run_section_link`` with ``dry_run=True`` and
  ``dry_run=False``. Skipped when ``SCIX_TEST_DSN`` is not set.

The DB layer is gated on ``SCIX_TEST_DSN`` so the pure-library tests run
anywhere; the integration tests run on workstations that have the test DB
provisioned.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Iterator

import psycopg
import pytest

from scix.aho_corasick import EntityRow, build_automaton
from scix.section_linker import (
    Section,
    SectionLinkCandidate,
    link_paper_sections,
    parse_sections,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tests.helpers import get_test_dsn  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jwst_automaton():
    """Single ``unique`` entity (no homograph gating concerns)."""
    rows = [
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
    return build_automaton(rows)


@pytest.fixture
def two_entity_automaton():
    """JWST (unique) + Kepler (unique). Used for cross-section dedupe checks."""
    rows = [
        EntityRow(
            entity_id=202,
            surface="JWST",
            canonical_name="James Webb Space Telescope",
            ambiguity_class="unique",
            is_alias=True,
        ),
        EntityRow(
            entity_id=303,
            surface="Kepler",
            canonical_name="Kepler",
            ambiguity_class="unique",
            is_alias=False,
        ),
    ]
    return build_automaton(rows)


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_none_returns_empty(self) -> None:
        assert parse_sections(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert parse_sections([]) == []

    def test_indices_are_zero_based_and_array_ordered(self) -> None:
        sections = parse_sections(
            [
                {"heading": "Introduction", "text": "intro body"},
                {"heading": "Methods", "text": "methods body"},
                {"heading": "Results", "text": "results body"},
            ]
        )
        assert [s.section_index for s in sections] == [0, 1, 2]
        assert [s.heading for s in sections] == [
            "Introduction",
            "Methods",
            "Results",
        ]

    def test_missing_heading_becomes_none(self) -> None:
        sections = parse_sections([{"text": "body without heading"}])
        assert len(sections) == 1
        assert sections[0].heading is None
        assert sections[0].text == "body without heading"

    def test_missing_text_becomes_empty_string(self) -> None:
        sections = parse_sections([{"heading": "Empty"}])
        assert len(sections) == 1
        assert sections[0].heading == "Empty"
        assert sections[0].text == ""

    def test_drops_entries_with_neither_heading_nor_text(self) -> None:
        sections = parse_sections(
            [
                {"heading": "Real", "text": "body"},
                {},
                {"heading": None, "text": ""},
                {"heading": "Tail", "text": "tail body"},
            ]
        )
        # The dropped entries are skipped, but the section_index of the
        # surviving entries is still computed from the SOURCE array
        # position so it stays aligned with what other consumers see.
        assert [s.section_index for s in sections] == [0, 3]
        assert [s.heading for s in sections] == ["Real", "Tail"]

    def test_non_dict_entries_are_skipped(self) -> None:
        sections = parse_sections(
            [
                {"heading": "Real", "text": "body"},
                "not a dict",  # type: ignore[list-item]
                None,  # type: ignore[list-item]
                ["also not a dict"],  # type: ignore[list-item]
                {"heading": "Tail", "text": "tail"},
            ]
        )
        assert [s.section_index for s in sections] == [0, 4]

    def test_empty_string_heading_normalized_to_none(self) -> None:
        sections = parse_sections([{"heading": "", "text": "body"}])
        assert len(sections) == 1
        assert sections[0].heading is None

    def test_returns_frozen_section_dataclass(self) -> None:
        sections = parse_sections([{"heading": "H", "text": "t"}])
        assert isinstance(sections[0], Section)
        with pytest.raises((AttributeError, Exception)):
            sections[0].text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# link_paper_sections
# ---------------------------------------------------------------------------


class TestLinkPaperSections:
    def test_empty_sections_yields_no_candidates(self, jwst_automaton) -> None:
        assert link_paper_sections([], jwst_automaton) == []

    def test_section_with_empty_text_is_skipped(self, jwst_automaton) -> None:
        sections = [Section(section_index=0, heading="Methods", text="")]
        assert link_paper_sections(sections, jwst_automaton) == []

    def test_section_with_no_match_yields_no_candidate(
        self, jwst_automaton
    ) -> None:
        sections = [
            Section(
                section_index=0,
                heading="Methods",
                text="We used ground-based telescopes only.",
            )
        ]
        assert link_paper_sections(sections, jwst_automaton) == []

    def test_single_match_returns_candidate_with_section_provenance(
        self, jwst_automaton
    ) -> None:
        sections = [
            Section(
                section_index=2,
                heading="Methods",
                text="JWST imaging was used.",
            )
        ]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        cand = candidates[0]
        assert isinstance(cand, SectionLinkCandidate)
        assert cand.section_index == 2
        assert cand.section_heading == "Methods"
        assert cand.section_role == "method"
        assert cand.entity_id == 202
        assert cand.candidate.matched_surface == "JWST"

    def test_multiple_surface_forms_in_one_section_dedupe_to_earliest(
        self, jwst_automaton
    ) -> None:
        # Both "James Webb Space Telescope" (alias=False) and "JWST"
        # (alias=True) hit the same entity_id=202 in this section. Dedupe
        # should keep the earliest start offset (the long-form, position 0).
        text = "James Webb Space Telescope (JWST) is a flagship mission."
        sections = [Section(section_index=0, heading="Intro", text=text)]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        # Earliest mention wins — the long-form starts at offset 0.
        assert candidates[0].candidate.start == 0
        assert candidates[0].candidate.matched_surface == (
            "James Webb Space Telescope"
        )

    def test_cross_section_duplicates_preserved(self, jwst_automaton) -> None:
        # Same entity in two sections — both kept; the whole point of
        # section-grain linking is to capture the section spread.
        sections = [
            Section(section_index=0, heading="Introduction", text="JWST background"),
            Section(section_index=1, heading="Methods", text="JWST observations"),
            Section(section_index=2, heading="Results", text="JWST detected"),
        ]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert {c.section_index for c in candidates} == {0, 1, 2}
        # Same entity, three different sections.
        assert {c.entity_id for c in candidates} == {202}
        # Roles are computed from the heading, not shared.
        assert {(c.section_index, c.section_role) for c in candidates} == {
            (0, "background"),
            (1, "method"),
            (2, "result"),
        }

    def test_two_entities_in_one_section_each_gets_own_row(
        self, two_entity_automaton
    ) -> None:
        sections = [
            Section(
                section_index=0,
                heading="Methods",
                text="We re-analyzed Kepler photometry alongside JWST spectra.",
            )
        ]
        candidates = link_paper_sections(sections, two_entity_automaton)
        assert {c.entity_id for c in candidates} == {202, 303}
        assert {c.section_index for c in candidates} == {0}

    def test_section_role_falls_back_to_other_for_unknown_heading(
        self, jwst_automaton
    ) -> None:
        sections = [
            Section(
                section_index=0,
                heading="Some Unrecognized Heading",
                text="JWST work.",
            )
        ]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        assert candidates[0].section_role == "other"

    def test_section_role_handles_none_heading(self, jwst_automaton) -> None:
        sections = [Section(section_index=0, heading=None, text="JWST work.")]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        assert candidates[0].section_role == "other"
        assert candidates[0].section_heading is None

    def test_numbered_heading_is_classified_after_numbering_strip(
        self, jwst_automaton
    ) -> None:
        sections = [
            Section(
                section_index=0,
                heading="2.1 Data Reduction",
                text="We reduced JWST data with the standard pipeline.",
            )
        ]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        assert candidates[0].section_role == "method"

    def test_results_and_discussion_resolves_to_conclusion(
        self, jwst_automaton
    ) -> None:
        # section_role priority: conclusion > method > result > background.
        sections = [
            Section(
                section_index=0,
                heading="Results and Discussion",
                text="JWST observations show...",
            )
        ]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        assert candidates[0].section_role == "conclusion"


# ---------------------------------------------------------------------------
# SectionLinkCandidate convenience
# ---------------------------------------------------------------------------


class TestSectionLinkCandidateProperty:
    def test_entity_id_property_proxies_underlying_candidate(
        self, jwst_automaton
    ) -> None:
        sections = [Section(section_index=4, heading="Methods", text="JWST works")]
        candidates = link_paper_sections(sections, jwst_automaton)
        assert len(candidates) == 1
        assert candidates[0].entity_id == candidates[0].candidate.entity_id == 202


# ---------------------------------------------------------------------------
# Integration fixture against scix_test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip(
            "SCIX_TEST_DSN must be set to a non-production DSN for "
            "section_linker integration tests"
        )
    return test_dsn


_SEED_BIBCODES: tuple[str, ...] = ("test_67e_001", "test_67e_002")

# Two papers, multi-section bodies. Paper 1 mentions JWST in three different
# sections (intro/methods/results) so we can assert cross-section preservation.
# Paper 2 mentions ALMA only in methods so we can assert role classification.
_SEED_FULLTEXT: dict[str, list[dict]] = {
    "test_67e_001": [
        {"heading": "Introduction", "text": "JWST is a flagship NASA mission."},
        {"heading": "Methods", "text": "We re-reduced the JWST NIRCam data."},
        {"heading": "Results", "text": "JWST detected water vapor."},
        {"heading": "Conclusions", "text": "These observations confirm the model."},
    ],
    "test_67e_002": [
        {"heading": "Introduction", "text": "We study a high-redshift quasar."},
        {"heading": "Methods", "text": "ALMA Observatory data was reduced with CASA."},
    ],
}

_SEED_ENTITIES: list[tuple[str, str, str]] = [
    # (canonical_name, source, ambiguity_class)
    ("James Webb Space Telescope", "unit_test_67e", "unique"),
    ("ALMA Observatory", "unit_test_67e", "domain_safe"),
]

_SEED_ALIASES: list[tuple[str, str]] = [
    ("James Webb Space Telescope", "JWST"),
    ("ALMA Observatory", "ALMA"),
]


def _cleanup(conn: psycopg.Connection) -> None:
    bibcodes = list(_SEED_BIBCODES)
    canonicals = [e[0] for e in _SEED_ENTITIES]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM section_entities WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM papers_fulltext WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM curated_entity_core "
            " WHERE entity_id IN ("
            "     SELECT id FROM entities "
            "      WHERE canonical_name = ANY(%s) "
            "        AND source = 'unit_test_67e'"
            " )",
            (canonicals,),
        )
        cur.execute(
            "DELETE FROM entities "
            " WHERE canonical_name = ANY(%s) AND source = 'unit_test_67e'",
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
        for bibcode in _SEED_BIBCODES:
            cur.execute(
                "INSERT INTO papers (bibcode, abstract) VALUES (%s, %s)",
                (bibcode, "fixture abstract for " + bibcode),
            )
            cur.execute(
                "INSERT INTO papers_fulltext "
                "  (bibcode, source, sections, inline_cites, parser_version) "
                "VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)",
                (
                    bibcode,
                    "test_67e",
                    json.dumps(_SEED_FULLTEXT[bibcode]),
                    "[]",
                    "test_67e_v1",
                ),
            )
        for canonical, source, ambiguity in _SEED_ENTITIES:
            cur.execute(
                "INSERT INTO entities "
                "  (canonical_name, entity_type, source, ambiguity_class) "
                "VALUES (%s, %s, %s, %s::entity_ambiguity_class) RETURNING id",
                (canonical, "test_type", source, ambiguity),
            )
            row = cur.fetchone()
            assert row is not None
            name_to_id[canonical] = int(row[0])
        for canonical, alias in _SEED_ALIASES:
            cur.execute(
                "INSERT INTO entity_aliases "
                "  (entity_id, alias, alias_source) "
                "VALUES (%s, %s, %s)",
                (name_to_id[canonical], alias, "test_seed_67e"),
            )
        for canonical in name_to_id:
            cur.execute(
                "INSERT INTO curated_entity_core "
                "  (entity_id, query_hits_14d) "
                "VALUES (%s, %s) ON CONFLICT (entity_id) DO NOTHING",
                (name_to_id[canonical], 1),
            )
    conn.commit()
    return name_to_id


@pytest.fixture()
def seeded_conn(
    dsn: str,
) -> Iterator[tuple[psycopg.Connection, dict[str, int]]]:
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


class TestSectionLinkerEndToEnd:
    def test_dry_run_returns_stats_and_writes_no_rows(
        self,
        seeded_conn: tuple[psycopg.Connection, dict[str, int]],
    ) -> None:
        # Late import — link_section_entities pulls in link_tier2 which
        # requires SCIX_TEST_DSN to be set; importing at module scope would
        # break the pure-library tests when no DB is configured.
        import link_section_entities

        conn, _ids = seeded_conn
        stats = link_section_entities.run_section_link(
            conn,
            workers=1,
            bibcode_prefix="test_67e_",
            dry_run=True,
        )
        # 2 seeded papers, both have sections. Each produced at least one
        # candidate (JWST x3 for paper 1, ALMA x1 for paper 2 = 4).
        assert stats.papers_scanned == 2
        assert stats.candidates_generated >= 4
        assert stats.entities_with_links >= 2

        # Dry run rolled back — section_entities should still be empty for
        # the test bibcodes.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM section_entities "
                " WHERE bibcode = ANY(%s)",
                (list(_SEED_BIBCODES),),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == 0

    def test_real_run_writes_section_entities_with_role_and_provenance(
        self,
        seeded_conn: tuple[psycopg.Connection, dict[str, int]],
    ) -> None:
        import link_section_entities

        conn, ids = seeded_conn
        stats = link_section_entities.run_section_link(
            conn,
            workers=1,
            bibcode_prefix="test_67e_",
            dry_run=False,
        )
        assert stats.rows_inserted >= 4

        jwst_id = ids["James Webb Space Telescope"]
        alma_id = ids["ALMA Observatory"]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, section_index, entity_id, section_role, "
                "       link_type, tier, match_method "
                "  FROM section_entities "
                " WHERE bibcode = ANY(%s) "
                " ORDER BY bibcode, section_index, entity_id",
                (list(_SEED_BIBCODES),),
            )
            rows = cur.fetchall()

        # Cross-section preservation: paper 1 mentions JWST in 3 sections.
        paper1_jwst_sections = {
            r[1] for r in rows if r[0] == "test_67e_001" and r[2] == jwst_id
        }
        assert paper1_jwst_sections == {0, 1, 2}

        # Role classification flows through to the row.
        roles_paper1_jwst = {
            (r[1], r[3]) for r in rows if r[0] == "test_67e_001" and r[2] == jwst_id
        }
        assert (0, "background") in roles_paper1_jwst  # Introduction
        assert (1, "method") in roles_paper1_jwst  # Methods
        assert (2, "result") in roles_paper1_jwst  # Results

        # Paper 2 ALMA in methods only.
        paper2 = [r for r in rows if r[0] == "test_67e_002"]
        assert len(paper2) == 1
        assert paper2[0][2] == alma_id
        assert paper2[0][3] == "method"

        # Provenance constants applied uniformly.
        assert {r[4] for r in rows} == {"section_match"}
        assert {r[5] for r in rows} == {2}
        assert {r[6] for r in rows} == {"aho_corasick_section"}

    def test_idempotent_writes(
        self,
        seeded_conn: tuple[psycopg.Connection, dict[str, int]],
    ) -> None:
        import link_section_entities

        conn, _ids = seeded_conn
        first = link_section_entities.run_section_link(
            conn, workers=1, bibcode_prefix="test_67e_", dry_run=False
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM section_entities "
                " WHERE bibcode = ANY(%s)",
                (list(_SEED_BIBCODES),),
            )
            row = cur.fetchone()
            assert row is not None
            after_first = row[0]

        # Re-run; ON CONFLICT DO NOTHING means the table count must not grow.
        link_section_entities.run_section_link(
            conn, workers=1, bibcode_prefix="test_67e_", dry_run=False
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM section_entities "
                " WHERE bibcode = ANY(%s)",
                (list(_SEED_BIBCODES),),
            )
            row = cur.fetchone()
            assert row is not None
            after_second = row[0]

        assert after_first >= 4
        assert after_second == after_first
        # Stats from the first run already covered candidates_generated.
        assert first.rows_inserted >= 4
