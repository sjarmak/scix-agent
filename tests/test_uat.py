"""Unit and integration tests for UAT concept hierarchy module.

Unit tests verify SKOS parsing, level computation, and relationship extraction
without any database or network access. Integration tests (marked with
@pytest.mark.integration) require a running scix database with migration 007.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import psycopg
import pytest
from helpers import DSN

from scix.uat import (
    UATConcept,
    UATRelationship,
    _pg_text_array,
    hierarchical_search,
    load_concepts,
    load_relationships,
    map_keywords_exact,
    parse_skos,
)

# ---------------------------------------------------------------------------
# SKOS fixture — a minimal 3-concept hierarchy
# ---------------------------------------------------------------------------

SKOS_FIXTURE = """\
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:skos="http://www.w3.org/2004/02/skos/core#">
  <skos:Concept rdf:about="http://astrothesaurus.org/uat/1">
    <skos:prefLabel>Astronomy</skos:prefLabel>
    <skos:narrower rdf:resource="http://astrothesaurus.org/uat/2"/>
  </skos:Concept>
  <skos:Concept rdf:about="http://astrothesaurus.org/uat/2">
    <skos:prefLabel>Galaxies</skos:prefLabel>
    <skos:altLabel>Galaxy</skos:altLabel>
    <skos:broader rdf:resource="http://astrothesaurus.org/uat/1"/>
    <skos:narrower rdf:resource="http://astrothesaurus.org/uat/3"/>
    <skos:definition>Large systems of stars</skos:definition>
  </skos:Concept>
  <skos:Concept rdf:about="http://astrothesaurus.org/uat/3">
    <skos:prefLabel>Spiral galaxies</skos:prefLabel>
    <skos:broader rdf:resource="http://astrothesaurus.org/uat/2"/>
  </skos:Concept>
</rdf:RDF>
"""

UAT_1 = "http://astrothesaurus.org/uat/1"
UAT_2 = "http://astrothesaurus.org/uat/2"
UAT_3 = "http://astrothesaurus.org/uat/3"


@pytest.fixture()
def skos_file(tmp_path: Path) -> Path:
    """Write the SKOS fixture to a temporary file and return its path."""
    p = tmp_path / "test_uat.rdf"
    p.write_text(SKOS_FIXTURE, encoding="utf-8")
    return p


@pytest.fixture()
def parsed(skos_file: Path) -> tuple[list[UATConcept], list[UATRelationship]]:
    """Parse the SKOS fixture."""
    return parse_skos(skos_file)


# ---------------------------------------------------------------------------
# Unit tests (no database, no network)
# ---------------------------------------------------------------------------


class TestParseSKOS:
    """Verify basic SKOS parsing: concept count, labels, definitions."""

    def test_concept_count(self, parsed: tuple) -> None:
        concepts, _ = parsed
        assert len(concepts) == 3

    def test_relationship_count(self, parsed: tuple) -> None:
        _, relationships = parsed
        assert len(relationships) == 2

    def test_preferred_labels(self, parsed: tuple) -> None:
        concepts, _ = parsed
        labels = {c.concept_id: c.preferred_label for c in concepts}
        assert labels[UAT_1] == "Astronomy"
        assert labels[UAT_2] == "Galaxies"
        assert labels[UAT_3] == "Spiral galaxies"

    def test_definition(self, parsed: tuple) -> None:
        concepts, _ = parsed
        by_id = {c.concept_id: c for c in concepts}
        assert by_id[UAT_2].definition == "Large systems of stars"
        assert by_id[UAT_1].definition is None
        assert by_id[UAT_3].definition is None


class TestConceptLevels:
    """Verify BFS level assignment from root concepts."""

    def test_root_level(self, parsed: tuple) -> None:
        concepts, _ = parsed
        by_id = {c.concept_id: c for c in concepts}
        assert by_id[UAT_1].level == 0

    def test_child_level(self, parsed: tuple) -> None:
        concepts, _ = parsed
        by_id = {c.concept_id: c for c in concepts}
        assert by_id[UAT_2].level == 1

    def test_grandchild_level(self, parsed: tuple) -> None:
        concepts, _ = parsed
        by_id = {c.concept_id: c for c in concepts}
        assert by_id[UAT_3].level == 2


class TestAlternateLabels:
    """Verify alternate label extraction."""

    def test_galaxies_has_alt_label(self, parsed: tuple) -> None:
        concepts, _ = parsed
        by_id = {c.concept_id: c for c in concepts}
        assert "Galaxy" in by_id[UAT_2].alternate_labels

    def test_astronomy_no_alt_labels(self, parsed: tuple) -> None:
        concepts, _ = parsed
        by_id = {c.concept_id: c for c in concepts}
        assert by_id[UAT_1].alternate_labels == ()

    def test_alt_labels_are_tuple(self, parsed: tuple) -> None:
        concepts, _ = parsed
        for c in concepts:
            assert isinstance(c.alternate_labels, tuple)


class TestRelationshipDirection:
    """Verify parent-child direction derived from skos:broader."""

    def test_astronomy_is_parent_of_galaxies(self, parsed: tuple) -> None:
        _, relationships = parsed
        rels = {(r.parent_id, r.child_id) for r in relationships}
        assert (UAT_1, UAT_2) in rels

    def test_galaxies_is_parent_of_spiral(self, parsed: tuple) -> None:
        _, relationships = parsed
        rels = {(r.parent_id, r.child_id) for r in relationships}
        assert (UAT_2, UAT_3) in rels

    def test_no_reverse_relationships(self, parsed: tuple) -> None:
        _, relationships = parsed
        rels = {(r.parent_id, r.child_id) for r in relationships}
        # narrower should NOT create reversed parent->child duplicates
        assert (UAT_2, UAT_1) not in rels
        assert (UAT_3, UAT_2) not in rels


class TestPgTextArray:
    """Verify PostgreSQL text array literal formatting."""

    def test_empty(self) -> None:
        assert _pg_text_array(()) == "{}"

    def test_single(self) -> None:
        assert _pg_text_array(("Galaxy",)) == '{"Galaxy"}'

    def test_multiple(self) -> None:
        result = _pg_text_array(("one", "two"))
        assert result == '{"one","two"}'

    def test_escapes_quotes(self) -> None:
        result = _pg_text_array(('say "hello"',))
        assert '\\"' in result

    def test_escapes_backslash(self) -> None:
        result = _pg_text_array(("back\\slash",))
        assert "\\\\" in result


class TestFrozenDataclasses:
    """Verify that data types are immutable."""

    def test_concept_is_frozen(self) -> None:
        c = UATConcept(
            concept_id="test",
            preferred_label="Test",
            alternate_labels=(),
            definition=None,
            level=0,
        )
        with pytest.raises(AttributeError):
            c.preferred_label = "Modified"  # type: ignore[misc]

    def test_relationship_is_frozen(self) -> None:
        r = UATRelationship(parent_id="a", child_id="b")
        with pytest.raises(AttributeError):
            r.parent_id = "c"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration tests (require database with migration 007)
# ---------------------------------------------------------------------------


def _has_uat_tables(conn: psycopg.Connection) -> bool:
    """Check if UAT tables exist (migration 007 applied)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('uat_concepts', 'uat_relationships', 'paper_uat_mappings')
        """)
        return cur.fetchone()[0] == 3


@pytest.fixture()
def db_conn():
    """Provide a database connection, skip if unavailable or tables missing."""
    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("Database not available")
        return

    if not _has_uat_tables(conn):
        conn.close()
        pytest.skip("UAT tables not found (migration 007 not applied)")
        return

    yield conn
    conn.close()


@pytest.mark.integration
class TestLoadConcepts:
    """Integration: load SKOS fixture concepts into database."""

    def test_load_and_count(self, db_conn: psycopg.Connection, skos_file: Path) -> None:
        concepts, _ = parse_skos(skos_file)

        # Clean up before test
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

        count = load_concepts(db_conn, concepts)
        assert count == 3

        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM uat_concepts")
            assert cur.fetchone()[0] == 3

        # Verify data integrity
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT preferred_label FROM uat_concepts WHERE concept_id = %s",
                (UAT_2,),
            )
            assert cur.fetchone()[0] == "Galaxies"

        # Clean up after test
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

    def test_load_relationships(self, db_conn: psycopg.Connection, skos_file: Path) -> None:
        concepts, relationships = parse_skos(skos_file)

        # Clean up before test
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

        load_concepts(db_conn, concepts)
        count = load_relationships(db_conn, relationships)
        assert count == 2

        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM uat_relationships")
            assert cur.fetchone()[0] == 2

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()


@pytest.mark.integration
class TestMapKeywords:
    """Integration: map paper keywords to UAT concepts."""

    def test_exact_match_count(self, db_conn: psycopg.Connection, skos_file: Path) -> None:
        concepts, relationships = parse_skos(skos_file)

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

        load_concepts(db_conn, concepts)
        load_relationships(db_conn, relationships)

        # Check if any papers have keywords that match our fixture
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM papers p, LATERAL unnest(p.keywords) AS kw
                WHERE lower(kw) IN ('astronomy', 'galaxies', 'galaxy', 'spiral galaxies')
            """)
            potential_matches = cur.fetchone()[0]

        count = map_keywords_exact(db_conn)

        if potential_matches > 0:
            assert count > 0
        else:
            assert count >= 0  # No matching papers is valid

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()


@pytest.mark.integration
class TestHierarchicalSearch:
    """Integration: verify hierarchical search returns expected results."""

    def test_search_from_root(self, db_conn: psycopg.Connection, skos_file: Path) -> None:
        concepts, relationships = parse_skos(skos_file)

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

        load_concepts(db_conn, concepts)
        load_relationships(db_conn, relationships)
        map_keywords_exact(db_conn)

        # Search from root — should include all descendants
        results = hierarchical_search(db_conn, UAT_1, limit=10)
        assert isinstance(results, list)

        # Each result should have expected keys
        for row in results:
            assert "bibcode" in row
            assert "title" in row
            assert "year" in row

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

    def test_search_returns_dicts(self, db_conn: psycopg.Connection, skos_file: Path) -> None:
        concepts, relationships = parse_skos(skos_file)

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings")
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()

        load_concepts(db_conn, concepts)
        load_relationships(db_conn, relationships)

        results = hierarchical_search(db_conn, UAT_1, limit=5)
        assert isinstance(results, list)
        for row in results:
            assert isinstance(row, dict)

        # Clean up
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM uat_relationships")
            cur.execute("DELETE FROM uat_concepts")
        db_conn.commit()
