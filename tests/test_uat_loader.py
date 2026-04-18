"""Integration test for ``scix.uat.load_relationships``.

Regression guard for the loader bug where 2,180 parsed relationships from the
UAT source produced zero rows in the ``uat_relationships`` table: the loader
returned ``cursor.rowcount`` from an ``INSERT ... ON CONFLICT DO NOTHING``,
which reports 0 on idempotent re-runs and hides silent FK drops.

The fix in ``src/scix/uat.py`` returns the count of input relationships
actually present in ``uat_relationships`` after the load; this test pins
that contract against a fresh ``scix_test`` database.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from helpers import get_test_dsn

from scix.uat import (
    UATConcept,
    UATRelationship,
    load_concepts,
    load_relationships,
)

# Fixture: 20 parent-child relationships rooted at concept 1, children 2..21.
# All 21 concepts are inserted into uat_concepts before load_relationships
# runs so the FK check cannot drop any rows.
PARENT_ID = "http://astrothesaurus.org/uat/1"
FIXTURE_SIZE = 20


def _concept(uri: str, label: str) -> UATConcept:
    return UATConcept(
        concept_id=uri,
        preferred_label=label,
        alternate_labels=(),
        definition=None,
        level=0,
    )


def _fixture_concepts() -> list[UATConcept]:
    concepts = [_concept(PARENT_ID, "Root")]
    for i in range(2, 2 + FIXTURE_SIZE):
        concepts.append(_concept(f"http://astrothesaurus.org/uat/{i}", f"Child {i}"))
    return concepts


def _fixture_relationships() -> list[UATRelationship]:
    return [
        UATRelationship(
            parent_id=PARENT_ID,
            child_id=f"http://astrothesaurus.org/uat/{i}",
        )
        for i in range(2, 2 + FIXTURE_SIZE)
    ]


def _has_uat_tables(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN (
                  'uat_concepts', 'uat_relationships', 'paper_uat_mappings'
              )
            """)
        return cur.fetchone()[0] == 3


def _wipe_uat_tables(conn: psycopg.Connection) -> None:
    """Wipe only the fixture rows so the test is side-effect free on re-runs."""
    with conn.cursor() as cur:
        fixture_uris = [PARENT_ID] + [
            f"http://astrothesaurus.org/uat/{i}" for i in range(2, 2 + FIXTURE_SIZE)
        ]
        cur.execute(
            "DELETE FROM paper_uat_mappings WHERE concept_id = ANY(%s)",
            (fixture_uris,),
        )
        cur.execute(
            "DELETE FROM uat_relationships WHERE parent_id = ANY(%s) OR child_id = ANY(%s)",
            (fixture_uris, fixture_uris),
        )
        cur.execute("DELETE FROM uat_concepts WHERE concept_id = ANY(%s)", (fixture_uris,))
    conn.commit()


@pytest.fixture()
def test_conn() -> Iterator[psycopg.Connection]:
    """Yield a connection to SCIX_TEST_DSN; skip if unset or production-shaped."""
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip(
            "SCIX_TEST_DSN not set (or points at production). "
            "Set SCIX_TEST_DSN=dbname=scix_test to run."
        )

    try:
        conn = psycopg.connect(dsn)
    except psycopg.OperationalError as exc:
        pytest.skip(f"scix_test database not available: {exc}")

    try:
        if not _has_uat_tables(conn):
            pytest.skip("UAT tables not found (migration 007 not applied)")

        _wipe_uat_tables(conn)
        yield conn
    finally:
        try:
            _wipe_uat_tables(conn)
        finally:
            conn.close()


@pytest.mark.integration
class TestLoadRelationships:
    """Regression test pinning the loader at 20 rows for a 20-row fixture."""

    def test_inserts_expected_row_count(self, test_conn: psycopg.Connection) -> None:
        concepts = _fixture_concepts()
        rels = _fixture_relationships()

        load_concepts(test_conn, concepts)
        returned = load_relationships(test_conn, rels)

        assert (
            returned == FIXTURE_SIZE
        ), f"load_relationships returned {returned}, expected {FIXTURE_SIZE}"

        with test_conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM uat_relationships
                WHERE parent_id = %s
                  AND child_id = ANY(%s)
                """,
                (
                    PARENT_ID,
                    [f"http://astrothesaurus.org/uat/{i}" for i in range(2, 2 + FIXTURE_SIZE)],
                ),
            )
            assert cur.fetchone()[0] == FIXTURE_SIZE

    def test_idempotent_re_run(self, test_conn: psycopg.Connection) -> None:
        """Re-running must still report 20 rows present, not 0."""
        concepts = _fixture_concepts()
        rels = _fixture_relationships()

        load_concepts(test_conn, concepts)
        first = load_relationships(test_conn, rels)
        second = load_relationships(test_conn, rels)

        assert first == FIXTURE_SIZE
        assert second == FIXTURE_SIZE, (
            "load_relationships returned 0 on re-run — regression in return-value "
            "semantics (ON CONFLICT DO NOTHING rowcount trap)."
        )

    def test_fk_missing_parent_is_skipped_not_raised(self, test_conn: psycopg.Connection) -> None:
        """A relationship referencing a concept that was never loaded must be
        silently dropped by the FK preflight, not raise a FK violation."""
        concepts = _fixture_concepts()
        rels = _fixture_relationships()
        # Inject one relationship whose parent is absent from uat_concepts.
        bogus = UATRelationship(
            parent_id="http://astrothesaurus.org/uat/does-not-exist",
            child_id="http://astrothesaurus.org/uat/2",
        )
        rels_with_bogus = rels + [bogus]

        load_concepts(test_conn, concepts)
        returned = load_relationships(test_conn, rels_with_bogus)

        # Valid rows still land; the bogus row is dropped.
        assert returned == FIXTURE_SIZE
        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM uat_relationships WHERE parent_id = %s",
                (bogus.parent_id,),
            )
            assert cur.fetchone()[0] == 0
