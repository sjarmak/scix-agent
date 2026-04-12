"""Tests for src/scix/citation_consistency.py (PRD §S1).

Integration tests — require SCIX_TEST_DSN pointing at a non-production DB
with migration 037 applied.

Closed-form fixture: a 5-paper network.

  P0 cites P1, P2, P3, P4
  P1 links to entity E=42
  P2 links to entity E=42
  P3 links to entity E=99 (unrelated)
  P4 has no links

Then consistency(P0, E=42) = |{P1, P2}| / |{P1,P2,P3,P4}| = 2/4 = 0.5.

A second entity E=99 gives consistency(P0, E=99) = 1/4 = 0.25.

Paper with no outbound citations (P1) -> None.
"""

from __future__ import annotations

import psycopg
import pytest

from scix.citation_consistency import ConsistencyResult, compute_consistency
from tests.helpers import get_test_dsn

pytestmark = pytest.mark.integration


P0 = "u14_cons_P0"
P1 = "u14_cons_P1"
P2 = "u14_cons_P2"
P3 = "u14_cons_P3"
P4 = "u14_cons_P4"
ALL_BIBCODES = (P0, P1, P2, P3, P4)

ENTITY_A = 4200142  # arbitrary test-range ids — avoid clobbering real entities
ENTITY_B = 4200199


@pytest.fixture()
def seeded_conn() -> psycopg.Connection:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    c = psycopg.connect(dsn)
    try:
        _cleanup(c)
        _seed(c)
        c.commit()
        yield c
    finally:
        _cleanup(c)
        c.commit()
        c.close()


def _cleanup(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (list(ALL_BIBCODES),),
        )
        cur.execute(
            "DELETE FROM citation_edges WHERE source_bibcode = ANY(%s) OR target_bibcode = ANY(%s)",
            (list(ALL_BIBCODES), list(ALL_BIBCODES)),
        )
        cur.execute(
            "DELETE FROM entities WHERE id = ANY(%s)",
            ([ENTITY_A, ENTITY_B],),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (list(ALL_BIBCODES),),
        )


def _seed(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        # Papers
        for b in ALL_BIBCODES:
            cur.execute(
                "INSERT INTO papers (bibcode, title) VALUES (%s, %s) "
                "ON CONFLICT (bibcode) DO NOTHING",
                (b, f"test paper {b}"),
            )
        # Citation edges: P0 -> {P1, P2, P3, P4}
        for tgt in (P1, P2, P3, P4):
            cur.execute(
                "INSERT INTO citation_edges (source_bibcode, target_bibcode) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (P0, tgt),
            )
        # Entities
        cur.execute(
            "INSERT INTO entities (id, canonical_name, entity_type, source) "
            "VALUES (%s, %s, %s, %s), (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (
                ENTITY_A,
                "Test Entity A u14",
                "instrument",
                "u14_test",
                ENTITY_B,
                "Test Entity B u14",
                "instrument",
                "u14_test",
            ),
        )
        # Links. NOTE: writes to document_entities here are test fixture
        # setup, not production resolver state. The AST lint only scans
        # src/ — tests/ is exempt.
        links = [
            (P1, ENTITY_A, "mention"),
            (P2, ENTITY_A, "mention"),
            (P3, ENTITY_B, "mention"),
            # P4 has no entity links
        ]
        for bib, eid, lt in links:
            cur.execute(
                "INSERT INTO document_entities "
                "(bibcode, entity_id, link_type, tier, tier_version, confidence) "
                "VALUES (%s, %s, %s, 0, 1, 1.0) "
                "ON CONFLICT DO NOTHING",
                (bib, eid, lt),
            )


def test_consistency_half(seeded_conn: psycopg.Connection) -> None:
    """AC: P0 citing 4 papers, 2 link to entity A -> 2/4 = 0.5."""
    result = compute_consistency(P0, ENTITY_A, conn=seeded_conn)
    assert isinstance(result, ConsistencyResult)
    assert result.total_cites == 4
    assert result.matching_cites == 2
    assert result.consistency == pytest.approx(0.5)


def test_consistency_quarter(seeded_conn: psycopg.Connection) -> None:
    """P0 citing 4 papers, 1 links to entity B -> 1/4 = 0.25."""
    result = compute_consistency(P0, ENTITY_B, conn=seeded_conn)
    assert result.total_cites == 4
    assert result.matching_cites == 1
    assert result.consistency == pytest.approx(0.25)


def test_consistency_no_outbound_cites_returns_none(
    seeded_conn: psycopg.Connection,
) -> None:
    """P1 has no outbound citations -> consistency is None, not 0.0."""
    result = compute_consistency(P1, ENTITY_A, conn=seeded_conn)
    assert result.total_cites == 0
    assert result.matching_cites == 0
    assert result.consistency is None


def test_consistency_unrelated_entity_is_zero(
    seeded_conn: psycopg.Connection,
) -> None:
    """Unknown entity -> zero matching but total still > 0."""
    unknown_entity = 999_999_999
    result = compute_consistency(P0, unknown_entity, conn=seeded_conn)
    assert result.total_cites == 4
    assert result.matching_cites == 0
    assert result.consistency == pytest.approx(0.0)
