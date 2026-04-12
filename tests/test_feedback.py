"""Tests for src/scix/feedback.py (PRD §S5)."""

from __future__ import annotations

import psycopg
import pytest

from scix.feedback import report_incorrect_link
from tests.helpers import get_test_dsn

pytestmark = pytest.mark.integration


TEST_BIBCODE = "u14_feedback_TEST1"
TEST_ENTITY_ID = 4200500


@pytest.fixture()
def conn() -> psycopg.Connection:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    c = psycopg.connect(dsn)
    try:
        _cleanup(c)
        c.commit()
        yield c
    finally:
        _cleanup(c)
        c.commit()
        c.close()


def _cleanup(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM entity_link_disputes WHERE bibcode = %s OR entity_id = %s",
            (TEST_BIBCODE, TEST_ENTITY_ID),
        )


def test_report_incorrect_link_writes_row(conn: psycopg.Connection) -> None:
    """AC4: report_incorrect_link writes a row to entity_link_disputes."""
    report_incorrect_link(
        bibcode=TEST_BIBCODE,
        entity_id=TEST_ENTITY_ID,
        reason="looks like a false positive from the NER pass",
        tier=1,
        conn=conn,
    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, entity_id, reason, tier, reported_at "
            "FROM entity_link_disputes "
            "WHERE bibcode = %s AND entity_id = %s",
            (TEST_BIBCODE, TEST_ENTITY_ID),
        )
        rows = cur.fetchall()

    assert len(rows) == 1
    bib, eid, reason, tier, reported_at = rows[0]
    assert bib == TEST_BIBCODE
    assert eid == TEST_ENTITY_ID
    assert reason == "looks like a false positive from the NER pass"
    assert tier == 1
    assert reported_at is not None


def test_report_incorrect_link_empty_reason_dropped(
    conn: psycopg.Connection,
) -> None:
    """Empty or whitespace reasons must not create rows."""
    report_incorrect_link(
        bibcode=TEST_BIBCODE,
        entity_id=TEST_ENTITY_ID,
        reason="   ",
        tier=0,
        conn=conn,
    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM entity_link_disputes WHERE bibcode = %s",
            (TEST_BIBCODE,),
        )
        count = cur.fetchone()[0]

    assert count == 0


def test_report_incorrect_link_multiple_rows(conn: psycopg.Connection) -> None:
    """Multiple reports on the same link should all persist (append-only)."""
    for i in range(3):
        report_incorrect_link(
            bibcode=TEST_BIBCODE,
            entity_id=TEST_ENTITY_ID,
            reason=f"report number {i}",
            tier=2,
            conn=conn,
        )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM entity_link_disputes " "WHERE bibcode = %s AND entity_id = %s",
            (TEST_BIBCODE, TEST_ENTITY_ID),
        )
        count = cur.fetchone()[0]

    assert count == 3
