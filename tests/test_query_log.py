"""Tests for src/scix/query_log.py and scripts/backfill_query_log.py (M3.5.0)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest

from scix import query_log as ql
from tests.helpers import get_test_dsn

pytestmark = pytest.mark.integration

SESSION_ID = "u07-test-session"
TEST_TOOL = "u07_test_tool"


@pytest.fixture()
def conn() -> psycopg.Connection:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    c = psycopg.connect(dsn)
    try:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM query_log WHERE tool = %s OR session_id = %s",
                (TEST_TOOL, SESSION_ID),
            )
        c.commit()
        yield c
    finally:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM query_log WHERE tool = %s OR session_id = %s",
                (TEST_TOOL, SESSION_ID),
            )
        c.commit()
        c.close()


def test_log_query_writes_row_with_recent_ts(conn: psycopg.Connection) -> None:
    """AC1: log_query writes a row and ts is approximately now()."""
    ql.log_query(
        tool=TEST_TOOL,
        query="hydrogen alpha line",
        result_count=42,
        session_id=SESSION_ID,
        is_test=True,
        conn=conn,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tool, query, result_count, session_id, is_test, ts,
                   tool_name, success
              FROM query_log
             WHERE session_id = %s AND tool = %s
            """,
            (SESSION_ID, TEST_TOOL),
        )
        row = cur.fetchone()

    assert row is not None, "log_query did not write a row"
    tool, query, result_count, session_id, is_test, ts, tool_name, success = row
    assert tool == TEST_TOOL
    assert query == "hydrogen alpha line"
    assert result_count == 42
    assert session_id == SESSION_ID
    assert is_test is True
    assert tool_name == TEST_TOOL, "legacy tool_name column must be populated"
    assert success is True, "legacy success column must be populated"

    now = datetime.now(timezone.utc)
    assert ts is not None
    delta = abs((now - ts).total_seconds())
    assert delta < 30, f"ts drift too large: {delta}s"


def test_log_query_zero_result(conn: psycopg.Connection) -> None:
    """Zero-result queries must still be persisted (they drive gap pass-1)."""
    ql.log_query(
        tool=TEST_TOOL,
        query="nonexistent_entity_xyz",
        result_count=0,
        session_id=SESSION_ID,
        is_test=True,
        conn=conn,
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM query_log WHERE session_id = %s AND result_count = 0",
            (SESSION_ID,),
        )
        (n,) = cur.fetchone()
    assert n == 1


def test_backfill_inserts_rows_from_fixture(conn: psycopg.Connection, tmp_path: Path) -> None:
    """AC2: backfill replays a fixture JSONL file into query_log."""
    from scripts.backfill_query_log import run_backfill

    fixture = tmp_path / "mcp_session.jsonl"
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    records = [
        {
            "tool": TEST_TOOL,
            "query": f"fixture query {i}",
            "result_count": i,
            "session_id": SESSION_ID,
            "ts": (base + timedelta(minutes=i)).isoformat(),
            "is_test": True,
        }
        for i in range(5)
    ]
    with fixture.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        # malformed line should be skipped, not abort the backfill
        f.write("{not json}\n")
        # missing 'tool' should be skipped
        f.write(json.dumps({"query": "no tool field"}) + "\n")

    n = run_backfill(fixture, conn, default_is_test=True)
    assert n == 5

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM query_log WHERE session_id = %s AND tool = %s",
            (SESSION_ID, TEST_TOOL),
        )
        (count,) = cur.fetchone()
    assert count >= 5
