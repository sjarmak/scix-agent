"""Tests for migration 057 — v_claim_edges materialized view.

Covers MH-2 of the SciX Deep Search v1 PRD:
  (a) migration applies
  (b) view exposes the expected columns and types
  (c) unique index supports REFRESH MATERIALIZED VIEW CONCURRENTLY
  (d) single-source-bibcode lookup uses an index scan

Integration tests run only when ``SCIX_TEST_DSN`` points at a non-production
database (per CLAUDE.md "Database Safety"). The migration-file static checks
run unconditionally.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from tests.helpers import get_test_dsn

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "migrations" / "057_v_claim_edges.sql"

EXPECTED_COLUMNS: dict[str, str] = {
    "source_bibcode": "text",
    "target_bibcode": "text",
    "context_snippet": "text",
    "intent": "text",
    "section_name": "text",
    "source_year": "smallint",
    "target_year": "smallint",
    "char_offset": "integer",
}


# ---------------------------------------------------------------------------
# Static migration-file checks (no DB required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def migration_sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


class TestMigrationFileText:
    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.is_file(), f"missing {MIGRATION_PATH}"

    def test_wrapped_in_transaction(self, migration_sql: str) -> None:
        assert "BEGIN;" in migration_sql
        assert "COMMIT;" in migration_sql

    def test_creates_materialized_view(self, migration_sql: str) -> None:
        assert "CREATE MATERIALIZED VIEW v_claim_edges" in migration_sql

    def test_idempotent_drop(self, migration_sql: str) -> None:
        assert "DROP MATERIALIZED VIEW IF EXISTS v_claim_edges" in migration_sql

    def test_unique_index_for_concurrent_refresh(self, migration_sql: str) -> None:
        # The unique index is mandatory for REFRESH ... CONCURRENTLY.
        assert "CREATE UNIQUE INDEX" in migration_sql
        assert "(source_bibcode, target_bibcode, char_offset)" in migration_sql

    def test_per_endpoint_intent_indexes(self, migration_sql: str) -> None:
        assert "(source_bibcode, intent)" in migration_sql
        assert "(target_bibcode, intent)" in migration_sql

    def test_context_snippet_capped_at_1000(self, migration_sql: str) -> None:
        assert "substring(cc.context_text FROM 1 FOR 1000)" in migration_sql
        assert "context_snippet" in migration_sql

    def test_joins_required_tables(self, migration_sql: str) -> None:
        # citation_contexts (cc), citation_edges (ce), papers (sp/tp)
        assert "FROM citation_contexts cc" in migration_sql
        assert "JOIN citation_edges ce" in migration_sql
        assert "JOIN papers sp" in migration_sql
        assert "JOIN papers tp" in migration_sql


# ---------------------------------------------------------------------------
# DB-backed integration tests (require SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


requires_test_dsn = pytest.mark.skipif(
    get_test_dsn() is None,
    reason="SCIX_TEST_DSN not set (or points at production) — skipping integration tests",
)


def _apply_migration(conn: psycopg.Connection) -> None:
    """Apply migration 057 (idempotent)."""
    sql_text = MIGRATION_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()


@pytest.fixture
def db_conn() -> psycopg.Connection:
    dsn = get_test_dsn()
    assert dsn is not None  # gated by requires_test_dsn at the test level
    conn = psycopg.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


@requires_test_dsn
class TestMigrationApplies:
    """(a) Migration applies cleanly to the test DB."""

    def test_apply_clean(self, db_conn: psycopg.Connection) -> None:
        _apply_migration(db_conn)
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_matviews WHERE schemaname='public' "
                "AND matviewname='v_claim_edges'"
            )
            assert cur.fetchone() is not None

    def test_idempotent_reapply(self, db_conn: psycopg.Connection) -> None:
        # Apply twice; the second pass must be a no-op (DROP IF EXISTS + CREATE).
        _apply_migration(db_conn)
        _apply_migration(db_conn)
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM pg_matviews WHERE schemaname='public' "
                "AND matviewname='v_claim_edges'"
            )
            assert cur.fetchone()[0] == 1


@requires_test_dsn
class TestViewSchema:
    """(b) View has the expected columns and types."""

    def test_columns_match_spec(self, db_conn: psycopg.Connection) -> None:
        _apply_migration(db_conn)
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.attname, format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'v_claim_edges'
                  AND n.nspname = 'public'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY a.attnum
                """
            )
            rows = cur.fetchall()
        actual = {name: dtype for name, dtype in rows}
        assert actual == EXPECTED_COLUMNS, (
            f"column set mismatch:\nexpected={EXPECTED_COLUMNS}\nactual={actual}"
        )


@requires_test_dsn
class TestConcurrentRefresh:
    """(c) Unique index supports REFRESH MATERIALIZED VIEW CONCURRENTLY."""

    def test_unique_index_present_on_view(self, db_conn: psycopg.Connection) -> None:
        _apply_migration(db_conn)
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname='public' AND tablename='v_claim_edges'
                """
            )
            indexes = {name: defn for name, defn in cur.fetchall()}
        unique_defs = [d for d in indexes.values() if "UNIQUE" in d.upper()]
        assert unique_defs, f"no UNIQUE index found on v_claim_edges: {indexes}"

    def test_concurrent_refresh_succeeds(self, db_conn: psycopg.Connection) -> None:
        _apply_migration(db_conn)
        # CONCURRENT refresh requires the connection to be in autocommit.
        dsn = get_test_dsn()
        assert dsn is not None
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY v_claim_edges")


@requires_test_dsn
class TestSingleBibcodeQueryUsesIndex:
    """(d) Single-source-bibcode lookup uses an index (no seq scan).

    We force the planner to choose an index by disabling sequential scans
    (``SET LOCAL enable_seqscan = off``); the assertion is that an index plan
    is *available*, not that the planner would prefer it on an empty MV. On a
    primed cache with real data, the per-endpoint btrees are the planner's
    natural choice for an equality predicate on (source_bibcode, intent).
    """

    def test_explain_uses_index(self, db_conn: psycopg.Connection) -> None:
        _apply_migration(db_conn)
        with db_conn.cursor() as cur:
            # SET LOCAL stays scoped to the current transaction.
            cur.execute("BEGIN")
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                """
                EXPLAIN (ANALYZE FALSE, FORMAT JSON)
                SELECT source_bibcode, target_bibcode, intent, context_snippet
                FROM v_claim_edges
                WHERE source_bibcode = %s
                LIMIT 5
                """,
                ("2024ApJ...999L..42X",),
            )
            row = cur.fetchone()
            cur.execute("ROLLBACK")
        plan_json = row[0]
        # psycopg returns JSON columns as a parsed list/dict already; allow both.
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)
        plan_text = json.dumps(plan_json)
        assert "Index Scan" in plan_text or "Bitmap" in plan_text, (
            f"single-bibcode lookup did not use an index — plan was:\n{plan_text}"
        )
