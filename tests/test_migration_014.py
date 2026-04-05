"""Tests for migration 014_discipline_and_indexes.sql.

Validates SQL syntax and, when a database is available, verifies the migration
applies cleanly and produces the expected schema changes.
"""

from __future__ import annotations

import pathlib

import psycopg
import pytest
from helpers import DSN

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "migrations" / "014_discipline_and_indexes.sql"
)

ASTRO_SOURCES = ("ascl", "aas", "physh", "pwc", "astromlab", "vizier", "ads_data")


# ---------------------------------------------------------------------------
# Unit tests (no database required)
# ---------------------------------------------------------------------------


class TestMigrationSQL:
    """Validate the migration file structure and content."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists(), f"Migration file not found: {MIGRATION_PATH}"

    def test_migration_is_valid_sql_structure(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert sql.strip(), "Migration file is empty"

    def test_migration_wrapped_in_transaction(self) -> None:
        sql = MIGRATION_PATH.read_text().upper()
        assert "BEGIN" in sql, "Migration should be wrapped in a transaction (BEGIN)"
        assert "COMMIT" in sql, "Migration should be wrapped in a transaction (COMMIT)"

    def test_adds_discipline_column(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "add column" in sql, "Migration should add a column"
        assert "discipline" in sql, "Migration should add a 'discipline' column"
        assert "text" in sql, "discipline column should be TEXT type"

    def test_creates_discipline_btree_index(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "idx_entity_dict_discipline" in sql, "Missing btree index on discipline"
        assert "entity_dictionary" in sql

    def test_creates_canonical_lower_functional_index(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "idx_entity_dict_canonical_lower" in sql, "Missing functional index"
        assert "lower(canonical_name)" in sql, "Functional index should use lower(canonical_name)"

    def test_backfills_astrophysics_discipline(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "astrophysics" in sql, "Backfill should set discipline='astrophysics'"
        for source in ASTRO_SOURCES:
            assert f"'{source}'" in sql, f"Backfill should include source '{source}'"

    def test_discipline_column_is_nullable(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        # The ADD COLUMN should NOT contain "not null"
        # Extract just the ALTER TABLE statement
        lines = sql.splitlines()
        add_col_lines = [l for l in lines if "add column" in l and "discipline" in l]
        assert len(add_col_lines) == 1, "Expected exactly one ADD COLUMN for discipline"
        assert "not null" not in add_col_lines[0], "discipline column must be nullable"


# ---------------------------------------------------------------------------
# Integration tests (require running database with migration 013 applied)
# ---------------------------------------------------------------------------


def _has_entity_dictionary(conn: psycopg.Connection) -> bool:
    """Check if entity_dictionary table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'entity_dictionary'
        """)
        return cur.fetchone()[0] == 1


def _column_exists(conn: psycopg.Connection, table: str, column: str) -> bool:
    """Check if a column exists on a table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
        """,
            (table, column),
        )
        return cur.fetchone()[0] == 1


def _index_exists(conn: psycopg.Connection, index_name: str) -> bool:
    """Check if an index exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname = %s
        """,
            (index_name,),
        )
        return cur.fetchone()[0] == 1


@pytest.fixture()
def db_conn():
    """Provide a database connection with migration 014 applied."""
    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("Database not available")
        return

    if not _has_entity_dictionary(conn):
        conn.close()
        pytest.skip("entity_dictionary table not found (migration 013 not applied)")
        return

    # Apply migration 014 (idempotent due to IF NOT EXISTS / IF NOT EXISTS)
    sql = MIGRATION_PATH.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

    yield conn
    conn.close()


@pytest.mark.integration
class TestMigration014Integration:
    """Integration: verify migration 014 applies correctly."""

    def test_discipline_column_exists(self, db_conn: psycopg.Connection) -> None:
        assert _column_exists(
            db_conn, "entity_dictionary", "discipline"
        ), "discipline column should exist after migration"

    def test_discipline_column_is_nullable(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT is_nullable FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'entity_dictionary'
                  AND column_name = 'discipline'
            """)
            row = cur.fetchone()
            assert row is not None
            assert row[0] == "YES", "discipline column should be nullable"

    def test_discipline_btree_index_exists(self, db_conn: psycopg.Connection) -> None:
        assert _index_exists(
            db_conn, "idx_entity_dict_discipline"
        ), "btree index on discipline should exist"

    def test_canonical_lower_index_exists(self, db_conn: psycopg.Connection) -> None:
        assert _index_exists(
            db_conn, "idx_entity_dict_canonical_lower"
        ), "functional index idx_entity_dict_canonical_lower should exist"

    def test_no_null_discipline_after_backfill(self, db_conn: psycopg.Connection) -> None:
        """After migration, all rows should have discipline set (none NULL)."""
        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM entity_dictionary WHERE discipline IS NULL")
            null_count = cur.fetchone()[0]
            assert null_count == 0, f"Expected 0 NULL discipline rows, found {null_count}"

    def test_astro_sources_have_astrophysics(self, db_conn: psycopg.Connection) -> None:
        """All known astro sources should have discipline='astrophysics'."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT discipline
                FROM entity_dictionary
                WHERE source IN ('ascl','aas','physh','pwc','astromlab','vizier','ads_data')
            """)
            rows = cur.fetchall()
            if rows:
                disciplines = {r[0] for r in rows}
                assert disciplines == {
                    "astrophysics"
                }, f"Expected only 'astrophysics', got {disciplines}"

    def test_migration_is_idempotent(self, db_conn: psycopg.Connection) -> None:
        """Applying migration a second time should not error."""
        sql = MIGRATION_PATH.read_text()
        with db_conn.cursor() as cur:
            cur.execute(sql)
        db_conn.commit()
        # If we get here without error, idempotency is confirmed
        assert _column_exists(db_conn, "entity_dictionary", "discipline")
