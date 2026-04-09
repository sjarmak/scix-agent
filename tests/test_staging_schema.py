"""Tests for migration 015: staging schema for extraction writes.

Two test modes:
- Static: always runs, verifies SQL content and structure
- Integration (pytest.mark.integration): runs against a real PostgreSQL database
  when SCIX_DSN is available
"""

import os
import pathlib

import psycopg
import pytest

from helpers import is_production_dsn

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "migrations" / "015_staging_schema.sql"
)
DSN = os.environ.get("SCIX_TEST_DSN") or os.environ.get("SCIX_DSN", "dbname=scix")


# ---------------------------------------------------------------------------
# Static tests — always run, no DB required
# ---------------------------------------------------------------------------


class TestMigrationSQL:
    """Verify the migration SQL contains the expected structural elements."""

    @pytest.fixture(scope="class")
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_creates_staging_schema(self, sql: str) -> None:
        assert "CREATE SCHEMA IF NOT EXISTS staging" in sql

    def test_creates_staging_extractions_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS staging.extractions" in sql

    def test_staging_table_has_no_fk_to_papers(self, sql: str) -> None:
        # The staging table definition should NOT reference papers
        # Extract just the staging table CREATE block
        start = sql.index("CREATE TABLE IF NOT EXISTS staging.extractions")
        end = sql.index(");", start) + 2
        staging_block = sql[start:end]
        assert "REFERENCES papers" not in staging_block

    def test_staging_table_has_unique_constraint(self, sql: str) -> None:
        assert "uq_staging_extractions_bibcode_type_version" in sql

    def test_promote_function_exists(self, sql: str) -> None:
        assert "CREATE OR REPLACE FUNCTION staging.promote_extractions()" in sql

    def test_promote_uses_on_conflict_upsert(self, sql: str) -> None:
        assert "ON CONFLICT (bibcode, extraction_type, extraction_version)" in sql
        assert "DO UPDATE SET" in sql

    def test_promote_truncates_staging(self, sql: str) -> None:
        assert "TRUNCATE staging.extractions" in sql

    def test_promote_returns_count(self, sql: str) -> None:
        assert "RETURNS INTEGER" in sql

    def test_wrapped_in_transaction(self, sql: str) -> None:
        assert sql.strip().startswith("--") or "BEGIN;" in sql
        assert "COMMIT;" in sql


# ---------------------------------------------------------------------------
# Integration tests — require a running PostgreSQL with scix database
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration


def _db_available() -> bool:
    """Check if we can connect to the database."""
    try:
        with psycopg.connect(DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


skip_no_db = pytest.mark.skipif(not _db_available(), reason="No database available")
skip_production = pytest.mark.skipif(
    is_production_dsn(DSN),
    reason="Staging integration tests TRUNCATE tables. Set SCIX_TEST_DSN to enable.",
)


@skip_no_db
@skip_production
class TestStagingIntegration:
    """Full cycle: insert into staging -> promote -> verify in public -> staging empty."""

    @pytest.fixture(scope="class")
    def conn(self):
        """Provide a connection; apply the migration if staging schema doesn't exist."""
        with psycopg.connect(DSN) as c:
            c.autocommit = False
            # Check if staging schema already exists; if not, apply migration
            with c.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'staging'"
                )
                if cur.fetchone() is None:
                    c.rollback()
                    c.autocommit = True
                    migration_sql = MIGRATION_PATH.read_text()
                    c.execute(migration_sql)
                    c.autocommit = False
            yield c
            c.rollback()

    @pytest.fixture(autouse=True)
    def _savepoint(self, conn):
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT staging_test_sp")
        yield
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT staging_test_sp")

    def _ensure_paper(self, cur: psycopg.Cursor, bibcode: str) -> None:
        """Insert a minimal paper row if it doesn't exist (needed for public FK)."""
        cur.execute(
            """
            INSERT INTO public.papers (bibcode, title, year, doctype)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (bibcode) DO NOTHING
            """,
            (bibcode, "Test Paper", 2024, "article"),
        )

    def test_full_promote_cycle(self, conn) -> None:
        """Insert into staging, promote, verify public has rows, staging is empty."""
        bibcode = "2024Test..stg..001A"
        with conn.cursor() as cur:
            # Ensure the paper exists in public (FK requirement for public.extractions)
            self._ensure_paper(cur, bibcode)

            # Insert into staging (no FK check)
            cur.execute(
                """
                INSERT INTO staging.extractions
                    (bibcode, extraction_type, extraction_version, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (bibcode, "entities", "v1.0", '{"entities": ["star", "galaxy"]}'),
            )

            # Verify row is in staging
            cur.execute("SELECT count(*) FROM staging.extractions WHERE bibcode = %s", (bibcode,))
            assert cur.fetchone()[0] == 1

            # Promote
            cur.execute("SELECT staging.promote_extractions()")
            promoted_count = cur.fetchone()[0]
            assert promoted_count >= 1

            # Verify row landed in public
            cur.execute(
                """
                SELECT payload FROM public.extractions
                WHERE bibcode = %s AND extraction_type = 'entities' AND extraction_version = 'v1.0'
                """,
                (bibcode,),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == {"entities": ["star", "galaxy"]}

            # Verify staging is empty
            cur.execute("SELECT count(*) FROM staging.extractions")
            assert cur.fetchone()[0] == 0

    def test_promote_upsert_updates_existing(self, conn) -> None:
        """Promoting a row that already exists in public updates payload."""
        bibcode = "2024Test..stg..002B"
        with conn.cursor() as cur:
            self._ensure_paper(cur, bibcode)

            # Insert original into public
            cur.execute(
                """
                INSERT INTO public.extractions
                    (bibcode, extraction_type, extraction_version, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (bibcode, "keywords", "v1.0", '{"keywords": ["old"]}'),
            )

            # Insert updated version into staging
            cur.execute(
                """
                INSERT INTO staging.extractions
                    (bibcode, extraction_type, extraction_version, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (bibcode, "keywords", "v1.0", '{"keywords": ["new", "updated"]}'),
            )

            # Promote
            cur.execute("SELECT staging.promote_extractions()")

            # Public should have updated payload
            cur.execute(
                """
                SELECT payload FROM public.extractions
                WHERE bibcode = %s AND extraction_type = 'keywords' AND extraction_version = 'v1.0'
                """,
                (bibcode,),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == {"keywords": ["new", "updated"]}

            # Staging should be empty
            cur.execute("SELECT count(*) FROM staging.extractions")
            assert cur.fetchone()[0] == 0

    def test_promote_empty_staging_returns_zero(self, conn) -> None:
        """Promoting with an empty staging table returns 0."""
        with conn.cursor() as cur:
            cur.execute("TRUNCATE staging.extractions")
            cur.execute("SELECT staging.promote_extractions()")
            assert cur.fetchone()[0] == 0
