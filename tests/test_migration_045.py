"""Schema tests for migration 045 (citation_diff).

Verifies:
- citation_diff table exists and is LOGGED (never UNLOGGED per feedback_unlogged_tables).
- Primary key is (source_bibcode, target_bibcode).
- Expected columns and types are present.
- Indexes exist for source, target, and provenance columns.
- Materialized views citation_diff_by_year and citation_diff_by_journal exist.
- Unique indexes on the materialized views exist (for REFRESH CONCURRENTLY).

These tests require SCIX_TEST_DSN to be set to a non-production database.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Iterator

import psycopg
import pytest

from tests.helpers import get_test_dsn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATION_FILE = REPO_ROOT / "migrations" / "045_citation_diff.sql"

# Prerequisites — these migrations must exist for the FK/join targets.
PREREQUISITE_MIGRATIONS = [
    "001_initial_schema.sql",
    "038_papers_external_ids.sql",
    "040_openalex_tables.sql",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip(
            "SCIX_TEST_DSN is not set or points at production — destructive "
            "migration tests require a dedicated test DB"
        )
    return test_dsn


@pytest.fixture(scope="module", autouse=True)
def ensure_migration_applied(dsn: str) -> None:
    """Apply prerequisite migrations and 045 before any test in this module."""
    all_migrations = PREREQUISITE_MIGRATIONS + ["045_citation_diff.sql"]
    for fname in all_migrations:
        path = REPO_ROOT / "migrations" / fname
        assert path.exists(), f"missing migration file: {fname}"
        result = subprocess.run(
            ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"failed to apply {fname}:\nstdout:\n{result.stdout}\n" f"stderr:\n{result.stderr}"
        )


@pytest.fixture(scope="module")
def conn(dsn: str) -> Iterator[psycopg.Connection]:
    """Autocommit connection for schema inspection."""
    c = psycopg.connect(dsn)
    c.autocommit = True
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# LOGGED-ness: critical check
# ---------------------------------------------------------------------------


class TestLoggedness:
    def test_citation_diff_is_logged(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT relpersistence FROM pg_class WHERE relname = 'citation_diff'")
            row = cur.fetchone()
            assert row is not None, "citation_diff table missing"
            assert row[0] == "p", (
                f"citation_diff must be LOGGED (relpersistence='p'), "
                f"got relpersistence='{row[0]}'. UNLOGGED tables truncate on crash."
            )


# ---------------------------------------------------------------------------
# citation_diff table structure
# ---------------------------------------------------------------------------


class TestCitationDiffSchema:
    def test_primary_key_is_composite(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.attname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_attribute a ON a.attrelid = c.conrelid
                    AND a.attnum = ANY(c.conkey)
                WHERE t.relname = 'citation_diff' AND c.contype = 'p'
                ORDER BY array_position(c.conkey, a.attnum)
                """)
            cols = [r[0] for r in cur.fetchall()]
            assert cols == ["source_bibcode", "target_bibcode"]

    def test_required_columns_present(self, conn: psycopg.Connection) -> None:
        expected = {
            "source_bibcode": "text",
            "target_bibcode": "text",
            "in_ads": "boolean",
            "in_openalex": "boolean",
            "source_attrs": "jsonb",
        }
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'citation_diff'
                """)
            actual = dict(cur.fetchall())
        for col, dtype in expected.items():
            assert col in actual, f"Missing column {col}"
            assert actual[col] == dtype, f"Column {col}: expected {dtype}, got {actual[col]}"

    def test_boolean_defaults(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, column_default, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'citation_diff'
                  AND column_name IN ('in_ads', 'in_openalex')
                """)
            rows = cur.fetchall()
        assert len(rows) == 2
        for name, default, nullable in rows:
            assert nullable == "NO", f"{name} should be NOT NULL"
            assert (
                default is not None and "false" in default.lower()
            ), f"{name} should default to false, got {default}"

    def test_indexes_exist(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'citation_diff'
                """)
            indexes = {r[0]: r[1] for r in cur.fetchall()}

        # Check for source, target, and provenance indexes
        assert any(
            "source_bibcode" in d for d in indexes.values()
        ), f"Missing index on source_bibcode. Found: {list(indexes.keys())}"
        assert any(
            "target_bibcode" in d for d in indexes.values()
        ), f"Missing index on target_bibcode. Found: {list(indexes.keys())}"
        assert any(
            "in_ads" in d and "in_openalex" in d for d in indexes.values()
        ), f"Missing provenance index. Found: {list(indexes.keys())}"


# ---------------------------------------------------------------------------
# Materialized views
# ---------------------------------------------------------------------------


class TestMaterializedViews:
    def test_by_year_view_exists(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relname, relkind
                FROM pg_class
                WHERE relname = 'citation_diff_by_year'
                  AND relnamespace = 'public'::regnamespace
                """)
            row = cur.fetchone()
            assert row is not None, "citation_diff_by_year materialized view missing"
            assert row[1] == "m", (
                f"citation_diff_by_year should be a materialized view (relkind='m'), "
                f"got '{row[1]}'"
            )

    def test_by_journal_view_exists(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relname, relkind
                FROM pg_class
                WHERE relname = 'citation_diff_by_journal'
                  AND relnamespace = 'public'::regnamespace
                """)
            row = cur.fetchone()
            assert row is not None, "citation_diff_by_journal materialized view missing"
            assert row[1] == "m", (
                f"citation_diff_by_journal should be a materialized view (relkind='m'), "
                f"got '{row[1]}'"
            )

    def test_by_year_has_unique_index(self, conn: psycopg.Connection) -> None:
        """REFRESH CONCURRENTLY requires a unique index."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'citation_diff_by_year'
                """)
            names = [r[0] for r in cur.fetchall()]
            assert any("pk" in n or "year" in n for n in names), (
                f"citation_diff_by_year needs a unique index for REFRESH CONCURRENTLY. "
                f"Found: {names}"
            )

    def test_by_journal_has_unique_index(self, conn: psycopg.Connection) -> None:
        """REFRESH CONCURRENTLY requires a unique index."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'citation_diff_by_journal'
                """)
            names = [r[0] for r in cur.fetchall()]
            assert any("pk" in n or "journal" in n for n in names), (
                f"citation_diff_by_journal needs a unique index for REFRESH CONCURRENTLY. "
                f"Found: {names}"
            )

    def test_by_year_columns(self, conn: psycopg.Connection) -> None:
        # Materialized views don't appear in information_schema.columns;
        # query pg_attribute instead.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.attname
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'citation_diff_by_year'
                  AND c.relnamespace = 'public'::regnamespace
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """)
            cols = {r[0] for r in cur.fetchall()}
        expected = {
            "pub_year",
            "total_edges",
            "both_count",
            "ads_only_count",
            "openalex_only_count",
            "overlap_pct",
        }
        missing = expected - cols
        assert not missing, f"Missing columns on citation_diff_by_year: {missing}"

    def test_by_journal_columns(self, conn: psycopg.Connection) -> None:
        # Materialized views don't appear in information_schema.columns;
        # query pg_attribute instead.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.attname
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'citation_diff_by_journal'
                  AND c.relnamespace = 'public'::regnamespace
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """)
            cols = {r[0] for r in cur.fetchall()}
        expected = {
            "journal",
            "total_edges",
            "both_count",
            "ads_only_count",
            "openalex_only_count",
            "overlap_pct",
        }
        missing = expected - cols
        assert not missing, f"Missing columns on citation_diff_by_journal: {missing}"
