"""Schema tests for migration 055 (paper_umap_2d).

Verifies:
- paper_umap_2d table exists after applying migration 055 to a fresh scix_test DB.
- Primary key is (bibcode).
- Foreign key from paper_umap_2d.bibcode references papers(bibcode).
- Columns (x, y, community_id, resolution, projected_at) exist with expected types
  and NOT NULL / default semantics.
- Index ix_paper_umap_2d_resolution_community exists on (resolution, community_id).
- Migration is idempotent — applying twice does not raise.

Two layers:
- `TestMigrationSQLFile` is a pure unit test that reads the .sql file and
  asserts structural fragments; it runs in every environment (no DB).
- The remainder are integration tests that require SCIX_TEST_DSN pointing at
  a non-production database. They skip otherwise.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import Iterator

import psycopg
import pytest

from tests.helpers import get_test_dsn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATION_FILE = REPO_ROOT / "migrations" / "055_paper_umap_2d.sql"

# Prerequisite migrations: paper_umap_2d has a FK to papers(bibcode), which is
# defined in the initial schema. Applying only 001 is enough — later migrations
# don't affect the paper_umap_2d table.
PREREQUISITE_MIGRATIONS = [
    "001_initial_schema.sql",
]


# ---------------------------------------------------------------------------
# Pure unit test — parses the .sql file, runs in every environment
# ---------------------------------------------------------------------------


class TestMigrationSQLFile:
    """Static checks on the migration file itself. No DB required."""

    def test_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"migration file missing: {MIGRATION_FILE}"

    def test_contains_expected_ddl_fragments(self) -> None:
        text = MIGRATION_FILE.read_text()
        expected_fragments = [
            "CREATE TABLE IF NOT EXISTS paper_umap_2d",
            "REFERENCES papers(bibcode)",
            "CREATE INDEX IF NOT EXISTS ix_paper_umap_2d_resolution_community",
        ]
        for fragment in expected_fragments:
            assert fragment in text, (
                f"migration 055 missing expected fragment: {fragment!r}"
            )

    def test_wrapped_in_transaction(self) -> None:
        text = MIGRATION_FILE.read_text()
        assert "BEGIN;" in text, "migration 055 should be wrapped in BEGIN/COMMIT"
        assert "COMMIT;" in text, "migration 055 should be wrapped in BEGIN/COMMIT"


# ---------------------------------------------------------------------------
# Integration fixtures — require SCIX_TEST_DSN pointing at a non-prod DB
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


def _apply_sql_file(dsn: str, path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def ensure_migration_applied(dsn: str) -> Iterator[None]:
    """Apply prerequisite migrations + 055 before any integration test in this
    module. Drop paper_umap_2d on teardown so scix_test is left clean.

    Not autouse: pure unit tests in `TestMigrationSQLFile` must run without a
    DB. Integration tests opt in by depending on the `conn` fixture below,
    which depends on this one.
    """
    # 1. Prerequisites + migration under test
    for fname in PREREQUISITE_MIGRATIONS + ["055_paper_umap_2d.sql"]:
        path = REPO_ROOT / "migrations" / fname
        assert path.exists(), f"missing migration file: {fname}"
        result = _apply_sql_file(dsn, path)
        assert result.returncode == 0, (
            f"failed to apply {fname}:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    try:
        yield
    finally:
        # Teardown — leave scix_test clean for subsequent runs.
        with psycopg.connect(dsn, autocommit=True) as cleanup_conn:
            with cleanup_conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS paper_umap_2d CASCADE")


@pytest.fixture(scope="module")
def conn(
    dsn: str, ensure_migration_applied: None
) -> Iterator[psycopg.Connection]:
    """Autocommit connection for schema inspection.

    Depends on `ensure_migration_applied` so integration tests that request
    `conn` automatically trigger the migration apply/teardown cycle.
    """
    c = psycopg.connect(dsn)
    c.autocommit = True
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Integration tests — schema shape
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPaperUmap2dSchema:
    def test_table_exists(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                  FROM information_schema.tables
                 WHERE table_schema = 'public'
                   AND table_name   = 'paper_umap_2d'
                """
            )
            assert cur.fetchone() is not None, "paper_umap_2d table missing"

    def test_primary_key_is_bibcode(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.attname
                  FROM pg_constraint c
                  JOIN pg_class t ON t.oid = c.conrelid
                  JOIN pg_attribute a ON a.attrelid = c.conrelid
                       AND a.attnum = ANY(c.conkey)
                 WHERE t.relname = 'paper_umap_2d' AND c.contype = 'p'
                 ORDER BY array_position(c.conkey, a.attnum)
                """
            )
            cols = [r[0] for r in cur.fetchall()]
            assert cols == ["bibcode"], f"expected PK=(bibcode), got {cols}"

    def test_foreign_key_to_papers(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rt.relname,
                       string_agg(ra.attname, ',' ORDER BY k.ord) AS referenced_cols,
                       string_agg(a.attname,  ',' ORDER BY k.ord) AS local_cols,
                       c.confdeltype
                  FROM pg_constraint c
                  JOIN pg_class  t  ON t.oid  = c.conrelid
                  JOIN pg_class  rt ON rt.oid = c.confrelid
                  JOIN unnest(c.conkey)  WITH ORDINALITY AS k(attnum, ord)  ON TRUE
                  JOIN pg_attribute  a  ON a.attrelid  = t.oid  AND a.attnum  = k.attnum
                  JOIN unnest(c.confkey) WITH ORDINALITY AS rk(attnum, ord) ON rk.ord = k.ord
                  JOIN pg_attribute  ra ON ra.attrelid = rt.oid AND ra.attnum = rk.attnum
                 WHERE t.relname = 'paper_umap_2d'
                   AND c.contype = 'f'
                 GROUP BY rt.relname, c.confdeltype
                """
            )
            rows = cur.fetchall()
        assert rows, "paper_umap_2d has no FK constraint"
        # Expect exactly one FK: local bibcode -> papers.bibcode, cascading on delete.
        assert len(rows) == 1, f"expected exactly one FK, got {rows}"
        ref_table, ref_cols, local_cols, on_delete = rows[0]
        assert ref_table == "papers"
        assert ref_cols == "bibcode"
        assert local_cols == "bibcode"
        # 'c' == CASCADE in pg_constraint.confdeltype
        assert on_delete == "c", f"expected ON DELETE CASCADE ('c'), got {on_delete!r}"

    def test_required_columns_present_with_types(
        self, conn: psycopg.Connection
    ) -> None:
        expected = {
            # column_name: (data_type, is_nullable)
            "bibcode":       ("text",                        "NO"),
            "x":             ("double precision",            "NO"),
            "y":             ("double precision",            "NO"),
            "community_id":  ("integer",                     "YES"),
            "resolution":    ("text",                        "NO"),
            "projected_at":  ("timestamp with time zone",    "NO"),
        }
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name   = 'paper_umap_2d'
                """
            )
            actual = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

        for col, (dtype, nullable) in expected.items():
            assert col in actual, f"missing column {col}. got: {sorted(actual)}"
            assert actual[col] == (dtype, nullable), (
                f"column {col}: expected ({dtype}, nullable={nullable}), "
                f"got {actual[col]}"
            )

    def test_projected_at_default_is_now(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_default
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name   = 'paper_umap_2d'
                   AND column_name  = 'projected_at'
                """
            )
            row = cur.fetchone()
        assert row is not None
        default = row[0] or ""
        assert "now()" in default.lower(), (
            f"projected_at should default to now(), got {default!r}"
        )

    def test_resolution_community_index_exists(
        self, conn: psycopg.Connection
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexdef
                  FROM pg_indexes
                 WHERE schemaname = 'public'
                   AND tablename  = 'paper_umap_2d'
                   AND indexname  = 'ix_paper_umap_2d_resolution_community'
                """
            )
            row = cur.fetchone()
        assert row is not None, (
            "ix_paper_umap_2d_resolution_community index missing"
        )
        indexdef = row[0]
        # Guard against typos where only one column made it into the index.
        assert "resolution" in indexdef and "community_id" in indexdef, (
            f"index does not cover (resolution, community_id): {indexdef}"
        )


@pytest.mark.integration
class TestMigrationIdempotency:
    def test_reapplying_migration_is_a_noop(
        self, dsn: str, ensure_migration_applied: None
    ) -> None:
        # ensure_migration_applied already ran the migration once.
        # Running it a second time must not raise.
        result = _apply_sql_file(dsn, MIGRATION_FILE)
        assert result.returncode == 0, (
            "re-applying migration 055 must succeed (idempotent); got:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Guardrails — make sure the integration tests are actually wired up to skip
# ---------------------------------------------------------------------------


def test_integration_suite_skips_without_test_dsn() -> None:
    """If SCIX_TEST_DSN is unset or points at prod, integration tests must
    skip rather than attempt destructive DDL against production."""
    if os.environ.get("SCIX_TEST_DSN") is None:
        # No test DSN — get_test_dsn() should return None and the dsn fixture
        # should skip.
        assert get_test_dsn() is None
    else:
        # A test DSN is set — if it points at prod, get_test_dsn must treat
        # it as unset; otherwise it returns the DSN and integration tests run.
        test_dsn = os.environ["SCIX_TEST_DSN"]
        from scix.db import is_production_dsn

        if is_production_dsn(test_dsn):
            assert get_test_dsn() is None
        else:
            assert get_test_dsn() == test_dsn
