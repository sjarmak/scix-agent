"""Integration tests for u01 schema migrations (028-031) + zombie cleanup.

Acceptance criteria are pinned to these numbers:
    028_entity_schema_hardening.sql         — tier, tier_version, ENUMs, PK
    029_ontology_version_pinning.sql        — source_version, supersedes_id
    030_staging_and_promote_harvest.sql     — *_staging tables + promote_harvest
    031_query_log.sql                       — extended query_log columns

The spec described these as 026-029; actual numbering was shifted to keep
migration versions contiguous (026/027 slots were already used).

Safety: these tests run destructive statements. They REQUIRE SCIX_TEST_DSN
to be set and to NOT point at the production scix database. Tests SKIP
otherwise.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import psycopg
import pytest

from tests.helpers import get_test_dsn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"
SCRIPT_PATH = REPO_ROOT / "scripts" / "cleanup_harvest_zombies.py"

NEW_MIGRATIONS = [
    "028_entity_schema_hardening.sql",
    "029_ontology_version_pinning.sql",
    "030_staging_and_promote_harvest.sql",
    "031_query_log.sql",
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
def ensure_migrations_applied(dsn: str) -> None:
    """Apply the u01 migrations before any test in this module runs.

    All migrations are idempotent, so re-applying is a no-op if they were
    already installed by scripts/setup_db.sh.
    """
    for fname in NEW_MIGRATIONS:
        path = MIGRATIONS_DIR / fname
        assert path.exists(), f"missing migration file: {fname}"
        result = subprocess.run(
            ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"failed to apply {fname}:\nstdout:\n{result.stdout}\n" f"stderr:\n{result.stderr}"
        )


@pytest.fixture()
def conn(dsn: str):
    with psycopg.connect(dsn) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _column_exists(conn: psycopg.Connection, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = %s
               AND column_name = %s
            """,
            (table, column),
        )
        return cur.fetchone() is not None


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
              FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_name = %s
            """,
            (table,),
        )
        return cur.fetchone() is not None


def _pk_columns(conn: psycopg.Connection, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
              FROM pg_constraint c
              JOIN pg_attribute a
                ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
             WHERE c.conrelid = ('public.' || %s)::regclass
               AND c.contype  = 'p'
             ORDER BY array_position(c.conkey, a.attnum)
            """,
            (table,),
        )
        return [row[0] for row in cur.fetchall()]


def _enum_values(conn: psycopg.Connection, typename: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.enumlabel
              FROM pg_type t
              JOIN pg_enum e ON e.enumtypid = t.oid
             WHERE t.typname = %s
             ORDER BY e.enumsortorder
            """,
            (typename,),
        )
        return [row[0] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Migration 028 — entity schema hardening
# ---------------------------------------------------------------------------


class TestMigration028:
    def test_tier_column_exists_and_not_null(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type, is_nullable, column_default
                  FROM information_schema.columns
                 WHERE table_name = 'document_entities' AND column_name = 'tier'
                """)
            row = cur.fetchone()
        assert row is not None, "tier column missing"
        data_type, is_nullable, _default = row
        assert data_type == "smallint"
        assert is_nullable == "NO"

    def test_tier_version_column_exists(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type, is_nullable, column_default
                  FROM information_schema.columns
                 WHERE table_name = 'document_entities' AND column_name = 'tier_version'
                """)
            row = cur.fetchone()
        assert row is not None, "tier_version column missing"
        data_type, is_nullable, default = row
        assert data_type == "integer"
        assert is_nullable == "NO"
        assert default is not None and "1" in default

    def test_pk_includes_tier(self, conn: psycopg.Connection) -> None:
        cols = _pk_columns(conn, "document_entities")
        assert cols == ["bibcode", "entity_id", "link_type", "tier"], cols

    def test_ambiguity_class_enum_values(self, conn: psycopg.Connection) -> None:
        values = _enum_values(conn, "entity_ambiguity_class")
        assert values == ["unique", "domain_safe", "homograph", "banned"]

    def test_link_policy_enum_values(self, conn: psycopg.Connection) -> None:
        values = _enum_values(conn, "entity_link_policy")
        assert values == ["open", "context_required", "llm_only", "banned"]

    def test_entities_has_new_enum_columns(self, conn: psycopg.Connection) -> None:
        assert _column_exists(conn, "entities", "ambiguity_class")
        assert _column_exists(conn, "entities", "link_policy")

    def test_tier_collision_allowed(self, conn: psycopg.Connection) -> None:
        """Same (bibcode, entity_id, link_type) at different tiers must coexist."""
        bibcode = "2099TEST.....1U01"
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO entities (canonical_name, entity_type, source)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
                        SET updated_at = now()
                    RETURNING id
                    """,
                    ("__u01_test_entity__", "test_type", "__u01_test_source__"),
                )
                entity_id = cur.fetchone()[0]

                # Clean up any pre-existing rows from previous failed runs.
                cur.execute(
                    "DELETE FROM document_entities WHERE bibcode = %s AND entity_id = %s",
                    (bibcode, entity_id),
                )

                cur.execute(
                    """
                    INSERT INTO document_entities
                        (bibcode, entity_id, link_type, tier, tier_version, confidence)
                    VALUES (%s, %s, 'mention', 1, 1, 0.9),
                           (%s, %s, 'mention', 2, 1, 0.5)
                    """,
                    (bibcode, entity_id, bibcode, entity_id),
                )

                cur.execute(
                    "SELECT count(*) FROM document_entities WHERE bibcode = %s AND entity_id = %s",
                    (bibcode, entity_id),
                )
                assert cur.fetchone()[0] == 2

                # Delete tier=2, should leave exactly one row (tier=1).
                cur.execute(
                    "DELETE FROM document_entities WHERE bibcode = %s AND entity_id = %s AND tier = 2",
                    (bibcode, entity_id),
                )

                cur.execute(
                    "SELECT tier FROM document_entities WHERE bibcode = %s AND entity_id = %s",
                    (bibcode, entity_id),
                )
                rows = cur.fetchall()
                assert len(rows) == 1
                assert rows[0][0] == 1
            finally:
                cur.execute("DELETE FROM document_entities WHERE bibcode = %s", (bibcode,))
                cur.execute(
                    """
                    DELETE FROM entities
                     WHERE canonical_name = %s
                       AND entity_type = %s
                       AND source = %s
                    """,
                    ("__u01_test_entity__", "test_type", "__u01_test_source__"),
                )
                conn.commit()


# ---------------------------------------------------------------------------
# Migration 029 — ontology version pinning
# ---------------------------------------------------------------------------


class TestMigration029:
    def test_source_version_column_exists(self, conn: psycopg.Connection) -> None:
        assert _column_exists(conn, "entities", "source_version")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type FROM information_schema.columns
                 WHERE table_name = 'entities' AND column_name = 'source_version'
                """)
            assert cur.fetchone()[0] == "text"

    def test_supersedes_id_column_exists(self, conn: psycopg.Connection) -> None:
        assert _column_exists(conn, "entities", "supersedes_id")

    def test_supersedes_self_fk_exists(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT conname
                  FROM pg_constraint
                 WHERE conrelid = 'public.entities'::regclass
                   AND contype  = 'f'
                   AND confrelid = 'public.entities'::regclass
                """)
            rows = [row[0] for row in cur.fetchall()]
        assert any("supersedes" in name for name in rows), rows


# ---------------------------------------------------------------------------
# Migration 030 — staging tables + promote_harvest stub
# ---------------------------------------------------------------------------


class TestMigration030:
    def test_staging_tables_exist(self, conn: psycopg.Connection) -> None:
        for t in ("entities_staging", "entity_aliases_staging", "entity_identifiers_staging"):
            assert _table_exists(conn, t), f"missing staging table {t}"

    def test_staging_run_id_column(self, conn: psycopg.Connection) -> None:
        for t in ("entities_staging", "entity_aliases_staging", "entity_identifiers_staging"):
            assert _column_exists(conn, t, "staging_run_id"), t

    def test_promote_harvest_function_exists(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.proname, pg_get_function_arguments(p.oid)
                  FROM pg_proc p
                  JOIN pg_namespace n ON n.oid = p.pronamespace
                 WHERE n.nspname = 'public'
                   AND p.proname = 'promote_harvest'
                """)
            row = cur.fetchone()
        assert row is not None, "promote_harvest function missing"
        _name, args = row
        assert "bigint" in args.lower()

    def test_promote_harvest_stub_callable(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT promote_harvest(0::bigint)")
            assert cur.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Migration 031 — query_log extension
# ---------------------------------------------------------------------------


class TestMigration031:
    @pytest.mark.parametrize(
        "column,expected_type",
        [
            ("ts", "timestamp with time zone"),
            ("tool", "text"),
            ("query", "text"),
            ("result_count", "integer"),
            ("session_id", "text"),
            ("is_test", "boolean"),
        ],
    )
    def test_columns_exist_with_types(
        self, conn: psycopg.Connection, column: str, expected_type: str
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data_type
                  FROM information_schema.columns
                 WHERE table_name = 'query_log' AND column_name = %s
                """,
                (column,),
            )
            row = cur.fetchone()
        assert row is not None, f"query_log.{column} missing"
        assert row[0] == expected_type, f"{column}: expected {expected_type}, got {row[0]}"

    def test_is_test_default_false(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_default, is_nullable
                  FROM information_schema.columns
                 WHERE table_name = 'query_log' AND column_name = 'is_test'
                """)
            default, is_nullable = cur.fetchone()
        assert is_nullable == "NO"
        assert default is not None and "false" in default.lower()


# ---------------------------------------------------------------------------
# Idempotency: re-apply all four migrations, expect no error
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_reapply_all_new_migrations(self, dsn: str) -> None:
        for fname in NEW_MIGRATIONS:
            path = MIGRATIONS_DIR / fname
            result = subprocess.run(
                ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"re-applying {fname} failed:\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )


# ---------------------------------------------------------------------------
# cleanup_harvest_zombies.py
# ---------------------------------------------------------------------------


class TestCleanupHarvestZombies:
    def test_marks_old_running_rows_as_aborted_zombie(
        self, conn: psycopg.Connection, dsn: str
    ) -> None:
        # Insert a fixture zombie row (started 10 hours ago).
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO harvest_runs (source, started_at, status)
                VALUES ('__u01_zombie_src__',
                        now() - interval '10 hours',
                        'running')
                RETURNING id
                """)
            zombie_id = cur.fetchone()[0]

            # Also insert a fresh running row that should NOT be touched.
            cur.execute("""
                INSERT INTO harvest_runs (source, started_at, status)
                VALUES ('__u01_fresh_src__', now(), 'running')
                RETURNING id
                """)
            fresh_id = cur.fetchone()[0]
        conn.commit()

        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--dsn", dsn],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"cleanup script failed: stdout={result.stdout!r} " f"stderr={result.stderr!r}"
            )
            assert "Aborted" in result.stdout
            assert str(zombie_id) in result.stdout

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, finished_at FROM harvest_runs WHERE id = %s",
                    (zombie_id,),
                )
                status, finished_at = cur.fetchone()
                assert status == "aborted_zombie"
                assert finished_at is not None

                cur.execute(
                    "SELECT status FROM harvest_runs WHERE id = %s",
                    (fresh_id,),
                )
                assert cur.fetchone()[0] == "running"
        finally:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM harvest_runs WHERE id IN (%s, %s)",
                    (zombie_id, fresh_id),
                )
            conn.commit()

    def test_script_dry_run_does_not_update(self, conn: psycopg.Connection, dsn: str) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO harvest_runs (source, started_at, status)
                VALUES ('__u01_zombie_dry__',
                        now() - interval '10 hours',
                        'running')
                RETURNING id
                """)
            zombie_id = cur.fetchone()[0]
        conn.commit()

        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--dsn", dsn, "--dry-run"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "Would abort" in result.stdout
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM harvest_runs WHERE id = %s", (zombie_id,))
                assert cur.fetchone()[0] == "running"
        finally:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM harvest_runs WHERE id = %s", (zombie_id,))
            conn.commit()
