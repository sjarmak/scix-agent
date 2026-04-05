"""Tests for migration 022: staging tables for entity graph.

Two test modes:
- Static: always runs, verifies SQL content and structure
- Integration (pytest.mark.integration): runs against a real PostgreSQL database
  when SCIX_DSN is available
"""

import os
import pathlib

import psycopg
import pytest

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "migrations" / "022_staging_entities.sql"
)
DSN = os.environ.get("SCIX_DSN", "dbname=scix")


# ---------------------------------------------------------------------------
# Static tests -- always run, no DB required
# ---------------------------------------------------------------------------


class TestMigrationFileExists:
    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists(), f"Migration file not found: {MIGRATION_PATH}"


class TestStagingEntitiesSQL:
    """Verify the migration SQL contains the expected structural elements."""

    @pytest.fixture(scope="class")
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    # -- Tables ---------------------------------------------------------------

    def test_defines_staging_entities_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS staging.entities" in sql

    def test_defines_staging_entity_identifiers_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS staging.entity_identifiers" in sql

    def test_defines_staging_entity_aliases_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS staging.entity_aliases" in sql

    def test_defines_public_entities_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS public.entities" in sql

    def test_defines_public_entity_identifiers_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS public.entity_identifiers" in sql

    def test_defines_public_entity_aliases_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS public.entity_aliases" in sql

    # -- Promote function -----------------------------------------------------

    def test_promote_function_exists(self, sql: str) -> None:
        assert "CREATE OR REPLACE FUNCTION staging.promote_entities()" in sql

    def test_promote_returns_integer(self, sql: str) -> None:
        assert "RETURNS INTEGER" in sql

    def test_promote_uses_on_conflict_for_entities(self, sql: str) -> None:
        assert "ON CONFLICT (canonical_name, entity_type, source)" in sql

    def test_promote_uses_on_conflict_for_identifiers(self, sql: str) -> None:
        assert "ON CONFLICT (id_scheme, external_id)" in sql

    def test_promote_uses_on_conflict_for_aliases(self, sql: str) -> None:
        assert "ON CONFLICT (entity_id, alias)" in sql

    def test_promote_truncates_staging_entities(self, sql: str) -> None:
        assert "TRUNCATE staging.entities" in sql

    def test_promote_truncates_staging_entity_identifiers(self, sql: str) -> None:
        assert "TRUNCATE staging.entity_identifiers" in sql

    def test_promote_truncates_staging_entity_aliases(self, sql: str) -> None:
        assert "TRUNCATE staging.entity_aliases" in sql

    # -- Transaction wrapping -------------------------------------------------

    def test_wrapped_in_transaction(self, sql: str) -> None:
        assert "BEGIN;" in sql
        assert "COMMIT;" in sql

    # -- Entity ID remapping --------------------------------------------------

    def test_identifiers_remap_through_natural_key(self, sql: str) -> None:
        """Identifiers must join staging.entities -> public.entities to remap IDs."""
        assert "JOIN staging.entities" in sql
        assert "JOIN public.entities" in sql

    # -- Staging tables have no FK --------------------------------------------

    def test_staging_entity_identifiers_no_fk(self, sql: str) -> None:
        start = sql.index("CREATE TABLE IF NOT EXISTS staging.entity_identifiers")
        end = sql.index(");", start) + 2
        block = sql[start:end]
        assert "REFERENCES" not in block

    def test_staging_entity_aliases_no_fk(self, sql: str) -> None:
        start = sql.index("CREATE TABLE IF NOT EXISTS staging.entity_aliases")
        end = sql.index(");", start) + 2
        block = sql[start:end]
        assert "REFERENCES" not in block


# ---------------------------------------------------------------------------
# Integration tests -- require a running PostgreSQL with scix database
# ---------------------------------------------------------------------------


def _db_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


skip_no_db = pytest.mark.skipif(not _db_available(), reason="No database available")


@skip_no_db
class TestStagingEntitiesIntegration:
    """Full cycle: insert into staging -> promote -> verify in public -> staging empty."""

    @pytest.fixture(scope="class")
    def conn(self):
        with psycopg.connect(DSN) as c:
            c.autocommit = True
            migration_sql = MIGRATION_PATH.read_text()
            c.execute(migration_sql)
            c.autocommit = False
            yield c
            c.rollback()

    @pytest.fixture(autouse=True)
    def _savepoint(self, conn):
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT staging_entity_test_sp")
        yield
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT staging_entity_test_sp")

    def test_full_promote_cycle(self, conn) -> None:
        with conn.cursor() as cur:
            # Insert into staging.entities
            cur.execute(
                """
                INSERT INTO staging.entities
                    (canonical_name, entity_type, source, discipline, properties)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                ("Milky Way", "galaxy", "test", "astronomy", '{"mass": "1e12"}'),
            )

            # Insert identifier referencing staging entity
            cur.execute("SELECT id FROM staging.entities WHERE canonical_name = 'Milky Way'")
            staging_eid = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO staging.entity_identifiers
                    (entity_id, id_scheme, external_id, is_primary)
                VALUES (%s, %s, %s, %s)
                """,
                (staging_eid, "wikidata", "Q321", True),
            )

            # Insert alias
            cur.execute(
                """
                INSERT INTO staging.entity_aliases
                    (entity_id, alias, alias_source)
                VALUES (%s, %s, %s)
                """,
                (staging_eid, "The Galaxy", "common_name"),
            )

            # Promote
            cur.execute("SELECT staging.promote_entities()")
            promoted = cur.fetchone()[0]
            assert promoted == 1

            # Verify public.entities
            cur.execute(
                "SELECT id, properties FROM public.entities WHERE canonical_name = 'Milky Way'"
            )
            row = cur.fetchone()
            assert row is not None
            public_eid = row[0]
            assert row[1] == {"mass": "1e12"}

            # Verify public.entity_identifiers with remapped entity_id
            cur.execute(
                "SELECT entity_id FROM public.entity_identifiers WHERE id_scheme = 'wikidata' AND external_id = 'Q321'"
            )
            id_row = cur.fetchone()
            assert id_row is not None
            assert id_row[0] == public_eid

            # Verify public.entity_aliases with remapped entity_id
            cur.execute(
                "SELECT entity_id, alias_source FROM public.entity_aliases WHERE alias = 'The Galaxy'"
            )
            alias_row = cur.fetchone()
            assert alias_row is not None
            assert alias_row[0] == public_eid

            # Staging tables should be empty
            for table in (
                "staging.entities",
                "staging.entity_identifiers",
                "staging.entity_aliases",
            ):
                cur.execute(f"SELECT count(*) FROM {table}")
                assert cur.fetchone()[0] == 0, f"{table} not empty after promote"

    def test_promote_upsert_updates_existing(self, conn) -> None:
        with conn.cursor() as cur:
            # Insert into public directly
            cur.execute(
                """
                INSERT INTO public.entities
                    (canonical_name, entity_type, source, properties)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                ("Alpha Centauri", "star", "test", '{"old": true}'),
            )

            # Insert updated version into staging
            cur.execute(
                """
                INSERT INTO staging.entities
                    (canonical_name, entity_type, source, properties)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                ("Alpha Centauri", "star", "test", '{"new": true}'),
            )

            cur.execute("SELECT staging.promote_entities()")

            cur.execute(
                "SELECT properties FROM public.entities WHERE canonical_name = 'Alpha Centauri'"
            )
            assert cur.fetchone()[0] == {"new": True}

    def test_promote_empty_staging_returns_zero(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE staging.entity_aliases, staging.entity_identifiers, staging.entities"
            )
            cur.execute("SELECT staging.promote_entities()")
            assert cur.fetchone()[0] == 0
