"""Tests for migration 020_harvest_runs.sql.

Validates SQL structure and content statically (no live DB needed).
"""

from __future__ import annotations

import pathlib

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "migrations" / "020_harvest_runs.sql"
)


class TestMigrationFileExists:
    """Verify the migration file is present."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists(), f"Migration file not found: {MIGRATION_PATH}"

    def test_migration_is_not_empty(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert sql.strip(), "Migration file is empty"


class TestTransactionWrapping:
    """Verify BEGIN/COMMIT transaction wrapping."""

    def test_migration_wrapped_in_transaction(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert "BEGIN;" in sql, "Migration should start with BEGIN;"
        assert "COMMIT;" in sql, "Migration should end with COMMIT;"


class TestTableStructure:
    """Verify harvest_runs table DDL."""

    def test_creates_harvest_runs_table(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert "CREATE TABLE IF NOT EXISTS harvest_runs" in sql

    def test_has_id_serial_primary_key(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "id serial primary key" in sql

    def test_has_source_text_not_null(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "source text not null" in sql

    def test_has_started_at(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "started_at timestamptz" in sql

    def test_has_finished_at(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "finished_at timestamptz" in sql

    def test_has_status_with_default(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "status text not null default 'running'" in sql

    def test_has_records_fetched(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "records_fetched int" in sql

    def test_has_records_upserted(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "records_upserted int" in sql

    def test_has_cursor_state_jsonb(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "cursor_state jsonb" in sql

    def test_has_error_message_text(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "error_message text" in sql

    def test_has_config_jsonb(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "config jsonb" in sql

    def test_has_counts_jsonb(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "counts jsonb" in sql


class TestIndexes:
    """Verify indexes on harvest_runs."""

    def test_has_source_index(self) -> None:
        sql = MIGRATION_PATH.read_text().lower()
        assert "idx_harvest_runs_source" in sql
        assert "harvest_runs(source)" in sql
