"""Tests for the migration runner (scripts/setup_db.sh)."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_db.sh"


class TestMigrationFileIntegrity:
    """Verify migration file numbering and naming conventions."""

    def _migration_files(self) -> list[pathlib.Path]:
        return sorted(MIGRATIONS_DIR.glob("*.sql"))

    def test_no_duplicate_version_numbers(self) -> None:
        versions: dict[int, list[str]] = {}
        for f in self._migration_files():
            match = re.match(r"^(\d+)_", f.name)
            assert match, f"Migration file {f.name} does not start with a numeric prefix"
            version = int(match.group(1))
            versions.setdefault(version, []).append(f.name)

        duplicates = {v: names for v, names in versions.items() if len(names) > 1}
        assert duplicates == {}, f"Duplicate migration versions: {duplicates}"

    def test_013_collision_resolved(self) -> None:
        """The old 013_query_log.sql should no longer exist."""
        assert not (MIGRATIONS_DIR / "013_query_log.sql").exists()
        assert (MIGRATIONS_DIR / "013_entity_dictionary.sql").exists()

    def test_migration_019_exists(self) -> None:
        path = MIGRATIONS_DIR / "019_schema_migrations.sql"
        assert path.exists()
        content = path.read_text()
        assert "schema_migrations" in content
        assert "version INT PRIMARY KEY" in content or "version INT" in content

    def test_all_migrations_contiguous(self) -> None:
        """Versions 1-19 should all be present with no gaps."""
        versions = set()
        for f in self._migration_files():
            match = re.match(r"^(\d+)_", f.name)
            if match:
                versions.add(int(match.group(1)))
        expected = set(range(1, 20))
        assert versions == expected, f"Missing: {expected - versions}, Extra: {versions - expected}"

    def test_016_is_query_log(self) -> None:
        """016_query_log.sql should be the canonical query_log migration."""
        path = MIGRATIONS_DIR / "016_query_log.sql"
        assert path.exists()
        content = path.read_text()
        assert "query_log" in content


class TestSetupScript:
    """Verify the setup script structure and behavior."""

    def test_script_exists_and_is_executable_content(self) -> None:
        assert SETUP_SCRIPT.exists()
        content = SETUP_SCRIPT.read_text()
        assert content.startswith("#!/usr/bin/env bash")

    def test_script_creates_schema_migrations_table(self) -> None:
        content = SETUP_SCRIPT.read_text()
        assert "schema_migrations" in content

    def test_script_iterates_migration_files(self) -> None:
        content = SETUP_SCRIPT.read_text()
        assert "migrations" in content.lower()
        # Should loop over .sql files
        assert ".sql" in content

    def test_script_checks_before_applying(self) -> None:
        """Runner should check schema_migrations before applying each migration."""
        content = SETUP_SCRIPT.read_text()
        assert "SELECT" in content
        assert "schema_migrations" in content

    def test_script_records_applied_migrations(self) -> None:
        content = SETUP_SCRIPT.read_text()
        assert "INSERT INTO schema_migrations" in content

    def test_script_handles_on_conflict(self) -> None:
        """INSERT should use ON CONFLICT DO NOTHING for idempotency."""
        content = SETUP_SCRIPT.read_text()
        assert "ON CONFLICT" in content

    def test_script_creates_db_if_not_exists(self) -> None:
        content = SETUP_SCRIPT.read_text()
        assert "CREATE DATABASE" in content

    def test_script_ensures_pgvector(self) -> None:
        content = SETUP_SCRIPT.read_text()
        assert "CREATE EXTENSION IF NOT EXISTS vector" in content

    def test_script_uses_set_euo_pipefail(self) -> None:
        content = SETUP_SCRIPT.read_text()
        assert "set -euo pipefail" in content


class TestMigration019Content:
    """Verify the schema_migrations migration SQL content."""

    def test_creates_table_with_correct_columns(self) -> None:
        content = (MIGRATIONS_DIR / "019_schema_migrations.sql").read_text()
        assert "CREATE TABLE IF NOT EXISTS schema_migrations" in content
        assert "version INT PRIMARY KEY" in content
        assert "applied_at TIMESTAMPTZ" in content
        assert "filename TEXT NOT NULL" in content

    def test_uses_transaction(self) -> None:
        content = (MIGRATIONS_DIR / "019_schema_migrations.sql").read_text()
        assert "BEGIN;" in content
        assert "COMMIT;" in content
