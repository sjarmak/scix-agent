"""Tests for migration files and schema_migrations bookkeeping.

The per-migration runner that lived in scripts/setup_db.sh was removed in
commit 16cc518 ("chore: strip internal docs, consolidate migrations into
schema.sql"); setup_db.sh now applies schema.sql directly. The TestSetupScript
class that asserted runner behaviour was deleted with it (scix_experiments-7n2v).
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"


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
        """All migration versions should be present with no gaps."""
        versions = set()
        for f in self._migration_files():
            match = re.match(r"^(\d+)_", f.name)
            if match:
                versions.add(int(match.group(1)))
        max_version = max(versions)
        expected = set(range(1, max_version + 1))
        assert versions == expected, f"Missing: {expected - versions}, Extra: {versions - expected}"

    def test_016_is_query_log(self) -> None:
        """016_query_log.sql should be the canonical query_log migration."""
        path = MIGRATIONS_DIR / "016_query_log.sql"
        assert path.exists()
        content = path.read_text()
        assert "query_log" in content


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
