"""Tests for migration 021_entity_graph.sql — entity graph schema."""

import os
import re

import pytest

MIGRATION_PATH = os.path.join(os.path.dirname(__file__), "..", "migrations", "021_entity_graph.sql")


@pytest.fixture(scope="module")
def sql_content() -> str:
    """Read the migration file once for all tests."""
    with open(MIGRATION_PATH) as f:
        return f.read()


class TestMigrationFileExists:
    def test_file_exists(self) -> None:
        assert os.path.isfile(MIGRATION_PATH), "021_entity_graph.sql must exist"

    def test_valid_sql_structure(self, sql_content: str) -> None:
        """Migration should be wrapped in BEGIN/COMMIT."""
        assert "BEGIN;" in sql_content
        assert "COMMIT;" in sql_content


class TestEntitiesTable:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS entities" in sql_content

    def test_columns(self, sql_content: str) -> None:
        for col in [
            "id SERIAL PRIMARY KEY",
            "canonical_name TEXT NOT NULL",
            "entity_type TEXT NOT NULL",
            "discipline TEXT",
            "source TEXT NOT NULL",
            "harvest_run_id INT REFERENCES harvest_runs(id)",
            "properties JSONB",
            "created_at TIMESTAMPTZ",
            "updated_at TIMESTAMPTZ",
        ]:
            assert col in sql_content, f"entities table missing column: {col}"

    def test_unique_constraint(self, sql_content: str) -> None:
        assert "UNIQUE (canonical_name, entity_type, source)" in sql_content

    def test_indexes(self, sql_content: str) -> None:
        assert "idx_entities_entity_type" in sql_content
        assert "idx_entities_discipline" in sql_content
        assert "idx_entities_canonical_lower" in sql_content
        assert "lower(canonical_name)" in sql_content
        assert "jsonb_path_ops" in sql_content


class TestEntityIdentifiers:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS entity_identifiers" in sql_content

    def test_columns(self, sql_content: str) -> None:
        for col in ["id_scheme TEXT NOT NULL", "external_id TEXT NOT NULL", "is_primary BOOLEAN"]:
            assert col in sql_content, f"entity_identifiers missing: {col}"

    def test_primary_key(self, sql_content: str) -> None:
        assert "PRIMARY KEY (id_scheme, external_id)" in sql_content

    def test_fk_cascade(self, sql_content: str) -> None:
        # entity_id FK with CASCADE in the entity_identifiers CREATE TABLE block
        start = sql_content.index("CREATE TABLE IF NOT EXISTS entity_identifiers")
        end = sql_content.index(");", start) + 2
        section = sql_content[start:end]
        assert "REFERENCES entities(id) ON DELETE CASCADE" in section

    def test_entity_id_index(self, sql_content: str) -> None:
        assert "idx_entity_identifiers_entity_id" in sql_content


class TestEntityAliases:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS entity_aliases" in sql_content

    def test_primary_key(self, sql_content: str) -> None:
        assert "PRIMARY KEY (entity_id, alias)" in sql_content

    def test_functional_index(self, sql_content: str) -> None:
        assert "idx_entity_aliases_lower" in sql_content
        assert "lower(alias)" in sql_content


class TestEntityRelationships:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS entity_relationships" in sql_content

    def test_columns(self, sql_content: str) -> None:
        for col in [
            "subject_entity_id INT REFERENCES entities(id)",
            "predicate TEXT NOT NULL",
            "object_entity_id INT REFERENCES entities(id)",
            "confidence REAL",
        ]:
            assert col in sql_content, f"entity_relationships missing: {col}"

    def test_unique_constraint(self, sql_content: str) -> None:
        assert "UNIQUE (subject_entity_id, predicate, object_entity_id)" in sql_content

    def test_object_index(self, sql_content: str) -> None:
        assert "idx_entity_relationships_object" in sql_content


class TestDocumentEntities:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS document_entities" in sql_content

    def test_primary_key(self, sql_content: str) -> None:
        assert "PRIMARY KEY (bibcode, entity_id, link_type)" in sql_content

    def test_no_bibcode_fk(self, sql_content: str) -> None:
        """bibcode should NOT have a REFERENCES constraint (matches citation_edges pattern)."""
        start = sql_content.index("CREATE TABLE IF NOT EXISTS document_entities")
        # Find the end of this CREATE TABLE block
        end = sql_content.index(");", start) + 2
        section = sql_content[start:end]
        # bibcode line should not have REFERENCES
        bibcode_lines = [
            l for l in section.split("\n") if "bibcode" in l.lower() and "REFERENCES" in l
        ]
        assert len(bibcode_lines) == 0, "bibcode must not have a REFERENCES constraint"

    def test_columns(self, sql_content: str) -> None:
        for col in [
            "link_type TEXT NOT NULL",
            "confidence REAL",
            "match_method TEXT",
            "evidence JSONB",
        ]:
            assert col in sql_content, f"document_entities missing: {col}"


class TestDatasetsTable:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS datasets" in sql_content

    def test_unique_constraint(self, sql_content: str) -> None:
        assert "UNIQUE (source, canonical_id)" in sql_content

    def test_columns(self, sql_content: str) -> None:
        for col in [
            "canonical_id TEXT NOT NULL",
            "temporal_start DATE",
            "temporal_end DATE",
        ]:
            assert col in sql_content, f"datasets missing: {col}"


class TestDatasetEntities:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS dataset_entities" in sql_content

    def test_primary_key(self, sql_content: str) -> None:
        assert "PRIMARY KEY (dataset_id, entity_id, relationship)" in sql_content


class TestDocumentDatasets:
    def test_create_table(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS document_datasets" in sql_content

    def test_primary_key(self, sql_content: str) -> None:
        assert "PRIMARY KEY (bibcode, dataset_id, link_type)" in sql_content

    def test_no_bibcode_fk(self, sql_content: str) -> None:
        start = sql_content.index("CREATE TABLE IF NOT EXISTS document_datasets")
        end = sql_content.index(");", start) + 2
        section = sql_content[start:end]
        bibcode_lines = [
            l for l in section.split("\n") if "bibcode" in l.lower() and "REFERENCES" in l
        ]
        assert len(bibcode_lines) == 0, "bibcode must not have a REFERENCES constraint"


class TestCompatView:
    def test_view_defined(self, sql_content: str) -> None:
        assert "CREATE OR REPLACE VIEW entity_dictionary_compat" in sql_content

    def test_view_columns(self, sql_content: str) -> None:
        """View must expose: id, canonical_name, entity_type, source, external_id, aliases, metadata."""
        view_start = sql_content.index("CREATE OR REPLACE VIEW entity_dictionary_compat")
        # Find the semicolon ending the view definition
        view_end = sql_content.index(";", view_start)
        view_sql = sql_content[view_start:view_end]
        for col in ["canonical_name", "entity_type", "external_id", "aliases", "metadata"]:
            assert col in view_sql, f"compat view missing column: {col}"


class TestSeedMigration:
    def test_seed_entities(self, sql_content: str) -> None:
        assert "INSERT INTO entities" in sql_content
        assert "FROM entity_dictionary" in sql_content

    def test_seed_aliases(self, sql_content: str) -> None:
        assert "INSERT INTO entity_aliases" in sql_content
        assert "unnest" in sql_content.lower()

    def test_seed_identifiers(self, sql_content: str) -> None:
        assert "INSERT INTO entity_identifiers" in sql_content
        assert "external_id IS NOT NULL" in sql_content


class TestAllConstraints:
    """Verify all FK, PK, and UNIQUE constraints are present."""

    def test_all_fk_references(self, sql_content: str) -> None:
        fk_patterns = [
            "REFERENCES harvest_runs(id)",
            "REFERENCES entities(id)",
            "REFERENCES datasets(id)",
        ]
        for pat in fk_patterns:
            assert pat in sql_content, f"Missing FK: {pat}"

    def test_cascade_deletes(self, sql_content: str) -> None:
        """All entity/dataset child tables should CASCADE on delete."""
        cascade_count = sql_content.count("ON DELETE CASCADE")
        # entity_identifiers, entity_aliases, entity_relationships (2x),
        # document_entities, dataset_entities (2x), document_datasets
        assert cascade_count >= 7, f"Expected at least 7 ON DELETE CASCADE, got {cascade_count}"
