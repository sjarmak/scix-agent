"""Tests for entity provenance metadata (migration 017).

Covers:
- ExtractionRow dataclass provenance defaults and custom values
- load_results_to_db() writes provenance columns
- Migration SQL structure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix.extract import (
    ExtractionRow,
    _parse_extraction_rows,
    load_results_to_db,
)

# ---------------------------------------------------------------------------
# ExtractionRow provenance fields
# ---------------------------------------------------------------------------


class TestExtractionRowProvenance:
    def test_default_provenance_values(self) -> None:
        row = ExtractionRow(
            bibcode="2024ApJ...1A",
            extraction_type="methods",
            extraction_version="v1",
            payload={"entities": ["PCA"]},
        )
        assert row.source == "llm"
        assert row.confidence_tier == "medium"
        assert row.extraction_model is None

    def test_custom_provenance_values(self) -> None:
        row = ExtractionRow(
            bibcode="2024ApJ...1A",
            extraction_type="methods",
            extraction_version="v1",
            payload={"entities": ["PCA"]},
            source="metadata",
            confidence_tier="high",
            extraction_model="claude-sonnet-4-20250514",
        )
        assert row.source == "metadata"
        assert row.confidence_tier == "high"
        assert row.extraction_model == "claude-sonnet-4-20250514"

    def test_frozen_provenance_fields(self) -> None:
        row = ExtractionRow(
            bibcode="bib",
            extraction_type="methods",
            extraction_version="v1",
            payload={"entities": []},
        )
        with pytest.raises(AttributeError):
            row.source = "ner"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            row.confidence_tier = "high"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            row.extraction_model = "model"  # type: ignore[misc]

    def test_all_valid_sources(self) -> None:
        """ExtractionRow accepts all valid source values."""
        valid_sources = ["metadata", "ner", "llm", "openalex", "citation_propagation"]
        for src in valid_sources:
            row = ExtractionRow(
                bibcode="bib",
                extraction_type="methods",
                extraction_version="v1",
                payload={"entities": []},
                source=src,
            )
            assert row.source == src

    def test_all_valid_confidence_tiers(self) -> None:
        """ExtractionRow accepts all valid confidence_tier values."""
        for tier in ["high", "medium", "low"]:
            row = ExtractionRow(
                bibcode="bib",
                extraction_type="methods",
                extraction_version="v1",
                payload={"entities": []},
                confidence_tier=tier,
            )
            assert row.confidence_tier == tier


# ---------------------------------------------------------------------------
# load_results_to_db provenance columns
# ---------------------------------------------------------------------------


class TestLoadResultsProvenanceColumns:
    def _make_jsonl(self, tmp_path: Path, lines: list[dict[str, Any]]) -> Path:
        p = tmp_path / "test_results.jsonl"
        with open(p, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return p

    def _sample_result_line(self, bibcode: str = "2024ApJ...1A") -> dict[str, Any]:
        return {
            "custom_id": bibcode,
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "extract_entities",
                            "input": {
                                "methods": ["PCA"],
                                "datasets": [],
                                "instruments": [],
                                "materials": [],
                            },
                        }
                    ]
                },
            },
        }

    def test_insert_sql_includes_provenance_columns(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [self._sample_result_line()])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        load_results_to_db(mock_conn, jsonl)

        sql = mock_cur.execute.call_args_list[0][0][0]
        assert "source" in sql
        assert "confidence_tier" in sql
        assert "extraction_model" in sql

    def test_insert_params_include_provenance_defaults(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [self._sample_result_line()])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        load_results_to_db(mock_conn, jsonl)

        # Params tuple: (bibcode, type, version, payload, source, confidence_tier, extraction_model)
        params = mock_cur.execute.call_args_list[0][0][1]
        assert len(params) == 7
        assert params[4] == "llm"  # source default
        assert params[5] == "medium"  # confidence_tier default
        assert params[6] is None  # extraction_model default

    def test_on_conflict_updates_provenance(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [self._sample_result_line()])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        load_results_to_db(mock_conn, jsonl)

        sql = mock_cur.execute.call_args_list[0][0][0]
        assert "EXCLUDED.source" in sql
        assert "EXCLUDED.confidence_tier" in sql
        assert "EXCLUDED.extraction_model" in sql


# ---------------------------------------------------------------------------
# _parse_extraction_rows preserves defaults
# ---------------------------------------------------------------------------


class TestParseExtractionRowsProvenance:
    def test_parsed_rows_have_default_provenance(self) -> None:
        line = {
            "custom_id": "2024ApJ...1A",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "extract_entities",
                            "input": {
                                "methods": ["MCMC"],
                                "datasets": [],
                                "instruments": [],
                                "materials": [],
                            },
                        }
                    ]
                },
            },
        }
        rows = _parse_extraction_rows(line, "v1")
        assert len(rows) == 1
        assert rows[0].source == "llm"
        assert rows[0].confidence_tier == "medium"
        assert rows[0].extraction_model is None


# ---------------------------------------------------------------------------
# Migration SQL validation
# ---------------------------------------------------------------------------


class TestMigrationSQL:
    def test_migration_file_exists(self) -> None:
        migration = Path(__file__).parent.parent / "migrations" / "017_entity_provenance.sql"
        assert migration.exists(), f"Migration file not found: {migration}"

    def test_migration_adds_columns(self) -> None:
        migration = Path(__file__).parent.parent / "migrations" / "017_entity_provenance.sql"
        sql = migration.read_text()
        assert "ADD COLUMN" in sql
        assert "source" in sql
        assert "confidence_tier" in sql
        assert "extraction_model" in sql

    def test_migration_has_check_constraints(self) -> None:
        migration = Path(__file__).parent.parent / "migrations" / "017_entity_provenance.sql"
        sql = migration.read_text()
        assert "chk_extractions_source" in sql
        assert "chk_extractions_confidence_tier" in sql

    def test_migration_has_indexes(self) -> None:
        migration = Path(__file__).parent.parent / "migrations" / "017_entity_provenance.sql"
        sql = migration.read_text()
        assert "idx_extractions_source" in sql
        assert "idx_extractions_confidence_tier" in sql

    def test_migration_wraps_in_transaction(self) -> None:
        migration = Path(__file__).parent.parent / "migrations" / "017_entity_provenance.sql"
        sql = migration.read_text()
        assert sql.strip().startswith("--") or "BEGIN" in sql
        assert "COMMIT" in sql

    def test_migration_default_values(self) -> None:
        migration = Path(__file__).parent.parent / "migrations" / "017_entity_provenance.sql"
        sql = migration.read_text()
        assert "DEFAULT 'llm'" in sql
        assert "DEFAULT 'medium'" in sql
