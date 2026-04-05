"""Tests for the AstroMLab concept vocabulary harvester.

Unit tests use mocked HTTP responses and tmp_path fixtures for deterministic
testing without network or database access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_astromlab import (
    _parse_csv,
    _parse_json,
    download_concepts,
    load_concepts,
    map_category_to_entity_type,
    parse_concepts,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Sample fixture data
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
concept,category,description
Hubble Space Telescope,Instrumentation,Space-based optical telescope
Sloan Digital Sky Survey,Survey,Multi-filter imaging and spectroscopic survey
Markov Chain Monte Carlo,Statistical Methods,Sampling algorithm for posterior distributions
Convolutional Neural Network,Machine Learning,Deep learning architecture for image data
Gaia satellite,Instrument,Astrometry mission
VLA,Telescope,Very Large Array radio telescope
2MASS,Catalog,Two Micron All Sky Survey
Spectral Energy Distribution,Observational Techniques,
"""

SAMPLE_JSON = json.dumps(
    [
        {
            "concept": "Hubble Space Telescope",
            "category": "Instrumentation",
            "description": "Space-based optical telescope",
        },
        {
            "concept": "Sloan Digital Sky Survey",
            "category": "Survey",
            "description": "Multi-filter imaging and spectroscopic survey",
        },
        {
            "concept": "Markov Chain Monte Carlo",
            "category": "Statistical Methods",
            "description": "Sampling algorithm for posterior distributions",
        },
        {
            "concept": "Convolutional Neural Network",
            "category": "Machine Learning",
        },
        {
            "concept": "Gaia satellite",
            "category": "Instrument",
        },
        {
            "concept": "VLA",
            "category": "Telescope",
        },
        {
            "concept": "2MASS",
            "category": "Catalog",
        },
        {
            "concept": "Spectral Energy Distribution",
            "category": "Observational Techniques",
        },
    ]
)


# ---------------------------------------------------------------------------
# Unit tests: map_category_to_entity_type
# ---------------------------------------------------------------------------


class TestMapCategoryToEntityType:
    """Unit tests for category -> entity_type mapping."""

    def test_instrument_keywords(self) -> None:
        assert map_category_to_entity_type("Instrumentation") == "instrument"
        assert map_category_to_entity_type("Instrument") == "instrument"
        assert map_category_to_entity_type("Telescope Design") == "instrument"
        assert map_category_to_entity_type("Detector Physics") == "instrument"
        assert map_category_to_entity_type("Instrumental Calibration") == "instrument"

    def test_dataset_keywords(self) -> None:
        assert map_category_to_entity_type("Survey") == "dataset"
        assert map_category_to_entity_type("Star Catalog") == "dataset"
        assert map_category_to_entity_type("Catalogue of Nebulae") == "dataset"
        assert map_category_to_entity_type("Database Management") == "dataset"
        assert map_category_to_entity_type("Data Release 3") == "dataset"
        assert map_category_to_entity_type("Archive Systems") == "dataset"

    def test_method_default(self) -> None:
        assert map_category_to_entity_type("Statistical Methods") == "method"
        assert map_category_to_entity_type("Machine Learning") == "method"
        assert map_category_to_entity_type("Observational Techniques") == "method"

    def test_case_insensitive(self) -> None:
        assert map_category_to_entity_type("INSTRUMENTATION") == "instrument"
        assert map_category_to_entity_type("survey") == "dataset"
        assert map_category_to_entity_type("TELESCOPE") == "instrument"

    def test_empty_string_returns_method(self) -> None:
        assert map_category_to_entity_type("") == "method"

    def test_whitespace_handling(self) -> None:
        assert map_category_to_entity_type("  Telescope  ") == "instrument"
        assert map_category_to_entity_type("  Survey  ") == "dataset"


# ---------------------------------------------------------------------------
# Unit tests: _parse_csv
# ---------------------------------------------------------------------------


class TestParseCsv:
    """Unit tests for CSV parsing."""

    def test_parses_all_rows(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        assert len(entries) == 8

    def test_entry_structure(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        for entry in entries:
            assert "canonical_name" in entry
            assert "entity_type" in entry
            assert "source" in entry
            assert "aliases" in entry
            assert "metadata" in entry
            assert entry["source"] == "astromlab"
            assert isinstance(entry["aliases"], list)

    def test_instrument_category_mapping(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        hst = next(e for e in entries if e["canonical_name"] == "Hubble Space Telescope")
        assert hst["entity_type"] == "instrument"

    def test_dataset_category_mapping(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        sdss = next(e for e in entries if e["canonical_name"] == "Sloan Digital Sky Survey")
        assert sdss["entity_type"] == "dataset"

    def test_method_category_mapping(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        mcmc = next(e for e in entries if e["canonical_name"] == "Markov Chain Monte Carlo")
        assert mcmc["entity_type"] == "method"

    def test_category_in_metadata(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        hst = next(e for e in entries if e["canonical_name"] == "Hubble Space Telescope")
        assert hst["metadata"]["category"] == "Instrumentation"

    def test_description_in_metadata(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        hst = next(e for e in entries if e["canonical_name"] == "Hubble Space Telescope")
        assert hst["metadata"]["description"] == "Space-based optical telescope"

    def test_empty_description_not_in_metadata(self) -> None:
        entries = _parse_csv(SAMPLE_CSV)
        sed = next(e for e in entries if e["canonical_name"] == "Spectral Energy Distribution")
        assert "description" not in sed["metadata"]

    def test_skips_empty_concept_names(self) -> None:
        csv_text = "concept,category\n,Machine Learning\nValid Concept,Methods\n"
        entries = _parse_csv(csv_text)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "Valid Concept"

    def test_flexible_column_names(self) -> None:
        csv_text = "name,topic\nFoo,Survey\n"
        entries = _parse_csv(csv_text)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "Foo"
        assert entries[0]["entity_type"] == "dataset"

    def test_no_headers_returns_empty(self) -> None:
        entries = _parse_csv("")
        assert entries == []


# ---------------------------------------------------------------------------
# Unit tests: _parse_json
# ---------------------------------------------------------------------------


class TestParseJson:
    """Unit tests for JSON parsing."""

    def test_parses_all_items(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        assert len(entries) == 8

    def test_entry_structure(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        for entry in entries:
            assert "canonical_name" in entry
            assert "entity_type" in entry
            assert "source" in entry
            assert "aliases" in entry
            assert "metadata" in entry
            assert entry["source"] == "astromlab"

    def test_instrument_category_mapping(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        hst = next(e for e in entries if e["canonical_name"] == "Hubble Space Telescope")
        assert hst["entity_type"] == "instrument"

    def test_dataset_category_mapping(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        sdss = next(e for e in entries if e["canonical_name"] == "Sloan Digital Sky Survey")
        assert sdss["entity_type"] == "dataset"

    def test_method_category_mapping(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        mcmc = next(e for e in entries if e["canonical_name"] == "Markov Chain Monte Carlo")
        assert mcmc["entity_type"] == "method"

    def test_description_in_metadata(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        hst = next(e for e in entries if e["canonical_name"] == "Hubble Space Telescope")
        assert hst["metadata"]["description"] == "Space-based optical telescope"

    def test_no_description_key_when_absent(self) -> None:
        entries = _parse_json(SAMPLE_JSON)
        cnn = next(e for e in entries if e["canonical_name"] == "Convolutional Neural Network")
        assert "description" not in cnn["metadata"]

    def test_skips_items_without_concept_name(self) -> None:
        raw = json.dumps([{"category": "Methods"}, {"concept": "Valid", "category": "Methods"}])
        entries = _parse_json(raw)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "Valid"

    def test_flexible_key_names(self) -> None:
        raw = json.dumps([{"concept_name": "Bar", "topic": "Catalog"}])
        entries = _parse_json(raw)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "Bar"
        assert entries[0]["entity_type"] == "dataset"


# ---------------------------------------------------------------------------
# Unit tests: parse_concepts (auto-detect format)
# ---------------------------------------------------------------------------


class TestParseConcepts:
    """Unit tests for parse_concepts with file-based auto-detection."""

    def test_parse_concepts_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "concepts.csv"
        csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
        entries = parse_concepts(csv_file)
        assert len(entries) == 8
        assert all(e["source"] == "astromlab" for e in entries)

    def test_parse_concepts_json(self, tmp_path: Path) -> None:
        json_file = tmp_path / "concepts.json"
        json_file.write_text(SAMPLE_JSON, encoding="utf-8")
        entries = parse_concepts(json_file)
        assert len(entries) == 8
        assert all(e["source"] == "astromlab" for e in entries)

    def test_auto_detect_json_without_extension(self, tmp_path: Path) -> None:
        """File with .txt extension but JSON content should be parsed as JSON."""
        txt_file = tmp_path / "concepts.txt"
        txt_file.write_text(SAMPLE_JSON, encoding="utf-8")
        entries = parse_concepts(txt_file)
        assert len(entries) == 8

    def test_auto_detect_csv_without_extension(self, tmp_path: Path) -> None:
        """File with .txt extension but CSV content should be parsed as CSV."""
        txt_file = tmp_path / "concepts.txt"
        txt_file.write_text(SAMPLE_CSV, encoding="utf-8")
        entries = parse_concepts(txt_file)
        assert len(entries) == 8

    def test_empty_file(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("", encoding="utf-8")
        entries = parse_concepts(empty_file)
        assert entries == []

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        ws_file = tmp_path / "whitespace.csv"
        ws_file.write_text("   \n  \n  ", encoding="utf-8")
        entries = parse_concepts(ws_file)
        assert entries == []


# ---------------------------------------------------------------------------
# Unit tests: download_concepts
# ---------------------------------------------------------------------------


class TestDownloadConcepts:
    """Unit tests for download_concepts with mocked HTTP."""

    def test_skip_existing(self, tmp_path: Path) -> None:
        existing = tmp_path / "concepts.csv"
        existing.write_text(SAMPLE_CSV, encoding="utf-8")
        result = download_concepts(dest=existing)
        assert result == existing

    @patch("harvest_astromlab._get_client")
    def test_fetches_when_missing(self, mock_get_client: MagicMock, tmp_path: Path) -> None:
        dest = tmp_path / "concepts.csv"
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_CSV.encode("utf-8")
        mock_client.get.return_value = mock_resp
        mock_get_client.return_value = mock_client

        result = download_concepts(dest=dest)
        assert result == dest
        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == SAMPLE_CSV

    @patch("harvest_astromlab._get_client")
    def test_raises_on_request_error(self, mock_get_client: MagicMock, tmp_path: Path) -> None:
        dest = tmp_path / "concepts.csv"
        mock_client = MagicMock()
        mock_client.get.side_effect = requests.RequestException("persistent failure")
        mock_get_client.return_value = mock_client
        with pytest.raises(requests.RequestException):
            download_concepts(dest=dest)

    def test_skip_empty_file(self, tmp_path: Path) -> None:
        """An empty file (0 bytes) should NOT be treated as existing."""
        existing = tmp_path / "concepts.csv"
        existing.write_text("", encoding="utf-8")
        with patch("harvest_astromlab._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_CSV.encode("utf-8")
            mock_client.get.return_value = mock_resp
            mock_get_client.return_value = mock_client
            result = download_concepts(dest=existing)
        assert result == existing
        assert existing.stat().st_size > 0


# ---------------------------------------------------------------------------
# Unit tests: run_pipeline (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Unit tests for run_pipeline with mocked download and DB."""

    @patch("harvest_astromlab.HarvestRunLog")
    @patch("harvest_astromlab.get_connection")
    @patch("harvest_astromlab.bulk_load")
    def test_pipeline_with_local_file(
        self,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        csv_file = tmp_path / "concepts.csv"
        csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        count = run_pipeline(data_path=csv_file, dsn="dbname=test")

        assert count == 8
        mock_bulk_load.assert_called_once()
        loaded_entries = mock_bulk_load.call_args[0][1]
        assert len(loaded_entries) == 8
        assert all(e["source"] == "astromlab" for e in loaded_entries)
        mock_conn.close.assert_called_once()

    @patch("harvest_astromlab.HarvestRunLog")
    @patch("harvest_astromlab.get_connection")
    @patch("harvest_astromlab.bulk_load")
    def test_pipeline_closes_connection_on_error(
        self,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        csv_file = tmp_path / "concepts.csv"
        csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError):
            run_pipeline(data_path=csv_file, dsn="dbname=test")

        mock_conn.close.assert_called_once()

    def test_pipeline_returns_zero_for_empty_file(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("", encoding="utf-8")

        count = run_pipeline(data_path=empty_file, dsn="dbname=test")

        assert count == 0

    def test_pipeline_raises_for_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            run_pipeline(data_path=missing, dsn="dbname=test")

    @patch("harvest_astromlab.HarvestRunLog")
    @patch("harvest_astromlab.get_connection")
    @patch("harvest_astromlab.bulk_load")
    def test_pipeline_entity_types_are_correct(
        self,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        csv_file = tmp_path / "concepts.csv"
        csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        run_pipeline(data_path=csv_file, dsn="dbname=test")

        loaded_entries = mock_bulk_load.call_args[0][1]
        by_name = {e["canonical_name"]: e for e in loaded_entries}

        assert by_name["Hubble Space Telescope"]["entity_type"] == "instrument"
        assert by_name["Sloan Digital Sky Survey"]["entity_type"] == "dataset"
        assert by_name["Markov Chain Monte Carlo"]["entity_type"] == "method"
        assert by_name["VLA"]["entity_type"] == "instrument"
        assert by_name["2MASS"]["entity_type"] == "dataset"

    @patch("harvest_astromlab.HarvestRunLog")
    @patch("harvest_astromlab.get_connection")
    @patch("harvest_astromlab.bulk_load")
    def test_pipeline_creates_harvest_run(
        self,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify harvest_runs record is created and completed."""
        csv_file = tmp_path / "concepts.csv"
        csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        run_pipeline(data_path=csv_file, dsn="dbname=test")

        mock_run_log_cls.assert_called_once_with(mock_conn, "astromlab")
        mock_run_log.start.assert_called_once()
        mock_run_log.complete.assert_called_once()
        _, kwargs = mock_run_log.complete.call_args
        assert kwargs["records_fetched"] == 8
        assert kwargs["records_upserted"] == 8
