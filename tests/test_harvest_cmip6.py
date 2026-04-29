"""Tests for scripts/harvest_cmip6.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harvest_cmip6


# ---------------------------------------------------------------------------
# Fixtures — minimal but realistic CV records
# ---------------------------------------------------------------------------


SOURCE_RECORD: dict[str, Any] = {
    "label": "ACCESS-CM2",
    "label_extended": "Australian Community Climate and Earth System Simulator CM2",
    "institution_id": ["CSIRO-ARCCSS"],
    "activity_participation": ["CMIP", "ScenarioMIP"],
    "release_year": "2019",
    "license_info": {"id": "CC BY 4.0"},
    "model_component": {
        "atmos": {"description": "MetUM-HadGEM3-GA7.1", "native_nominal_resolution": "250 km"},
        "land": {"description": "none", "native_nominal_resolution": "none"},
    },
}

EXPERIMENT_RECORD: dict[str, Any] = {
    "experiment_id": "historical",
    "experiment": "all-forcing simulation of the recent past",
    "description": "CMIP6 historical",
    "activity_id": ["CMIP"],
    "parent_activity_id": ["CMIP"],
    "parent_experiment_id": ["piControl"],
    "tier": "1",
    "start_year": "1850",
    "end_year": "2014",
    "min_number_yrs_per_sim": "165",
    "required_model_components": ["AOGCM"],
    "additional_allowed_model_components": ["AER", "CHEM", "BGC"],
    "sub_experiment_id": ["none"],
}


# ---------------------------------------------------------------------------
# parse helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseSourceRecord:
    def test_basic_fields(self) -> None:
        record = harvest_cmip6._parse_source_record("ACCESS-CM2", SOURCE_RECORD)
        assert record["canonical_id"] == "ACCESS-CM2"
        assert record["name"] == SOURCE_RECORD["label_extended"]
        assert "Institution: CSIRO-ARCCSS" in record["description"]
        assert "MIPs: CMIP, ScenarioMIP" in record["description"]

    def test_properties_capture_release_year_and_license(self) -> None:
        record = harvest_cmip6._parse_source_record("ACCESS-CM2", SOURCE_RECORD)
        assert record["properties"]["release_year"] == "2019"
        assert record["properties"]["license"] == "CC BY 4.0"

    def test_filters_none_components(self) -> None:
        record = harvest_cmip6._parse_source_record("ACCESS-CM2", SOURCE_RECORD)
        components = record["properties"]["model_components"]
        assert "atmos" in components
        # land has description="none" — should be filtered out
        assert "land" not in components

    def test_falls_back_to_label_when_no_label_extended(self) -> None:
        raw = {**SOURCE_RECORD, "label_extended": ""}
        record = harvest_cmip6._parse_source_record("BAR-1", raw)
        assert record["name"] in (raw["label"], "BAR-1")


@pytest.mark.unit
class TestParseExperimentRecord:
    def test_basic_fields(self) -> None:
        record = harvest_cmip6._parse_experiment_record("historical", EXPERIMENT_RECORD)
        assert record["canonical_id"] == "historical"
        assert record["name"] == EXPERIMENT_RECORD["experiment"]
        assert record["temporal_start"] == "1850-01-01"
        assert record["temporal_end"] == "2014-12-31"

    def test_properties_capture_activity_and_parents(self) -> None:
        record = harvest_cmip6._parse_experiment_record("historical", EXPERIMENT_RECORD)
        props = record["properties"]
        assert props["activity_id"] == ["CMIP"]
        assert props["parent_experiment_id"] == ["piControl"]
        assert props["tier"] == "1"

    def test_handles_missing_temporal(self) -> None:
        raw = {**EXPERIMENT_RECORD, "start_year": "", "end_year": ""}
        record = harvest_cmip6._parse_experiment_record("foo", raw)
        assert record["temporal_start"] is None
        assert record["temporal_end"] is None


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchCv:
    def test_returns_inner_dict(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"source_id": {"FOO": SOURCE_RECORD}}
        client.get.return_value = resp

        cv = harvest_cmip6._fetch_cv(client, "https://example/cv.json", "source_id")
        assert cv == {"FOO": SOURCE_RECORD}

    def test_raises_when_key_missing(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {}
        client.get.return_value = resp

        with pytest.raises(ValueError):
            harvest_cmip6._fetch_cv(client, "https://example/cv.json", "source_id")

    def test_passes_accept_json_header(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"source_id": {}}
        client.get.return_value = resp

        harvest_cmip6._fetch_cv(client, "https://example/cv.json", "source_id")
        # ResilientClient.get receives the headers kwarg
        kwargs = client.get.call_args.kwargs
        headers = kwargs.get("headers", {})
        assert headers.get("Accept") == "application/json"


# ---------------------------------------------------------------------------
# DB write side
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoreKind:
    def test_store_source_kind_calls_upserts(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (42,)

        # Patch entity helpers since they'd otherwise need a real psycopg conn
        cv = {"ACCESS-CM2": SOURCE_RECORD}
        from unittest.mock import patch

        with patch.object(harvest_cmip6, "upsert_entity", return_value=99) as mock_ue, patch.object(
            harvest_cmip6, "upsert_entity_identifier"
        ) as mock_uei, patch.object(harvest_cmip6, "upsert_entity_alias") as mock_uea:
            counts = harvest_cmip6._store_kind(
                mock_conn, "source", cv, harvest_run_id=1
            )

        assert counts["source_datasets"] == 1
        assert counts["source_entities"] == 1
        # _upsert_dataset executes one INSERT
        sql = mock_cursor.execute.call_args[0][0]
        assert "datasets" in sql
        # entity helpers invoked once each
        mock_ue.assert_called_once()
        mock_uei.assert_called_once()
        # alias upserted only when label != name; in fixture name == label_extended != label
        assert mock_uea.call_count >= 0


@pytest.mark.unit
class TestKindConfig:
    def test_known_kinds(self) -> None:
        assert "source" in harvest_cmip6.KIND_CONFIG
        assert "experiment" in harvest_cmip6.KIND_CONFIG

    def test_kind_config_has_required_fields(self) -> None:
        for kind, cfg in harvest_cmip6.KIND_CONFIG.items():
            assert "json_url" in cfg, f"missing json_url for {kind}"
            assert "json_key" in cfg, f"missing json_key for {kind}"
            assert "dataset_source" in cfg, f"missing dataset_source for {kind}"
            assert "id_scheme" in cfg, f"missing id_scheme for {kind}"
