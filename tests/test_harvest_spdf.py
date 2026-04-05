"""Tests for scripts/harvest_spdf.py — SPDF/CDAWeb harvester."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harvest_spdf

# ---------------------------------------------------------------------------
# Fixtures — realistic CDAWeb API JSON responses
# ---------------------------------------------------------------------------

MOCK_DATASETS_RESPONSE: dict[str, Any] = {
    "DatasetDescription": [
        {
            "Id": "AC_H0_MFI",
            "Label": "ACE Magnetic Field 16-Second Level 2 Data",
            "TimeInterval": {
                "Start": "1997-09-02T00:00:12.000Z",
                "End": "2024-12-31T23:59:44.000Z",
            },
            "ObservatoryGroup": ["ACE"],
            "InstrumentType": ["Magnetic Fields (Satellite)"],
            "Notes": "ACE Magnetometer Instrument 16-second data",
            "ResourceId": "spase://NASA/NumericalData/ACE/MAG/L2/PT16S",
        },
        {
            "Id": "THA_L2_FGM",
            "Label": "THEMIS-A FGM Level 2 Data",
            "TimeInterval": {
                "Start": "2007-02-23T00:00:00.000Z",
                "End": "2025-01-15T23:59:59.000Z",
            },
            "ObservatoryGroup": ["THEMIS"],
            "InstrumentType": ["Magnetic Fields (Satellite)"],
            "Notes": "THEMIS-A Fluxgate Magnetometer Level 2 data",
        },
        {
            "Id": "WI_H0_SWE",
            "Label": "Wind Solar Wind Experiment (SWE)",
            "TimeInterval": {
                "Start": "1994-11-01T00:00:00.000Z",
                "End": "2024-06-30T23:59:59.000Z",
            },
            "ObservatoryGroup": ["Wind"],
            "InstrumentType": ["Plasma and Solar Wind"],
        },
    ],
}

MOCK_OBSERVATORIES_RESPONSE: dict[str, Any] = {
    "ObservatoryGroupDescription": [
        {
            "Name": "ACE",
            "ObservatoryDescription": [
                {
                    "Name": "ACE",
                    "ResourceId": "spase://SMWG/Observatory/ACE",
                },
            ],
        },
        {
            "Name": "THEMIS",
            "ObservatoryDescription": [
                {
                    "Name": "THEMIS-A",
                    "ResourceId": "spase://SMWG/Observatory/THEMIS/A",
                },
                {
                    "Name": "THEMIS-B",
                    "ResourceId": "spase://SMWG/Observatory/THEMIS/B",
                },
            ],
        },
        {
            "Name": "Wind",
            "ObservatoryDescription": {
                "Name": "Wind",
                "ResourceId": "spase://SMWG/Observatory/Wind",
            },
        },
    ],
}

MOCK_INSTRUMENTS_RESPONSE: dict[str, Any] = {
    "InstrumentTypeDescription": [
        {
            "Name": "Magnetic Fields (Satellite)",
            "InstrumentDescription": [
                {
                    "Name": "ACE MAG",
                    "ResourceId": "spase://SMWG/Instrument/ACE/MAG",
                },
                {
                    "Name": "THEMIS-A FGM",
                    "ResourceId": "spase://SMWG/Instrument/THEMIS/A/FGM",
                },
            ],
        },
        {
            "Name": "Plasma and Solar Wind",
            "InstrumentDescription": [
                {
                    "Name": "Wind SWE",
                    "ResourceId": "spase://SMWG/Instrument/Wind/SWE",
                },
            ],
        },
    ],
}


def _mock_get(url: str, **kwargs: Any) -> MagicMock:
    """Return a mock response based on the URL."""
    resp = MagicMock()
    if "/datasets" in url:
        resp.json.return_value = MOCK_DATASETS_RESPONSE
    elif "/observatories" in url:
        resp.json.return_value = MOCK_OBSERVATORIES_RESPONSE
    elif "/instruments" in url:
        resp.json.return_value = MOCK_INSTRUMENTS_RESPONSE
    else:
        resp.json.return_value = {}
    resp.status_code = 200
    return resp


# ---------------------------------------------------------------------------
# Test: ResilientClient is used
# ---------------------------------------------------------------------------


class TestResilientClientUsed:
    """Verify that ResilientClient is used for all HTTP calls."""

    def test_make_client_returns_resilient_client(self) -> None:
        client = harvest_spdf._make_client()
        assert isinstance(client, harvest_spdf.ResilientClient)

    @patch.object(harvest_spdf.ResilientClient, "get", side_effect=_mock_get)
    def test_fetch_all_uses_client_get(self, mock_get: MagicMock) -> None:
        client = harvest_spdf._make_client()
        result = harvest_spdf.fetch_all(client)

        assert mock_get.call_count == 3
        urls = [call.args[0] for call in mock_get.call_args_list]
        assert any("/datasets" in u for u in urls)
        assert any("/observatories" in u for u in urls)
        assert any("/instruments" in u for u in urls)

        # Verify Accept header
        for call in mock_get.call_args_list:
            headers = call.kwargs.get("headers", {})
            assert headers.get("Accept") == "application/json"

    @patch.object(harvest_spdf.ResilientClient, "get", side_effect=_mock_get)
    def test_fetch_datasets_returns_list(self, mock_get: MagicMock) -> None:
        client = harvest_spdf._make_client()
        datasets = harvest_spdf.fetch_datasets(client)
        assert len(datasets) == 3
        assert datasets[0]["Id"] == "AC_H0_MFI"


# ---------------------------------------------------------------------------
# Test: Parse functions
# ---------------------------------------------------------------------------


class TestParsing:
    """Verify parsing logic extracts correct records."""

    def test_parse_observatories(self) -> None:
        groups = MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        obs = harvest_spdf.parse_observatories(groups)

        names = {o["name"] for o in obs}
        assert "ACE" in names
        assert "THEMIS-A" in names
        assert "THEMIS-B" in names
        assert "Wind" in names
        assert len(obs) == 4

        ace = next(o for o in obs if o["name"] == "ACE")
        assert ace["spase_resource_id"] == "spase://SMWG/Observatory/ACE"
        assert ace["group"] == "ACE"

    def test_parse_observatories_deduplicates(self) -> None:
        groups = [
            {
                "Name": "Group1",
                "ObservatoryDescription": [{"Name": "Obs1", "ResourceId": "id1"}],
            },
            {
                "Name": "Group2",
                "ObservatoryDescription": [{"Name": "Obs1", "ResourceId": "id2"}],
            },
        ]
        obs = harvest_spdf.parse_observatories(groups)
        assert len(obs) == 1

    def test_parse_observatories_handles_single_dict(self) -> None:
        """When there's one observatory, the API may return a dict instead of list."""
        groups = MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        # Wind group has a single dict instead of list
        wind_group = next(g for g in groups if g["Name"] == "Wind")
        assert isinstance(wind_group["ObservatoryDescription"], dict)

        obs = harvest_spdf.parse_observatories(groups)
        assert "Wind" in {o["name"] for o in obs}

    def test_parse_instruments(self) -> None:
        types = MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        instruments = harvest_spdf.parse_instruments(types)

        names = {i["name"] for i in instruments}
        assert "ACE MAG" in names
        assert "THEMIS-A FGM" in names
        assert "Wind SWE" in names
        assert len(instruments) == 3

        ace_mag = next(i for i in instruments if i["name"] == "ACE MAG")
        assert ace_mag["spase_resource_id"] == "spase://SMWG/Instrument/ACE/MAG"
        assert ace_mag["instrument_type"] == "Magnetic Fields (Satellite)"

    def test_parse_datasets(self) -> None:
        raw = MOCK_DATASETS_RESPONSE["DatasetDescription"]
        datasets = harvest_spdf.parse_datasets(raw)

        assert len(datasets) == 3

        ace = next(d for d in datasets if d["id"] == "AC_H0_MFI")
        assert ace["label"] == "ACE Magnetic Field 16-Second Level 2 Data"
        assert ace["start_date"] == "1997-09-02"
        assert ace["end_date"] == "2024-12-31"
        assert ace["description"] == "ACE Magnetometer Instrument 16-second data"
        assert "ACE" in ace["observatory_groups"]
        assert "Magnetic Fields (Satellite)" in ace["instrument_types"]
        assert ace["spase_resource_id"] == "spase://NASA/NumericalData/ACE/MAG/L2/PT16S"

    def test_parse_datasets_missing_spase(self) -> None:
        raw = MOCK_DATASETS_RESPONSE["DatasetDescription"]
        datasets = harvest_spdf.parse_datasets(raw)

        wind = next(d for d in datasets if d["id"] == "WI_H0_SWE")
        assert "spase_resource_id" not in wind

    def test_parse_datasets_skips_empty_id(self) -> None:
        raw = [{"Id": "", "Label": "Empty"}]
        datasets = harvest_spdf.parse_datasets(raw)
        assert len(datasets) == 0


# ---------------------------------------------------------------------------
# Test: Database storage
# ---------------------------------------------------------------------------


def _make_mock_conn() -> MagicMock:
    """Create a mock database connection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    # Track entity IDs by call count
    _entity_counter = {"value": 0}
    _dataset_counter = {"value": 0}
    _run_id = 42

    def fetchone_side_effect() -> tuple[int]:
        # First call is harvest_run creation
        nonlocal _entity_counter, _dataset_counter, _run_id
        _entity_counter["value"] += 1
        return (_entity_counter["value"],)

    cursor.fetchone.side_effect = fetchone_side_effect

    return conn


class TestStoreHarvest:
    """Verify database writes for entities, datasets, and relationships."""

    def test_store_creates_harvest_run(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )
        inst = harvest_spdf.parse_instruments(
            MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        )
        ds = harvest_spdf.parse_datasets(MOCK_DATASETS_RESPONSE["DatasetDescription"])

        harvest_spdf.store_harvest(conn, observatories=obs, instruments=inst, datasets=ds)

        # Check that INSERT INTO harvest_runs was called
        all_sql = [call.args[0].strip() for call in cursor.execute.call_args_list]
        harvest_run_inserts = [s for s in all_sql if "INSERT INTO harvest_runs" in s]
        assert len(harvest_run_inserts) >= 1

    def test_store_creates_entities_for_observatories(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )

        harvest_spdf.store_harvest(conn, observatories=obs, instruments=[], datasets=[])

        all_sql = [call.args[0].strip() for call in cursor.execute.call_args_list]
        entity_inserts = [s for s in all_sql if "INSERT INTO entities" in s]
        # 4 observatories
        assert len(entity_inserts) >= 4

    def test_store_creates_entities_for_instruments(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        inst = harvest_spdf.parse_instruments(
            MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        )

        harvest_spdf.store_harvest(conn, observatories=[], instruments=inst, datasets=[])

        all_sql = [call.args[0].strip() for call in cursor.execute.call_args_list]
        entity_inserts = [s for s in all_sql if "INSERT INTO entities" in s]
        # 3 instruments
        assert len(entity_inserts) >= 3

    def test_store_creates_spase_identifiers(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )

        harvest_spdf.store_harvest(conn, observatories=obs, instruments=[], datasets=[])

        all_calls = cursor.execute.call_args_list
        identifier_inserts = [
            call for call in all_calls if "INSERT INTO entity_identifiers" in call.args[0]
        ]
        # All 4 observatories have SPASE ResourceIDs
        assert len(identifier_inserts) >= 4

        # Check id_scheme is spase_resource_id
        for call in identifier_inserts:
            params = call.args[1]
            assert params[1] == "spase_resource_id"

    def test_store_creates_datasets_with_correct_source(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        ds = harvest_spdf.parse_datasets(MOCK_DATASETS_RESPONSE["DatasetDescription"])

        harvest_spdf.store_harvest(conn, observatories=[], instruments=[], datasets=ds)

        all_calls = cursor.execute.call_args_list
        dataset_inserts = [call for call in all_calls if "INSERT INTO datasets" in call.args[0]]
        assert len(dataset_inserts) == 3

        for call in dataset_inserts:
            params = call.args[1]
            # params: name, discipline, source, canonical_id, ...
            assert params[2] == "spdf"  # source

    def test_store_creates_dataset_instrument_links(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        inst = harvest_spdf.parse_instruments(
            MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        )
        ds = harvest_spdf.parse_datasets(MOCK_DATASETS_RESPONSE["DatasetDescription"])

        harvest_spdf.store_harvest(conn, observatories=[], instruments=inst, datasets=ds)

        all_calls = cursor.execute.call_args_list
        de_inserts = [call for call in all_calls if "INSERT INTO dataset_entities" in call.args[0]]
        # AC_H0_MFI and THA_L2_FGM -> "Magnetic Fields (Satellite)" -> ACE MAG, THEMIS-A FGM
        # WI_H0_SWE -> "Plasma and Solar Wind" -> Wind SWE
        assert len(de_inserts) > 0

        # All should have relationship='from_instrument'
        for call in de_inserts:
            params = call.args[1]
            assert params[2] == "from_instrument"

    def test_store_creates_instrument_observatory_relationships(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )
        inst = harvest_spdf.parse_instruments(
            MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        )
        ds = harvest_spdf.parse_datasets(MOCK_DATASETS_RESPONSE["DatasetDescription"])

        harvest_spdf.store_harvest(conn, observatories=obs, instruments=inst, datasets=ds)

        all_calls = cursor.execute.call_args_list
        rel_inserts = [
            call for call in all_calls if "INSERT INTO entity_relationships" in call.args[0]
        ]
        assert len(rel_inserts) > 0

        # All should have predicate='at_observatory'
        for call in rel_inserts:
            params = call.args[1]
            assert params[1] == "at_observatory"

    def test_store_completes_harvest_run(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )
        inst = harvest_spdf.parse_instruments(
            MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        )
        ds = harvest_spdf.parse_datasets(MOCK_DATASETS_RESPONSE["DatasetDescription"])

        counts = harvest_spdf.store_harvest(conn, observatories=obs, instruments=inst, datasets=ds)

        # Verify harvest run was completed
        all_calls = cursor.execute.call_args_list
        update_calls = [
            call
            for call in all_calls
            if "UPDATE harvest_runs" in call.args[0] and "completed" in call.args[0]
        ]
        assert len(update_calls) == 1

        # Check counts
        assert counts["observatories"] == 4
        assert counts["instruments"] == 3
        assert counts["datasets"] == 3

    def test_store_marks_failed_on_error(self) -> None:
        conn = _make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value

        # Make entity insert raise an error after harvest_run creation
        call_count = {"value": 0}
        _id_counter = {"value": 0}

        def failing_execute(sql: str, params: Any = None) -> None:
            if "INSERT INTO entities" in sql:
                raise RuntimeError("DB write failed")

        def fetchone_fn() -> tuple[int]:
            _id_counter["value"] += 1
            return (_id_counter["value"],)

        cursor.execute.side_effect = failing_execute
        cursor.fetchone.side_effect = fetchone_fn

        with pytest.raises(RuntimeError, match="DB write failed"):
            harvest_spdf.store_harvest(
                conn, observatories=[{"name": "X", "group": "G"}], instruments=[], datasets=[]
            )

        # Verify harvest run was marked as failed
        fail_calls = [
            call
            for call in cursor.execute.call_args_list
            if isinstance(call.args[0], str)
            and "UPDATE harvest_runs" in call.args[0]
            and "failed" in call.args[0]
        ]
        assert len(fail_calls) == 1


# ---------------------------------------------------------------------------
# Test: entity_dictionary backward compat
# ---------------------------------------------------------------------------


class TestDictionaryCompat:
    """Verify backward-compat entity_dictionary entries."""

    def test_build_dictionary_entries_observatories(self) -> None:
        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )
        entries = harvest_spdf._build_dictionary_entries(obs, [])

        assert len(entries) == 4
        for entry in entries:
            assert entry["entity_type"] == "instrument"  # compat mapping
            assert entry["source"] == "spdf"
            assert entry["metadata"]["spdf_type"] == "observatory"

    def test_build_dictionary_entries_instruments(self) -> None:
        inst = harvest_spdf.parse_instruments(
            MOCK_INSTRUMENTS_RESPONSE["InstrumentTypeDescription"]
        )
        entries = harvest_spdf._build_dictionary_entries([], inst)

        assert len(entries) == 3
        for entry in entries:
            assert entry["entity_type"] == "instrument"
            assert entry["source"] == "spdf"
            assert entry["metadata"]["spdf_type"] == "instrument"

    def test_build_dictionary_entries_has_spase_external_id(self) -> None:
        obs = harvest_spdf.parse_observatories(
            MOCK_OBSERVATORIES_RESPONSE["ObservatoryGroupDescription"]
        )
        entries = harvest_spdf._build_dictionary_entries(obs, [])

        ace = next(e for e in entries if e["canonical_name"] == "ACE")
        assert ace["external_id"] == "spase://SMWG/Observatory/ACE"


# ---------------------------------------------------------------------------
# Test: CLI / dry-run
# ---------------------------------------------------------------------------


class TestCLI:
    """Verify CLI behavior."""

    @patch.object(harvest_spdf.ResilientClient, "get", side_effect=_mock_get)
    def test_dry_run_does_not_write_db(self, mock_get: MagicMock) -> None:
        counts = harvest_spdf.run_harvest(dry_run=True)

        assert counts["observatories"] == 4
        assert counts["instruments"] == 3
        assert counts["datasets"] == 3

    @patch.object(harvest_spdf.ResilientClient, "get", side_effect=_mock_get)
    def test_main_with_dry_run(self, mock_get: MagicMock, capsys: Any) -> None:
        rc = harvest_spdf.main(["--dry-run"])
        assert rc == 0

        captured = capsys.readouterr()
        assert "Dry run" in captured.out

    def test_main_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            harvest_spdf.main(["--help"])
        assert exc_info.value.code == 0
