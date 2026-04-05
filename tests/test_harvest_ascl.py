"""Tests for the ASCL software catalog harvester.

Unit tests use mocked HTTP responses for deterministic testing.
Integration tests (marked @pytest.mark.integration) require a running scix
database with migration 013 applied.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_ascl import download_ascl_catalog, parse_ascl_entries, run_harvest

# ---------------------------------------------------------------------------
# Sample ASCL data
# ---------------------------------------------------------------------------

SAMPLE_ASCL_ENTRIES: list[dict[str, Any]] = [
    {
        "title": "Astropy",
        "ascl_id": "1304.002",
        "bibcode": "2013A&A...558A..33A",
        "credit": "Astropy Collaboration",
    },
    {
        "title": "NumPy",
        "ascl_id": "1309.006",
        "bibcode": "2020Natur.585..357H",
        "credit": "Harris, C. R. et al.",
    },
    {
        "title": "SciPy",
        "ascl_id": "1309.007",
        "bibcode": "2020NatMe..17..261V",
        "credit": "Virtanen, P. et al.",
    },
    {
        "title": "TOPCAT",
        "ascl_id": "1101.010",
        "bibcode": "2005ASPC..347...29T",
        "credit": "Taylor, M. B.",
    },
    {
        "title": "emcee",
        "ascl_id": "1303.002",
        "bibcode": "2013PASP..125..306F",
        "credit": "Foreman-Mackey, D. et al.",
    },
]


def _make_urlopen_response(entries: list[dict[str, Any]]) -> MagicMock:
    """Create a mock urllib response that returns JSON-encoded entries."""
    data = json.dumps(entries).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Unit tests: parse_ascl_entries
# ---------------------------------------------------------------------------


class TestParseAsclEntries:
    """Unit tests for parse_ascl_entries."""

    def test_parses_all_valid_entries(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        assert len(entries) == 5

    def test_entry_has_required_keys(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        for entry in entries:
            assert "canonical_name" in entry
            assert "entity_type" in entry
            assert "source" in entry
            assert "external_id" in entry
            assert "aliases" in entry
            assert "metadata" in entry

    def test_entity_type_is_software(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        for entry in entries:
            assert entry["entity_type"] == "software"

    def test_source_is_ascl(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        for entry in entries:
            assert entry["source"] == "ascl"

    def test_external_id_is_ascl_id(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        ids = {e["external_id"] for e in entries}
        assert "1304.002" in ids
        assert "1309.006" in ids

    def test_canonical_name_from_title(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        names = {e["canonical_name"] for e in entries}
        assert "Astropy" in names
        assert "NumPy" in names
        assert "emcee" in names

    def test_bibcode_in_metadata(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        astropy = next(e for e in entries if e["canonical_name"] == "Astropy")
        assert "bibcode" in astropy["metadata"]
        assert astropy["metadata"]["bibcode"] == "2013A&A...558A..33A"

    def test_credit_in_metadata(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        numpy_entry = next(e for e in entries if e["canonical_name"] == "NumPy")
        assert "credit" in numpy_entry["metadata"]
        assert numpy_entry["metadata"]["credit"] == "Harris, C. R. et al."

    def test_aliases_include_lowercase(self) -> None:
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        astropy = next(e for e in entries if e["canonical_name"] == "Astropy")
        assert "astropy" in astropy["aliases"]

    def test_no_duplicate_lowercase_alias(self) -> None:
        """When title is already lowercase, aliases should not duplicate it."""
        entries = parse_ascl_entries(SAMPLE_ASCL_ENTRIES)
        emcee = next(e for e in entries if e["canonical_name"] == "emcee")
        # "emcee" is already lowercase, so no alias needed
        assert "emcee" not in emcee["aliases"]

    def test_skips_entries_without_title(self) -> None:
        raw = [
            {"ascl_id": "9999.001", "bibcode": "2025test"},
            {"title": "", "ascl_id": "9999.002"},
            {"title": "Valid", "ascl_id": "9999.003", "bibcode": "2025valid"},
        ]
        entries = parse_ascl_entries(raw)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "Valid"

    def test_skips_entries_without_ascl_id(self) -> None:
        raw = [
            {"title": "NoID"},
            {"title": "HasID", "ascl_id": "9999.001", "bibcode": "2025x"},
        ]
        entries = parse_ascl_entries(raw)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "HasID"

    def test_entry_without_bibcode_has_empty_metadata_bibcode(self) -> None:
        raw = [{"title": "NoBib", "ascl_id": "9999.010"}]
        entries = parse_ascl_entries(raw)
        assert len(entries) == 1
        assert "bibcode" not in entries[0]["metadata"]

    def test_empty_input_returns_empty(self) -> None:
        entries = parse_ascl_entries([])
        assert entries == []


# ---------------------------------------------------------------------------
# Unit tests: download_ascl_catalog
# ---------------------------------------------------------------------------


class TestDownloadAsclCatalog:
    """Unit tests for download_ascl_catalog with mocked HTTP."""

    @patch("harvest_ascl.urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response(SAMPLE_ASCL_ENTRIES)
        result = download_ascl_catalog()
        assert len(result) == 5
        assert result[0]["title"] == "Astropy"

    @patch("harvest_ascl.urllib.request.urlopen")
    def test_retries_on_failure(self, mock_urlopen: MagicMock) -> None:
        """First call fails, second succeeds."""
        mock_urlopen.side_effect = [
            urllib.error.URLError("temporary failure"),
            _make_urlopen_response(SAMPLE_ASCL_ENTRIES),
        ]
        with patch("harvest_ascl.time.sleep"):
            result = download_ascl_catalog()
        assert len(result) == 5

    @patch("harvest_ascl.urllib.request.urlopen")
    def test_raises_after_max_retries(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("persistent failure")
        with patch("harvest_ascl.time.sleep"):
            with pytest.raises(urllib.error.URLError):
                download_ascl_catalog()


# Need to import urllib.error for the test above
import urllib.error  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests: run_harvest (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestRunHarvest:
    """Unit tests for run_harvest with mocked download and DB."""

    @patch("harvest_ascl.get_connection")
    @patch("harvest_ascl.bulk_load")
    @patch("harvest_ascl.download_ascl_catalog")
    def test_run_harvest_calls_bulk_load(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_ASCL_ENTRIES
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5

        count = run_harvest(dsn="dbname=test")

        assert count == 5
        mock_download.assert_called_once()
        mock_bulk_load.assert_called_once()
        # Verify entries passed to bulk_load
        loaded_entries = mock_bulk_load.call_args[0][1]
        assert len(loaded_entries) == 5
        assert all(e["entity_type"] == "software" for e in loaded_entries)
        assert all(e["source"] == "ascl" for e in loaded_entries)

    @patch("harvest_ascl.get_connection")
    @patch("harvest_ascl.bulk_load")
    @patch("harvest_ascl.download_ascl_catalog")
    def test_run_harvest_closes_connection(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_ASCL_ENTRIES
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5

        run_harvest()

        mock_conn.close.assert_called_once()

    @patch("harvest_ascl.get_connection")
    @patch("harvest_ascl.bulk_load")
    @patch("harvest_ascl.download_ascl_catalog")
    def test_run_harvest_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_ASCL_ENTRIES
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_harvest()

        mock_conn.close.assert_called_once()

    @patch("harvest_ascl.get_connection")
    @patch("harvest_ascl.bulk_load")
    @patch("harvest_ascl.download_ascl_catalog")
    def test_astropy_entry_has_ascl_id_and_bibcode(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Verify that an astropy-like entry has the correct structure."""
        mock_download.return_value = SAMPLE_ASCL_ENTRIES
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5

        run_harvest()

        loaded_entries = mock_bulk_load.call_args[0][1]
        astropy = next(e for e in loaded_entries if e["canonical_name"] == "Astropy")
        assert astropy["external_id"] == "1304.002"
        assert astropy["metadata"]["bibcode"] == "2013A&A...558A..33A"
        assert "astropy" in astropy["aliases"]


# ---------------------------------------------------------------------------
# Large catalog simulation
# ---------------------------------------------------------------------------


class TestLargeCatalog:
    """Test that parsing handles a catalog with >3500 entries."""

    def test_parses_large_catalog(self) -> None:
        """Simulate a catalog with 4000 entries."""
        raw = [
            {
                "title": f"Software_{i}",
                "ascl_id": f"{1000 + i // 1000}.{i % 1000:03d}",
                "bibcode": f"2025test{i:04d}",
            }
            for i in range(4000)
        ]
        entries = parse_ascl_entries(raw)
        assert len(entries) == 4000
        assert len(entries) > 3500
