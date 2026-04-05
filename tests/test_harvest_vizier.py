"""Tests for the VizieR catalog harvester.

Unit tests use mocked HTTP responses with sample VOTable XML for
deterministic testing. Integration tests (marked @pytest.mark.integration)
require a running scix database with migration 013 applied.
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_vizier import (
    build_dictionary_entries,
    parse_votable_catalogs,
    query_tap_vizier,
    run_harvest,
)

# ---------------------------------------------------------------------------
# Sample VOTable XML
# ---------------------------------------------------------------------------

SAMPLE_VOTABLE_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
  <RESOURCE type="results">
    <TABLE>
      <FIELD name="table_name" datatype="char" arraysize="*"/>
      <FIELD name="description" datatype="char" arraysize="*"/>
      <FIELD name="utype" datatype="char" arraysize="*"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>J/A+A/680/A81</TD>
            <TD>SDSS-V Milky Way Mapper targets</TD>
            <TD></TD>
          </TR>
          <TR>
            <TD>J/ApJ/923/67</TD>
            <TD>Proper motions of stars in Sculptor dSph</TD>
            <TD>catalog:main</TD>
          </TR>
          <TR>
            <TD>II/246</TD>
            <TD>2MASS All-Sky Catalog of Point Sources</TD>
            <TD>catalog:main</TD>
          </TR>
          <TR>
            <TD>J/MNRAS/504/3029</TD>
            <TD>Galaxy morphologies from DECaLS</TD>
            <TD></TD>
          </TR>
          <TR>
            <TD>VII/275</TD>
            <TD>SDSS Photometric Catalogue, Release 12</TD>
            <TD>catalog:main</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""

SAMPLE_VOTABLE_EMPTY = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
  <RESOURCE type="results">
    <TABLE>
      <FIELD name="table_name" datatype="char" arraysize="*"/>
      <FIELD name="description" datatype="char" arraysize="*"/>
      <FIELD name="utype" datatype="char" arraysize="*"/>
      <DATA>
        <TABLEDATA>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""

SAMPLE_VOTABLE_MISSING_DESC = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
  <RESOURCE type="results">
    <TABLE>
      <FIELD name="table_name" datatype="char" arraysize="*"/>
      <FIELD name="description" datatype="char" arraysize="*"/>
      <FIELD name="utype" datatype="char" arraysize="*"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>J/A+A/999/A1</TD>
            <TD></TD>
            <TD></TD>
          </TR>
          <TR>
            <TD>II/999</TD>
            <TD>A catalog with a description</TD>
            <TD>catalog:main</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""


def _make_urlopen_response(data: bytes) -> MagicMock:
    """Create a mock urllib response that returns the given bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Unit tests: parse_votable_catalogs
# ---------------------------------------------------------------------------


class TestParseVotableCatalogs:
    """Unit tests for VOTable XML parsing."""

    def test_parses_all_rows(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_XML)
        assert len(catalogs) == 5

    def test_extracts_table_name(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_XML)
        names = {c["table_name"] for c in catalogs}
        assert "J/A+A/680/A81" in names
        assert "II/246" in names
        assert "VII/275" in names

    def test_extracts_description(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_XML)
        descs = {c["description"] for c in catalogs}
        assert "2MASS All-Sky Catalog of Point Sources" in descs
        assert "SDSS-V Milky Way Mapper targets" in descs

    def test_extracts_utype(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_XML)
        two_mass = next(c for c in catalogs if c["table_name"] == "II/246")
        assert two_mass["utype"] == "catalog:main"

    def test_empty_utype(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_XML)
        sdss_v = next(c for c in catalogs if c["table_name"] == "J/A+A/680/A81")
        assert sdss_v["utype"] == ""

    def test_empty_tabledata_returns_empty(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_EMPTY)
        assert catalogs == []

    def test_missing_description_returns_empty_string(self) -> None:
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_MISSING_DESC)
        no_desc = next(c for c in catalogs if c["table_name"] == "J/A+A/999/A1")
        assert no_desc["description"] == ""


# ---------------------------------------------------------------------------
# Unit tests: build_dictionary_entries
# ---------------------------------------------------------------------------


class TestBuildDictionaryEntries:
    """Unit tests for converting catalog dicts to entity_dictionary records."""

    def _sample_catalogs(self) -> list[dict[str, str]]:
        return parse_votable_catalogs(SAMPLE_VOTABLE_XML)

    def test_builds_correct_count(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        assert len(entries) == 5

    def test_entity_type_is_dataset(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        for entry in entries:
            assert entry["entity_type"] == "dataset"

    def test_source_is_vizier(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        for entry in entries:
            assert entry["source"] == "vizier"

    def test_external_id_is_table_name(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        ids = {e["external_id"] for e in entries}
        assert "J/A+A/680/A81" in ids
        assert "II/246" in ids

    def test_canonical_name_from_description(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        two_mass = next(e for e in entries if e["external_id"] == "II/246")
        assert two_mass["canonical_name"] == "2MASS All-Sky Catalog of Point Sources"

    def test_canonical_name_fallback_to_table_name(self) -> None:
        """When description is empty, use table_name as canonical_name."""
        catalogs = parse_votable_catalogs(SAMPLE_VOTABLE_MISSING_DESC)
        entries = build_dictionary_entries(catalogs)
        no_desc = next(e for e in entries if e["external_id"] == "J/A+A/999/A1")
        assert no_desc["canonical_name"] == "J/A+A/999/A1"

    def test_utype_in_metadata(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        two_mass = next(e for e in entries if e["external_id"] == "II/246")
        assert two_mass["metadata"]["utype"] == "catalog:main"

    def test_empty_utype_not_in_metadata(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        sdss_v = next(e for e in entries if e["external_id"] == "J/A+A/680/A81")
        assert "utype" not in sdss_v["metadata"]

    def test_aliases_empty(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        for entry in entries:
            assert entry["aliases"] == []

    def test_required_keys_present(self) -> None:
        entries = build_dictionary_entries(self._sample_catalogs())
        for entry in entries:
            assert "canonical_name" in entry
            assert "entity_type" in entry
            assert "source" in entry
            assert "external_id" in entry
            assert "aliases" in entry
            assert "metadata" in entry

    def test_skips_entries_without_table_name(self) -> None:
        catalogs = [
            {"table_name": "", "description": "No name", "utype": ""},
            {"table_name": "II/1", "description": "Has name", "utype": ""},
        ]
        entries = build_dictionary_entries(catalogs)
        assert len(entries) == 1
        assert entries[0]["external_id"] == "II/1"

    def test_empty_input_returns_empty(self) -> None:
        entries = build_dictionary_entries([])
        assert entries == []


# ---------------------------------------------------------------------------
# Unit tests: query_tap_vizier
# ---------------------------------------------------------------------------


class TestQueryTapVizier:
    """Unit tests for TAP query with mocked HTTP."""

    @patch("harvest_vizier.urllib.request.urlopen")
    def test_returns_response_bytes(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response(SAMPLE_VOTABLE_XML)
        result = query_tap_vizier()
        assert result == SAMPLE_VOTABLE_XML

    @patch("harvest_vizier.urllib.request.urlopen")
    def test_retries_on_failure(self, mock_urlopen: MagicMock) -> None:
        """First call fails, second succeeds."""
        mock_urlopen.side_effect = [
            urllib.error.URLError("temporary failure"),
            _make_urlopen_response(SAMPLE_VOTABLE_XML),
        ]
        with patch("harvest_vizier.time.sleep"):
            result = query_tap_vizier()
        assert result == SAMPLE_VOTABLE_XML

    @patch("harvest_vizier.urllib.request.urlopen")
    def test_raises_after_max_retries(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("persistent failure")
        with patch("harvest_vizier.time.sleep"):
            with pytest.raises(urllib.error.URLError):
                query_tap_vizier()

    @patch("harvest_vizier.urllib.request.urlopen")
    def test_sends_post_request(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response(SAMPLE_VOTABLE_XML)
        query_tap_vizier()
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert request_obj.get_method() == "POST"
        assert request_obj.data is not None


# ---------------------------------------------------------------------------
# Unit tests: run_harvest (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestRunHarvest:
    """Unit tests for run_harvest with mocked download and DB."""

    @patch("harvest_vizier.get_connection")
    @patch("harvest_vizier.bulk_load")
    @patch("harvest_vizier.query_tap_vizier")
    def test_run_harvest_calls_bulk_load(
        self,
        mock_query: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_query.return_value = SAMPLE_VOTABLE_XML
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5

        count = run_harvest(dsn="dbname=test")

        assert count == 5
        mock_query.assert_called_once()
        mock_bulk_load.assert_called_once()
        loaded_entries = mock_bulk_load.call_args[0][1]
        assert len(loaded_entries) == 5
        assert all(e["entity_type"] == "dataset" for e in loaded_entries)
        assert all(e["source"] == "vizier" for e in loaded_entries)

    @patch("harvest_vizier.get_connection")
    @patch("harvest_vizier.bulk_load")
    @patch("harvest_vizier.query_tap_vizier")
    def test_run_harvest_closes_connection(
        self,
        mock_query: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_query.return_value = SAMPLE_VOTABLE_XML
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5

        run_harvest()

        mock_conn.close.assert_called_once()

    @patch("harvest_vizier.get_connection")
    @patch("harvest_vizier.bulk_load")
    @patch("harvest_vizier.query_tap_vizier")
    def test_run_harvest_closes_connection_on_error(
        self,
        mock_query: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_query.return_value = SAMPLE_VOTABLE_XML
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_harvest()

        mock_conn.close.assert_called_once()

    @patch("harvest_vizier.get_connection")
    @patch("harvest_vizier.bulk_load")
    @patch("harvest_vizier.query_tap_vizier")
    def test_two_mass_entry_structure(
        self,
        mock_query: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Verify that a 2MASS-like entry has the correct structure."""
        mock_query.return_value = SAMPLE_VOTABLE_XML
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5

        run_harvest()

        loaded_entries = mock_bulk_load.call_args[0][1]
        two_mass = next(e for e in loaded_entries if e["external_id"] == "II/246")
        assert two_mass["canonical_name"] == "2MASS All-Sky Catalog of Point Sources"
        assert two_mass["entity_type"] == "dataset"
        assert two_mass["source"] == "vizier"
        assert two_mass["metadata"]["utype"] == "catalog:main"


# ---------------------------------------------------------------------------
# Large catalog simulation
# ---------------------------------------------------------------------------


class TestLargeCatalog:
    """Test that parsing and building handles >25000 entries."""

    def _build_large_votable(self, n: int) -> bytes:
        """Build a VOTable XML with n rows."""
        header = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
            '  <RESOURCE type="results">\n'
            "    <TABLE>\n"
            '      <FIELD name="table_name" datatype="char" arraysize="*"/>\n'
            '      <FIELD name="description" datatype="char" arraysize="*"/>\n'
            '      <FIELD name="utype" datatype="char" arraysize="*"/>\n'
            "      <DATA>\n"
            "        <TABLEDATA>\n"
        )
        rows = ""
        for i in range(n):
            rows += (
                f"          <TR>"
                f"<TD>J/A+A/{i // 100 + 1}/A{i % 100 + 1}</TD>"
                f"<TD>Catalog number {i}</TD>"
                f"<TD></TD>"
                f"</TR>\n"
            )
        footer = (
            "        </TABLEDATA>\n"
            "      </DATA>\n"
            "    </TABLE>\n"
            "  </RESOURCE>\n"
            "</VOTABLE>"
        )
        return (header + rows + footer).encode("utf-8")

    def test_parses_large_catalog(self) -> None:
        """Simulate a catalog with 30000 entries."""
        xml_bytes = self._build_large_votable(30000)
        catalogs = parse_votable_catalogs(xml_bytes)
        assert len(catalogs) == 30000
        assert len(catalogs) > 25000

    def test_builds_large_dictionary(self) -> None:
        """Build dictionary entries from 30000 catalogs."""
        xml_bytes = self._build_large_votable(30000)
        catalogs = parse_votable_catalogs(xml_bytes)
        entries = build_dictionary_entries(catalogs)
        assert len(entries) == 30000
        assert len(entries) > 25000
        assert all(e["entity_type"] == "dataset" for e in entries)
        assert all(e["source"] == "vizier" for e in entries)
