"""Tests for the PDS4 context products dictionary harvester.

Unit tests use synthetic PDS API response fixtures.
No network or database access required.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_pds4 import (
    ALL_PRODUCT_TYPES,
    PRODUCT_TYPE_MAP,
    download_pds4_context,
    extract_aliases,
    fetch_pds4_page,
    parse_pds4_products,
    run_harvest,
)

# ---------------------------------------------------------------------------
# Synthetic PDS API response fixtures
# ---------------------------------------------------------------------------


def _make_product(
    lid: str,
    title: str,
    *,
    pds_type: str = "",
    description: str = "",
    alternate_title: str | None = None,
    alias_titles: list[str] | None = None,
) -> dict[str, Any]:
    """Build a synthetic PDS API product dict."""
    versioned_id = f"{lid}::1.0"
    properties: dict[str, Any] = {
        "pds:Identification_Area.pds:logical_identifier": [lid],
        "pds:Identification_Area.pds:title": [title],
    }
    if alternate_title is not None:
        properties["pds:Identification_Area.pds:alternate_title"] = [alternate_title]
    else:
        properties["pds:Identification_Area.pds:alternate_title"] = ["null"]

    if alias_titles:
        properties["pds:Alias_List.pds:Alias.pds:alternate_title"] = alias_titles

    # Add type/description based on URN segment
    if "investigation:" in lid:
        if pds_type:
            properties["pds:Investigation.pds:type"] = [pds_type]
        if description:
            properties["pds:Investigation.pds:description"] = [description]
    elif "instrument:" in lid:
        if pds_type:
            properties["pds:Instrument.pds:type"] = [pds_type]
        if description:
            properties["pds:Instrument.pds:description"] = [description]
    elif "target:" in lid:
        if pds_type:
            properties["pds:Target.pds:type"] = [pds_type]
        if description:
            properties["pds:Target.pds:description"] = [description]

    return {
        "id": versioned_id,
        "type": "Product_Context",
        "title": title,
        "properties": properties,
    }


SAMPLE_INVESTIGATIONS: list[dict[str, Any]] = [
    _make_product(
        "urn:nasa:pds:context:investigation:mission.cassini-huygens",
        "Cassini-Huygens",
        pds_type="Mission",
        description="The Cassini-Huygens mission to Saturn.",
    ),
    _make_product(
        "urn:nasa:pds:context:investigation:mission.mars-exploration-rover",
        "Mars Exploration Rover (MER)",
        pds_type="Mission",
        description="Twin rover mission to Mars.",
    ),
    _make_product(
        "urn:nasa:pds:context:investigation:field_campaign.dd_nnss-nv_2019",
        "Dust Devil Field Campaign, Nevada National Security Site (NNSS), Nevada, 2019",
        pds_type="Field Campaign",
        description="A dust devil field campaign.",
        alternate_title="DD NNSS 2019",
    ),
]

SAMPLE_INSTRUMENTS: list[dict[str, Any]] = [
    _make_product(
        "urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.cirs",
        "Composite Infrared Spectrometer (CIRS)",
        pds_type="Spectrometer",
        description="Far and mid-infrared spectrometer.",
    ),
    _make_product(
        "urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.iss-nac",
        "Imaging Science Subsystem - Narrow Angle Camera (ISS-NAC)",
        pds_type="Imager",
    ),
]

SAMPLE_TARGETS: list[dict[str, Any]] = [
    _make_product(
        "urn:nasa:pds:context:target:planet.mars",
        "Mars",
        pds_type="Planet",
        description="The fourth planet from the Sun.",
    ),
    _make_product(
        "urn:nasa:pds:context:target:satellite.titan",
        "Titan",
        pds_type="Satellite",
        description="Saturn's largest moon.",
        alias_titles=["Saturn VI", "S VI Titan"],
    ),
    _make_product(
        "urn:nasa:pds:context:target:asteroid.433_eros",
        "433 Eros",
        pds_type="Asteroid",
    ),
]


def _make_api_response(
    products: list[dict[str, Any]],
    total_hits: int | None = None,
) -> dict[str, Any]:
    """Build a synthetic PDS API JSON response."""
    return {
        "summary": {
            "hits": total_hits if total_hits is not None else len(products),
            "properties": [],
            "facets": [],
        },
        "data": products,
    }


# ---------------------------------------------------------------------------
# Tests: extract_aliases
# ---------------------------------------------------------------------------


class TestExtractAliases:
    """Tests for alias extraction from titles and properties."""

    def test_abbreviation_from_title_parens(self) -> None:
        props: dict[str, Any] = {"pds:Identification_Area.pds:alternate_title": ["null"]}
        aliases = extract_aliases("Mars Exploration Rover (MER)", props)
        assert "MER" in aliases

    def test_multiple_abbreviations_from_title(self) -> None:
        props: dict[str, Any] = {}
        aliases = extract_aliases(
            "Imaging Science Subsystem - Narrow Angle Camera (ISS-NAC)", props
        )
        assert "ISS-NAC" in aliases

    def test_alternate_title_from_properties(self) -> None:
        props: dict[str, Any] = {
            "pds:Identification_Area.pds:alternate_title": ["DD NNSS 2019"],
        }
        aliases = extract_aliases("Dust Devil Campaign", props)
        assert "DD NNSS 2019" in aliases

    def test_alias_list_entries(self) -> None:
        props: dict[str, Any] = {
            "pds:Alias_List.pds:Alias.pds:alternate_title": ["Saturn VI", "S VI Titan"],
        }
        aliases = extract_aliases("Titan", props)
        assert "Saturn VI" in aliases
        assert "S VI Titan" in aliases

    def test_null_alternate_title_ignored(self) -> None:
        props: dict[str, Any] = {
            "pds:Identification_Area.pds:alternate_title": ["null"],
        }
        aliases = extract_aliases("Mars", props)
        assert aliases == []

    def test_deduplication(self) -> None:
        props: dict[str, Any] = {
            "pds:Identification_Area.pds:alternate_title": ["MER"],
            "pds:Alias_List.pds:Alias.pds:alternate_title": ["MER", "mer"],
        }
        aliases = extract_aliases("Mars Exploration Rover (MER)", props)
        assert aliases.count("MER") == 1

    def test_title_not_in_aliases(self) -> None:
        props: dict[str, Any] = {
            "pds:Identification_Area.pds:alternate_title": ["Mars"],
        }
        aliases = extract_aliases("Mars", props)
        assert "Mars" not in aliases

    def test_empty_properties(self) -> None:
        aliases = extract_aliases("Simple Name", {})
        assert aliases == []

    def test_lowercase_not_matched_as_abbreviation(self) -> None:
        """Parenthesised lowercase words should not be extracted as abbreviations."""
        props: dict[str, Any] = {}
        aliases = extract_aliases("Something (with lowercase)", props)
        assert aliases == []


# ---------------------------------------------------------------------------
# Tests: parse_pds4_products — Investigation
# ---------------------------------------------------------------------------


class TestParseInvestigations:
    """Tests for parsing Investigation context products."""

    def test_returns_correct_count(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        assert len(entries) == 3

    def test_entity_type_is_mission(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        for entry in entries:
            assert entry["entity_type"] == "mission"

    def test_source_is_pds4(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        for entry in entries:
            assert entry["source"] == "pds4"

    def test_external_id_is_urn(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        for entry in entries:
            assert entry["external_id"].startswith("urn:nasa:pds:context:investigation:")

    def test_external_id_has_no_version(self) -> None:
        """logical_identifier (no version suffix) used as external_id."""
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        for entry in entries:
            assert "::" not in entry["external_id"]

    def test_canonical_name_from_title(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        names = {e["canonical_name"] for e in entries}
        assert "Cassini-Huygens" in names

    def test_abbreviation_extracted_as_alias(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        mer = next(e for e in entries if "Mars Exploration" in e["canonical_name"])
        assert "MER" in mer["aliases"]

    def test_alternate_title_in_aliases(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        dd = next(e for e in entries if "Dust Devil" in e["canonical_name"])
        assert "DD NNSS 2019" in dd["aliases"]

    def test_description_in_metadata(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        cassini = next(e for e in entries if e["canonical_name"] == "Cassini-Huygens")
        assert "Saturn" in cassini["metadata"]["description"]

    def test_pds_type_in_metadata(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        cassini = next(e for e in entries if e["canonical_name"] == "Cassini-Huygens")
        assert cassini["metadata"]["pds_type"] == "Mission"

    def test_versioned_id_in_metadata(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        cassini = next(e for e in entries if e["canonical_name"] == "Cassini-Huygens")
        assert "::1.0" in cassini["metadata"]["pds_versioned_id"]

    def test_skips_product_missing_title(self) -> None:
        bad = [{"id": "urn:nasa:pds:context:investigation:x::1.0", "title": "", "properties": {}}]
        entries = parse_pds4_products(bad, "investigation")
        assert len(entries) == 0

    def test_skips_product_missing_lid(self) -> None:
        bad = [
            {
                "id": "urn:nasa:pds:context:investigation:x::1.0",
                "title": "Test",
                "properties": {},
            }
        ]
        entries = parse_pds4_products(bad, "investigation")
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# Tests: parse_pds4_products — Instrument
# ---------------------------------------------------------------------------


class TestParseInstruments:
    """Tests for parsing Instrument context products."""

    def test_returns_correct_count(self) -> None:
        entries = parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument")
        assert len(entries) == 2

    def test_entity_type_is_instrument(self) -> None:
        entries = parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument")
        for entry in entries:
            assert entry["entity_type"] == "instrument"

    def test_source_is_pds4(self) -> None:
        entries = parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument")
        for entry in entries:
            assert entry["source"] == "pds4"

    def test_external_id_is_instrument_urn(self) -> None:
        entries = parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument")
        for entry in entries:
            assert entry["external_id"].startswith("urn:nasa:pds:context:instrument:")

    def test_abbreviation_extracted(self) -> None:
        entries = parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument")
        cirs = next(e for e in entries if "Composite Infrared" in e["canonical_name"])
        assert "CIRS" in cirs["aliases"]


# ---------------------------------------------------------------------------
# Tests: parse_pds4_products — Target
# ---------------------------------------------------------------------------


class TestParseTargets:
    """Tests for parsing Target context products."""

    def test_returns_correct_count(self) -> None:
        entries = parse_pds4_products(SAMPLE_TARGETS, "target")
        assert len(entries) == 3

    def test_entity_type_is_target(self) -> None:
        entries = parse_pds4_products(SAMPLE_TARGETS, "target")
        for entry in entries:
            assert entry["entity_type"] == "target"

    def test_source_is_pds4(self) -> None:
        entries = parse_pds4_products(SAMPLE_TARGETS, "target")
        for entry in entries:
            assert entry["source"] == "pds4"

    def test_external_id_is_target_urn(self) -> None:
        entries = parse_pds4_products(SAMPLE_TARGETS, "target")
        for entry in entries:
            assert entry["external_id"].startswith("urn:nasa:pds:context:target:")

    def test_alias_list_in_aliases(self) -> None:
        entries = parse_pds4_products(SAMPLE_TARGETS, "target")
        titan = next(e for e in entries if e["canonical_name"] == "Titan")
        assert "Saturn VI" in titan["aliases"]
        assert "S VI Titan" in titan["aliases"]

    def test_description_none_filtered(self) -> None:
        """Products with description='none' should not have it in metadata."""
        product = _make_product(
            "urn:nasa:pds:context:target:asteroid.test",
            "Test Asteroid",
            pds_type="Asteroid",
            description="none",
        )
        entries = parse_pds4_products([product], "target")
        assert "description" not in entries[0]["metadata"]


# ---------------------------------------------------------------------------
# Tests: fetch_pds4_page
# ---------------------------------------------------------------------------


class TestFetchPds4Page:
    """Tests for fetch_pds4_page with mocked HTTP."""

    @patch("harvest_pds4.urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen: MagicMock) -> None:
        response_data = _make_api_response(SAMPLE_INVESTIGATIONS[:1])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = fetch_pds4_page("investigation", start=0, limit=10)
        assert result["summary"]["hits"] == 1
        assert len(result["data"]) == 1

    @patch("harvest_pds4.urllib.request.urlopen")
    def test_retries_on_failure(self, mock_urlopen: MagicMock) -> None:
        response_data = _make_api_response(SAMPLE_INVESTIGATIONS[:1])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("temporary failure"),
            mock_resp,
        ]
        with patch("harvest_pds4.time.sleep"):
            result = fetch_pds4_page("investigation")
        assert result["summary"]["hits"] == 1

    @patch("harvest_pds4.urllib.request.urlopen")
    def test_raises_after_max_retries(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("persistent failure")
        with patch("harvest_pds4.time.sleep"):
            with pytest.raises(urllib.error.URLError):
                fetch_pds4_page("investigation")


# ---------------------------------------------------------------------------
# Tests: download_pds4_context
# ---------------------------------------------------------------------------


class TestDownloadPds4Context:
    """Tests for download_pds4_context with mocked fetch."""

    @patch("harvest_pds4.fetch_pds4_page")
    def test_downloads_all_types(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = [
            _make_api_response(SAMPLE_INVESTIGATIONS),
            _make_api_response(SAMPLE_INSTRUMENTS),
            _make_api_response(SAMPLE_TARGETS),
        ]
        result = download_pds4_context(ALL_PRODUCT_TYPES)
        assert set(result.keys()) == {"investigation", "instrument", "target"}
        assert len(result["investigation"]) == 3
        assert len(result["instrument"]) == 2
        assert len(result["target"]) == 3

    @patch("harvest_pds4.fetch_pds4_page")
    def test_handles_pagination(self, mock_fetch: MagicMock) -> None:
        """Two pages of results should be concatenated."""
        page1 = _make_api_response(SAMPLE_INVESTIGATIONS[:2], total_hits=3)
        page2 = _make_api_response(SAMPLE_INVESTIGATIONS[2:], total_hits=3)
        mock_fetch.side_effect = [page1, page2]

        result = download_pds4_context(("investigation",))
        assert len(result["investigation"]) == 3

    @patch("harvest_pds4.fetch_pds4_page")
    def test_single_type_filter(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_api_response(SAMPLE_TARGETS)
        result = download_pds4_context(("target",))
        assert "target" in result
        assert "investigation" not in result

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown product type"):
            download_pds4_context(("bogus",))


# ---------------------------------------------------------------------------
# Tests: run_harvest
# ---------------------------------------------------------------------------


class TestRunHarvest:
    """Tests for run_harvest with mocked download and DB."""

    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_loads_all_types(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
            "instrument": SAMPLE_INSTRUMENTS,
            "target": SAMPLE_TARGETS,
        }
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8

        counts = run_harvest(dsn="dbname=test")

        assert counts["mission"] == 3
        assert counts["instrument"] == 2
        assert counts["target"] == 3
        mock_bulk_load.assert_called_once()

        # Verify discipline passed
        _, kwargs = mock_bulk_load.call_args
        assert kwargs["discipline"] == "planetary_science"

    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_dry_run_skips_db(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
        }

        counts = run_harvest(dry_run=True, product_types=("investigation",))

        assert counts["mission"] == 3
        mock_bulk_load.assert_not_called()
        mock_get_conn.assert_not_called()

    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_closes_connection(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 3

        run_harvest(product_types=("investigation",))

        mock_conn.close.assert_called_once()

    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_harvest(product_types=("investigation",))

        mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: CLI --help
# ---------------------------------------------------------------------------


class TestCLI:
    """Test that the CLI is well-formed."""

    def test_help_flag(self) -> None:
        """--help should exit 0 (argparse SystemExit)."""
        import harvest_pds4

        with pytest.raises(SystemExit) as exc_info:
            harvest_pds4.main.__wrapped__ if hasattr(harvest_pds4.main, "__wrapped__") else None
            # Simulate --help by patching sys.argv
            with patch("sys.argv", ["harvest_pds4.py", "--help"]):
                harvest_pds4.main()
        assert exc_info.value.code == 0
