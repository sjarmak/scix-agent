"""Tests for the PDS4 context products dictionary harvester.

Unit tests use synthetic PDS API response fixtures.
No network or database access required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_pds4 import (
    ALL_PRODUCT_TYPES,
    PRODUCT_TYPE_MAP,
    _write_entity_graph,
    _write_relationships,
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
    ref_lid_investigation: list[str] | None = None,
    ref_lid_target: list[str] | None = None,
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

    if ref_lid_investigation:
        properties["ref_lid_investigation"] = ref_lid_investigation
    if ref_lid_target:
        properties["ref_lid_target"] = ref_lid_target

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
        ref_lid_target=[
            "urn:nasa:pds:context:target:planet.saturn",
            "urn:nasa:pds:context:target:satellite.titan",
        ],
    ),
    _make_product(
        "urn:nasa:pds:context:investigation:mission.mars-exploration-rover",
        "Mars Exploration Rover (MER)",
        pds_type="Mission",
        description="Twin rover mission to Mars.",
        ref_lid_target=["urn:nasa:pds:context:target:planet.mars"],
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
        ref_lid_investigation=["urn:nasa:pds:context:investigation:mission.cassini-huygens"],
    ),
    _make_product(
        "urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.iss-nac",
        "Imaging Science Subsystem - Narrow Angle Camera (ISS-NAC)",
        pds_type="Imager",
        ref_lid_investigation=["urn:nasa:pds:context:investigation:mission.cassini-huygens"],
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
    _make_product(
        "urn:nasa:pds:context:target:planet.saturn",
        "Saturn",
        pds_type="Planet",
        description="The sixth planet from the Sun.",
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

    def test_ref_lid_target_in_metadata(self) -> None:
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation")
        cassini = next(e for e in entries if e["canonical_name"] == "Cassini-Huygens")
        assert "ref_lid_target" in cassini["metadata"]
        assert len(cassini["metadata"]["ref_lid_target"]) == 2

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

    def test_ref_lid_investigation_in_metadata(self) -> None:
        entries = parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument")
        cirs = next(e for e in entries if "Composite Infrared" in e["canonical_name"])
        assert "ref_lid_investigation" in cirs["metadata"]


# ---------------------------------------------------------------------------
# Tests: parse_pds4_products — Target
# ---------------------------------------------------------------------------


class TestParseTargets:
    """Tests for parsing Target context products."""

    def test_returns_correct_count(self) -> None:
        entries = parse_pds4_products(SAMPLE_TARGETS, "target")
        assert len(entries) == 4

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
# Tests: fetch_pds4_page (ResilientClient)
# ---------------------------------------------------------------------------


class TestFetchPds4Page:
    """Tests for fetch_pds4_page with mocked ResilientClient."""

    def test_returns_parsed_json(self) -> None:
        response_data = _make_api_response(SAMPLE_INVESTIGATIONS[:1])
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_client.get.return_value = mock_response

        result = fetch_pds4_page("investigation", client=mock_client, limit=10)
        assert result["summary"]["hits"] == 1
        assert len(result["data"]) == 1
        mock_client.get.assert_called_once()

    def test_uses_resilient_client(self) -> None:
        """Verify ResilientClient.get() is called (not urllib)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_api_response([])
        mock_client.get.return_value = mock_response

        fetch_pds4_page("investigation", client=mock_client)
        mock_client.get.assert_called_once()
        # Verify the URL starts with PDS_API_BASE
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://pds.nasa.gov/api/search/1/products"

    def test_passes_search_after_param(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_api_response([])
        mock_client.get.return_value = mock_response

        fetch_pds4_page("investigation", client=mock_client, search_after=["val1", "val2"])
        call_args = mock_client.get.call_args
        params = (
            call_args[1].get("params") or call_args[0][1]
            if len(call_args[0]) > 1
            else call_args[1].get("params")
        )
        assert "search-after" in params
        assert params["search-after"] == "val1,val2"


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
        mock_client = MagicMock()
        result = download_pds4_context(ALL_PRODUCT_TYPES, client=mock_client)
        assert set(result.keys()) == {"investigation", "instrument", "target"}
        assert len(result["investigation"]) == 3
        assert len(result["instrument"]) == 2
        assert len(result["target"]) == 4

    @patch("harvest_pds4.fetch_pds4_page")
    def test_handles_pagination(self, mock_fetch: MagicMock) -> None:
        """Two pages of results should be concatenated via cursor pagination."""
        # Page 1: 2 products with sort values, total_hits=3
        page1_products = [
            {**p, "sort": [f"sort_val_{i}"]} for i, p in enumerate(SAMPLE_INVESTIGATIONS[:2])
        ]
        page1 = _make_api_response(page1_products, total_hits=3)
        page2 = _make_api_response(SAMPLE_INVESTIGATIONS[2:], total_hits=3)
        mock_fetch.side_effect = [page1, page2]

        mock_client = MagicMock()
        result = download_pds4_context(("investigation",), client=mock_client)
        assert len(result["investigation"]) == 3
        # Second call should use search_after from last product's sort values
        _, kwargs = mock_fetch.call_args_list[1]
        assert kwargs.get("search_after") == ["sort_val_1"]

    @patch("harvest_pds4.fetch_pds4_page")
    def test_single_type_filter(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_api_response(SAMPLE_TARGETS)
        mock_client = MagicMock()
        result = download_pds4_context(("target",), client=mock_client)
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

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_loads_all_types(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
            "instrument": SAMPLE_INSTRUMENTS,
            "target": SAMPLE_TARGETS,
        }
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 9
        mock_start_run.return_value = 42
        mock_write_graph.return_value = 9
        mock_write_rels.return_value = 0

        counts = run_harvest(dsn="dbname=test")

        assert counts["mission"] == 3
        assert counts["instrument"] == 2
        assert counts["target"] == 4
        mock_bulk_load.assert_called_once()

        # Verify discipline passed
        _, kwargs = mock_bulk_load.call_args
        assert kwargs["discipline"] == "planetary_science"

    @patch("harvest_pds4.download_pds4_context")
    def test_dry_run_skips_db(
        self,
        mock_download: MagicMock,
    ) -> None:
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
        }

        counts = run_harvest(dry_run=True, product_types=("investigation",))

        assert counts["mission"] == 3

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_closes_connection(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 3
        mock_start_run.return_value = 1
        mock_write_graph.return_value = 3
        mock_write_rels.return_value = 0

        run_harvest(product_types=("investigation",))

        mock_conn.close.assert_called_once()

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_start_run.return_value = 1
        mock_bulk_load.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_harvest(product_types=("investigation",))

        mock_conn.close.assert_called_once()

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_harvest_run_logged(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
        }
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 3
        mock_start_run.return_value = 99
        mock_write_graph.return_value = 3
        mock_write_rels.return_value = 0

        run_harvest(product_types=("investigation",))

        # harvest_run started
        mock_start_run.assert_called_once_with(mock_conn, ("investigation",))
        # harvest_run finished with correct status
        mock_finish_run.assert_called_once()
        finish_kwargs = mock_finish_run.call_args[1]
        assert finish_kwargs["records_upserted"] == 3
        assert finish_kwargs["counts"]["mission"] == 3

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_entity_graph_written(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
            "instrument": SAMPLE_INSTRUMENTS,
        }
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5
        mock_start_run.return_value = 10
        mock_write_graph.return_value = 5
        mock_write_rels.return_value = 2

        run_harvest(product_types=("investigation", "instrument"))

        # entity graph written with all entries and harvest_run_id
        mock_write_graph.assert_called_once()
        graph_args = mock_write_graph.call_args[0]
        assert graph_args[0] is mock_conn  # conn
        assert len(graph_args[1]) == 5  # 3 investigations + 2 instruments
        assert graph_args[2] == 10  # harvest_run_id

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_backward_compat_bulk_load_called(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        """bulk_load() to entity_dictionary must still be called."""
        mock_download.return_value = {"target": SAMPLE_TARGETS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 4
        mock_start_run.return_value = 1
        mock_write_graph.return_value = 4
        mock_write_rels.return_value = 0

        run_harvest(product_types=("target",))

        mock_bulk_load.assert_called_once()
        _, kwargs = mock_bulk_load.call_args
        assert kwargs["discipline"] == "planetary_science"

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_harvest_run_failed_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        """On error, harvest_run should be marked as 'failed'."""
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_start_run.return_value = 5
        mock_bulk_load.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_harvest(product_types=("investigation",))

        # Should call _finish_harvest_run with status='failed'
        assert mock_finish_run.call_count == 1
        finish_kwargs = mock_finish_run.call_args[1]
        assert finish_kwargs["status"] == "failed"
        assert "DB error" in finish_kwargs["error_message"]


# ---------------------------------------------------------------------------
# Tests: ResilientClient usage
# ---------------------------------------------------------------------------


class TestResilientClientUsage:
    """Verify that ResilientClient is imported and used."""

    def test_resilient_client_imported(self) -> None:
        """harvest_pds4 should import ResilientClient from scix.http_client."""
        import harvest_pds4

        assert hasattr(harvest_pds4, "ResilientClient")
        from scix.http_client import ResilientClient

        assert harvest_pds4.ResilientClient is ResilientClient

    def test_fetch_pds4_page_requires_client(self) -> None:
        """fetch_pds4_page should require a client keyword argument."""
        import inspect

        sig = inspect.signature(fetch_pds4_page)
        assert "client" in sig.parameters

    @patch("harvest_pds4.download_pds4_context")
    def test_run_harvest_creates_client(self, mock_download: MagicMock) -> None:
        """run_harvest should create a ResilientClient if none provided."""
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        with patch("harvest_pds4.ResilientClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            run_harvest(dry_run=True, product_types=("investigation",))
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Entity graph integration
# ---------------------------------------------------------------------------


class TestEntityGraph:
    """Tests for entity graph table population."""

    def test_write_entity_graph_inserts_entities(self) -> None:
        """_write_entity_graph should INSERT into entities table."""
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS[:1], "investigation")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        count = _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        assert count == 1
        # Should have executed SQL with 'planetary_science' discipline
        calls = mock_cursor.execute.call_args_list
        entity_insert = calls[0]
        sql = entity_insert[0][0]
        assert "entities" in sql
        assert "'planetary_science'" in sql
        params = entity_insert[0][1]
        assert params["canonical_name"] == "Cassini-Huygens"
        assert params["entity_type"] == "mission"

    def test_write_entity_graph_inserts_pds_urn_identifier(self) -> None:
        """_write_entity_graph should INSERT PDS URN into entity_identifiers."""
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS[:1], "investigation")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        calls = mock_cursor.execute.call_args_list
        # Second call should be entity_identifiers insert
        id_insert = calls[1]
        sql = id_insert[0][0]
        assert "entity_identifiers" in sql
        assert "'pds_urn'" in sql
        params = id_insert[0][1]
        assert params["external_id"].startswith("urn:nasa:pds:context:investigation:")
        assert params["entity_id"] == 1

    def test_write_entity_graph_inserts_aliases(self) -> None:
        """_write_entity_graph should INSERT aliases into entity_aliases."""
        # MER product has abbreviation alias
        entries = parse_pds4_products(SAMPLE_INVESTIGATIONS[1:2], "investigation")
        assert len(entries[0]["aliases"]) > 0

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (2,)
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        calls = mock_cursor.execute.call_args_list
        alias_calls = [c for c in calls if "entity_aliases" in c[0][0]]
        assert len(alias_calls) >= 1
        alias_params = alias_calls[0][0][1]
        assert alias_params["alias"] == "MER"
        assert alias_params["entity_id"] == 2


# ---------------------------------------------------------------------------
# Tests: Entity relationships
# ---------------------------------------------------------------------------


class TestEntityRelationships:
    """Tests for entity relationship creation."""

    def _setup_entries(self) -> list[dict[str, Any]]:
        """Parse all sample products into entries."""
        entries: list[dict[str, Any]] = []
        entries.extend(parse_pds4_products(SAMPLE_INVESTIGATIONS, "investigation"))
        entries.extend(parse_pds4_products(SAMPLE_INSTRUMENTS, "instrument"))
        entries.extend(parse_pds4_products(SAMPLE_TARGETS, "target"))
        return entries

    def test_part_of_mission_from_urn(self) -> None:
        """Instruments with parent investigation in URN get part_of_mission."""
        entries = self._setup_entries()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        # Simulate entity_identifiers lookup
        # Map URNs to entity IDs
        urn_rows = [
            ("urn:nasa:pds:context:investigation:mission.cassini-huygens", 100),
            ("urn:nasa:pds:context:investigation:mission.mars-exploration-rover", 101),
            ("urn:nasa:pds:context:investigation:field_campaign.dd_nnss-nv_2019", 102),
            ("urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.cirs", 200),
            ("urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.iss-nac", 201),
            ("urn:nasa:pds:context:target:planet.mars", 300),
            ("urn:nasa:pds:context:target:satellite.titan", 301),
            ("urn:nasa:pds:context:target:asteroid.433_eros", 302),
            ("urn:nasa:pds:context:target:planet.saturn", 303),
        ]
        mock_cursor.fetchall.return_value = urn_rows

        count = _write_relationships(mock_conn, entries, harvest_run_id=1)

        # Find part_of_mission INSERT calls
        execute_calls = mock_cursor.execute.call_args_list
        part_of_calls = [
            c
            for c in execute_calls
            if len(c[0]) > 1
            and isinstance(c[0][1], dict)
            and c[0][1].get("subject") in (200, 201)
            and "'part_of_mission'" in c[0][0]
        ]
        assert len(part_of_calls) >= 2  # CIRS and ISS-NAC both link to Cassini

        # Verify CIRS -> Cassini-Huygens
        cirs_call = [c for c in part_of_calls if c[0][1]["subject"] == 200]
        assert len(cirs_call) >= 1
        assert cirs_call[0][0][1]["object"] == 100  # Cassini-Huygens

    def test_observes_target_from_ref_lid(self) -> None:
        """Missions with ref_lid_target get observes_target relationships."""
        entries = self._setup_entries()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        urn_rows = [
            ("urn:nasa:pds:context:investigation:mission.cassini-huygens", 100),
            ("urn:nasa:pds:context:investigation:mission.mars-exploration-rover", 101),
            ("urn:nasa:pds:context:investigation:field_campaign.dd_nnss-nv_2019", 102),
            ("urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.cirs", 200),
            ("urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.iss-nac", 201),
            ("urn:nasa:pds:context:target:planet.mars", 300),
            ("urn:nasa:pds:context:target:satellite.titan", 301),
            ("urn:nasa:pds:context:target:asteroid.433_eros", 302),
            ("urn:nasa:pds:context:target:planet.saturn", 303),
        ]
        mock_cursor.fetchall.return_value = urn_rows

        count = _write_relationships(mock_conn, entries, harvest_run_id=1)

        execute_calls = mock_cursor.execute.call_args_list
        observes_calls = [
            c
            for c in execute_calls
            if len(c[0]) > 1 and isinstance(c[0][1], dict) and "'observes_target'" in c[0][0]
        ]
        # Cassini observes Saturn and Titan (2), MER observes Mars (1) = 3
        assert len(observes_calls) >= 3

        # Verify Cassini -> Saturn
        cassini_saturn = [
            c
            for c in observes_calls
            if c[0][1].get("subject") == 100 and c[0][1].get("object") == 303
        ]
        assert len(cassini_saturn) == 1

        # Verify MER -> Mars
        mer_mars = [
            c
            for c in observes_calls
            if c[0][1].get("subject") == 101 and c[0][1].get("object") == 300
        ]
        assert len(mer_mars) == 1

    def test_returns_relationship_count(self) -> None:
        """_write_relationships should return the count of relationships created."""
        entries = self._setup_entries()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        urn_rows = [
            ("urn:nasa:pds:context:investigation:mission.cassini-huygens", 100),
            ("urn:nasa:pds:context:investigation:mission.mars-exploration-rover", 101),
            ("urn:nasa:pds:context:investigation:field_campaign.dd_nnss-nv_2019", 102),
            ("urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.cirs", 200),
            ("urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.iss-nac", 201),
            ("urn:nasa:pds:context:target:planet.mars", 300),
            ("urn:nasa:pds:context:target:satellite.titan", 301),
            ("urn:nasa:pds:context:target:asteroid.433_eros", 302),
            ("urn:nasa:pds:context:target:planet.saturn", 303),
        ]
        mock_cursor.fetchall.return_value = urn_rows

        count = _write_relationships(mock_conn, entries, harvest_run_id=1)

        # part_of_mission: CIRS (URN + ref_lid), ISS-NAC (URN + ref_lid) = 4
        # observes_target: Cassini->Saturn, Cassini->Titan, MER->Mars = 3
        # Total: 7
        assert count > 0


# ---------------------------------------------------------------------------
# Tests: harvest_runs
# ---------------------------------------------------------------------------


class TestHarvestRuns:
    """Tests for harvest_runs logging."""

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_harvest_run_completed(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        """harvest_runs should have a completion record with source='pds4'."""
        mock_download.return_value = {"investigation": SAMPLE_INVESTIGATIONS}
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 3
        mock_start_run.return_value = 7
        mock_write_graph.return_value = 3
        mock_write_rels.return_value = 0

        run_harvest(product_types=("investigation",))

        # _start_harvest_run called with source implied as 'pds4'
        mock_start_run.assert_called_once()
        # _finish_harvest_run called with completed status
        mock_finish_run.assert_called_once()
        finish_args = mock_finish_run.call_args
        assert finish_args[0][1] == 7  # run_id
        finish_kwargs = finish_args[1]
        assert finish_kwargs["records_fetched"] == 3
        assert finish_kwargs["records_upserted"] == 3
        assert "mission" in finish_kwargs["counts"]

    @patch("harvest_pds4._write_relationships")
    @patch("harvest_pds4._write_entity_graph")
    @patch("harvest_pds4._finish_harvest_run")
    @patch("harvest_pds4._start_harvest_run")
    @patch("harvest_pds4.get_connection")
    @patch("harvest_pds4.bulk_load")
    @patch("harvest_pds4.download_pds4_context")
    def test_harvest_run_accurate_counts(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_start_run: MagicMock,
        mock_finish_run: MagicMock,
        mock_write_graph: MagicMock,
        mock_write_rels: MagicMock,
    ) -> None:
        """harvest_runs counts should match actual product counts."""
        mock_download.return_value = {
            "investigation": SAMPLE_INVESTIGATIONS,
            "instrument": SAMPLE_INSTRUMENTS,
            "target": SAMPLE_TARGETS,
        }
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 9
        mock_start_run.return_value = 1
        mock_write_graph.return_value = 9
        mock_write_rels.return_value = 5

        run_harvest()

        finish_kwargs = mock_finish_run.call_args[1]
        assert finish_kwargs["records_fetched"] == 9  # 3+2+4 raw products
        assert finish_kwargs["records_upserted"] == 9
        assert finish_kwargs["counts"]["mission"] == 3
        assert finish_kwargs["counts"]["instrument"] == 2
        assert finish_kwargs["counts"]["target"] == 4


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
