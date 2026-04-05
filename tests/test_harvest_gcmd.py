"""Tests for the GCMD unified keyword harvester.

Unit tests use synthetic JSON fixtures mimicking GCMD hierarchy structure.
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

from harvest_gcmd import (
    SCHEME_CONFIGS,
    SchemeConfig,
    download_github_scheme,
    download_kms_scheme,
    harvest_all,
    harvest_scheme,
    parse_github_scheme,
    parse_kms_scheme,
    run_harvest,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures: GitHub hierarchy schemes
# ---------------------------------------------------------------------------

SAMPLE_INSTRUMENTS: list[dict[str, Any]] = [
    {
        "uuid": "root-instruments-uuid",
        "label": "Instruments",
        "children": [
            {
                "uuid": "cat-solar-uuid",
                "label": "Solar/Space Observing Instruments",
                "broader": "root-instruments-uuid",
                "children": [
                    {
                        "uuid": "inst-coronagraph-uuid",
                        "label": "Coronagraphs",
                        "broader": "cat-solar-uuid",
                        "children": [
                            {
                                "uuid": "inst-lasco-uuid",
                                "label": "LASCO",
                                "broader": "inst-coronagraph-uuid",
                            },
                        ],
                    },
                    {
                        "uuid": "inst-magnetograph-uuid",
                        "label": "Magnetographs",
                        "broader": "cat-solar-uuid",
                    },
                ],
            },
            {
                "uuid": "cat-radar-uuid",
                "label": "Active Remote Sensing",
                "broader": "root-instruments-uuid",
                "children": [
                    {
                        "uuid": "inst-altimeter-uuid",
                        "label": "Altimeters",
                        "broader": "cat-radar-uuid",
                        "children": [
                            {
                                "uuid": "inst-ra2-uuid",
                                "label": "RA-2",
                                "broader": "inst-altimeter-uuid",
                            },
                        ],
                    },
                    {
                        "uuid": "inst-lidar-uuid",
                        "label": "Lidars",
                        "broader": "cat-radar-uuid",
                    },
                ],
            },
        ],
    }
]

SAMPLE_PLATFORMS: list[dict[str, Any]] = [
    {
        "uuid": "root-platforms-uuid",
        "label": "Platforms",
        "children": [
            {
                "uuid": "cat-satellite-uuid",
                "label": "Earth Observation Satellites",
                "broader": "root-platforms-uuid",
                "children": [
                    {
                        "uuid": "plat-terra-uuid",
                        "label": "Terra",
                        "broader": "cat-satellite-uuid",
                    },
                    {
                        "uuid": "plat-aqua-uuid",
                        "label": "Aqua",
                        "broader": "cat-satellite-uuid",
                    },
                ],
            },
        ],
    }
]

SAMPLE_SCIENCEKEYWORDS: list[dict[str, Any]] = [
    {
        "uuid": "root-es-uuid",
        "label": "EARTH SCIENCE",
        "children": [
            {
                "uuid": "cat-atmo-uuid",
                "label": "ATMOSPHERE",
                "broader": "root-es-uuid",
                "children": [
                    {
                        "uuid": "topic-winds-uuid",
                        "label": "ATMOSPHERIC WINDS",
                        "broader": "cat-atmo-uuid",
                        "children": [
                            {
                                "uuid": "var-surface-winds-uuid",
                                "label": "SURFACE WINDS",
                                "broader": "topic-winds-uuid",
                            },
                            {
                                "uuid": "var-upper-winds-uuid",
                                "label": "UPPER LEVEL WINDS",
                                "broader": "topic-winds-uuid",
                            },
                        ],
                    },
                    {
                        "uuid": "topic-clouds-uuid",
                        "label": "CLOUDS",
                        "broader": "cat-atmo-uuid",
                    },
                ],
            },
            {
                "uuid": "cat-ocean-uuid",
                "label": "OCEANS",
                "broader": "root-es-uuid",
                "children": [
                    {
                        "uuid": "topic-sst-uuid",
                        "label": "OCEAN TEMPERATURE",
                        "broader": "cat-ocean-uuid",
                        "children": [
                            {
                                "uuid": "var-sst-uuid",
                                "label": "SEA SURFACE TEMPERATURE",
                                "broader": "topic-sst-uuid",
                            },
                        ],
                    },
                ],
            },
        ],
    }
]

# Fixture with duplicate leaf names across different hierarchy branches
SAMPLE_SCIENCEKEYWORDS_DUPES: list[dict[str, Any]] = [
    {
        "uuid": "root-es-uuid",
        "label": "EARTH SCIENCE",
        "children": [
            {
                "uuid": "cat-atmo-uuid",
                "label": "ATMOSPHERE",
                "broader": "root-es-uuid",
                "children": [
                    {
                        "uuid": "atmo-temp-uuid",
                        "label": "TEMPERATURE",
                        "broader": "cat-atmo-uuid",
                    },
                ],
            },
            {
                "uuid": "cat-ocean-uuid",
                "label": "OCEANS",
                "broader": "root-es-uuid",
                "children": [
                    {
                        "uuid": "ocean-temp-uuid",
                        "label": "TEMPERATURE",
                        "broader": "cat-ocean-uuid",
                    },
                ],
            },
        ],
    }
]


# Synthetic fixture: KMS API
SAMPLE_KMS_PROVIDERS: list[dict[str, Any]] = [
    {
        "uuid": "prov-noaa-uuid",
        "prefLabel": "DOC/NOAA",
        "scheme": {"shortName": "providers", "longName": "Providers"},
        "definitions": [{"text": "National Oceanic and Atmospheric Administration"}],
    },
    {
        "uuid": "prov-nasa-uuid",
        "prefLabel": "NASA/GSFC",
        "scheme": {"shortName": "providers", "longName": "Providers"},
        "definitions": [{"text": "Goddard Space Flight Center"}],
    },
    {
        "uuid": "prov-esa-uuid",
        "prefLabel": "ESA",
        "scheme": {"shortName": "providers", "longName": "Providers"},
        "definitions": [{"text": "European Space Agency"}],
    },
]

SAMPLE_KMS_PROJECTS: list[dict[str, Any]] = [
    {
        "uuid": "proj-modis-uuid",
        "prefLabel": "MODIS",
        "scheme": {"shortName": "projects", "longName": "Projects"},
        "definitions": [{"text": "Moderate Resolution Imaging Spectroradiometer"}],
    },
    {
        "uuid": "proj-ceres-uuid",
        "prefLabel": "CERES",
        "scheme": {"shortName": "projects", "longName": "Projects"},
        "definitions": [],
    },
]


# ---------------------------------------------------------------------------
# Tests: parse_github_scheme (instruments)
# ---------------------------------------------------------------------------


class TestParseGithubInstruments:
    """Parse instruments scheme from GitHub hierarchy."""

    def _config(self) -> SchemeConfig:
        return SCHEME_CONFIGS["instruments"]

    def test_returns_all_nodes(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        # 2 categories + 2 subcategories + 4 leaf instruments = 8
        assert len(entries) == 8

    def test_entity_type_is_instrument(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        for entry in entries:
            assert entry["entity_type"] == "instrument"

    def test_source_is_gcmd(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        for entry in entries:
            assert entry["source"] == "gcmd"

    def test_external_id_is_uuid(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        lasco = next(e for e in entries if "LASCO" in e["canonical_name"])
        assert lasco["external_id"] == "inst-lasco-uuid"

    def test_metadata_has_gcmd_scheme(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        for entry in entries:
            assert entry["metadata"]["gcmd_scheme"] == "instruments"

    def test_metadata_has_gcmd_hierarchy(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        lasco = next(e for e in entries if "LASCO" in e["canonical_name"])
        assert "Instruments" in lasco["metadata"]["gcmd_hierarchy"]
        assert "Coronagraphs" in lasco["metadata"]["gcmd_hierarchy"]

    def test_metadata_has_uuid(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        for entry in entries:
            assert "uuid" in entry["metadata"]
            assert entry["metadata"]["uuid"] == entry["external_id"]

    def test_metadata_has_short_name(self) -> None:
        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, self._config())
        for entry in entries:
            assert "short_name" in entry["metadata"]


# ---------------------------------------------------------------------------
# Tests: parse_github_scheme (platforms)
# ---------------------------------------------------------------------------


class TestParseGithubPlatforms:
    """Parse platforms scheme from GitHub hierarchy."""

    def _config(self) -> SchemeConfig:
        return SCHEME_CONFIGS["platforms"]

    def test_returns_correct_count(self) -> None:
        entries = parse_github_scheme(SAMPLE_PLATFORMS, self._config())
        # 1 category + 2 platforms = 3
        assert len(entries) == 3

    def test_entity_type_is_instrument(self) -> None:
        entries = parse_github_scheme(SAMPLE_PLATFORMS, self._config())
        for entry in entries:
            assert entry["entity_type"] == "instrument"

    def test_gcmd_scheme_is_platforms(self) -> None:
        entries = parse_github_scheme(SAMPLE_PLATFORMS, self._config())
        for entry in entries:
            assert entry["metadata"]["gcmd_scheme"] == "platforms"


# ---------------------------------------------------------------------------
# Tests: parse_github_scheme (sciencekeywords — leaves only)
# ---------------------------------------------------------------------------


class TestParseGithubScienceKeywords:
    """Parse science keywords scheme — only leaf nodes emitted."""

    def _config(self) -> SchemeConfig:
        return SCHEME_CONFIGS["sciencekeywords"]

    def test_only_leaf_nodes_emitted(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS, self._config())
        names = {e["canonical_name"] for e in entries}
        # Leaves: SURFACE WINDS, UPPER LEVEL WINDS, CLOUDS, SEA SURFACE TEMPERATURE
        assert len(entries) == 4
        assert "SURFACE WINDS" in names
        assert "UPPER LEVEL WINDS" in names
        assert "CLOUDS" in names
        assert "SEA SURFACE TEMPERATURE" in names

    def test_branch_nodes_excluded(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS, self._config())
        names = {e["canonical_name"] for e in entries}
        assert "ATMOSPHERE" not in names
        assert "ATMOSPHERIC WINDS" not in names
        assert "OCEANS" not in names
        assert "OCEAN TEMPERATURE" not in names

    def test_entity_type_is_observable(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS, self._config())
        for entry in entries:
            assert entry["entity_type"] == "observable"

    def test_hierarchy_path_in_metadata(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS, self._config())
        sw = next(e for e in entries if e["canonical_name"] == "SURFACE WINDS")
        hierarchy = sw["metadata"]["gcmd_hierarchy"]
        assert "EARTH SCIENCE" in hierarchy
        assert "ATMOSPHERE" in hierarchy
        assert "ATMOSPHERIC WINDS" in hierarchy
        assert "SURFACE WINDS" in hierarchy


# ---------------------------------------------------------------------------
# Tests: duplicate disambiguation
# ---------------------------------------------------------------------------


class TestDuplicateDisambiguation:
    """Duplicate leaf names are disambiguated with parent prefix."""

    def _config(self) -> SchemeConfig:
        return SCHEME_CONFIGS["sciencekeywords"]

    def test_duplicate_names_get_parent_prefix(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS_DUPES, self._config())
        names = [e["canonical_name"] for e in entries]
        # Both are "TEMPERATURE" but from different parents
        assert "ATMOSPHERE > TEMPERATURE" in names
        assert "OCEANS > TEMPERATURE" in names

    def test_duplicate_original_name_in_aliases(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS_DUPES, self._config())
        for entry in entries:
            assert "TEMPERATURE" in entry["aliases"]

    def test_disambiguated_entries_have_unique_names(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS_DUPES, self._config())
        names = [e["canonical_name"] for e in entries]
        assert len(names) == len(set(names))

    def test_disambiguated_entries_keep_correct_uuid(self) -> None:
        entries = parse_github_scheme(SAMPLE_SCIENCEKEYWORDS_DUPES, self._config())
        atmo_temp = next(e for e in entries if e["canonical_name"] == "ATMOSPHERE > TEMPERATURE")
        ocean_temp = next(e for e in entries if e["canonical_name"] == "OCEANS > TEMPERATURE")
        assert atmo_temp["external_id"] == "atmo-temp-uuid"
        assert ocean_temp["external_id"] == "ocean-temp-uuid"


# ---------------------------------------------------------------------------
# Tests: parse_kms_scheme (providers)
# ---------------------------------------------------------------------------


class TestParseKmsProviders:
    """Parse KMS API providers scheme."""

    def _config(self) -> SchemeConfig:
        return SCHEME_CONFIGS["providers"]

    def test_returns_correct_count(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        assert len(entries) == 3

    def test_entity_type_is_mission(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        for entry in entries:
            assert entry["entity_type"] == "mission"

    def test_source_is_gcmd(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        for entry in entries:
            assert entry["source"] == "gcmd"

    def test_external_id_is_uuid(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        noaa = next(e for e in entries if "NOAA" in e["canonical_name"])
        assert noaa["external_id"] == "prov-noaa-uuid"

    def test_metadata_has_gcmd_scheme(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        for entry in entries:
            assert entry["metadata"]["gcmd_scheme"] == "providers"

    def test_metadata_has_gcmd_hierarchy(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        noaa = next(e for e in entries if "NOAA" in e["canonical_name"])
        assert noaa["metadata"]["gcmd_hierarchy"] == "DOC > NOAA"

    def test_slash_separated_alias(self) -> None:
        """Providers with slash-separated names get short_name as alias."""
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        noaa = next(e for e in entries if "NOAA" in e["canonical_name"])
        assert "NOAA" in noaa["aliases"]

    def test_simple_name_no_extra_alias(self) -> None:
        """Single-part provider names don't get redundant alias."""
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        esa = next(e for e in entries if e["canonical_name"] == "ESA")
        assert esa["aliases"] == []

    def test_long_name_in_metadata(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROVIDERS, self._config())
        noaa = next(e for e in entries if "NOAA" in e["canonical_name"])
        assert "long_name" in noaa["metadata"]
        assert "National Oceanic" in noaa["metadata"]["long_name"]


# ---------------------------------------------------------------------------
# Tests: parse_kms_scheme (projects)
# ---------------------------------------------------------------------------


class TestParseKmsProjects:
    """Parse KMS API projects scheme."""

    def _config(self) -> SchemeConfig:
        return SCHEME_CONFIGS["projects"]

    def test_returns_correct_count(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROJECTS, self._config())
        assert len(entries) == 2

    def test_entity_type_is_mission(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROJECTS, self._config())
        for entry in entries:
            assert entry["entity_type"] == "mission"

    def test_metadata_gcmd_scheme_is_projects(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROJECTS, self._config())
        for entry in entries:
            assert entry["metadata"]["gcmd_scheme"] == "projects"

    def test_missing_definition_no_long_name(self) -> None:
        entries = parse_kms_scheme(SAMPLE_KMS_PROJECTS, self._config())
        ceres = next(e for e in entries if e["canonical_name"] == "CERES")
        assert "long_name" not in ceres["metadata"]


# ---------------------------------------------------------------------------
# Tests: download_github_scheme (uses ResilientClient)
# ---------------------------------------------------------------------------


class TestDownloadGithubScheme:
    """Download from GitHub with mocked ResilientClient."""

    @patch("harvest_gcmd._get_client")
    def test_downloads_and_parses(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_INSTRUMENTS
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = download_github_scheme("https://example.com/test.json")
        assert len(result) == 1
        assert result[0]["label"] == "Instruments"
        mock_client.get.assert_called_once_with("https://example.com/test.json")

    @patch("harvest_gcmd._get_client")
    def test_uses_resilient_client(self, mock_get_client: MagicMock) -> None:
        """Verify ResilientClient is used instead of raw requests/urllib."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_INSTRUMENTS
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        download_github_scheme("https://example.com/test.json")
        mock_client.get.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: download_kms_scheme (uses ResilientClient)
# ---------------------------------------------------------------------------


class TestDownloadKmsScheme:
    """Download from KMS API with mocked ResilientClient and pagination."""

    @patch("harvest_gcmd._get_client")
    def test_single_page(self, mock_get_client: MagicMock) -> None:
        payload = {
            "hits": 3,
            "page_num": 1,
            "page_size": 2000,
            "concepts": SAMPLE_KMS_PROVIDERS,
        }
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = download_kms_scheme("https://example.com/kms?format=json")
        assert len(result) == 3

    @patch("harvest_gcmd._get_client")
    def test_multi_page(self, mock_get_client: MagicMock) -> None:
        page1 = {
            "hits": 4,
            "page_num": 1,
            "page_size": 2,
            "concepts": SAMPLE_KMS_PROVIDERS[:2],
        }
        page2 = {
            "hits": 4,
            "page_num": 2,
            "page_size": 2,
            "concepts": SAMPLE_KMS_PROVIDERS[2:] + SAMPLE_KMS_PROJECTS[:1],
        }

        mock_client = MagicMock()
        resp1 = MagicMock()
        resp1.json.return_value = page1
        resp2 = MagicMock()
        resp2.json.return_value = page2
        mock_client.get.side_effect = [resp1, resp2]
        mock_get_client.return_value = mock_client

        result = download_kms_scheme("https://example.com/kms?format=json", page_size=2)
        assert len(result) == 4

    @patch("harvest_gcmd._get_client")
    def test_uses_resilient_client(self, mock_get_client: MagicMock) -> None:
        """Verify ResilientClient is used for KMS downloads."""
        payload = {"hits": 0, "concepts": []}
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        download_kms_scheme("https://example.com/kms?format=json")
        mock_client.get.assert_called()


# ---------------------------------------------------------------------------
# Tests: harvest_all / harvest_scheme
# ---------------------------------------------------------------------------


class TestHarvestAll:
    """Integration tests for harvest_all with mocked downloads."""

    @patch("harvest_gcmd.download_kms_scheme")
    @patch("harvest_gcmd.download_github_scheme")
    def test_harvest_single_scheme(
        self,
        mock_github: MagicMock,
        mock_kms: MagicMock,
    ) -> None:
        mock_github.return_value = SAMPLE_INSTRUMENTS
        entries = harvest_all(schemes=["instruments"])
        assert len(entries) == 8
        assert all(e["entity_type"] == "instrument" for e in entries)
        mock_kms.assert_not_called()

    @patch("harvest_gcmd.download_kms_scheme")
    @patch("harvest_gcmd.download_github_scheme")
    def test_harvest_kms_scheme(
        self,
        mock_github: MagicMock,
        mock_kms: MagicMock,
    ) -> None:
        mock_kms.return_value = SAMPLE_KMS_PROVIDERS
        entries = harvest_all(schemes=["providers"])
        assert len(entries) == 3
        assert all(e["entity_type"] == "mission" for e in entries)
        mock_github.assert_not_called()

    @patch("harvest_gcmd.download_kms_scheme")
    @patch("harvest_gcmd.download_github_scheme")
    def test_harvest_multiple_schemes(
        self,
        mock_github: MagicMock,
        mock_kms: MagicMock,
    ) -> None:
        mock_github.return_value = SAMPLE_INSTRUMENTS
        mock_kms.return_value = SAMPLE_KMS_PROJECTS
        entries = harvest_all(schemes=["instruments", "projects"])
        types = {e["entity_type"] for e in entries}
        assert "instrument" in types
        assert "mission" in types


# ---------------------------------------------------------------------------
# Tests: run_harvest
# ---------------------------------------------------------------------------


class TestRunHarvest:
    """Tests for run_harvest with mocked download and DB."""

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_calls_bulk_load_with_discipline(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log
        mock_write_graph.return_value = 8

        count = run_harvest(dsn="dbname=test", schemes=["instruments"])

        assert count == 8
        mock_bulk_load.assert_called_once()
        # Verify discipline parameter
        _, kwargs = mock_bulk_load.call_args
        assert kwargs.get("discipline") == "earth_science"

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_closes_connection(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log
        mock_write_graph.return_value = 8

        run_harvest(schemes=["instruments"])
        mock_conn.close.assert_called_once()

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError):
            run_harvest(schemes=["instruments"])
        mock_conn.close.assert_called_once()

    @patch("harvest_gcmd.download_github_scheme")
    def test_dry_run_skips_db(
        self,
        mock_download: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        count = run_harvest(schemes=["instruments"], dry_run=True)
        assert count == 8

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_creates_harvest_run(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        """Verify harvest_runs record is created and completed."""
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 42
        mock_run_log.run_id = 42
        mock_run_log_cls.return_value = mock_run_log
        mock_write_graph.return_value = 8

        run_harvest(schemes=["instruments"])

        mock_run_log_cls.assert_called_once_with(mock_conn, "gcmd")
        mock_run_log.start.assert_called_once()
        mock_run_log.complete.assert_called_once()
        _, kwargs = mock_run_log.complete.call_args
        assert kwargs["records_fetched"] == 8
        assert kwargs["records_upserted"] == 8

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_writes_entity_graph(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        """Verify entity graph tables are written after bulk_load."""
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log
        mock_write_graph.return_value = 8

        run_harvest(schemes=["instruments"])

        mock_write_graph.assert_called_once()
        args = mock_write_graph.call_args[0]
        assert args[0] is mock_conn  # conn
        assert len(args[1]) == 8  # entries
        assert args[2] == 1  # harvest_run_id


# ---------------------------------------------------------------------------
# Tests: _write_entity_graph
# ---------------------------------------------------------------------------


class TestWriteEntityGraph:
    """Tests for _write_entity_graph with mocked DB cursor."""

    def _make_mock_conn(self) -> MagicMock:
        """Create a mock connection with a cursor that returns entity IDs."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # fetchone returns entity_id for each INSERT INTO entities
        mock_cursor.fetchone.return_value = (100,)
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn

    def test_writes_entities_with_discipline(self) -> None:
        from harvest_gcmd import _write_entity_graph

        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, SCHEME_CONFIGS["instruments"])
        mock_conn = self._make_mock_conn()

        count = _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        assert count == 8
        # Verify execute calls include source='gcmd' (positional params via upsert_entity)
        cursor = mock_conn.cursor.return_value
        calls = cursor.execute.call_args_list
        entity_inserts = [c for c in calls if "INTO entities" in str(c)]
        assert len(entity_inserts) == 8
        for call_obj in entity_inserts:
            params = call_obj[0][1]
            # upsert_entity uses positional tuple:
            # (canonical_name, entity_type, discipline, source, harvest_run_id, properties)
            assert params[3] == "gcmd"  # source

    def test_writes_entity_identifiers_gcmd_uuid(self) -> None:
        from harvest_gcmd import _write_entity_graph

        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, SCHEME_CONFIGS["instruments"])
        mock_conn = self._make_mock_conn()

        _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        cursor = mock_conn.cursor.return_value
        calls = cursor.execute.call_args_list
        id_inserts = [c for c in calls if "entity_identifiers" in str(c)]
        assert len(id_inserts) == 8  # All 8 entries have external_id
        for call_obj in id_inserts:
            params = call_obj[0][1]
            # upsert_entity_identifier uses positional tuple:
            # (entity_id, id_scheme, external_id, is_primary)
            assert params[1] == "gcmd_uuid"
            assert params[2] is not None  # external_id

    def test_writes_entity_aliases(self) -> None:
        from harvest_gcmd import _write_entity_graph

        # Use duplicates fixture — each entry gets an alias
        entries = parse_github_scheme(
            SAMPLE_SCIENCEKEYWORDS_DUPES, SCHEME_CONFIGS["sciencekeywords"]
        )
        mock_conn = self._make_mock_conn()

        _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        cursor = mock_conn.cursor.return_value
        calls = cursor.execute.call_args_list
        alias_inserts = [c for c in calls if "entity_aliases" in str(c)]
        assert len(alias_inserts) == 2  # Two "TEMPERATURE" aliases

    def test_properties_contain_gcmd_scheme_and_hierarchy(self) -> None:
        from harvest_gcmd import _write_entity_graph

        entries = parse_github_scheme(SAMPLE_INSTRUMENTS, SCHEME_CONFIGS["instruments"])
        mock_conn = self._make_mock_conn()

        _write_entity_graph(mock_conn, entries, harvest_run_id=1)

        cursor = mock_conn.cursor.return_value
        calls = cursor.execute.call_args_list
        entity_inserts = [c for c in calls if "INTO entities" in str(c)]
        for call_obj in entity_inserts:
            params = call_obj[0][1]
            # upsert_entity uses positional tuple; properties JSON is at index 5
            props = json.loads(params[5])
            assert "gcmd_scheme" in props
            assert "gcmd_hierarchy" in props
            assert props["gcmd_scheme"] == "instruments"


# ---------------------------------------------------------------------------
# Tests: all entries have discipline='earth_science' metadata key
# ---------------------------------------------------------------------------


class TestDisciplineMetadata:
    """Verify discipline is passed to bulk_load for all schemes."""

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_kms_scheme")
    @patch("harvest_gcmd.download_github_scheme")
    def test_bulk_load_receives_earth_science_discipline(
        self,
        mock_github: MagicMock,
        mock_kms: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        mock_github.return_value = SAMPLE_INSTRUMENTS
        mock_kms.return_value = SAMPLE_KMS_PROVIDERS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 9
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log
        mock_write_graph.return_value = 9

        run_harvest(schemes=["instruments", "providers"])

        _, kwargs = mock_bulk_load.call_args
        assert kwargs["discipline"] == "earth_science"


# ---------------------------------------------------------------------------
# Tests: scheme configs cover all 5 schemes
# ---------------------------------------------------------------------------


class TestSchemeConfigs:
    """Verify scheme configuration completeness."""

    def test_five_schemes_configured(self) -> None:
        assert len(SCHEME_CONFIGS) == 5

    def test_instruments_config(self) -> None:
        cfg = SCHEME_CONFIGS["instruments"]
        assert cfg.entity_type == "instrument"
        assert cfg.source_kind == "github"
        assert not cfg.leaves_only

    def test_platforms_config(self) -> None:
        cfg = SCHEME_CONFIGS["platforms"]
        assert cfg.entity_type == "instrument"
        assert cfg.source_kind == "github"
        assert not cfg.leaves_only

    def test_sciencekeywords_config(self) -> None:
        cfg = SCHEME_CONFIGS["sciencekeywords"]
        assert cfg.entity_type == "observable"
        assert cfg.source_kind == "github"
        assert cfg.leaves_only

    def test_providers_config(self) -> None:
        cfg = SCHEME_CONFIGS["providers"]
        assert cfg.entity_type == "mission"
        assert cfg.source_kind == "kms"

    def test_projects_config(self) -> None:
        cfg = SCHEME_CONFIGS["projects"]
        assert cfg.entity_type == "mission"
        assert cfg.source_kind == "kms"


# ---------------------------------------------------------------------------
# Tests: ResilientClient usage verification
# ---------------------------------------------------------------------------


class TestResilientClientUsage:
    """Verify that ResilientClient is used instead of urllib/raw requests."""

    def test_no_urllib_import_in_harvester(self) -> None:
        """The harvester should not use urllib for HTTP requests."""
        import harvest_gcmd

        source = Path(harvest_gcmd.__file__).read_text()
        assert "urllib.request" not in source
        assert "urllib.error" not in source

    def test_resilient_client_imported(self) -> None:
        """The harvester should import ResilientClient."""
        import harvest_gcmd

        assert hasattr(harvest_gcmd, "ResilientClient")


# ---------------------------------------------------------------------------
# Tests: harvest_runs tracking
# ---------------------------------------------------------------------------


class TestHarvestRunsTracking:
    """Verify harvest_runs records are created and updated."""

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_harvest_run_completed_with_counts(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 7
        mock_run_log.run_id = 7
        mock_run_log_cls.return_value = mock_run_log
        mock_write_graph.return_value = 8

        run_harvest(schemes=["instruments"])

        mock_run_log.complete.assert_called_once()
        _, kwargs = mock_run_log.complete.call_args
        assert kwargs["records_fetched"] == 8
        assert kwargs["records_upserted"] == 8
        assert "instrument" in kwargs["counts"]

    @patch("harvest_gcmd._write_entity_graph")
    @patch("harvest_gcmd.HarvestRunLog")
    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_harvest_run_failed_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        mock_write_graph: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 7
        mock_run_log.run_id = 7
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError):
            run_harvest(schemes=["instruments"])

        mock_run_log.fail.assert_called_once()
        assert "DB error" in mock_run_log.fail.call_args[0][0]

    def test_dry_run_does_not_create_harvest_run(self) -> None:
        """Dry run should not touch the database at all."""
        with patch("harvest_gcmd.download_github_scheme") as mock_dl:
            mock_dl.return_value = SAMPLE_INSTRUMENTS
            with patch("harvest_gcmd.HarvestRunLog") as mock_run_log_cls:
                run_harvest(schemes=["instruments"], dry_run=True)
                mock_run_log_cls.assert_not_called()
