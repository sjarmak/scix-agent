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
# Tests: download_github_scheme
# ---------------------------------------------------------------------------


class TestDownloadGithubScheme:
    """Download from GitHub with mocked HTTP."""

    @patch("harvest_gcmd.urllib.request.urlopen")
    def test_downloads_and_parses(self, mock_urlopen: MagicMock) -> None:
        raw = json.dumps(SAMPLE_INSTRUMENTS).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_github_scheme("https://example.com/test.json")
        assert len(result) == 1
        assert result[0]["label"] == "Instruments"

    @patch("harvest_gcmd.urllib.request.urlopen")
    def test_retries_on_failure(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        raw = json.dumps(SAMPLE_INSTRUMENTS).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("temporary failure"),
            mock_resp,
        ]
        with patch("harvest_gcmd.time.sleep"):
            result = download_github_scheme("https://example.com/test.json")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: download_kms_scheme
# ---------------------------------------------------------------------------


class TestDownloadKmsScheme:
    """Download from KMS API with mocked HTTP and pagination."""

    @patch("harvest_gcmd.urllib.request.urlopen")
    def test_single_page(self, mock_urlopen: MagicMock) -> None:
        payload = {
            "hits": 3,
            "page_num": 1,
            "page_size": 2000,
            "concepts": SAMPLE_KMS_PROVIDERS,
        }
        raw = json.dumps(payload).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = download_kms_scheme("https://example.com/kms?format=json")
        assert len(result) == 3

    @patch("harvest_gcmd.urllib.request.urlopen")
    def test_multi_page(self, mock_urlopen: MagicMock) -> None:
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

        def make_resp(payload: dict) -> MagicMock:
            raw = json.dumps(payload).encode("utf-8")
            resp = MagicMock()
            resp.read.return_value = raw
            resp.__enter__ = lambda self: self
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_urlopen.side_effect = [make_resp(page1), make_resp(page2)]
        result = download_kms_scheme("https://example.com/kms?format=json", page_size=2)
        assert len(result) == 4


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

    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_calls_bulk_load_with_discipline(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8

        count = run_harvest(dsn="dbname=test", schemes=["instruments"])

        assert count == 8
        mock_bulk_load.assert_called_once()
        # Verify discipline parameter
        _, kwargs = mock_bulk_load.call_args
        assert kwargs.get("discipline") == "earth_science"

    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_closes_connection(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 8

        run_harvest(schemes=["instruments"])
        mock_conn.close.assert_called_once()

    @patch("harvest_gcmd.get_connection")
    @patch("harvest_gcmd.bulk_load")
    @patch("harvest_gcmd.download_github_scheme")
    def test_run_harvest_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_INSTRUMENTS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")

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


# ---------------------------------------------------------------------------
# Tests: all entries have discipline='earth_science' metadata key
# ---------------------------------------------------------------------------


class TestDisciplineMetadata:
    """Verify discipline is passed to bulk_load for all schemes."""

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
    ) -> None:
        mock_github.return_value = SAMPLE_INSTRUMENTS
        mock_kms.return_value = SAMPLE_KMS_PROVIDERS
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 9

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
