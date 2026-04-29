"""Tests for the CRAN package catalog harvester."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_cran import (
    CRAN_ALL_URL,
    SOURCE,
    _is_production_dsn,
    _normalize_text,
    _normalize_url,
    download_catalog,
    parse_catalog,
    run_harvest,
)

SAMPLE_CRAN: dict[str, Any] = {
    "ggplot2": {
        "name": "ggplot2",
        "title": "Create Elegant Data Visualisations Using the Grammar of Graphics",
        "latest": "4.0.3",
        "versions": {
            "4.0.3": {
                "Package": "ggplot2",
                "Title": "Create Elegant Data Visualisations\n  Using the Grammar of Graphics",
                "Description": "A system for declaratively creating graphics.",
                "License": "MIT + file LICENSE",
                "URL": "https://ggplot2.tidyverse.org,\nhttps://github.com/tidyverse/ggplot2",
                "BugReports": "https://github.com/tidyverse/ggplot2/issues",
                "Author": "Hadley Wickham [aut]",
            },
        },
    },
    "dplyr": {
        "name": "dplyr",
        "latest": "1.1.0",
        "versions": {
            "1.1.0": {
                "Package": "dplyr",
                "Title": "A Grammar of Data Manipulation",
                "License": "MIT",
                "URL": "https://dplyr.tidyverse.org",
            },
        },
    },
    "MASS": {
        "name": "MASS",
        "latest": "7.3-60",
        "versions": {
            "7.3-60": {
                "Package": "MASS",
                "Title": "Support Functions and Datasets",
                "License": "GPL-2 | GPL-3",
            },
        },
    },
}


def _mock_response(payload: Any) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestNormalizers:
    def test_normalize_text_collapses_whitespace(self) -> None:
        assert _normalize_text("hello\n  world") == "hello world"

    def test_normalize_text_strips(self) -> None:
        assert _normalize_text("  abc  ") == "abc"

    def test_normalize_text_none(self) -> None:
        assert _normalize_text(None) is None
        assert _normalize_text("") is None

    def test_normalize_url_picks_first(self) -> None:
        url = "https://a.example,\nhttps://b.example"
        assert _normalize_url(url) == "https://a.example"

    def test_normalize_url_strips_whitespace(self) -> None:
        assert _normalize_url("  https://a.example\n") == "https://a.example"

    def test_normalize_url_none(self) -> None:
        assert _normalize_url(None) is None
        assert _normalize_url("") is None


class TestProductionDsn:
    def test_production_dsn_detected(self) -> None:
        assert _is_production_dsn("dbname=scix")
        assert _is_production_dsn("host=localhost dbname=scix port=5432")

    def test_test_dsn_safe(self) -> None:
        assert not _is_production_dsn("dbname=scix_test")
        assert not _is_production_dsn("dbname=other")
        assert not _is_production_dsn("")


class TestParseCatalog:
    def test_parses_all_valid_entries(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        names = {e["canonical_name"] for e in entries}
        assert names == {"ggplot2", "dplyr", "MASS"}

    def test_required_keys_present(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        for entry in entries:
            for key in (
                "canonical_name",
                "entity_type",
                "source",
                "external_id",
                "aliases",
                "properties",
                "metadata",
            ):
                assert key in entry, f"missing key: {key}"

    def test_entity_type_is_software(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        assert all(e["entity_type"] == "software" for e in entries)

    def test_source_is_cran(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        assert all(e["source"] == SOURCE for e in entries)

    def test_external_id_is_package_name(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        for entry in entries:
            assert entry["external_id"] == entry["canonical_name"]

    def test_homepage_in_properties(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        ggplot2 = next(e for e in entries if e["canonical_name"] == "ggplot2")
        assert ggplot2["properties"]["homepage"] == "https://ggplot2.tidyverse.org"

    def test_license_in_properties(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        ggplot2 = next(e for e in entries if e["canonical_name"] == "ggplot2")
        assert ggplot2["properties"]["license"] == "MIT + file LICENSE"

    def test_title_collapses_newlines(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        ggplot2 = next(e for e in entries if e["canonical_name"] == "ggplot2")
        assert "\n" not in ggplot2["properties"]["title"]
        assert "Grammar of Graphics" in ggplot2["properties"]["title"]

    def test_lowercase_alias_when_mixed_case(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        mass = next(e for e in entries if e["canonical_name"] == "MASS")
        assert "mass" in mass["aliases"]

    def test_no_duplicate_alias_when_already_lowercase(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        dplyr = next(e for e in entries if e["canonical_name"] == "dplyr")
        assert "dplyr" not in dplyr["aliases"]

    def test_skips_entries_missing_name(self) -> None:
        bad = {"": {"latest": "1.0", "versions": {}}}
        assert parse_catalog(bad) == []

    def test_empty_input(self) -> None:
        assert parse_catalog({}) == []

    def test_metadata_mirrors_properties(self) -> None:
        entries = parse_catalog(SAMPLE_CRAN)
        ggplot2 = next(e for e in entries if e["canonical_name"] == "ggplot2")
        assert ggplot2["metadata"] == ggplot2["properties"]


class TestDownloadCatalog:
    @patch("harvest_cran._get_client")
    def test_returns_dict(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_response(SAMPLE_CRAN)
        mock_get_client.return_value = mock_client
        result = download_catalog()
        assert isinstance(result, dict)
        assert "ggplot2" in result

    @patch("harvest_cran._get_client")
    def test_raises_on_non_dict(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_response(["not", "a", "dict"])
        mock_get_client.return_value = mock_client
        with pytest.raises(RuntimeError):
            download_catalog()


class TestRunHarvest:
    @patch("harvest_cran.HarvestRunLog")
    @patch("harvest_cran.get_connection")
    @patch("harvest_cran._write_entity_graph")
    @patch("harvest_cran.bulk_load")
    @patch("harvest_cran.download_catalog")
    def test_calls_pipeline(
        self,
        mock_download: MagicMock,
        mock_bulk: MagicMock,
        mock_write_graph: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_CRAN
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk.return_value = 3
        mock_write_graph.return_value = 3
        mock_run_log = MagicMock()
        mock_run_log.run_id = 42
        mock_run_log_cls.return_value = mock_run_log

        count = run_harvest(dsn="dbname=test")

        assert count == 3
        mock_download.assert_called_once()
        mock_bulk.assert_called_once()
        mock_write_graph.assert_called_once()
        mock_run_log.start.assert_called_once()
        mock_run_log.complete.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("harvest_cran.HarvestRunLog")
    @patch("harvest_cran.get_connection")
    @patch("harvest_cran._write_entity_graph")
    @patch("harvest_cran.bulk_load")
    @patch("harvest_cran.download_catalog")
    def test_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk: MagicMock,
        mock_write_graph: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_CRAN
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk.side_effect = RuntimeError("boom")
        mock_run_log = MagicMock()
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError):
            run_harvest()

        mock_conn.close.assert_called_once()
        mock_run_log.fail.assert_called_once()

    @patch("harvest_cran.download_catalog")
    def test_dry_run_skips_db(self, mock_download: MagicMock) -> None:
        mock_download.return_value = SAMPLE_CRAN
        count = run_harvest(dry_run=True)
        assert count == 3


class TestLargeCatalog:
    def test_parses_large_catalog(self) -> None:
        raw = {
            f"pkg_{i}": {
                "name": f"pkg_{i}",
                "latest": "1.0",
                "versions": {
                    "1.0": {
                        "Package": f"pkg_{i}",
                        "Title": f"Title {i}",
                        "License": "MIT",
                    }
                },
            }
            for i in range(20000)
        }
        entries = parse_catalog(raw)
        assert len(entries) == 20000
