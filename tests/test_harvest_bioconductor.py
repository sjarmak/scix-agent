"""Tests for the Bioconductor package catalog harvester."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_bioconductor import (
    DEFAULT_RELEASE,
    SOURCE,
    _is_production_dsn,
    catalog_url,
    download_catalog,
    parse_catalog,
    run_harvest,
)

SAMPLE_BIOC: dict[str, Any] = {
    "DESeq2": {
        "Package": "DESeq2",
        "Version": "1.42.0",
        "Title": "Differential gene expression analysis based on the negative binomial distribution",
        "Description": "Estimate variance-mean dependence in count data.",
        "License": "LGPL (>= 3)",
        "URL": "https://github.com/thelovelab/DESeq2",
        "biocViews": ["Bayesian", "ChIPSeq", "DifferentialExpression", "GeneExpression"],
        "Author": "Michael Love et al.",
        "Maintainer": "Michael Love",
        "git_url": "https://git.bioconductor.org/packages/DESeq2",
    },
    "limma": {
        "Package": "limma",
        "Version": "3.58.1",
        "Title": "Linear Models for Microarray Data",
        "License": "GPL (>=2)",
        "biocViews": ["Microarray", "DifferentialExpression"],
        "URL": "https://bioinf.wehi.edu.au/limma",
    },
    "edgeR": {
        "Package": "edgeR",
        "Version": "4.0.0",
        "Title": "Empirical Analysis of Digital Gene Expression Data",
        "License": "GPL (>=2)",
        "biocViews": "RNASeq, DifferentialExpression",
    },
}


def _mock_response(payload: Any) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestCatalogURL:
    def test_default_url(self) -> None:
        url = catalog_url()
        assert url == f"https://bioconductor.org/packages/json/{DEFAULT_RELEASE}/bioc/packages.json"

    def test_custom_release(self) -> None:
        url = catalog_url(release="3.18")
        assert "3.18" in url

    def test_custom_repo(self) -> None:
        url = catalog_url(repo="data/annotation")
        assert "data/annotation" in url


class TestProductionDsn:
    def test_production_dsn_detected(self) -> None:
        assert _is_production_dsn("dbname=scix")

    def test_test_dsn_safe(self) -> None:
        assert not _is_production_dsn("dbname=scix_test")


class TestParseCatalog:
    def test_parses_all_valid_entries(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        names = {e["canonical_name"] for e in entries}
        assert names == {"DESeq2", "limma", "edgeR"}

    def test_required_keys(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
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
                assert key in entry

    def test_entity_type_software(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        assert all(e["entity_type"] == "software" for e in entries)

    def test_source_bioconductor(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        assert all(e["source"] == SOURCE for e in entries)

    def test_biocviews_in_properties(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        deseq2 = next(e for e in entries if e["canonical_name"] == "DESeq2")
        assert "biocviews" in deseq2["properties"]
        assert "DifferentialExpression" in deseq2["properties"]["biocviews"]

    def test_biocviews_string_split(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        edger = next(e for e in entries if e["canonical_name"] == "edgeR")
        assert sorted(edger["properties"]["biocviews"]) == [
            "DifferentialExpression",
            "RNASeq",
        ]

    def test_homepage_in_properties(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        deseq2 = next(e for e in entries if e["canonical_name"] == "DESeq2")
        assert deseq2["properties"]["homepage"] == "https://github.com/thelovelab/DESeq2"

    def test_release_recorded(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC, release="3.18")
        for entry in entries:
            assert entry["properties"]["release"] == "3.18"

    def test_lowercase_alias_when_mixed_case(self) -> None:
        entries = parse_catalog(SAMPLE_BIOC)
        deseq2 = next(e for e in entries if e["canonical_name"] == "DESeq2")
        assert "deseq2" in deseq2["aliases"]

    def test_skips_missing_name(self) -> None:
        bad = {"": {"Version": "1.0"}}
        assert parse_catalog(bad) == []

    def test_empty(self) -> None:
        assert parse_catalog({}) == []


class TestDownloadCatalog:
    @patch("harvest_bioconductor._get_client")
    def test_returns_dict(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_response(SAMPLE_BIOC)
        mock_get_client.return_value = mock_client
        result = download_catalog()
        assert "DESeq2" in result

    @patch("harvest_bioconductor._get_client")
    def test_raises_on_non_dict(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_response([1, 2, 3])
        mock_get_client.return_value = mock_client
        with pytest.raises(RuntimeError):
            download_catalog()


class TestRunHarvest:
    @patch("harvest_bioconductor.HarvestRunLog")
    @patch("harvest_bioconductor.get_connection")
    @patch("harvest_bioconductor._write_entity_graph")
    @patch("harvest_bioconductor.bulk_load")
    @patch("harvest_bioconductor.download_catalog")
    def test_pipeline(
        self,
        mock_download: MagicMock,
        mock_bulk: MagicMock,
        mock_write_graph: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_BIOC
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
        mock_run_log.complete.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("harvest_bioconductor.download_catalog")
    def test_dry_run(self, mock_download: MagicMock) -> None:
        mock_download.return_value = SAMPLE_BIOC
        count = run_harvest(dry_run=True)
        assert count == 3

    @patch("harvest_bioconductor.HarvestRunLog")
    @patch("harvest_bioconductor.get_connection")
    @patch("harvest_bioconductor._write_entity_graph")
    @patch("harvest_bioconductor.bulk_load")
    @patch("harvest_bioconductor.download_catalog")
    def test_closes_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk: MagicMock,
        mock_write_graph: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_BIOC
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk.side_effect = RuntimeError("boom")
        mock_run_log = MagicMock()
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError):
            run_harvest()

        mock_conn.close.assert_called_once()
