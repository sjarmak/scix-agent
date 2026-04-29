"""Tests for the scientific-PyPI harvester."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_pypi_scientific import (
    SOURCE,
    _extract_topics,
    _has_scientific_classifier,
    _is_production_dsn,
    download_simple_index,
    fetch_package_json,
    harvest_candidates,
    parse_pypi_doc,
)

SAMPLE_NUMPY: dict[str, Any] = {
    "info": {
        "name": "numpy",
        "version": "2.0.0",
        "summary": "Fundamental package for array computing in Python",
        "license": "BSD",
        "home_page": "https://numpy.org",
        "author": "Travis E. Oliphant et al.",
        "classifiers": [
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
    }
}

SAMPLE_REQUESTS_NONSCI: dict[str, Any] = {
    "info": {
        "name": "requests",
        "version": "2.31.0",
        "summary": "Python HTTP for Humans.",
        "license": "Apache 2.0",
        "home_page": "https://requests.readthedocs.io",
        "classifiers": [
            "Topic :: Internet :: WWW/HTTP",
            "Topic :: Software Development :: Libraries",
        ],
    }
}

SAMPLE_PROJECT_URLS_FALLBACK: dict[str, Any] = {
    "info": {
        "name": "scikit-learn",
        "version": "1.4.0",
        "summary": "A set of python modules for machine learning",
        "license": "new BSD",
        "home_page": "",
        "project_urls": {
            "Homepage": "https://scikit-learn.org",
            "Source": "https://github.com/scikit-learn/scikit-learn",
        },
        "classifiers": [
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    }
}


class TestProductionDsn:
    def test_production_dsn_detected(self) -> None:
        assert _is_production_dsn("dbname=scix")

    def test_test_dsn_safe(self) -> None:
        assert not _is_production_dsn("dbname=scix_test")


class TestClassifierPredicates:
    def test_scientific_classifier_present(self) -> None:
        assert _has_scientific_classifier(["Topic :: Scientific/Engineering"])
        assert _has_scientific_classifier(["Topic :: Scientific/Engineering :: Astronomy"])

    def test_scientific_classifier_absent(self) -> None:
        assert not _has_scientific_classifier(["Topic :: Internet :: WWW/HTTP"])

    def test_no_classifiers(self) -> None:
        assert not _has_scientific_classifier([])

    def test_extract_topics(self) -> None:
        topics = _extract_topics(
            [
                "Topic :: Scientific/Engineering",
                "Topic :: Scientific/Engineering :: Mathematics",
                "Development Status :: 5",
            ]
        )
        assert "Scientific/Engineering" in topics
        assert "Scientific/Engineering / Mathematics" in topics
        assert all("Development Status" not in t for t in topics)


class TestParsePypiDoc:
    def test_parses_scientific_package(self) -> None:
        record = parse_pypi_doc(SAMPLE_NUMPY, fallback_name="numpy")
        assert record is not None
        assert record["canonical_name"] == "numpy"
        assert record["source"] == SOURCE
        assert record["external_id"] == "numpy"
        assert record["properties"]["homepage"] == "https://numpy.org"
        assert record["properties"]["license"] == "BSD"
        assert "topics" in record["properties"]
        assert record["properties"]["scientific_classifier"] is True

    def test_rejects_non_scientific_without_mentions(self) -> None:
        record = parse_pypi_doc(SAMPLE_REQUESTS_NONSCI, fallback_name="requests")
        assert record is None

    def test_accepts_non_scientific_when_mentioned(self) -> None:
        record = parse_pypi_doc(
            SAMPLE_REQUESTS_NONSCI,
            fallback_name="requests",
            mention_count=42,
        )
        assert record is not None
        assert record["canonical_name"] == "requests"
        assert record["properties"]["paper_mentions"] == 42
        assert "scientific_classifier" not in record["properties"]

    def test_homepage_falls_back_to_project_urls(self) -> None:
        record = parse_pypi_doc(
            SAMPLE_PROJECT_URLS_FALLBACK,
            fallback_name="scikit-learn",
        )
        assert record is not None
        assert record["properties"]["homepage"] == "https://scikit-learn.org"

    def test_returns_none_on_empty_info(self) -> None:
        assert parse_pypi_doc({"info": None}, fallback_name="x") is None
        assert parse_pypi_doc({}, fallback_name="x") is None

    def test_metadata_mirrors_properties(self) -> None:
        record = parse_pypi_doc(SAMPLE_NUMPY, fallback_name="numpy")
        assert record is not None
        assert record["metadata"] == record["properties"]

    def test_lowercase_alias_for_mixed_case(self) -> None:
        doc = {
            "info": {
                "name": "MyPackage",
                "summary": "x",
                "classifiers": ["Topic :: Scientific/Engineering"],
            }
        }
        record = parse_pypi_doc(doc, fallback_name="MyPackage")
        assert record is not None
        assert "mypackage" in record["aliases"]


class TestDownloadSimpleIndex:
    @patch("harvest_pypi_scientific._get_client")
    def test_returns_lowercase_set(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "projects": [
                {"name": "NumPy"},
                {"name": "scikit-learn"},
                {"name": ""},
            ]
        }
        mock_client.get.return_value = mock_resp
        mock_get_client.return_value = mock_client

        names = download_simple_index()

        assert "numpy" in names
        assert "scikit-learn" in names
        assert "" not in names


class TestFetchPackageJson:
    @patch("harvest_pypi_scientific._get_client")
    def test_returns_doc_on_200(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = SAMPLE_NUMPY
        mock_client.get.return_value = resp
        mock_get_client.return_value = mock_client

        doc = fetch_package_json("numpy")
        assert doc == SAMPLE_NUMPY

    @patch("harvest_pypi_scientific._get_client")
    def test_returns_none_on_404(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 404
        mock_client.get.return_value = resp
        mock_get_client.return_value = mock_client

        assert fetch_package_json("nonexistent") is None

    @patch("harvest_pypi_scientific._get_client")
    def test_returns_none_on_exception(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get.side_effect = RuntimeError("network error")
        mock_get_client.return_value = mock_client

        assert fetch_package_json("numpy") is None


class TestHarvestCandidates:
    @patch("harvest_pypi_scientific.fetch_package_json")
    def test_filters_non_scientific_unless_mentioned(
        self,
        mock_fetch: MagicMock,
    ) -> None:
        def fake_fetch(name: str) -> dict[str, Any] | None:
            if name == "numpy":
                return SAMPLE_NUMPY
            if name == "requests":
                return SAMPLE_REQUESTS_NONSCI
            return None

        mock_fetch.side_effect = fake_fetch

        candidates = [("numpy", 0), ("requests", 0)]
        records = harvest_candidates(candidates, workers=2, progress_every=10)
        names = {r["canonical_name"] for r in records}
        assert names == {"numpy"}

    @patch("harvest_pypi_scientific.fetch_package_json")
    def test_keeps_mentioned_packages(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = lambda name: SAMPLE_REQUESTS_NONSCI

        candidates = [("requests", 100)]
        records = harvest_candidates(candidates, workers=1, progress_every=10)
        assert len(records) == 1
        assert records[0]["properties"]["paper_mentions"] == 100

    @patch("harvest_pypi_scientific.fetch_package_json")
    def test_deduplicates_by_canonical_name(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = lambda name: SAMPLE_NUMPY

        candidates = [("numpy", 5), ("NumPy", 5)]
        records = harvest_candidates(candidates, workers=2, progress_every=10)
        assert len(records) == 1
