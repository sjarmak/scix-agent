"""Tests for the PhySH techniques dictionary harvester.

Unit tests use a synthetic JSON-LD fixture mimicking PhySH structure.
No network or database access required.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_physh import (
    PHYSH_URL,
    TECHNIQUES_FACET_ID,
    download_physh,
    parse_physh_techniques,
    run_harvest,
)

# ---------------------------------------------------------------------------
# Synthetic PhySH JSON-LD fixture
# ---------------------------------------------------------------------------

_DOI = "https://doi.org/10.29172"

# Sub-facet IDs matching the constants in harvest_physh
_EXPERIMENTAL_FACET = f"{_DOI}/705f7ed8-6d0e-4b5a-a65a-4a16ca09c040"
_COMPUTATIONAL_FACET = f"{_DOI}/1e0c099a-4cd7-42c4-8a0e-8aeb0e501882"

# Concept IDs
_NUMERICAL_TECHNIQUES = f"{_DOI}/aaaa0001-0000-0000-0000-000000000001"
_MONTE_CARLO = f"{_DOI}/aaaa0002-0000-0000-0000-000000000002"
_QUANTUM_MC = f"{_DOI}/aaaa0003-0000-0000-0000-000000000003"
_DIFFUSION_QMC = f"{_DOI}/aaaa0004-0000-0000-0000-000000000004"
_SPECTROSCOPY = f"{_DOI}/aaaa0005-0000-0000-0000-000000000005"
_XRAY_SPECTROSCOPY = f"{_DOI}/aaaa0006-0000-0000-0000-000000000006"
_DENSITY_FUNCTIONAL = f"{_DOI}/aaaa0007-0000-0000-0000-000000000007"
_ORPHAN_CONCEPT = f"{_DOI}/aaaa0008-0000-0000-0000-000000000008"

SAMPLE_JSONLD: dict[str, Any] = {
    "@context": {
        "physh_rdf": "http://physh.org/rdf#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
    },
    "@graph": [
        # Techniques facet (top level)
        {
            "@id": TECHNIQUES_FACET_ID,
            "@type": "physh_rdf:Facet",
            "skos:prefLabel": {"@value": "Techniques"},
        },
        # Experimental Techniques sub-facet
        {
            "@id": _EXPERIMENTAL_FACET,
            "@type": "physh_rdf:Facet",
            "skos:prefLabel": {"@value": "Experimental Techniques"},
            "physh_rdf:contains": [
                {"@id": _SPECTROSCOPY},
            ],
        },
        # Computational Techniques sub-facet
        {
            "@id": _COMPUTATIONAL_FACET,
            "@type": "physh_rdf:Facet",
            "skos:prefLabel": {"@value": "Computational Techniques"},
            "physh_rdf:contains": [
                {"@id": _NUMERICAL_TECHNIQUES},
                {"@id": _DENSITY_FUNCTIONAL},
            ],
        },
        # Concepts: Computational tree
        {
            "@id": _NUMERICAL_TECHNIQUES,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Numerical techniques"},
            "skos:altLabel": [
                {"@value": "Numerical methods"},
                {"@value": "Computational methods"},
            ],
        },
        {
            "@id": _MONTE_CARLO,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Monte Carlo methods"},
            "skos:broader": {"@id": _NUMERICAL_TECHNIQUES},
            "skos:altLabel": {"@value": "MC methods"},
        },
        {
            "@id": _QUANTUM_MC,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Quantum Monte Carlo"},
            "skos:broader": {"@id": _MONTE_CARLO},
            "skos:scopeNote": {"@value": "A class of stochastic methods for quantum systems"},
        },
        {
            "@id": _DIFFUSION_QMC,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Diffusion quantum Monte Carlo"},
            "skos:broader": {"@id": _QUANTUM_MC},
        },
        {
            "@id": _DENSITY_FUNCTIONAL,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Density functional theory"},
            "skos:altLabel": {"@value": "DFT"},
        },
        # Concepts: Experimental tree
        {
            "@id": _SPECTROSCOPY,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Spectroscopy"},
        },
        {
            "@id": _XRAY_SPECTROSCOPY,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "X-ray spectroscopy"},
            "skos:broader": {"@id": _SPECTROSCOPY},
        },
        # An orphan concept NOT in any facet (should NOT be harvested)
        {
            "@id": _ORPHAN_CONCEPT,
            "@type": "physh_rdf:Concept",
            "skos:prefLabel": {"@value": "Orphan concept"},
        },
    ],
}


# ---------------------------------------------------------------------------
# Unit tests: parse_physh_techniques
# ---------------------------------------------------------------------------


class TestParsePhyshTechniques:
    """Unit tests for parse_physh_techniques."""

    def test_returns_correct_count(self) -> None:
        """Should find 7 technique concepts (not the orphan, not facets)."""
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        assert len(entries) == 7

    def test_excludes_orphan(self) -> None:
        names = {e["canonical_name"] for e in parse_physh_techniques(SAMPLE_JSONLD)}
        assert "Orphan concept" not in names

    def test_entity_type_is_method(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        for entry in entries:
            assert entry["entity_type"] == "method"

    def test_source_is_physh(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        for entry in entries:
            assert entry["source"] == "physh"

    def test_external_id_is_doi_uri(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        for entry in entries:
            assert entry["external_id"].startswith("https://doi.org/10.29172/")

    def test_aliases_from_alt_labels(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        numerical = next(e for e in entries if e["canonical_name"] == "Numerical techniques")
        assert "Numerical methods" in numerical["aliases"]
        assert "Computational methods" in numerical["aliases"]

    def test_single_alt_label(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        mc = next(e for e in entries if e["canonical_name"] == "Monte Carlo methods")
        assert "MC methods" in mc["aliases"]

    def test_no_alt_label_empty_aliases(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        spec = next(e for e in entries if e["canonical_name"] == "Spectroscopy")
        assert spec["aliases"] == []

    def test_parent_child_in_metadata(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        mc = next(e for e in entries if e["canonical_name"] == "Monte Carlo methods")
        assert "Numerical techniques" in mc["metadata"]["parent_names"]
        assert _NUMERICAL_TECHNIQUES in mc["metadata"]["parent_ids"]

    def test_children_in_metadata(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        numerical = next(e for e in entries if e["canonical_name"] == "Numerical techniques")
        assert "Monte Carlo methods" in numerical["metadata"]["child_names"]

    def test_hierarchy_traversal_depth(self) -> None:
        """BFS should reach Diffusion QMC (3 levels deep from seed)."""
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        names = {e["canonical_name"] for e in entries}
        assert "Diffusion quantum Monte Carlo" in names

    def test_scope_note_in_metadata(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        qmc = next(e for e in entries if e["canonical_name"] == "Quantum Monte Carlo")
        assert "stochastic" in qmc["metadata"]["description"]

    def test_facet_label_in_metadata(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        numerical = next(e for e in entries if e["canonical_name"] == "Numerical techniques")
        assert numerical["metadata"]["facet"] == "Computational Techniques"

    def test_experimental_facet_label(self) -> None:
        entries = parse_physh_techniques(SAMPLE_JSONLD)
        spec = next(e for e in entries if e["canonical_name"] == "Spectroscopy")
        assert spec["metadata"]["facet"] == "Experimental Techniques"

    def test_empty_graph_returns_empty(self) -> None:
        entries = parse_physh_techniques({"@graph": []})
        assert entries == []

    def test_missing_graph_returns_empty(self) -> None:
        entries = parse_physh_techniques({})
        assert entries == []


# ---------------------------------------------------------------------------
# Unit tests: download_physh
# ---------------------------------------------------------------------------


class TestDownloadPhysh:
    """Unit tests for download_physh with mocked HTTP and cache."""

    def test_cache_hit_skips_download(self, tmp_path: Path) -> None:
        """If cache file exists, should read from it without network."""
        cache_file = tmp_path / "physh.json.gz"
        raw = json.dumps(SAMPLE_JSONLD).encode("utf-8")
        with gzip.open(cache_file, "wb") as f:
            f.write(raw)

        result = download_physh(cache_dir=tmp_path)
        assert "@graph" in result
        assert len(result["@graph"]) == len(SAMPLE_JSONLD["@graph"])

    @patch("harvest_physh._get_client")
    def test_downloads_and_decompresses(self, mock_get_client: MagicMock) -> None:
        raw = json.dumps(SAMPLE_JSONLD).encode("utf-8")
        compressed = gzip.compress(raw)
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = compressed
        mock_client.get.return_value = mock_resp
        mock_get_client.return_value = mock_client

        result = download_physh()
        assert "@graph" in result

    @patch("harvest_physh._get_client")
    def test_caches_after_download(self, mock_get_client: MagicMock, tmp_path: Path) -> None:
        raw = json.dumps(SAMPLE_JSONLD).encode("utf-8")
        compressed = gzip.compress(raw)
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = compressed
        mock_client.get.return_value = mock_resp
        mock_get_client.return_value = mock_client

        download_physh(cache_dir=tmp_path)
        assert (tmp_path / "physh.json.gz").exists()

    @patch("harvest_physh._get_client")
    def test_raises_on_request_error(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get.side_effect = requests.RequestException("persistent failure")
        mock_get_client.return_value = mock_client
        with pytest.raises(requests.RequestException):
            download_physh()


# ---------------------------------------------------------------------------
# Unit tests: run_harvest (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestRunHarvest:
    """Unit tests for run_harvest with mocked download and DB."""

    @patch("harvest_physh.HarvestRunLog")
    @patch("harvest_physh.get_connection")
    @patch("harvest_physh.bulk_load")
    @patch("harvest_physh.download_physh")
    def test_run_harvest_calls_bulk_load(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_JSONLD
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 7
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        count = run_harvest(dsn="dbname=test")

        assert count == 7
        mock_download.assert_called_once()
        mock_bulk_load.assert_called_once()
        loaded_entries = mock_bulk_load.call_args[0][1]
        assert len(loaded_entries) == 7
        assert all(e["entity_type"] == "method" for e in loaded_entries)
        assert all(e["source"] == "physh" for e in loaded_entries)

    @patch("harvest_physh.HarvestRunLog")
    @patch("harvest_physh.get_connection")
    @patch("harvest_physh.bulk_load")
    @patch("harvest_physh.download_physh")
    def test_run_harvest_closes_connection(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_JSONLD
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 7
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        run_harvest()

        mock_conn.close.assert_called_once()

    @patch("harvest_physh.HarvestRunLog")
    @patch("harvest_physh.get_connection")
    @patch("harvest_physh.bulk_load")
    @patch("harvest_physh.download_physh")
    def test_run_harvest_closes_connection_on_error(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_download.return_value = SAMPLE_JSONLD
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.side_effect = RuntimeError("DB error")
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError):
            run_harvest()

        mock_conn.close.assert_called_once()

    @patch("harvest_physh.HarvestRunLog")
    @patch("harvest_physh.get_connection")
    @patch("harvest_physh.bulk_load")
    @patch("harvest_physh.download_physh")
    def test_run_harvest_creates_harvest_run(
        self,
        mock_download: MagicMock,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        """Verify harvest_runs record is created and completed."""
        mock_download.return_value = SAMPLE_JSONLD
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 7
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        run_harvest()

        mock_run_log_cls.assert_called_once_with(mock_conn, "physh")
        mock_run_log.start.assert_called_once()
        mock_run_log.complete.assert_called_once()
        _, kwargs = mock_run_log.complete.call_args
        assert kwargs["records_fetched"] == 7
        assert kwargs["records_upserted"] == 7
