"""Tests for scripts/metadata_coverage.py — metadata vs extraction coverage analysis."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metadata_coverage import (
    CoverageReport,
    TypeCoverage,
    build_delta_summary,
    compute_coverage,
    extract_extraction_entities,
    extract_metadata_entities,
    fetch_extractions_for_bibcodes,
    fetch_sample_papers,
    run_analysis,
)

# ---------------------------------------------------------------------------
# extract_metadata_entities
# ---------------------------------------------------------------------------


class TestExtractMetadataEntities:
    def test_extracts_from_columns(self) -> None:
        columns = {
            "facility": ["HST", "Keck Observatory"],
            "data": ["SDSS DR16"],
            "keyword_norm": ["Monte Carlo"],
            "bibgroup": ["CfA"],
        }
        result = extract_metadata_entities(None, columns)

        assert "instruments" in result
        assert "hubble space telescope" in result["instruments"]
        assert "keck observatory" in result["instruments"]
        # bibgroup also maps to instruments
        assert "cfa" in result["instruments"]
        assert "datasets" in result
        assert "methods" in result

    def test_extracts_from_raw_jsonb(self) -> None:
        raw = {
            "facility": ["ALMA"],
            "data": ["2MASS"],
        }
        columns = {"facility": None, "data": None, "keyword_norm": None, "bibgroup": None}
        result = extract_metadata_entities(raw, columns)

        assert "instruments" in result
        assert "atacama large millimeter array" in result["instruments"]
        assert "datasets" in result
        assert "two micron all sky survey" in result["datasets"]

    def test_merges_columns_and_raw(self) -> None:
        raw = {"facility": ["VLT"]}
        columns = {
            "facility": ["HST"],
            "data": None,
            "keyword_norm": None,
            "bibgroup": None,
        }
        result = extract_metadata_entities(raw, columns)

        assert "hubble space telescope" in result["instruments"]
        assert "very large telescope" in result["instruments"]

    def test_empty_inputs(self) -> None:
        columns = {"facility": None, "data": None, "keyword_norm": None, "bibgroup": None}
        result = extract_metadata_entities(None, columns)

        # Should return dict with empty sets or missing keys
        for etype in result.values():
            assert len(etype) == 0

    def test_skips_empty_strings(self) -> None:
        columns = {
            "facility": ["", "  ", "HST"],
            "data": None,
            "keyword_norm": None,
            "bibgroup": None,
        }
        result = extract_metadata_entities(None, columns)

        assert "instruments" in result
        assert "" not in result["instruments"]
        assert "hubble space telescope" in result["instruments"]

    def test_handles_non_string_values_in_arrays(self) -> None:
        columns = {
            "facility": [123, None, "HST"],
            "data": None,
            "keyword_norm": None,
            "bibgroup": None,
        }
        result = extract_metadata_entities(None, columns)

        assert "hubble space telescope" in result["instruments"]


# ---------------------------------------------------------------------------
# extract_extraction_entities
# ---------------------------------------------------------------------------


class TestExtractExtractionEntities:
    def test_basic_extraction(self) -> None:
        payload = {
            "entities": [
                {"name": "Hubble Space Telescope", "type": "instruments"},
                {"name": "MCMC", "type": "methods"},
            ]
        }
        result = extract_extraction_entities(payload)

        assert "instruments" in result
        assert "hubble space telescope" in result["instruments"]
        assert "methods" in result
        assert "markov chain monte carlo" in result["methods"]

    def test_empty_payload(self) -> None:
        assert extract_extraction_entities(None) == {}
        assert extract_extraction_entities({}) == {}

    def test_missing_entities_key(self) -> None:
        assert extract_extraction_entities({"other": "data"}) == {}

    def test_skips_malformed_entities(self) -> None:
        payload = {
            "entities": [
                {"name": "", "type": "instruments"},
                {"name": "HST", "type": ""},
                {"name": "VLT", "type": "instruments"},
                "not a dict",
                42,
            ]
        }
        result = extract_extraction_entities(payload)

        assert "instruments" in result
        assert "very large telescope" in result["instruments"]
        assert len(result["instruments"]) == 1

    def test_normalizes_type_to_lowercase(self) -> None:
        payload = {
            "entities": [
                {"name": "ALMA", "type": "Instruments"},
            ]
        }
        result = extract_extraction_entities(payload)
        assert "instruments" in result


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_full_overlap(self) -> None:
        meta = {"instruments": {"hubble space telescope", "alma"}}
        extr = {"instruments": {"hubble space telescope", "alma"}}
        result = compute_coverage(meta, extr)

        assert result["instruments"].overlap_count == 2
        assert result["instruments"].metadata_only_count == 0
        assert result["instruments"].extraction_only_count == 0

    def test_no_overlap(self) -> None:
        meta = {"instruments": {"hubble space telescope"}}
        extr = {"instruments": {"very large telescope"}}
        result = compute_coverage(meta, extr)

        assert result["instruments"].overlap_count == 0
        assert result["instruments"].metadata_only_count == 1
        assert result["instruments"].extraction_only_count == 1

    def test_partial_overlap(self) -> None:
        meta = {"instruments": {"hst_norm", "alma_norm"}}
        extr = {"instruments": {"alma_norm", "vlt_norm"}}
        result = compute_coverage(meta, extr)

        assert result["instruments"].overlap_count == 1
        assert result["instruments"].metadata_only_count == 1
        assert result["instruments"].extraction_only_count == 1

    def test_disjoint_types(self) -> None:
        meta = {"instruments": {"hst"}}
        extr = {"methods": {"mcmc"}}
        result = compute_coverage(meta, extr)

        assert "instruments" in result
        assert "methods" in result
        assert result["instruments"].metadata_only_count == 1
        assert result["instruments"].extraction_only_count == 0
        assert result["methods"].metadata_only_count == 0
        assert result["methods"].extraction_only_count == 1

    def test_empty_inputs(self) -> None:
        result = compute_coverage({}, {})
        assert result == {}


# ---------------------------------------------------------------------------
# build_delta_summary
# ---------------------------------------------------------------------------


class TestBuildDeltaSummary:
    def test_basic_summary(self) -> None:
        per_type = {
            "instruments": TypeCoverage(
                metadata_only_count=5,
                extraction_only_count=10,
                overlap_count=3,
            ),
            "methods": TypeCoverage(
                metadata_only_count=2,
                extraction_only_count=8,
                overlap_count=1,
            ),
        }
        summary = build_delta_summary(per_type)

        assert summary["total_metadata_entities"] == 5 + 3 + 2 + 1  # 11
        assert summary["total_extraction_entities"] == 10 + 3 + 8 + 1  # 22
        assert summary["total_overlap"] == 3 + 1  # 4
        assert summary["total_metadata_only"] == 5 + 2  # 7
        assert summary["total_extraction_only"] == 10 + 8  # 18
        # lift = extraction_only / total_metadata = 18 / 11
        assert summary["extraction_lift_ratio"] == round(18 / 11, 4)

    def test_empty_per_type(self) -> None:
        summary = build_delta_summary({})
        assert summary["total_metadata_entities"] == 0
        assert summary["total_extraction_entities"] == 0
        assert summary["extraction_lift_ratio"] == 0.0

    def test_zero_metadata(self) -> None:
        per_type = {
            "methods": TypeCoverage(
                metadata_only_count=0,
                extraction_only_count=5,
                overlap_count=0,
            ),
        }
        summary = build_delta_summary(per_type)
        assert summary["extraction_lift_ratio"] == 0.0


# ---------------------------------------------------------------------------
# TypeCoverage / CoverageReport serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_type_coverage_to_dict(self) -> None:
        tc = TypeCoverage(metadata_only_count=1, extraction_only_count=2, overlap_count=3)
        d = tc.to_dict()
        assert d == {
            "metadata_only_count": 1,
            "extraction_only_count": 2,
            "overlap_count": 3,
        }

    def test_coverage_report_to_dict(self) -> None:
        report = CoverageReport(
            sample_size=100,
            per_type_coverage={
                "instruments": TypeCoverage(
                    metadata_only_count=1,
                    extraction_only_count=2,
                    overlap_count=3,
                )
            },
            delta_summary={"total_metadata_entities": 4},
        )
        d = report.to_dict()
        assert d["sample_size"] == 100
        assert "instruments" in d["per_type_coverage"]
        assert d["per_type_coverage"]["instruments"]["overlap_count"] == 3
        assert d["delta_summary"]["total_metadata_entities"] == 4

    def test_report_json_serializable(self) -> None:
        report = CoverageReport(
            sample_size=50,
            per_type_coverage={
                "methods": TypeCoverage(
                    metadata_only_count=0,
                    extraction_only_count=5,
                    overlap_count=0,
                )
            },
            delta_summary={"extraction_lift_ratio": 0.0},
        )
        output = json.dumps(report.to_dict())
        parsed = json.loads(output)
        assert parsed["sample_size"] == 50


# ---------------------------------------------------------------------------
# fetch_sample_papers (mocked DB)
# ---------------------------------------------------------------------------


class TestFetchSamplePapers:
    def test_parses_rows_correctly(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor.fetchall.return_value = [
            (
                "2024ApJ...001",
                ["HST"],
                ["SDSS"],
                ["Monte Carlo"],
                ["CfA"],
                {"facility": ["ALMA"]},
            ),
        ]

        papers = fetch_sample_papers(mock_conn, sample_size=10)

        assert len(papers) == 1
        assert papers[0]["bibcode"] == "2024ApJ...001"
        assert papers[0]["columns"]["facility"] == ["HST"]
        assert papers[0]["raw"]["facility"] == ["ALMA"]

    def test_handles_null_raw(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor.fetchall.return_value = [
            ("2024ApJ...002", None, None, None, None, None),
        ]

        papers = fetch_sample_papers(mock_conn, sample_size=10)

        assert len(papers) == 1
        assert papers[0]["raw"] is None

    def test_handles_string_raw_json(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor.fetchall.return_value = [
            ("2024ApJ...003", None, None, None, None, '{"facility": ["VLT"]}'),
        ]

        papers = fetch_sample_papers(mock_conn, sample_size=10)

        assert papers[0]["raw"] == {"facility": ["VLT"]}


# ---------------------------------------------------------------------------
# fetch_extractions_for_bibcodes (mocked DB)
# ---------------------------------------------------------------------------


class TestFetchExtractions:
    def test_groups_by_bibcode(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor.fetchall.return_value = [
            ("bib1", {"entities": [{"name": "HST", "type": "instruments"}]}),
            ("bib1", {"entities": [{"name": "MCMC", "type": "methods"}]}),
            ("bib2", {"entities": [{"name": "VLT", "type": "instruments"}]}),
        ]

        result = fetch_extractions_for_bibcodes(mock_conn, ["bib1", "bib2"])

        assert len(result["bib1"]) == 2
        assert len(result["bib2"]) == 1

    def test_empty_bibcodes(self) -> None:
        mock_conn = MagicMock()
        result = fetch_extractions_for_bibcodes(mock_conn, [])
        assert result == {}


# ---------------------------------------------------------------------------
# run_analysis (integration with mocked DB)
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    def test_end_to_end_with_mocks(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # First call: fetch_sample_papers
        # Second call: fetch_extractions_for_bibcodes
        mock_cursor.fetchall.side_effect = [
            # Papers query
            [
                (
                    "2024ApJ...001",
                    ["HST", "Keck"],
                    None,
                    ["Monte Carlo"],
                    None,
                    None,
                ),
                (
                    "2024ApJ...002",
                    None,
                    ["SDSS"],
                    None,
                    ["CfA"],
                    {"facility": ["ALMA"]},
                ),
            ],
            # Extractions query
            [
                (
                    "2024ApJ...001",
                    {
                        "entities": [
                            {"name": "HST", "type": "instruments"},
                            {"name": "Random Forest", "type": "methods"},
                        ]
                    },
                ),
                (
                    "2024ApJ...002",
                    {
                        "entities": [
                            {"name": "SDSS", "type": "datasets"},
                            {"name": "VLT", "type": "instruments"},
                        ]
                    },
                ),
            ],
        ]

        report = run_analysis(mock_conn, sample_size=2)

        assert report.sample_size == 2
        assert "instruments" in report.per_type_coverage
        assert report.delta_summary["total_overlap"] >= 0

        # Verify JSON-serializable
        d = report.to_dict()
        json.dumps(d)

    def test_required_report_keys(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor.fetchall.side_effect = [[], []]

        report = run_analysis(mock_conn, sample_size=10)
        d = report.to_dict()

        assert "sample_size" in d
        assert "per_type_coverage" in d
        assert "delta_summary" in d
