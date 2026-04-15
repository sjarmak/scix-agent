"""Unit tests for scripts/analyze_citation_diff.py.

Tests the pure-logic functions (dataclass construction, markdown formatting,
coverage math) using mock data. No database connection required.

Integration tests for the populate/report SQL queries are in
test_migration_045.py and require SCIX_TEST_DSN.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analyze_citation_diff import (
    JournalBucket,
    OverallStats,
    YearBucket,
    format_markdown,
    overall_stats,
)

# ---------------------------------------------------------------------------
# OverallStats dataclass tests
# ---------------------------------------------------------------------------


class TestOverallStats:
    def test_frozen(self) -> None:
        stats = OverallStats(
            total_edges=100,
            ads_only=20,
            openalex_only=30,
            both=50,
            ads_coverage_pct=62.5,
            openalex_coverage_pct=71.43,
            overlap_pct=50.0,
        )
        with pytest.raises(AttributeError):
            stats.total_edges = 999  # type: ignore[misc]

    def test_asdict_roundtrip(self) -> None:
        stats = OverallStats(
            total_edges=100,
            ads_only=20,
            openalex_only=30,
            both=50,
            ads_coverage_pct=62.5,
            openalex_coverage_pct=71.43,
            overlap_pct=50.0,
        )
        d = asdict(stats)
        assert d["total_edges"] == 100
        assert d["both"] == 50
        assert d["ads_coverage_pct"] == 62.5

    def test_coverage_math(self) -> None:
        """Verify the coverage percentage formulas match expectations."""
        # 50 in both, 30 OA-only => ADS covers 50/(50+30) = 62.5% of OA edges
        # 50 in both, 20 ADS-only => OA covers 50/(50+20) = 71.43% of ADS edges
        stats = OverallStats(
            total_edges=100,
            ads_only=20,
            openalex_only=30,
            both=50,
            ads_coverage_pct=round(50 / max(50 + 30, 1) * 100, 2),
            openalex_coverage_pct=round(50 / max(50 + 20, 1) * 100, 2),
            overlap_pct=round(50 / max(100, 1) * 100, 2),
        )
        assert stats.ads_coverage_pct == 62.5
        assert stats.openalex_coverage_pct == 71.43
        assert stats.overlap_pct == 50.0


# ---------------------------------------------------------------------------
# YearBucket / JournalBucket dataclass tests
# ---------------------------------------------------------------------------


class TestBucketDataclasses:
    def test_year_bucket_frozen(self) -> None:
        yb = YearBucket(
            pub_year=2023,
            total_edges=1000,
            both_count=800,
            ads_only_count=100,
            openalex_only_count=100,
            overlap_pct=80.0,
        )
        assert yb.pub_year == 2023
        with pytest.raises(AttributeError):
            yb.total_edges = 0  # type: ignore[misc]

    def test_year_bucket_none_year(self) -> None:
        yb = YearBucket(
            pub_year=None,
            total_edges=50,
            both_count=25,
            ads_only_count=15,
            openalex_only_count=10,
            overlap_pct=50.0,
        )
        assert yb.pub_year is None

    def test_journal_bucket_frozen(self) -> None:
        jb = JournalBucket(
            journal="The Astrophysical Journal",
            total_edges=5000,
            both_count=4500,
            ads_only_count=300,
            openalex_only_count=200,
            overlap_pct=90.0,
        )
        assert jb.journal == "The Astrophysical Journal"
        with pytest.raises(AttributeError):
            jb.total_edges = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# overall_stats with mock cursor
# ---------------------------------------------------------------------------


class TestOverallStatsFromDb:
    def _make_mock_conn(self, row: tuple[int, ...] | None) -> MagicMock:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = row
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_empty_table(self) -> None:
        conn = self._make_mock_conn((0, 0, 0, 0))
        stats = overall_stats(conn)
        assert stats.total_edges == 0
        assert stats.ads_coverage_pct == 0.0
        assert stats.openalex_coverage_pct == 0.0
        assert stats.overlap_pct == 0.0

    def test_none_row(self) -> None:
        conn = self._make_mock_conn(None)
        stats = overall_stats(conn)
        assert stats.total_edges == 0

    def test_typical_values(self) -> None:
        # total=1000, ads_only=200, oa_only=300, both=500
        conn = self._make_mock_conn((1000, 200, 300, 500))
        stats = overall_stats(conn)
        assert stats.total_edges == 1000
        assert stats.ads_only == 200
        assert stats.openalex_only == 300
        assert stats.both == 500
        # ads_coverage = 500 / (500+300) * 100 = 62.5
        assert stats.ads_coverage_pct == 62.5
        # openalex_coverage = 500 / (500+200) * 100 = 71.43
        assert stats.openalex_coverage_pct == 71.43
        # overlap = 500 / 1000 * 100 = 50.0
        assert stats.overlap_pct == 50.0

    def test_all_overlap(self) -> None:
        # Every edge is in both sources
        conn = self._make_mock_conn((500, 0, 0, 500))
        stats = overall_stats(conn)
        assert stats.ads_coverage_pct == 100.0
        assert stats.openalex_coverage_pct == 100.0
        assert stats.overlap_pct == 100.0

    def test_no_overlap(self) -> None:
        # Zero edges overlap
        conn = self._make_mock_conn((700, 400, 300, 0))
        stats = overall_stats(conn)
        assert stats.ads_coverage_pct == 0.0
        assert stats.openalex_coverage_pct == 0.0
        assert stats.overlap_pct == 0.0


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    @pytest.fixture()
    def sample_report(self) -> dict[str, Any]:
        return {
            "overall": {
                "total_edges": 1000,
                "ads_only": 200,
                "openalex_only": 300,
                "both": 500,
                "ads_coverage_pct": 62.5,
                "openalex_coverage_pct": 71.43,
                "overlap_pct": 50.0,
            },
            "by_year": [
                {
                    "pub_year": 2022,
                    "total_edges": 400,
                    "both_count": 200,
                    "ads_only_count": 100,
                    "openalex_only_count": 100,
                    "overlap_pct": 50.0,
                },
                {
                    "pub_year": 2023,
                    "total_edges": 600,
                    "both_count": 300,
                    "ads_only_count": 100,
                    "openalex_only_count": 200,
                    "overlap_pct": 50.0,
                },
            ],
            "by_journal": [
                {
                    "journal": "The Astrophysical Journal",
                    "total_edges": 500,
                    "both_count": 400,
                    "ads_only_count": 50,
                    "openalex_only_count": 50,
                    "overlap_pct": 80.0,
                },
            ],
        }

    def test_contains_title(self, sample_report: dict[str, Any]) -> None:
        md = format_markdown(sample_report)
        assert "# Citation Graph Cross-Validation: ADS vs OpenAlex" in md

    def test_contains_overall_stats(self, sample_report: dict[str, Any]) -> None:
        md = format_markdown(sample_report)
        assert "1,000" in md  # total_edges formatted with comma
        assert "62.50%" in md  # ads_coverage_pct
        assert "71.43%" in md  # openalex_coverage_pct

    def test_contains_year_table(self, sample_report: dict[str, Any]) -> None:
        md = format_markdown(sample_report)
        assert "## Per-Year Edge Coverage" in md
        assert "2022" in md
        assert "2023" in md

    def test_contains_journal_table(self, sample_report: dict[str, Any]) -> None:
        md = format_markdown(sample_report)
        assert "## Per-Journal Edge Coverage" in md
        assert "The Astrophysical Journal" in md

    def test_empty_report(self) -> None:
        report: dict[str, Any] = {
            "overall": {
                "total_edges": 0,
                "ads_only": 0,
                "openalex_only": 0,
                "both": 0,
                "ads_coverage_pct": 0.0,
                "openalex_coverage_pct": 0.0,
                "overlap_pct": 0.0,
            },
            "by_year": [],
            "by_journal": [],
        }
        md = format_markdown(report)
        assert "# Citation Graph Cross-Validation" in md
        assert "Per-Year" not in md
        assert "Per-Journal" not in md

    def test_null_year_renders_as_na(self) -> None:
        report: dict[str, Any] = {
            "overall": {
                "total_edges": 10,
                "ads_only": 5,
                "openalex_only": 3,
                "both": 2,
                "ads_coverage_pct": 40.0,
                "openalex_coverage_pct": 28.57,
                "overlap_pct": 20.0,
            },
            "by_year": [
                {
                    "pub_year": None,
                    "total_edges": 10,
                    "both_count": 2,
                    "ads_only_count": 5,
                    "openalex_only_count": 3,
                    "overlap_pct": 20.0,
                },
            ],
            "by_journal": [],
        }
        md = format_markdown(report)
        assert "N/A" in md

    def test_null_journal_renders_as_na(self) -> None:
        report: dict[str, Any] = {
            "overall": {
                "total_edges": 10,
                "ads_only": 5,
                "openalex_only": 3,
                "both": 2,
                "ads_coverage_pct": 40.0,
                "openalex_coverage_pct": 28.57,
                "overlap_pct": 20.0,
            },
            "by_year": [],
            "by_journal": [
                {
                    "journal": None,
                    "total_edges": 10,
                    "both_count": 2,
                    "ads_only_count": 5,
                    "openalex_only_count": 3,
                    "overlap_pct": 20.0,
                },
            ],
        }
        md = format_markdown(report)
        assert "N/A" in md


# ---------------------------------------------------------------------------
# Populate dry-run test
# ---------------------------------------------------------------------------


class TestPopulateDryRun:
    def test_dry_run_does_not_execute(self) -> None:
        """--dry-run should log SQL but not call cursor.execute()."""
        from analyze_citation_diff import populate

        mock_conn = MagicMock()
        populate(mock_conn, dry_run=True)
        # In dry-run mode, we should never open a cursor
        mock_conn.cursor.assert_not_called()


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    def test_no_mode_raises_error(self) -> None:
        """Running without --populate or --report should fail."""
        from analyze_citation_diff import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["analyze_citation_diff.py"]):
                main()
        assert exc_info.value.code == 2  # argparse error exit code


# ---------------------------------------------------------------------------
# JSON serialization roundtrip
# ---------------------------------------------------------------------------


class TestJsonRoundtrip:
    def test_report_is_json_serializable(self) -> None:
        """The report dict from generate_report should serialize cleanly."""
        report = {
            "overall": asdict(
                OverallStats(
                    total_edges=100,
                    ads_only=20,
                    openalex_only=30,
                    both=50,
                    ads_coverage_pct=62.5,
                    openalex_coverage_pct=71.43,
                    overlap_pct=50.0,
                )
            ),
            "by_year": [
                asdict(
                    YearBucket(
                        pub_year=2023,
                        total_edges=100,
                        both_count=50,
                        ads_only_count=20,
                        openalex_only_count=30,
                        overlap_pct=50.0,
                    )
                )
            ],
            "by_journal": [
                asdict(
                    JournalBucket(
                        journal="ApJ",
                        total_edges=80,
                        both_count=60,
                        ads_only_count=10,
                        openalex_only_count=10,
                        overlap_pct=75.0,
                    )
                )
            ],
        }
        serialized = json.dumps(report, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["overall"]["total_edges"] == 100
        assert deserialized["by_year"][0]["pub_year"] == 2023
        assert deserialized["by_journal"][0]["journal"] == "ApJ"
