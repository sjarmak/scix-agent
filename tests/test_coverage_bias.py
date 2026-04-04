"""Tests for coverage bias analysis script.

Unit tests mock DB queries and verify report generation logic.
No database required. Matplotlib is mocked so tests pass without it installed.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test via sys.path manipulation matching the script convention
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Ensure matplotlib is available (mocked if not installed) before importing the script.
# The script only imports matplotlib lazily in _get_plt(), but tests that call
# generate_figures need a mock in place.
try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    _mpl_mod = types.ModuleType("matplotlib")
    _mpl_mod.use = MagicMock()  # type: ignore[attr-defined]
    _pyplot_mod = types.ModuleType("matplotlib.pyplot")
    _mock_fig = MagicMock()
    _mock_ax = MagicMock()
    _pyplot_mod.subplots = MagicMock(return_value=(_mock_fig, _mock_ax))  # type: ignore[attr-defined]
    _pyplot_mod.close = MagicMock()  # type: ignore[attr-defined]
    sys.modules["matplotlib"] = _mpl_mod
    sys.modules["matplotlib.pyplot"] = _pyplot_mod

from coverage_bias_analysis import (
    DistributionRow,
    generate_figures,
    generate_report,
    get_arxiv_distribution,
    get_citation_distribution,
    get_journal_distribution,
    get_year_distribution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_year_dist() -> list[DistributionRow]:
    return [
        DistributionRow(
            label="2021", total=1000, with_body=200, without_body=800, pct_with_body=20.0
        ),
        DistributionRow(
            label="2022", total=1100, with_body=330, without_body=770, pct_with_body=30.0
        ),
        DistributionRow(
            label="2023", total=1200, with_body=240, without_body=960, pct_with_body=20.0
        ),
    ]


@pytest.fixture()
def sample_arxiv_dist() -> list[DistributionRow]:
    return [
        DistributionRow(
            label="astro-ph.GA", total=500, with_body=150, without_body=350, pct_with_body=30.0
        ),
        DistributionRow(
            label="astro-ph.SR", total=400, with_body=80, without_body=320, pct_with_body=20.0
        ),
        DistributionRow(
            label="hep-th", total=300, with_body=30, without_body=270, pct_with_body=10.0
        ),
    ]


@pytest.fixture()
def sample_citation_dist() -> list[DistributionRow]:
    return [
        DistributionRow(
            label="0", total=2000, with_body=200, without_body=1800, pct_with_body=10.0
        ),
        DistributionRow(
            label="1-5", total=1500, with_body=300, without_body=1200, pct_with_body=20.0
        ),
        DistributionRow(
            label="6-20", total=800, with_body=240, without_body=560, pct_with_body=30.0
        ),
    ]


@pytest.fixture()
def sample_journal_dist() -> list[DistributionRow]:
    return [
        DistributionRow(
            label="ApJ", total=600, with_body=180, without_body=420, pct_with_body=30.0
        ),
        DistributionRow(
            label="MNRAS", total=500, with_body=100, without_body=400, pct_with_body=20.0
        ),
        DistributionRow(label="A&A", total=400, with_body=60, without_body=340, pct_with_body=15.0),
    ]


# ---------------------------------------------------------------------------
# Distribution row tests
# ---------------------------------------------------------------------------


class TestDistributionRow:
    def test_frozen(self) -> None:
        row = DistributionRow(
            label="2021", total=100, with_body=20, without_body=80, pct_with_body=20.0
        )
        with pytest.raises(AttributeError):
            row.label = "2022"  # type: ignore[misc]

    def test_fields(self) -> None:
        row = DistributionRow(
            label="test", total=50, with_body=10, without_body=40, pct_with_body=20.0
        )
        assert row.label == "test"
        assert row.total == 50
        assert row.with_body == 10
        assert row.without_body == 40
        assert row.pct_with_body == 20.0


# ---------------------------------------------------------------------------
# DB query function tests (mocked)
# ---------------------------------------------------------------------------


def _mock_conn_with_rows(rows: list[tuple]) -> MagicMock:
    """Create a mock psycopg connection returning the given rows."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


class TestGetYearDistribution:
    def test_returns_distribution_rows(self) -> None:
        conn = _mock_conn_with_rows([(2021, 1000, 200, 800), (2022, 1100, 330, 770)])
        result = get_year_distribution(conn)
        assert len(result) == 2
        assert result[0].label == "2021"
        assert result[0].total == 1000
        assert result[0].with_body == 200
        assert result[0].pct_with_body == 20.0

    def test_empty_result(self) -> None:
        conn = _mock_conn_with_rows([])
        result = get_year_distribution(conn)
        assert result == []

    def test_zero_total_no_division_error(self) -> None:
        conn = _mock_conn_with_rows([(2021, 0, 0, 0)])
        result = get_year_distribution(conn)
        assert result[0].pct_with_body == 0.0


class TestGetArxivDistribution:
    def test_returns_distribution_rows(self) -> None:
        conn = _mock_conn_with_rows([("astro-ph.GA", 500, 150, 350)])
        result = get_arxiv_distribution(conn)
        assert len(result) == 1
        assert result[0].label == "astro-ph.GA"
        assert result[0].pct_with_body == 30.0

    def test_empty_result(self) -> None:
        conn = _mock_conn_with_rows([])
        result = get_arxiv_distribution(conn)
        assert result == []


class TestGetCitationDistribution:
    def test_returns_distribution_rows(self) -> None:
        conn = _mock_conn_with_rows([("0", 2000, 200, 1800), ("1-5", 1500, 300, 1200)])
        result = get_citation_distribution(conn)
        assert len(result) == 2
        assert result[0].label == "0"
        assert result[1].label == "1-5"

    def test_empty_result(self) -> None:
        conn = _mock_conn_with_rows([])
        result = get_citation_distribution(conn)
        assert result == []


class TestGetJournalDistribution:
    def test_returns_distribution_rows(self) -> None:
        conn = _mock_conn_with_rows([("ApJ", 600, 180, 420)])
        result = get_journal_distribution(conn)
        assert len(result) == 1
        assert result[0].label == "ApJ"
        assert result[0].pct_with_body == 30.0

    def test_empty_result(self) -> None:
        conn = _mock_conn_with_rows([])
        result = get_journal_distribution(conn)
        assert result == []


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_contains_all_sections(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        report = generate_report(
            sample_year_dist, sample_arxiv_dist, sample_citation_dist, sample_journal_dist
        )
        assert "# Full-Text Coverage Bias Analysis" in report
        assert "## Year Distribution" in report
        assert "## arXiv Class Distribution" in report
        assert "## Citation Count Distribution" in report
        assert "## Journal Distribution" in report

    def test_report_contains_summary_stats(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        report = generate_report(
            sample_year_dist, sample_arxiv_dist, sample_citation_dist, sample_journal_dist
        )
        assert "## Summary" in report
        assert "3,300" in report  # total papers: 1000+1100+1200
        assert "770" in report  # with_body: 200+330+240

    def test_report_contains_table_rows(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        report = generate_report(
            sample_year_dist, sample_arxiv_dist, sample_citation_dist, sample_journal_dist
        )
        # Year table
        assert "| 2021 |" in report
        assert "| 2022 |" in report
        # arXiv table
        assert "| astro-ph.GA |" in report
        # Journal table
        assert "| ApJ |" in report
        assert "| MNRAS |" in report

    def test_report_writes_to_file(
        self,
        tmp_path: Path,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        output_path = tmp_path / "report.md"
        report = generate_report(
            sample_year_dist,
            sample_arxiv_dist,
            sample_citation_dist,
            sample_journal_dist,
            output_path=output_path,
        )
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert content == report

    def test_report_includes_figure_refs_when_figures_dir(
        self,
        tmp_path: Path,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        figures_dir = tmp_path / "figures"
        report = generate_report(
            sample_year_dist,
            sample_arxiv_dist,
            sample_citation_dist,
            sample_journal_dist,
            figures_dir=figures_dir,
        )
        assert "![Year Distribution]" in report
        assert "![arXiv Distribution]" in report
        assert "![Citation Distribution]" in report
        assert "![Journal Distribution]" in report

    def test_report_no_figure_refs_without_figures_dir(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        report = generate_report(
            sample_year_dist, sample_arxiv_dist, sample_citation_dist, sample_journal_dist
        )
        assert "![" not in report

    def test_empty_distributions(self) -> None:
        report = generate_report([], [], [], [])
        assert "# Full-Text Coverage Bias Analysis" in report
        assert "**Total papers**: 0" in report


# ---------------------------------------------------------------------------
# Figure generation tests
# ---------------------------------------------------------------------------


class TestGenerateFigures:
    @staticmethod
    def _mock_plt(figures_dir: Path) -> MagicMock:
        """Build a mock plt whose savefig creates real empty PNG files."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        def _savefig(path: str | Path, **kwargs: object) -> None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")

        mock_fig.savefig.side_effect = _savefig
        return mock_plt

    def test_creates_figure_files(
        self,
        tmp_path: Path,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        figures_dir = tmp_path / "figures"
        mock_plt = self._mock_plt(figures_dir)
        with patch("coverage_bias_analysis._get_plt", return_value=mock_plt):
            paths = generate_figures(
                sample_year_dist,
                sample_arxiv_dist,
                sample_citation_dist,
                sample_journal_dist,
                figures_dir,
            )
        assert figures_dir.exists()
        # Should create 8 figures (2 per dimension: counts + pct)
        assert len(paths) == 8
        for path in paths.values():
            assert path.exists()
            assert path.suffix == ".png"

    def test_empty_distributions_no_figures(self, tmp_path: Path) -> None:
        figures_dir = tmp_path / "figures"
        paths = generate_figures([], [], [], [], figures_dir)
        assert len(paths) == 0

    def test_partial_distributions(
        self,
        tmp_path: Path,
        sample_year_dist: list[DistributionRow],
    ) -> None:
        figures_dir = tmp_path / "figures"
        mock_plt = self._mock_plt(figures_dir)
        with patch("coverage_bias_analysis._get_plt", return_value=mock_plt):
            paths = generate_figures(sample_year_dist, [], [], [], figures_dir)
        # Only year charts should be generated
        assert len(paths) == 2
        assert "year_counts" in paths
        assert "year_pct" in paths


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImport:
    def test_module_importable(self) -> None:
        """Verify the script can be imported as a module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "coverage_bias_analysis",
            str(Path(__file__).resolve().parent.parent / "scripts" / "coverage_bias_analysis.py"),
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert mod is not None
