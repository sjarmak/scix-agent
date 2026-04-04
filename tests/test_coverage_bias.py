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
    CitationCompletenessResult,
    CorpusSummary,
    DistributionRow,
    FieldCompletenessRow,
    generate_figures,
    generate_report,
    get_arxiv_distribution,
    get_citation_completeness,
    get_citation_distribution,
    get_corpus_summary,
    get_database_distribution,
    get_doctype_distribution,
    get_field_completeness,
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


@pytest.fixture()
def sample_doctype_dist() -> list[DistributionRow]:
    return [
        DistributionRow(
            label="article", total=2000, with_body=600, without_body=1400, pct_with_body=30.0
        ),
        DistributionRow(
            label="eprint", total=1000, with_body=400, without_body=600, pct_with_body=40.0
        ),
        DistributionRow(
            label="catalog", total=100, with_body=0, without_body=100, pct_with_body=0.0
        ),
    ]


@pytest.fixture()
def sample_database_dist() -> list[DistributionRow]:
    return [
        DistributionRow(
            label="astronomy", total=2500, with_body=700, without_body=1800, pct_with_body=28.0
        ),
        DistributionRow(
            label="physics", total=1200, with_body=200, without_body=1000, pct_with_body=16.67
        ),
        DistributionRow(
            label="general", total=300, with_body=10, without_body=290, pct_with_body=3.33
        ),
    ]


@pytest.fixture()
def sample_field_completeness() -> list[FieldCompletenessRow]:
    return [
        FieldCompletenessRow(
            field="title", total=3000, non_null=2990, null_count=10, pct_populated=99.67
        ),
        FieldCompletenessRow(
            field="abstract", total=3000, non_null=2700, null_count=300, pct_populated=90.0
        ),
        FieldCompletenessRow(
            field="body", total=3000, non_null=600, null_count=2400, pct_populated=20.0
        ),
        FieldCompletenessRow(
            field="keywords", total=3000, non_null=1500, null_count=1500, pct_populated=50.0
        ),
    ]


@pytest.fixture()
def sample_citation_completeness() -> CitationCompletenessResult:
    return CitationCompletenessResult(
        total_edges=50000,
        edges_target_in_corpus=20000,
        edges_target_missing=30000,
        pct_target_in_corpus=40.0,
        unique_targets=30000,
        unique_targets_in_corpus=10000,
        unique_targets_missing=20000,
        pct_unique_in_corpus=33.33,
    )


@pytest.fixture()
def sample_corpus_summary() -> CorpusSummary:
    return CorpusSummary(
        total_papers=3300,
        total_with_body=770,
        total_citation_edges=50000,
        total_embeddings=3000,
        year_min=2021,
        year_max=2023,
        median_citation_count=5.0,
        median_reference_count=25.0,
    )


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


class TestGetDoctypeDistribution:
    def test_returns_distribution_rows(self) -> None:
        conn = _mock_conn_with_rows([("article", 2000, 600, 1400), ("eprint", 1000, 400, 600)])
        result = get_doctype_distribution(conn)
        assert len(result) == 2
        assert result[0].label == "article"
        assert result[0].pct_with_body == 30.0

    def test_empty_result(self) -> None:
        conn = _mock_conn_with_rows([])
        result = get_doctype_distribution(conn)
        assert result == []


class TestGetDatabaseDistribution:
    def test_returns_distribution_rows(self) -> None:
        conn = _mock_conn_with_rows([("astronomy", 2500, 700, 1800)])
        result = get_database_distribution(conn)
        assert len(result) == 1
        assert result[0].label == "astronomy"
        assert result[0].pct_with_body == 28.0

    def test_empty_result(self) -> None:
        conn = _mock_conn_with_rows([])
        result = get_database_distribution(conn)
        assert result == []


class TestGetFieldCompleteness:
    def test_returns_completeness_rows(self) -> None:
        # The query returns a single row with total + one count per field (23 fields)
        # total=1000, then 23 populated counts
        populated_counts = [
            990,  # title
            900,  # abstract
            200,  # body
            980,  # year
            970,  # doctype
            960,  # pub
            950,  # first_author
            800,  # citation_count
            700,  # read_count
            600,  # reference_count
            950,  # pubdate
            300,  # lang
            200,  # copyright
            940,  # authors
            500,  # affiliations
            400,  # keywords
            350,  # arxiv_class
            900,  # database
            600,  # doi
            700,  # bibstem
            100,  # bibgroup
            150,  # orcid_pub
            100,  # orcid_user
        ]
        row = (1000, *populated_counts)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = row
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = get_field_completeness(mock_conn)
        assert len(result) == 23
        assert result[0].field == "title"
        assert result[0].total == 1000
        assert result[0].non_null == 990
        assert result[0].null_count == 10
        assert result[0].pct_populated == 99.0

    def test_empty_table(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = get_field_completeness(mock_conn)
        assert result == []


class TestGetCitationCompleteness:
    def test_returns_result_exact(self) -> None:
        """Test exact computation path (small table / sample_size=0)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First fetchone: total count; second: exact query with all 4 columns
        mock_cursor.fetchone.side_effect = [
            (100,),  # total edges (small enough for exact path)
            (100, 40, 60, 20),  # exact: total, in_corpus, unique, unique_in
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = get_citation_completeness(mock_conn, sample_size=0)
        assert result.total_edges == 100
        assert result.edges_target_in_corpus == 40
        assert result.edges_target_missing == 60
        assert result.pct_target_in_corpus == 40.0
        assert result.unique_targets == 60
        assert result.unique_targets_in_corpus == 20

    def test_returns_result_sampled(self) -> None:
        """Test sampling path (large table)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First fetchone: total count; second: sample results
        mock_cursor.fetchone.side_effect = [
            (1_000_000,),  # total edges (large, triggers sampling)
            (100000, 40000, 60000, 20000),  # sample: edges, in_corpus, unique, unique_in
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = get_citation_completeness(mock_conn, sample_size=100_000)
        assert result.total_edges == 1_000_000
        assert result.pct_target_in_corpus == 40.0
        assert result.pct_unique_in_corpus == 33.33

    def test_empty_result(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = get_citation_completeness(mock_conn)
        assert result.total_edges == 0
        assert result.pct_target_in_corpus == 0.0


class TestGetCorpusSummary:
    def test_returns_summary(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First call: main papers query
        # Second call: edge count
        # Third call: embedding count
        mock_cursor.fetchone.side_effect = [
            (5000, 1000, 2021, 2025, 5.0, 25.0),  # papers summary
            (50000,),  # edge count
            (4000,),  # embedding count
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = get_corpus_summary(mock_conn)
        assert result.total_papers == 5000
        assert result.total_with_body == 1000
        assert result.total_citation_edges == 50000
        assert result.total_embeddings == 4000
        assert result.year_min == 2021
        assert result.year_max == 2025
        assert result.median_citation_count == 5.0
        assert result.median_reference_count == 25.0


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
        assert "# SciX Corpus Coverage Bias Analysis" in report
        assert "## Year Distribution" in report
        assert "## arXiv Class Distribution" in report
        assert "## Citation Count Distribution" in report
        assert "## Journal Distribution" in report

    def test_report_contains_corpus_summary_from_year_dist(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
    ) -> None:
        report = generate_report(
            sample_year_dist, sample_arxiv_dist, sample_citation_dist, sample_journal_dist
        )
        assert "## Corpus Summary" in report
        assert "3,300" in report  # total papers: 1000+1100+1200
        assert "770" in report  # with_body: 200+330+240

    def test_report_with_corpus_summary_object(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
        sample_corpus_summary: CorpusSummary,
    ) -> None:
        report = generate_report(
            sample_year_dist,
            sample_arxiv_dist,
            sample_citation_dist,
            sample_journal_dist,
            corpus_summary=sample_corpus_summary,
        )
        assert "3,300" in report
        assert "50,000" in report  # citation edges
        assert "3,000" in report  # embeddings
        assert "2021 -- 2023" in report

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
        assert "# SciX Corpus Coverage Bias Analysis" in report
        assert "**Total papers**: 0" in report

    def test_report_with_doctype_and_database(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
        sample_doctype_dist: list[DistributionRow],
        sample_database_dist: list[DistributionRow],
    ) -> None:
        report = generate_report(
            sample_year_dist,
            sample_arxiv_dist,
            sample_citation_dist,
            sample_journal_dist,
            doctype_dist=sample_doctype_dist,
            database_dist=sample_database_dist,
        )
        assert "## Document Type Distribution" in report
        assert "| article |" in report
        assert "## Database (Discipline) Distribution" in report
        assert "| astronomy |" in report

    def test_report_with_field_completeness(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
        sample_field_completeness: list[FieldCompletenessRow],
    ) -> None:
        report = generate_report(
            sample_year_dist,
            sample_arxiv_dist,
            sample_citation_dist,
            sample_journal_dist,
            field_completeness=sample_field_completeness,
        )
        assert "## Field Completeness" in report
        assert "| title |" in report
        assert "| abstract |" in report
        assert "99.7%" in report  # title pct_populated

    def test_report_with_citation_completeness(
        self,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
        sample_citation_completeness: CitationCompletenessResult,
    ) -> None:
        report = generate_report(
            sample_year_dist,
            sample_arxiv_dist,
            sample_citation_dist,
            sample_journal_dist,
            citation_completeness=sample_citation_completeness,
        )
        assert "## Citation Network Completeness" in report
        assert "50,000" in report  # total edges
        assert "40.0%" in report  # pct in corpus


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

    def test_creates_all_figures_with_new_dimensions(
        self,
        tmp_path: Path,
        sample_year_dist: list[DistributionRow],
        sample_arxiv_dist: list[DistributionRow],
        sample_citation_dist: list[DistributionRow],
        sample_journal_dist: list[DistributionRow],
        sample_doctype_dist: list[DistributionRow],
        sample_database_dist: list[DistributionRow],
        sample_field_completeness: list[FieldCompletenessRow],
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
                doctype_dist=sample_doctype_dist,
                database_dist=sample_database_dist,
                field_completeness=sample_field_completeness,
            )
        # 8 original + 2 doctype + 2 database + 1 field_completeness = 13
        assert len(paths) == 13
        assert "doctype_counts" in paths
        assert "database_pct" in paths
        assert "field_completeness" in paths

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
