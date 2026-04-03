"""Unit tests for src/scix/benchmark.py.

Tests dataclasses, percentile calculation, and report generation.
No database required.
"""

from __future__ import annotations

import pytest

from scix.benchmark import (
    BenchmarkResult,
    IngestionMetrics,
    PRDBenchmarkReport,
    _fmt_bytes,
    generate_markdown_report,
    percentile,
)


class TestPercentile:
    def test_empty(self) -> None:
        assert percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 99) == 5.0

    def test_two_values(self) -> None:
        assert percentile([1.0, 3.0], 50) == 2.0

    def test_p50_odd_count(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p95(self) -> None:
        data = list(range(1, 101))  # 1..100
        p95 = percentile([float(x) for x in data], 95)
        assert 95.0 <= p95 <= 96.0

    def test_p99(self) -> None:
        data = [float(x) for x in range(1, 101)]
        p99 = percentile(data, 99)
        assert 99.0 <= p99 <= 100.0

    def test_unsorted_input(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 50) == 2.0


class TestBenchmarkResult:
    def test_frozen(self) -> None:
        r = BenchmarkResult(
            name="test",
            passed=True,
            target="p95 < 100ms",
            p50_ms=10.0,
            p95_ms=50.0,
            p99_ms=80.0,
            iterations=100,
        )
        with pytest.raises(AttributeError):
            r.name = "changed"  # type: ignore[misc]

    def test_default_details(self) -> None:
        r = BenchmarkResult(
            name="test",
            passed=None,
            target="informational",
            p50_ms=0,
            p95_ms=0,
            p99_ms=0,
            iterations=0,
        )
        assert r.details == {}

    def test_with_details(self) -> None:
        r = BenchmarkResult(
            name="test",
            passed=True,
            target="p95 < 100ms",
            p50_ms=10.0,
            p95_ms=50.0,
            p99_ms=80.0,
            iterations=100,
            details={"recall_at_10": "5/5"},
        )
        assert r.details["recall_at_10"] == "5/5"


class TestIngestionMetrics:
    def test_frozen(self) -> None:
        m = IngestionMetrics(
            total_papers=5_000_000,
            total_embeddings=5_000_000,
            total_citation_edges=82_000_000,
            papers_without_abstracts=700_000,
            hnsw_index_bytes=20_000_000_000,
            table_total_bytes=40_000_000_000,
            system_ram_bytes=64_000_000_000,
            hnsw_pct_of_ram=31.3,
        )
        with pytest.raises(AttributeError):
            m.total_papers = 0  # type: ignore[misc]


class TestFmtBytes:
    def test_bytes(self) -> None:
        assert _fmt_bytes(500) == "500.0 B"

    def test_kb(self) -> None:
        assert _fmt_bytes(2048) == "2.0 KB"

    def test_gb(self) -> None:
        result = _fmt_bytes(20_000_000_000)
        assert "GB" in result

    def test_zero(self) -> None:
        assert _fmt_bytes(0) == "0.0 B"


class TestGenerateReport:
    @pytest.fixture()
    def sample_report(self) -> PRDBenchmarkReport:
        return PRDBenchmarkReport(
            timestamp="2026-04-01 12:00:00",
            results=[
                BenchmarkResult(
                    name="semantic_search",
                    passed=True,
                    target="p95 < 100ms",
                    p50_ms=15.0,
                    p95_ms=45.0,
                    p99_ms=78.0,
                    iterations=50,
                    details={"recall_at_10": "5/5", "recall_fraction": 1.0},
                ),
                BenchmarkResult(
                    name="faceted_filtering",
                    passed=False,
                    target="p95 < 50ms",
                    p50_ms=30.0,
                    p95_ms=65.0,
                    p99_ms=90.0,
                    iterations=20,
                    details={"matching_papers": 1234},
                ),
            ],
            ingestion=IngestionMetrics(
                total_papers=5_000_000,
                total_embeddings=5_000_000,
                total_citation_edges=82_000_000,
                papers_without_abstracts=700_000,
                hnsw_index_bytes=20_000_000_000,
                table_total_bytes=40_000_000_000,
                system_ram_bytes=64_000_000_000,
                hnsw_pct_of_ram=31.3,
            ),
        )

    def test_contains_header(self, sample_report: PRDBenchmarkReport) -> None:
        md = generate_markdown_report(sample_report)
        assert "# SciX PRD Benchmark Validation Report" in md

    def test_contains_timestamp(self, sample_report: PRDBenchmarkReport) -> None:
        md = generate_markdown_report(sample_report)
        assert "2026-04-01 12:00:00" in md

    def test_contains_pass_fail(self, sample_report: PRDBenchmarkReport) -> None:
        md = generate_markdown_report(sample_report)
        assert "PASS" in md
        assert "**FAIL**" in md

    def test_contains_ingestion_metrics(self, sample_report: PRDBenchmarkReport) -> None:
        md = generate_markdown_report(sample_report)
        assert "5,000,000" in md
        assert "82,000,000" in md
        assert "31.3%" in md

    def test_contains_details(self, sample_report: PRDBenchmarkReport) -> None:
        md = generate_markdown_report(sample_report)
        assert "recall_at_10" in md
        assert "5/5" in md

    def test_nested_details(self) -> None:
        report = PRDBenchmarkReport(
            timestamp="2026-04-01",
            results=[
                BenchmarkResult(
                    name="citation_2hop",
                    passed=None,
                    target="informational",
                    p50_ms=10.0,
                    p95_ms=200.0,
                    p99_ms=200.0,
                    iterations=10,
                    details={
                        "typical": {
                            "bibcode": "2021Test...",
                            "p95_ms": 10.0,
                        },
                        "highly_cited": {
                            "bibcode": "2023Test...",
                            "p95_ms": 200.0,
                        },
                    },
                ),
            ],
            ingestion=IngestionMetrics(
                total_papers=100,
                total_embeddings=100,
                total_citation_edges=500,
                papers_without_abstracts=10,
                hnsw_index_bytes=1024,
                table_total_bytes=2048,
                system_ram_bytes=64_000_000_000,
                hnsw_pct_of_ram=0.0,
            ),
        )
        md = generate_markdown_report(report)
        assert "typical" in md
        assert "highly_cited" in md
        assert "N/A" in md  # passed=None
