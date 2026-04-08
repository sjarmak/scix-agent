"""Unit tests for scripts/stress_test_hnsw.py — pure logic, no DB required."""

from __future__ import annotations

from scripts.stress_test_hnsw import IngestStats, ScenarioResult, print_report


class TestScenarioResult:
    def test_frozen_dataclass(self) -> None:
        r = ScenarioResult(
            name="test",
            build_seconds=1.0,
            rows_before=100,
            rows_after=150,
            ingest_seconds=2.0,
            ingest_rows=50,
            ingest_errors=[],
            deadlocks=0,
            index_valid=True,
            ann_query_ok=True,
            ann_latency_ms=5.5,
        )
        assert r.name == "test"
        assert r.build_seconds == 1.0
        assert r.ingest_rows == 50

    def test_immutable(self) -> None:
        r = ScenarioResult(
            name="test",
            build_seconds=1.0,
            rows_before=100,
            rows_after=100,
            ingest_seconds=None,
            ingest_rows=0,
            ingest_errors=[],
            deadlocks=0,
            index_valid=True,
            ann_query_ok=True,
            ann_latency_ms=1.0,
        )
        try:
            r.name = "other"  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestIngestStats:
    def test_defaults(self) -> None:
        s = IngestStats()
        assert s.rows_inserted == 0
        assert s.seconds == 0.0
        assert s.errors == []
        assert s.deadlocks == 0

    def test_mutable(self) -> None:
        s = IngestStats()
        s.rows_inserted = 5000
        s.deadlocks = 1
        s.errors.append("deadlock detected")
        assert s.rows_inserted == 5000
        assert len(s.errors) == 1


class TestPrintReport:
    def _make_results(self, *, index_valid: bool = True, ann_ok: bool = True, deadlocks: int = 0) -> list[ScenarioResult]:
        serial = ScenarioResult(
            name="Serial build (baseline)",
            build_seconds=10.0,
            rows_before=150000,
            rows_after=150000,
            ingest_seconds=None,
            ingest_rows=0,
            ingest_errors=[],
            deadlocks=0,
            index_valid=True,
            ann_query_ok=True,
            ann_latency_ms=3.0,
        )
        parallel = ScenarioResult(
            name="Parallel build (7 workers)",
            build_seconds=2.5,
            rows_before=150000,
            rows_after=150000,
            ingest_seconds=None,
            ingest_rows=0,
            ingest_errors=[],
            deadlocks=0,
            index_valid=True,
            ann_query_ok=True,
            ann_latency_ms=2.8,
        )
        concurrent = ScenarioResult(
            name="Parallel build (7 workers) + concurrent ingest",
            build_seconds=3.0,
            rows_before=150000,
            rows_after=200000,
            ingest_seconds=5.0,
            ingest_rows=50000,
            ingest_errors=[],
            deadlocks=deadlocks,
            index_valid=index_valid,
            ann_query_ok=ann_ok,
            ann_latency_ms=4.0,
        )
        return [serial, parallel, concurrent]

    def test_safe_verdict(self) -> None:
        report = print_report(self._make_results())
        assert report["verdict"] == "SAFE"
        assert report["issues"] == []
        assert len(report["scenarios"]) == 3
        assert report["speedup"] == 4.0  # 10.0 / 2.5

    def test_caution_invalid_index(self) -> None:
        report = print_report(self._make_results(index_valid=False))
        assert report["verdict"] == "CAUTION"
        assert any("INVALID" in i for i in report["issues"])

    def test_caution_ann_failure(self) -> None:
        report = print_report(self._make_results(ann_ok=False))
        assert report["verdict"] == "CAUTION"
        assert any("ANN" in i for i in report["issues"])

    def test_caution_deadlocks(self) -> None:
        report = print_report(self._make_results(deadlocks=3))
        assert report["verdict"] == "CAUTION"
        assert any("deadlock" in i.lower() for i in report["issues"])
