"""Unit tests for scripts/bench_pgvectorscale_concurrent.py.

Covers the acceptance criteria:
  (a) production-DSN refusal (ValueError containing 'production' or 'refuse')
  (b) latency percentile math matches numpy.percentile within 1e-6
  (c) argparse wiring — --help lists --dsn, --thread-counts, --duration-seconds, --out
  (d) JSON schema shape via --dry-run
  (e) parse_thread_counts edge cases
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import bench_pgvectorscale_concurrent as mod


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "bench_pgvectorscale_concurrent.py"


REQUIRED_ENTRY_KEYS = {
    "index_name",
    "thread_count",
    "qps",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "queries_executed",
}


# ---------------------------------------------------------------------------
# Production-DSN refusal
# ---------------------------------------------------------------------------


class TestAssertPilotDsn:
    def test_refuses_production_keyvalue_dsn(self) -> None:
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn("dbname=scix")
        msg = str(exc.value).lower()
        assert "refuse" in msg or "production" in msg

    def test_refuses_production_uri_dsn(self) -> None:
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn("postgresql://user@localhost/scix")
        msg = str(exc.value).lower()
        assert "refuse" in msg or "production" in msg

    def test_refuses_empty_dsn(self) -> None:
        with pytest.raises(ValueError):
            mod.assert_pilot_dsn("")

    def test_allows_pilot_dsn(self) -> None:
        mod.assert_pilot_dsn("dbname=scix_pgvs_pilot")

    def test_allows_test_dsn(self) -> None:
        mod.assert_pilot_dsn("dbname=scix_test")

    def test_main_returns_2_on_production_dsn(self, tmp_path: Path) -> None:
        """End-to-end: main(--dsn dbname=scix ...) returns exit code 2."""
        out = tmp_path / "stress.json"
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix",
                "--dry-run",
                "--out",
                str(out),
            ]
        )
        assert rc == 2
        # No JSON or MD written — refusal happened before write.
        assert not out.exists()
        assert not mod._sibling_md_path(out).exists()


# ---------------------------------------------------------------------------
# Percentile math
# ---------------------------------------------------------------------------


class TestComputePercentiles:
    def test_matches_numpy_percentile_1_to_100(self) -> None:
        durations = [float(x) for x in range(1, 101)]
        got = mod.compute_percentiles(durations)

        expected_p50 = float(np.percentile(durations, 50))
        expected_p95 = float(np.percentile(durations, 95))
        expected_p99 = float(np.percentile(durations, 99))

        assert abs(got["p50"] - expected_p50) < 1e-6
        assert abs(got["p95"] - expected_p95) < 1e-6
        assert abs(got["p99"] - expected_p99) < 1e-6

    def test_matches_numpy_percentile_random(self) -> None:
        rng = np.random.default_rng(42)
        durations = rng.uniform(0.1, 500.0, size=1000).tolist()
        got = mod.compute_percentiles(durations)

        expected_p50 = float(np.percentile(durations, 50))
        expected_p95 = float(np.percentile(durations, 95))
        expected_p99 = float(np.percentile(durations, 99))

        assert abs(got["p50"] - expected_p50) < 1e-6
        assert abs(got["p95"] - expected_p95) < 1e-6
        assert abs(got["p99"] - expected_p99) < 1e-6

    def test_empty_input_returns_zeros(self) -> None:
        got = mod.compute_percentiles([])
        assert got == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_single_sample(self) -> None:
        got = mod.compute_percentiles([12.5])
        assert abs(got["p50"] - 12.5) < 1e-6
        assert abs(got["p95"] - 12.5) < 1e-6
        assert abs(got["p99"] - 12.5) < 1e-6

    def test_returns_float_type(self) -> None:
        got = mod.compute_percentiles([1.0, 2.0, 3.0])
        for key in ("p50", "p95", "p99"):
            assert isinstance(got[key], float)


# ---------------------------------------------------------------------------
# parse_thread_counts
# ---------------------------------------------------------------------------


class TestParseThreadCounts:
    def test_default(self) -> None:
        assert mod.parse_thread_counts("10,50") == [10, 50]

    def test_single_value(self) -> None:
        assert mod.parse_thread_counts("8") == [8]

    def test_ignores_whitespace(self) -> None:
        assert mod.parse_thread_counts(" 4 , 16 , 64 ") == [4, 16, 64]

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            mod.parse_thread_counts("0,1")

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            mod.parse_thread_counts("-1,5")

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(ValueError):
            mod.parse_thread_counts("10,abc")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            mod.parse_thread_counts("")


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_help_lists_required_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        out = result.stdout
        for flag in ("--dsn", "--thread-counts", "--duration-seconds", "--out"):
            assert flag in out, f"Expected flag {flag!r} in --help output"

    def test_parse_args_defaults(self) -> None:
        ns = mod.parse_args(["--dsn", "dbname=scix_test"])
        assert ns.dsn == "dbname=scix_test"
        assert ns.thread_counts == "10,50"
        assert ns.duration_seconds == 60
        assert ns.dry_run is False

    def test_parse_args_thread_counts_override(self) -> None:
        ns = mod.parse_args(
            [
                "--dsn",
                "dbname=scix_test",
                "--thread-counts",
                "4,8,16",
                "--duration-seconds",
                "5",
                "--dry-run",
            ]
        )
        assert ns.thread_counts == "4,8,16"
        assert ns.duration_seconds == 5
        assert ns.dry_run is True


# ---------------------------------------------------------------------------
# Dry-run JSON shape
# ---------------------------------------------------------------------------


class TestDryRunSchema:
    def test_dry_run_writes_schema_complete_json(self, tmp_path: Path) -> None:
        out = tmp_path / "stress.json"
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--thread-counts",
                "10,50",
                "--duration-seconds",
                "1",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        payload = json.loads(out.read_text())

        # Top-level
        for key in (
            "run_id",
            "timestamp",
            "thread_counts",
            "duration_seconds",
            "model_name",
            "dry_run",
            "entries",
        ):
            assert key in payload, f"Missing top-level key: {key}"

        assert payload["dry_run"] is True
        assert payload["thread_counts"] == [10, 50]
        assert payload["duration_seconds"] == 1
        assert isinstance(payload["entries"], list)

        # Entries: one per (index, thread_count).
        n_indexes = len(mod.DEFAULT_INDEX_LABELS)
        n_threads = 2
        assert len(payload["entries"]) == n_indexes * n_threads

        # Every entry must carry the required metric keys.
        for entry in payload["entries"]:
            missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
            assert not missing, f"Entry missing keys {missing}: {entry}"
            # Dry-run invariants
            assert entry["qps"] == 0.0
            assert entry["p50_ms"] == 0.0
            assert entry["p95_ms"] == 0.0
            assert entry["p99_ms"] == 0.0
            assert entry["queries_executed"] == 0
            # thread_count is one of the requested values
            assert entry["thread_count"] in (10, 50)
            # index_name is non-empty string
            assert isinstance(entry["index_name"], str) and entry["index_name"]

    def test_dry_run_writes_markdown(self, tmp_path: Path) -> None:
        out = tmp_path / "stress.json"
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        md_path = mod._sibling_md_path(out)
        assert md_path.exists()
        text = md_path.read_text()
        assert "Concurrent Stress Benchmark" in text
        assert "| Index |" in text

    def test_dry_run_creates_parent_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "stress.json"
        assert not out.parent.exists()
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        assert out.parent.is_dir()

    def test_dry_run_rejects_bad_thread_counts(self, tmp_path: Path) -> None:
        out = tmp_path / "stress.json"
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--thread-counts",
                "0,-1",
                "--out",
                str(out),
            ]
        )
        assert rc == 2
        assert not out.exists()


# ---------------------------------------------------------------------------
# Result shaping helpers
# ---------------------------------------------------------------------------


class TestResultShaping:
    def test_entry_to_json_roundtrip(self) -> None:
        entry = mod.ResultEntry(
            index_name="idx_foo",
            thread_count=10,
            qps=123.45,
            p50_ms=1.0,
            p95_ms=2.0,
            p99_ms=3.0,
            queries_executed=1000,
            errors=0,
            duration_seconds=8.1,
        )
        d = mod.entry_to_json(entry, "foo")
        assert d["index_label"] == "foo"
        assert d["index_name"] == "idx_foo"
        assert d["thread_count"] == 10
        assert d["qps"] == 123.45
        assert d["queries_executed"] == 1000

    def test_build_result_document_shape(self) -> None:
        doc = mod.build_result_document(
            entries=[],
            dsn="dbname=scix_pgvs_pilot",
            thread_counts=[10, 50],
            duration_seconds=60,
            dry_run=True,
        )
        assert doc["thread_counts"] == [10, 50]
        assert doc["duration_seconds"] == 60
        assert doc["dry_run"] is True
        assert doc["dsn_dbname"] == "scix_pgvs_pilot"
        assert doc["model_name"] == "indus"
        assert doc["entries"] == []

    def test_render_markdown_produces_table(self) -> None:
        doc = mod.build_result_document(
            entries=[
                {
                    "index_label": "hnsw_baseline",
                    "index_name": "idx_hnsw_baseline_indus",
                    "thread_count": 10,
                    "qps": 42.0,
                    "p50_ms": 1.0,
                    "p95_ms": 2.0,
                    "p99_ms": 3.0,
                    "queries_executed": 420,
                    "errors": 0,
                    "duration_seconds": 10.0,
                }
            ],
            dsn="dbname=scix_pgvs_pilot",
            thread_counts=[10],
            duration_seconds=10,
            dry_run=False,
        )
        md = mod.render_markdown(doc)
        assert "hnsw_baseline" in md
        assert "42.000" in md
        assert "| Index |" in md

    def test_sibling_md_path(self) -> None:
        p = Path("/tmp/results/concurrent_stress.json")
        got = mod._sibling_md_path(p)
        assert got == Path("/tmp/results/concurrent_stress.md")

    def test_dry_run_entries_cover_all_cells(self) -> None:
        entries = mod.dry_run_entries(
            [1, 2], {"a": "idx_a", "b": "idx_b"}
        )
        assert len(entries) == 4
        labels = {e["index_label"] for e in entries}
        assert labels == {"a", "b"}
        thread_values = {e["thread_count"] for e in entries}
        assert thread_values == {1, 2}
