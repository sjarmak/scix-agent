"""Unit tests for scripts/bench_pgvectorscale_coldstart.py.

Exercises:
  * Production-DSN refusal (key=value and URI, plus end-to-end via main()).
  * compute_cold_warm_ratio math on hand-computed values.
  * percentile helper correctness.
  * summarise_index_results shape + ratio.
  * argparse wiring (flags present, defaults right, --dry-run shell).
  * --help exits zero and mentions the Postgres-restart pre-requisite.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "bench_pgvectorscale_coldstart.py"

REQUIRED_INDEX_KEYS = {
    "cold_query_latencies_ms",
    "warm_p50_ms",
    "warm_p95_ms",
    "cold_warm_ratio",
}

REQUIRED_PAYLOAD_KEYS = {
    "run_id",
    "timestamp",
    "dsn",
    "n_cold",
    "n_warm",
    "indexes",
    "env",
    "dry_run",
}


def _load_module():
    """Import the script as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "bench_pgvectorscale_coldstart", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


# ---------------------------------------------------------------------------
# Production-DSN refusal
# ---------------------------------------------------------------------------

class TestAssertPilotDsn:
    def test_refuses_production_keyvalue_dsn(self, module) -> None:
        with pytest.raises(ValueError) as exc:
            module.assert_pilot_dsn("dbname=scix")
        msg = str(exc.value).lower()
        assert "refuse" in msg or "production" in msg

    def test_refuses_production_uri_dsn(self, module) -> None:
        with pytest.raises(ValueError) as exc:
            module.assert_pilot_dsn("postgresql://user@localhost/scix")
        msg = str(exc.value).lower()
        assert "refuse" in msg or "production" in msg

    def test_refuses_empty_dsn(self, module) -> None:
        with pytest.raises(ValueError):
            module.assert_pilot_dsn("")

    def test_allows_pilot_dsn(self, module) -> None:
        module.assert_pilot_dsn("dbname=scix_pgvs_pilot")

    def test_allows_test_dsn(self, module) -> None:
        module.assert_pilot_dsn("dbname=scix_test")

    def test_main_refuses_production_dry_run(
        self, module, tmp_path: Path
    ) -> None:
        """End-to-end: invoking main(--dry-run) with production DSN exits 2."""
        out_json = tmp_path / "cold.json"
        out_md = tmp_path / "cold.md"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix",
                "--dry-run",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        assert rc == 2
        # No artifacts written when production DSN is refused.
        assert not out_json.exists()
        assert not out_md.exists()

    def test_main_refuses_production_non_dry_run(
        self, module, tmp_path: Path
    ) -> None:
        """Non-dry-run with production DSN also exits 2 without touching DB."""
        out_json = tmp_path / "cold.json"
        out_md = tmp_path / "cold.md"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        assert rc == 2
        assert not out_json.exists()
        assert not out_md.exists()


# ---------------------------------------------------------------------------
# Pure math — compute_cold_warm_ratio + percentile
# ---------------------------------------------------------------------------

class TestComputeColdWarmRatio:
    def test_simple_ratio_10x(self, module) -> None:
        # cold_ms_list[0] = 100 ms, warm_p50 = 10 ms → ratio = 10.0
        assert module.compute_cold_warm_ratio([100.0], 10.0) == pytest.approx(10.0)

    def test_ratio_uses_only_first_cold(self, module) -> None:
        # Only cold[0] feeds the ratio; later values are ignored.
        assert module.compute_cold_warm_ratio(
            [50.0, 5.0, 5.0, 5.0], 25.0
        ) == pytest.approx(2.0)

    def test_ratio_one_when_cold_equals_warm(self, module) -> None:
        assert module.compute_cold_warm_ratio([7.5], 7.5) == pytest.approx(1.0)

    def test_ratio_handles_fractional_values(self, module) -> None:
        # 3.3 / 1.1 = 3.0
        assert module.compute_cold_warm_ratio([3.3], 1.1) == pytest.approx(3.0)

    def test_raises_on_empty_cold(self, module) -> None:
        with pytest.raises(ValueError):
            module.compute_cold_warm_ratio([], 10.0)

    def test_raises_on_zero_warm(self, module) -> None:
        with pytest.raises(ValueError):
            module.compute_cold_warm_ratio([100.0], 0.0)

    def test_raises_on_negative_warm(self, module) -> None:
        with pytest.raises(ValueError):
            module.compute_cold_warm_ratio([100.0], -1.0)


class TestPercentile:
    def test_p50_of_sorted_odd_length(self, module) -> None:
        assert module.percentile([1, 2, 3, 4, 5], 50.0) == pytest.approx(3.0)

    def test_p95_on_hundred_values(self, module) -> None:
        data = list(range(1, 101))  # 1..100
        # linear interp: k = 99 * 0.95 = 94.05, values[94]=95, values[95]=96
        assert module.percentile(data, 95.0) == pytest.approx(95.05, abs=0.01)

    def test_p50_of_single_value(self, module) -> None:
        assert module.percentile([42.0], 50.0) == pytest.approx(42.0)

    def test_raises_on_empty(self, module) -> None:
        with pytest.raises(ValueError):
            module.percentile([], 50.0)


class TestSummariseIndexResults:
    def test_shape_and_ratio(self, module) -> None:
        cold = [100.0] + [10.0] * 9
        warm = [10.0] * 100
        summary = module.summarise_index_results(cold, warm)

        missing = REQUIRED_INDEX_KEYS - set(summary.keys())
        assert not missing, f"Missing keys: {missing}"

        assert len(summary["cold_query_latencies_ms"]) == 10
        assert summary["warm_p50_ms"] == pytest.approx(10.0)
        assert summary["warm_p95_ms"] == pytest.approx(10.0)
        assert summary["cold_warm_ratio"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# argparse wiring + dry-run
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

    def test_help_mentions_postgres_restart(self) -> None:
        """ACC-2: --help output must clearly state the restart pre-requisite."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        out = result.stdout.lower()
        # Either IMPORTANT or PRE-REQUISITE must appear.
        assert "important" in out or "pre-requisite" in out or "prerequisite" in out
        # And the explicit restart command must be referenced.
        assert "systemctl restart postgresql" in out
        # And the script must claim it does NOT restart Postgres itself.
        assert "does not restart" in out or "not restart postgres" in out

    def test_help_lists_required_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        out = result.stdout
        for flag in (
            "--dsn",
            "--indexes",
            "--out-json",
            "--out-md",
            "--n-cold",
            "--n-warm",
            "--dry-run",
        ):
            assert flag in out, f"Expected flag {flag!r} in --help"

    def test_parse_args_defaults(self, module) -> None:
        ns = module.parse_args(["--dsn", "dbname=scix_test"])
        assert ns.dsn == "dbname=scix_test"
        assert ns.dry_run is False
        assert ns.n_cold == module.N_COLD
        assert ns.n_warm == module.N_WARM
        # Default indexes list is non-empty.
        assert list(ns.indexes) == list(module.DEFAULT_INDEXES)

    def test_parse_args_overrides(self, module) -> None:
        ns = module.parse_args(
            [
                "--dsn",
                "dbname=scix_test",
                "--indexes",
                "idx_a",
                "idx_b",
                "--n-cold",
                "3",
                "--n-warm",
                "7",
                "--dry-run",
                "--out-json",
                "/tmp/out.json",
                "--out-md",
                "/tmp/out.md",
            ]
        )
        assert ns.dry_run is True
        assert list(ns.indexes) == ["idx_a", "idx_b"]
        assert ns.n_cold == 3
        assert ns.n_warm == 7
        assert str(ns.out_json) == "/tmp/out.json"
        assert str(ns.out_md) == "/tmp/out.md"


class TestDryRunOutputs:
    def test_dry_run_json_schema(self, module, tmp_path: Path) -> None:
        out_json = tmp_path / "cold.json"
        out_md = tmp_path / "cold.md"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--indexes",
                "idx_foo",
                "idx_bar",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        assert rc == 0
        assert out_json.exists()
        assert out_md.exists()

        payload = json.loads(out_json.read_text())
        missing = REQUIRED_PAYLOAD_KEYS - set(payload.keys())
        assert not missing, f"Missing top-level keys: {missing}"

        assert payload["dry_run"] is True
        assert payload["n_cold"] == module.N_COLD
        assert payload["n_warm"] == module.N_WARM
        assert len(payload["indexes"]) == 2

        for entry in payload["indexes"]:
            entry_missing = REQUIRED_INDEX_KEYS - set(entry.keys())
            assert not entry_missing, f"Missing index keys: {entry_missing}"
            assert len(entry["cold_query_latencies_ms"]) == module.N_COLD
            assert entry["warm_p50_ms"] == 0.0
            assert entry["warm_p95_ms"] == 0.0
            assert entry["cold_warm_ratio"] == 0.0

    def test_dry_run_md_mentions_restart(
        self, module, tmp_path: Path
    ) -> None:
        out_json = tmp_path / "cold.json"
        out_md = tmp_path / "cold.md"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        assert rc == 0
        md = out_md.read_text().lower()
        assert "systemctl restart postgresql" in md
        assert "important" in md

    def test_dry_run_creates_parent_dir(
        self, module, tmp_path: Path
    ) -> None:
        out_json = tmp_path / "a" / "b" / "cold.json"
        out_md = tmp_path / "a" / "b" / "cold.md"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        assert rc == 0
        assert out_json.exists()
        assert out_md.exists()
