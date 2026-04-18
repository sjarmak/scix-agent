"""Unit tests for scripts/build_hnsw_baseline.py.

Exercises:
 - Production-DSN refusal
 - JSON schema shape in --dry-run mode
 - Dry-run creates parent directory
 - argparse wiring (--help lists required flags)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_hnsw_baseline.py"

REQUIRED_KEYS = {
    "run_id",
    "timestamp",
    "index_name",
    "params",
    "build_wall_seconds",
    "peak_rss_bytes",
    "index_size_bytes",
    "total_relation_size_bytes",
    "explain_plan",
    "postgres_version",
}

REQUIRED_PARAM_KEYS = {"m", "ef_construction", "opclass"}


def _load_module():
    """Import the script as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "build_hnsw_baseline", SCRIPT_PATH
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
        # Must not raise.
        module.assert_pilot_dsn("dbname=scix_pgvs_pilot")

    def test_allows_test_dsn(self, module) -> None:
        module.assert_pilot_dsn("dbname=scix_test")

    def test_dry_run_refuses_production(self, module, tmp_path: Path) -> None:
        """End-to-end: invoking main(--dry-run) with production DSN exits 2."""
        out = tmp_path / "out.json"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix",
                "--dry-run",
                "--out",
                str(out),
            ]
        )
        assert rc == 2
        # No JSON written — dry-run bailed before write.
        assert not out.exists()
        # Suffix form also should not exist.
        assert not module._default_dry_run_out(out).exists()


# ---------------------------------------------------------------------------
# JSON schema shape via --dry-run
# ---------------------------------------------------------------------------

class TestDryRunJsonSchema:
    def test_dry_run_writes_schema_complete_json(
        self, module, tmp_path: Path
    ) -> None:
        out = tmp_path / "baseline.json"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out",
                str(out),
                "--index-name",
                "idx_test_hnsw",
            ]
        )
        assert rc == 0

        written = module._default_dry_run_out(out)
        assert written.exists(), f"Expected dry-run output at {written}"
        payload = json.loads(written.read_text())

        missing = REQUIRED_KEYS - set(payload.keys())
        assert not missing, f"Missing keys: {missing}"

        # Dry-run invariants.
        assert payload["build_wall_seconds"] == 0
        assert payload["peak_rss_bytes"] == 0
        assert payload["index_name"] == "idx_test_hnsw"
        assert payload["explain_plan"] is None

        # params sub-object
        assert isinstance(payload["params"], dict)
        missing_params = REQUIRED_PARAM_KEYS - set(payload["params"].keys())
        assert not missing_params, f"Missing params: {missing_params}"
        assert payload["params"]["m"] == 16
        assert payload["params"]["ef_construction"] == 64
        assert payload["params"]["opclass"] == "halfvec_cosine_ops"

        # run_id is a non-empty string
        assert isinstance(payload["run_id"], str) and payload["run_id"]
        # timestamp parseable
        assert isinstance(payload["timestamp"], str) and payload["timestamp"]

    def test_dry_run_creates_parent_dir(
        self, module, tmp_path: Path
    ) -> None:
        """--out nested in a missing dir should be auto-created."""
        out = tmp_path / "a" / "b" / "c" / "baseline.json"
        assert not out.parent.exists()
        rc = module.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        written = module._default_dry_run_out(out)
        assert written.exists()
        assert written.parent.is_dir()

    def test_dry_run_suffix_applied(
        self, module, tmp_path: Path
    ) -> None:
        """Dry-run appends '-dry-run' before the extension."""
        out = tmp_path / "baseline.json"
        rc = module.main(
            [
                "--dsn",
                "dbname=scix_pgvs_pilot",
                "--dry-run",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        assert not out.exists()
        assert (tmp_path / "baseline-dry-run.json").exists()


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
        for flag in ("--dsn", "--index-name", "--dry-run", "--concurrent"):
            assert flag in out, f"Expected flag {flag!r} in --help output"

    def test_parse_args_defaults(self, module) -> None:
        ns = module.parse_args(["--dsn", "dbname=scix_test"])
        assert ns.dsn == "dbname=scix_test"
        assert ns.index_name == module.DEFAULT_INDEX_NAME
        assert ns.dry_run is False
        assert ns.concurrent is False

    def test_parse_args_flags(self, module) -> None:
        ns = module.parse_args(
            [
                "--dsn",
                "dbname=scix_test",
                "--index-name",
                "idx_foo",
                "--dry-run",
                "--concurrent",
                "--out",
                "/tmp/x.json",
            ]
        )
        assert ns.dry_run is True
        assert ns.concurrent is True
        assert ns.index_name == "idx_foo"
        assert str(ns.out) == "/tmp/x.json"


# ---------------------------------------------------------------------------
# DDL / EXPLAIN helpers (pure functions — no DB required)
# ---------------------------------------------------------------------------

class TestDdlHelpers:
    def test_ddl_contains_required_clauses(self, module) -> None:
        ddl = module.build_ddl("idx_test", concurrent=False)
        assert "CREATE INDEX idx_test" in ddl
        assert "paper_embeddings" in ddl
        assert "hnsw" in ddl
        assert "halfvec_cosine_ops" in ddl
        assert "m = 16" in ddl
        assert "ef_construction = 64" in ddl
        assert "model_name = 'indus'" in ddl

    def test_ddl_concurrent_flag(self, module) -> None:
        ddl = module.build_ddl("idx_test", concurrent=True)
        assert "CONCURRENTLY" in ddl

    def test_explain_sql_shape(self, module) -> None:
        q = module.build_explain_sql()
        assert "EXPLAIN" in q
        assert "paper_embeddings" in q
        assert "halfvec" in q
        assert "LIMIT 10" in q
