"""Tests for scripts/copy_indus_to_pilot.py.

These tests do NOT require a running database — all psycopg interactions
are monkey-patched. They cover:

* production-DSN safety guard (``assert_pilot_dsn``)
* ``--dry-run`` short-circuits before opening a connection
* argparse wiring exposes the required flags
* ImportError message when pgvs_bench_env is unavailable

Run with:
    pytest tests/test_copy_indus_to_pilot.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# --------------------------------------------------------------------------- #
# Import the script as a module (scripts/ is not a package).
# --------------------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "copy_indus_to_pilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "copy_indus_to_pilot", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy_indus_to_pilot"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    return _load_module()


# --------------------------------------------------------------------------- #
# (a) Production-DSN safety guard
# --------------------------------------------------------------------------- #


class TestAssertPilotDsn:
    def test_rejects_production_dbname_scix(self, mod, monkeypatch):
        monkeypatch.delenv("SCIX_DSN", raising=False)
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn(
                dest_dsn="dbname=scix",
                source_dsn="dbname=scix",
            )
        msg = str(exc.value).lower()
        assert "production" in msg
        assert "refuse" in msg

    def test_rejects_empty_dbname(self, mod, monkeypatch):
        monkeypatch.delenv("SCIX_DSN", raising=False)
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn(
                dest_dsn="host=localhost port=5432",
                source_dsn="dbname=scix",
            )
        assert "production" in str(exc.value).lower()
        assert "refuse" in str(exc.value).lower()

    def test_rejects_empty_dsn(self, mod, monkeypatch):
        monkeypatch.delenv("SCIX_DSN", raising=False)
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn(dest_dsn="", source_dsn="dbname=scix")
        assert "refuse" in str(exc.value).lower()

    def test_rejects_dest_equal_to_source(self, mod, monkeypatch):
        monkeypatch.delenv("SCIX_DSN", raising=False)
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn(
                dest_dsn="dbname=scix_pilot",
                source_dsn="dbname=scix_pilot",
            )
        assert "refuse" in str(exc.value).lower()

    def test_rejects_when_dest_equals_scix_dsn_env(self, mod, monkeypatch):
        monkeypatch.setenv("SCIX_DSN", "dbname=prod_secret host=prod")
        with pytest.raises(ValueError) as exc:
            mod.assert_pilot_dsn(
                dest_dsn="dbname=prod_secret host=prod",
                source_dsn="dbname=other",
            )
        assert "refuse" in str(exc.value).lower()

    def test_allows_pilot_dsn(self, mod, monkeypatch):
        monkeypatch.delenv("SCIX_DSN", raising=False)
        # Should not raise.
        mod.assert_pilot_dsn(
            dest_dsn="dbname=scix_pilot host=localhost",
            source_dsn="dbname=scix host=prod.example",
        )


# --------------------------------------------------------------------------- #
# (b) Dry-run short-circuits before any psycopg.connect call
# --------------------------------------------------------------------------- #


class TestDryRun:
    def test_dry_run_never_opens_connection(self, mod, monkeypatch, capsys):
        monkeypatch.delenv("SCIX_DSN", raising=False)

        connect_spy = MagicMock(
            side_effect=AssertionError("psycopg.connect should not be called")
        )
        # Patch any reachable psycopg.connect: the script imports psycopg
        # lazily inside _run_copy, so patching the already-loaded module is
        # sufficient if it's been imported; otherwise we stub the module.
        import psycopg  # noqa: F401

        monkeypatch.setattr("psycopg.connect", connect_spy)

        # Stub capture_env lazy-import to avoid requiring the helper.
        monkeypatch.setattr(
            mod, "_lazy_import_capture_env", lambda: (lambda: {"git_sha": "x"})
        )

        rc = mod.main(
            [
                "--source-dsn",
                "dbname=scix host=prod",
                "--dest-dsn",
                "dbname=scix_pilot host=localhost",
                "--dry-run",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr().out
        assert "dry-run" in captured
        assert "no connections opened" in captured
        connect_spy.assert_not_called()

    def test_dry_run_refuses_production_dest_before_printing(
        self, mod, monkeypatch
    ):
        """Safety guard fires even with --dry-run — prod dest is always rejected."""
        monkeypatch.delenv("SCIX_DSN", raising=False)
        with pytest.raises(ValueError) as exc:
            mod.main(
                [
                    "--source-dsn",
                    "dbname=scix host=prod",
                    "--dest-dsn",
                    "dbname=scix",
                    "--dry-run",
                ]
            )
        assert "refuse" in str(exc.value).lower()
        assert "production" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# (c) Argparse wiring
# --------------------------------------------------------------------------- #


class TestArgparse:
    def test_help_exits_zero(self, mod, capsys):
        parser = mod.build_arg_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0
        help_text = capsys.readouterr().out
        for flag in ("--source-dsn", "--dest-dsn", "--dry-run", "--batch-size", "--limit"):
            assert flag in help_text, f"missing {flag} in --help output"

    def test_default_source_dsn_from_env(self, mod, monkeypatch):
        monkeypatch.setenv("SCIX_DSN", "dbname=from_env")
        # Re-parse a fresh parser each time because argparse freezes defaults
        # at parser construction.
        fresh = _load_module()
        parsed = fresh.build_arg_parser().parse_args(
            ["--dest-dsn", "dbname=scix_pilot"]
        )
        assert parsed.source_dsn == "dbname=from_env"
        assert parsed.dry_run is False
        assert parsed.batch_size == 100_000
        assert parsed.limit is None

    def test_limit_and_batch_size_parse(self, mod):
        parsed = mod.build_arg_parser().parse_args(
            [
                "--source-dsn",
                "dbname=src",
                "--dest-dsn",
                "dbname=pilot",
                "--limit",
                "1000",
                "--batch-size",
                "500",
            ]
        )
        assert parsed.limit == 1000
        assert parsed.batch_size == 500


# --------------------------------------------------------------------------- #
# (d) SQL builders
# --------------------------------------------------------------------------- #


class TestSqlBuilders:
    def test_source_sql_includes_where_and_format_binary(self, mod):
        sql = mod.build_source_copy_sql(
            ["bibcode", "model_name", "embedding"], limit=None
        )
        assert "paper_embeddings" in sql
        assert "model_name = 'indus'" in sql
        assert "FORMAT BINARY" in sql
        assert "LIMIT" not in sql  # None

    def test_source_sql_with_limit(self, mod):
        sql = mod.build_source_copy_sql(["bibcode"], limit=42)
        assert "LIMIT 42" in sql

    def test_dest_sql_uses_from_stdin_binary(self, mod):
        sql = mod.build_dest_copy_sql(["bibcode", "embedding"])
        assert "COPY paper_embeddings" in sql
        assert "FROM STDIN" in sql
        assert "FORMAT BINARY" in sql


# --------------------------------------------------------------------------- #
# (e) Lazy capture_env ImportError message
# --------------------------------------------------------------------------- #


class TestCaptureEnvImport:
    def test_missing_pgvs_bench_env_raises_clear_error(self, mod, monkeypatch):
        # Remove any cached module.
        monkeypatch.delitem(sys.modules, "pgvs_bench_env", raising=False)

        original_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict
        ) else __builtins__.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pgvs_bench_env":
                raise ImportError("simulated missing module")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", fake_import)

        with pytest.raises(ImportError) as exc:
            mod._lazy_import_capture_env()
        assert "pgvs_bench_env" in str(exc.value)
        assert "capture_env" in str(exc.value)


# --------------------------------------------------------------------------- #
# (f) Result writer emits the expected shape
# --------------------------------------------------------------------------- #


class TestWriteResult:
    def test_write_result_shape(self, mod, tmp_path):
        result = mod.CopyResult(
            run_id="abc123",
            timestamp="2026-04-18T00:00:00+00:00",
            source_dsn_sanitized="dbname=src",
            dest_dsn_sanitized="dbname=pilot",
            rows_copied=1234,
            bytes_copied=5678,
            wall_seconds=1.5,
            git_sha="deadbeef",
            columns=["bibcode", "model_name", "embedding"],
        )
        out = tmp_path / "copy_indus.json"
        mod._write_result(result, path=out)
        payload = json.loads(out.read_text())
        assert payload["run_id"] == "abc123"
        assert payload["rows_copied"] == 1234
        assert payload["bytes_copied"] == 5678
        assert payload["wall_seconds"] == 1.5
        assert payload["git_sha"] == "deadbeef"
        assert payload["source_dsn_sanitized"] == "dbname=src"
        assert payload["dest_dsn_sanitized"] == "dbname=pilot"
        assert payload["columns"] == ["bibcode", "model_name", "embedding"]


# --------------------------------------------------------------------------- #
# (g) Sanitizer strips passwords
# --------------------------------------------------------------------------- #


class TestSanitizeDsn:
    def test_strips_uri_password(self, mod):
        s = mod._sanitize_dsn("postgres://user:supersecret@host:5432/db")
        assert "supersecret" not in s
        assert "user" in s

    def test_keyvalue_drops_password_token(self, mod):
        s = mod._sanitize_dsn("dbname=scix user=u password=SECRET host=h")
        assert "SECRET" not in s
        assert "dbname=scix" in s
