"""Tests for scripts/pgvs_bench_env.py.

All subprocess and psycopg interactions are mocked — tests never hit
the real DB and must pass on a clean machine without vectorscale.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pgvs_bench_env.py"


def _load_module() -> ModuleType:
    """Load pgvs_bench_env from scripts/ as a module for testing."""
    spec = importlib.util.spec_from_file_location("pgvs_bench_env", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pgvs_bench_env"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def env_mod() -> ModuleType:
    return _load_module()


# ---------------------------------------------------------------------------
# Fake helpers for subprocess and psycopg
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _make_fake_run(
    sha: str = "abc123def",
    branch: str = "main",
    lsblk_json: str | None = None,
    lsblk_fail: bool = False,
) -> Any:
    """Build a fake subprocess.run that dispatches on argv."""
    if lsblk_json is None:
        lsblk_json = json.dumps(
            {
                "blockdevices": [
                    {"name": "nvme0n1", "model": "Samsung SSD 990", "size": "2T"},
                ]
            }
        )

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if args[:2] == ["git", "rev-parse"]:
            if "--abbrev-ref" in args:
                return _FakeCompleted(branch)
            return _FakeCompleted(sha)
        if args[0] == "lsblk":
            if lsblk_fail:
                raise FileNotFoundError("lsblk missing")
            return _FakeCompleted(lsblk_json)
        return _FakeCompleted("", returncode=1, stderr="unknown command")

    return fake_run


class _FakeCursor:
    def __init__(self, version: str | None, ext_rows: list[tuple[str, str]]) -> None:
        self._version = version
        self._ext_rows = ext_rows
        self._last_query: str | None = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: str, *args: Any, **kwargs: Any) -> None:
        self._last_query = query

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._last_query and "version()" in self._last_query:
            return (self._version,) if self._version is not None else None
        return None

    def fetchall(self) -> list[tuple[str, str]]:
        if self._last_query and "pg_extension" in self._last_query:
            return self._ext_rows
        return []


class _FakeConnection:
    def __init__(self, version: str, ext_rows: list[tuple[str, str]]) -> None:
        self._cursor = _FakeCursor(version, ext_rows)

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return self._cursor


def _install_fake_psycopg(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "PostgreSQL 16.4 on x86_64-linux-gnu",
    ext_rows: list[tuple[str, str]] | None = None,
    raise_on_connect: Exception | None = None,
) -> None:
    if ext_rows is None:
        ext_rows = [("vector", "0.8.2")]

    def fake_connect(dsn: str, **kwargs: Any) -> _FakeConnection:
        if raise_on_connect is not None:
            raise raise_on_connect
        return _FakeConnection(version, ext_rows)

    fake_mod = SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "psycopg", fake_mod)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_capture_env_shape_fully_mocked(
    env_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """capture_env returns all required keys with correct types when everything works."""
    monkeypatch.setattr(env_mod.subprocess, "run", _make_fake_run())
    _install_fake_psycopg(
        monkeypatch,
        ext_rows=[("vector", "0.8.2"), ("vectorscale", "0.5.0")],
    )

    env = env_mod.capture_env(dsn="dbname=fake")

    required_keys = {
        "run_id",
        "timestamp",
        "git_sha",
        "postgres_version",
        "extensions",
        "cpu",
        "ram_gb",
        "disk",
        "git_branch",
    }
    assert required_keys.issubset(env.keys())
    assert env["git_sha"] == "abc123def"
    assert env["git_branch"] == "main"
    assert env["postgres_version"].startswith("PostgreSQL")
    assert env["extensions"]["vector"] == "0.8.2"
    assert env["extensions"]["vectorscale"] == "0.5.0"
    assert isinstance(env["run_id"], str) and len(env["run_id"]) >= 32
    assert "T" in env["timestamp"]  # ISO format
    # disk present when lsblk returns JSON
    assert env["disk"] is not None
    assert env["disk"]["name"] == "nvme0n1"


def test_missing_vectorscale_is_null(env_mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """extensions['vectorscale'] is None when the extension is not installed."""
    monkeypatch.setattr(env_mod.subprocess, "run", _make_fake_run())
    _install_fake_psycopg(monkeypatch, ext_rows=[("vector", "0.8.2")])

    env = env_mod.capture_env(dsn="dbname=fake")

    assert env["extensions"]["vector"] == "0.8.2"
    assert env["extensions"]["vectorscale"] is None
    # Not an error — just missing
    assert "_db_error" not in env


def test_unreachable_db_degrades_gracefully(
    env_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """capture_env returns a valid dict when DB is unreachable."""
    monkeypatch.setattr(env_mod.subprocess, "run", _make_fake_run())
    _install_fake_psycopg(
        monkeypatch,
        raise_on_connect=ConnectionRefusedError("db down"),
    )

    env = env_mod.capture_env(dsn="dbname=nope")

    assert env["postgres_version"] is None
    assert env["extensions"] == {"vector": None, "vectorscale": None}
    assert "_db_error" in env
    assert "ConnectionRefusedError" in env["_db_error"]
    # Other fields still populated
    assert env["git_sha"] == "abc123def"
    assert env["run_id"]


def test_dsn_none_no_db_probe(env_mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """When dsn=None, no DB connect is attempted; fields remain null."""
    monkeypatch.setattr(env_mod.subprocess, "run", _make_fake_run())

    def _boom(*_: Any, **__: Any) -> None:
        raise AssertionError("should not be called when dsn is None")

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=_boom))

    env = env_mod.capture_env(dsn=None)

    assert env["postgres_version"] is None
    assert env["extensions"]["vector"] is None
    assert env["extensions"]["vectorscale"] is None
    assert env.get("_db_error") == "no dsn provided"


def test_lsblk_failure_disk_null(env_mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disk probe failure yields None without raising."""
    monkeypatch.setattr(env_mod.subprocess, "run", _make_fake_run(lsblk_fail=True))
    _install_fake_psycopg(monkeypatch)

    env = env_mod.capture_env(dsn="dbname=fake")
    assert env["disk"] is None


def test_out_writes_valid_json_creates_parent(
    env_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Running main() with --out writes valid JSON and creates parent dir."""
    monkeypatch.setattr(env_mod.subprocess, "run", _make_fake_run())
    _install_fake_psycopg(monkeypatch)

    out_path = tmp_path / "nested" / "dir" / "env.json"
    assert not out_path.parent.exists()

    rc = env_mod.main(["--out", str(out_path), "--dsn", "dbname=fake"])
    assert rc == 0
    assert out_path.exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    # Acceptance criteria: all required keys present
    for key in (
        "run_id",
        "timestamp",
        "git_sha",
        "postgres_version",
        "extensions",
        "cpu",
        "ram_gb",
        "disk",
        "git_branch",
    ):
        assert key in data, f"missing key: {key}"
    assert "vector" in data["extensions"]
    assert "vectorscale" in data["extensions"]


def test_run_command_timeout_returns_none(
    env_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_run_command swallows TimeoutExpired and returns None."""

    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
    assert env_mod._run_command(["git", "rev-parse", "HEAD"]) is None
