"""Smoke tests for ``scripts/run_chunk_pass.py``.

These tests exercise the CLI surface without requiring a live database, a
GPU, or a running Qdrant — every path that would touch one of those
exits before the side-effect happens (``--help`` short-circuits argparse;
the missing-``QDRANT_URL`` path returns exit code ``2`` from the gate).

Coverage:

* ``--help`` exits 0 and lists every documented flag.
* Missing ``QDRANT_URL`` (no ``--dry-run``) exits with status ``2``.
* Missing ``QDRANT_URL`` + ``--dry-run`` does NOT trip the QDRANT_URL gate
  (the run may still fail on a downstream concern, but it must not return
  the ``2`` reserved for the env-var check).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_chunk_pass.py"

REQUIRED_FLAGS = (
    "--batch-size",
    "--inference-batch",
    "--since-bibcode",
    "--max-papers",
    "--collection",
    "--parser-version",
    "--dry-run",
    "--require-batch-scope",
    "--dsn",
    "--verbose",
)


def _env_without(*keys: str) -> dict[str, str]:
    """Copy ``os.environ`` minus the given keys."""
    drop = {k.upper() for k in keys}
    return {k: v for k, v in os.environ.items() if k.upper() not in drop}


def test_help_exits_zero_and_lists_all_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in REQUIRED_FLAGS:
        assert flag in result.stdout, f"missing {flag} in --help output"


def test_missing_qdrant_url_exits_2() -> None:
    env = _env_without("QDRANT_URL")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--max-papers", "1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2, (
        f"expected exit 2 when QDRANT_URL missing; got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "QDRANT_URL" in result.stderr


def test_missing_qdrant_url_with_dry_run_does_not_trip_gate() -> None:
    """``--dry-run`` waives the QDRANT_URL requirement.

    The run may still exit non-zero (e.g. DB connect failure under an
    invalid DSN), but it must NOT return the ``2`` reserved for the
    QDRANT_URL gate.
    """
    env = _env_without("QDRANT_URL")
    env["SCIX_DSN"] = "postgresql://invalid:0/does_not_exist"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--max-papers", "1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    assert result.returncode != 2, (
        f"--dry-run should waive QDRANT_URL gate; got exit 2\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_require_batch_scope_without_systemd_scope_exits_2() -> None:
    env = _env_without("SYSTEMD_SCOPE")
    # Set QDRANT_URL so we don't trip the URL gate first.
    env["QDRANT_URL"] = "http://localhost:6333"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--require-batch-scope",
            "--max-papers",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2, (
        f"expected exit 2 when --require-batch-scope set without SYSTEMD_SCOPE; "
        f"got {result.returncode}\nstderr: {result.stderr}"
    )
    assert "SYSTEMD_SCOPE" in result.stderr
