"""Smoke tests for ``scripts/viz/run.sh`` — the viz-demo one-command runner.

These tests exercise the shell entrypoint directly. They are intentionally
lightweight:

* three fast checks on the script file itself (executable bit, bash
  syntax, help-text content);
* one integration-marked end-to-end check that invokes ``--build-only
  --synthetic`` and asserts the two JSON payloads land on disk under
  ``data/viz/`` in the repository root.

The integration test uses ``--synthetic`` so it never touches Postgres and
never needs any embedding backend beyond ``umap-learn``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
RUN_SH: Path = REPO_ROOT / "scripts" / "viz" / "run.sh"
SANKEY_JSON: Path = REPO_ROOT / "data" / "viz" / "sankey.json"
UMAP_JSON: Path = REPO_ROOT / "data" / "viz" / "umap.json"


def test_run_sh_exists() -> None:
    """run.sh must exist at the documented location."""
    assert RUN_SH.is_file(), f"missing {RUN_SH}"


def test_run_sh_executable() -> None:
    """The script must be executable so ``./scripts/viz/run.sh`` works."""
    assert os.access(RUN_SH, os.X_OK), f"{RUN_SH} is not executable"


def test_run_sh_syntax() -> None:
    """``bash -n`` must accept the script (no shell syntax errors)."""
    result = subprocess.run(
        ["bash", "-n", str(RUN_SH)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"bash -n failed: rc={result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_run_sh_help() -> None:
    """``--help`` must print usage covering --port, --host, --no-build."""
    result = subprocess.run(
        [str(RUN_SH), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    combined = result.stdout + result.stderr
    for flag in ("--port", "--host", "--no-build"):
        assert flag in combined, f"expected {flag!r} in --help output, got: {combined!r}"


@pytest.mark.integration
def test_build_only_synthetic_produces_json() -> None:
    """``--build-only --synthetic`` must produce both JSON payloads.

    The test deletes any pre-existing ``data/viz/{sankey,umap}.json`` (they
    are regenerable artifacts), runs the script, and asserts both files
    exist and parse as JSON. It leaves the freshly-generated files in
    place — a subsequent ``make viz-demo`` or repeat test run will reuse
    or overwrite them idempotently.
    """
    for p in (SANKEY_JSON, UMAP_JSON):
        if p.exists():
            p.unlink()

    # If REPO_ROOT/.venv/bin/python is missing (e.g. running inside a
    # git worktree that shares the main repo's interpreter), point the
    # runner at whichever python is executing this test.
    env = os.environ.copy()
    default_venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if "VENV_PY" not in env and not default_venv_py.is_file():
        import sys as _sys
        env["VENV_PY"] = _sys.executable

    result = subprocess.run(
        [str(RUN_SH), "--build-only", "--synthetic"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, (
        f"run.sh --build-only --synthetic failed: rc={result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )

    assert SANKEY_JSON.is_file(), f"expected {SANKEY_JSON} after build"
    assert UMAP_JSON.is_file(), f"expected {UMAP_JSON} after build"

    with SANKEY_JSON.open(encoding="utf-8") as fh:
        sankey = json.load(fh)
    assert isinstance(sankey, dict)
    assert "nodes" in sankey and "links" in sankey
    assert isinstance(sankey["nodes"], list) and len(sankey["nodes"]) > 0

    with UMAP_JSON.open(encoding="utf-8") as fh:
        umap_payload = json.load(fh)
    assert isinstance(umap_payload, list) and len(umap_payload) > 0
    first = umap_payload[0]
    for key in ("bibcode", "x", "y", "community_id", "resolution"):
        assert key in first, f"umap.json row missing {key!r}: {first!r}"
