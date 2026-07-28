"""Shell-level tests for scripts/daily_sync.sh step decoupling (GOAL W7, bead dxa).

Two defects are pinned here:

  (i)  ``set -euo pipefail`` made a Step 5 (embed) failure abort the script, so
       Step 6 (v_claim_edges refresh) never ran. That is why the materialized
       view went 12 days stale alongside the 2026-07 GPU outage: unrelated work
       killed by an unrelated failure.
  (ii) ``-v`` on the harvest/ingest steps set the *root* logger to DEBUG, which
       turned on urllib3's per-request dumps. Those scripts have no debug
       output of their own, so the flag bought nothing and cost ~1700 lines.

The tests run the real ``daily_sync.sh`` against a sandbox tree: ``SCIX_REPO_DIR``
points it at a tmp_path and ``SCIX_PYTHON`` at a shim that execs the stub scripts
directly. Nothing touches the production database, Qdrant, or ADS.

The logger clamp for (ii)'s embed half lives in ``scix.embed`` and is unit-tested
at the bottom of this module.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DAILY_SYNC = REPO_ROOT / "scripts" / "daily_sync.sh"

# Every script daily_sync.sh invokes, and what each stub must do beyond
# recording its own call. Keys are the basenames the script calls.
_HARVEST_BODY = """
mkdir -p data/daily_harvest
printf '{"bibcode":"2026test..1"}\\n' | gzip -c \\
    > "data/daily_harvest/ads_daily_$(date -u +%Y-%m-%d).jsonl.gz"
"""

_STUBS: dict[str, str] = {
    "harvest_daily.py": _HARVEST_BODY,
    "ingest.py": "",
    "backfill_recent_from_ads.py": "",
    "embed.py": "",
    "refresh_v_claim_edges.py": "",
    "check_pipeline_health.py": "",
}


def _write_exec(path: pathlib.Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _build_sandbox(tmp_path: pathlib.Path, exit_codes: dict[str, int]) -> pathlib.Path:
    """Create a fake repo tree whose scripts/ are recording stubs."""
    sandbox = tmp_path / "repo"
    (sandbox / "scripts").mkdir(parents=True)
    (sandbox / "logs").mkdir()
    (sandbox / "data" / "daily_harvest").mkdir(parents=True)

    for name, body in _STUBS.items():
        rc = exit_codes.get(name, 0)
        _write_exec(
            sandbox / "scripts" / name,
            "#!/bin/sh\n"
            f'printf "%s %s\\n" "{name}" "$*" >> "$DAILY_SYNC_TEST_CALLS"\n'
            f'echo "stub {name} invoked"\n'
            f"{body}\n"
            f"exit {rc}\n",
        )

    # $PYTHON in daily_sync.sh; the stubs are executables, so exec'ing the
    # argument list runs them with the same argv the real python would see.
    _write_exec(sandbox / "python-shim", '#!/bin/sh\nexec "$@"\n')
    return sandbox


class SyncRun:
    """Result of one daily_sync.sh invocation against the sandbox."""

    def __init__(self, proc: subprocess.CompletedProcess[str], sandbox: pathlib.Path) -> None:
        self.proc = proc
        self.sandbox = sandbox
        calls_file = sandbox / "calls.txt"
        self.calls = (
            calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
        )

    @property
    def returncode(self) -> int:
        return self.proc.returncode

    @property
    def stdout(self) -> str:
        return self.proc.stdout

    def called(self, script: str) -> bool:
        return any(line.startswith(f"{script} ") for line in self.calls)

    def argv_for(self, script: str) -> list[str]:
        """Arguments of the first call to ``script``."""
        for line in self.calls:
            if line.startswith(f"{script} "):
                return line[len(script) + 1 :].split()
        raise AssertionError(f"{script} was never called; calls={self.calls}")

    def status(self) -> dict:
        path = self.sandbox / "logs" / "daily_sync_status.json"
        assert path.exists(), f"no status file written; stdout=\n{self.stdout}"
        return json.loads(path.read_text(encoding="utf-8"))


# The sandbox exists only because daily_sync.sh honours these two env vars. If a
# checkout without them is ever tested (stash, bisect, older worktree), the
# script falls back to REPO_DIR=/home/ds/projects/scix_experiments and
# PYTHON=.venv/bin/python3 — i.e. it harvests from ADS and ingests into the
# production `scix` database. That has happened once. Assert the seams exist
# before spawning anything, and fail rather than fall through.
_REQUIRED_SEAMS = ("SCIX_REPO_DIR", "SCIX_PYTHON")


def _assert_sandbox_seams_present(script_path: pathlib.Path) -> None:
    script = script_path.read_text(encoding="utf-8")
    missing = [seam for seam in _REQUIRED_SEAMS if seam not in script]
    if missing:
        raise AssertionError(
            f"{script_path} does not honour {missing} — running it would hit the "
            "PRODUCTION database and ADS. Refusing to spawn it. This test file "
            "must be run against a daily_sync.sh that carries the test seams."
        )


def _run_sync(tmp_path: pathlib.Path, **exit_codes: int) -> SyncRun:
    """Run daily_sync.sh in a sandbox. Keyword args are per-stub exit codes,
    e.g. ``_run_sync(tmp_path, embed=1)``."""
    _assert_sandbox_seams_present(DAILY_SYNC)
    by_script = {f"{name}.py": rc for name, rc in exit_codes.items()}
    sandbox = _build_sandbox(tmp_path, by_script)
    env = {
        **os.environ,
        "SCIX_REPO_DIR": str(sandbox),
        "SCIX_PYTHON": str(sandbox / "python-shim"),
        "DAILY_SYNC_TEST_CALLS": str(sandbox / "calls.txt"),
        "SCIX_BATCH": "",
    }
    proc = subprocess.run(
        ["bash", str(DAILY_SYNC)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return SyncRun(proc, sandbox)


pytestmark = pytest.mark.skipif(
    shutil.which("gzip") is None or shutil.which("zcat") is None,
    reason="daily_sync.sh needs gzip/zcat",
)


# ---------------------------------------------------------------------------
# (i) Step 5 failure must not take Step 6 down with it
# ---------------------------------------------------------------------------


class TestSandboxGuard:
    """The seam guard is what stands between this test file and production."""

    def test_guard_refuses_a_script_without_the_seams(self, tmp_path: pathlib.Path) -> None:
        seamless = tmp_path / "daily_sync.sh"
        seamless.write_text(
            '#!/bin/bash\nREPO_DIR="/home/ds/projects/scix_experiments"\n'
            'PYTHON=".venv/bin/python3"\n',
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="PRODUCTION"):
            _assert_sandbox_seams_present(seamless)

    def test_guard_accepts_the_real_script(self) -> None:
        _assert_sandbox_seams_present(DAILY_SYNC)


class TestStepDecoupling:
    def test_step6_runs_when_embed_fails(self, tmp_path: pathlib.Path) -> None:
        """The regression: embed exits non-zero, v_claim_edges must still refresh."""
        run = _run_sync(tmp_path, embed=1)
        assert run.called("refresh_v_claim_edges.py"), (
            "Step 6 was skipped after a Step 5 failure — the set -e coupling is back.\n"
            f"calls={run.calls}\nstdout=\n{run.stdout}"
        )
        steps = run.status()["steps"]
        assert steps["5"] == "failed"
        assert steps["6"] == "ok"

    def test_embed_failure_still_exits_non_zero(self, tmp_path: pathlib.Path) -> None:
        """Decoupling must not paper over the failure."""
        run = _run_sync(tmp_path, embed=1)
        assert run.returncode != 0
        assert run.status()["failed_steps"] == [5]
        assert "FAILED steps: 5" in run.stdout

    def test_harvest_failure_skips_ingest_but_still_refreshes_view(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Dependency order survives: a partial harvest is not ingested, yet the
        independent Step 6 still runs."""
        run = _run_sync(tmp_path, harvest_daily=1)
        assert run.status()["steps"]["1"] == "failed"
        assert not run.called("ingest.py")
        assert run.called("refresh_v_claim_edges.py")
        assert run.returncode != 0

    def test_view_refresh_failure_is_reported(self, tmp_path: pathlib.Path) -> None:
        run = _run_sync(tmp_path, refresh_v_claim_edges=1)
        assert run.status()["steps"]["6"] == "failed"
        assert run.returncode != 0

    def test_clean_run_exits_zero_with_all_steps_recorded(self, tmp_path: pathlib.Path) -> None:
        run = _run_sync(tmp_path)
        assert run.returncode == 0, run.stdout
        status = run.status()
        assert status["failed_steps"] == []
        assert set(status["steps"]) == {"1", "2", "3", "4", "5", "6"}
        assert all(v in {"ok", "skipped"} for v in status["steps"].values())
        assert status["harvest_records"] == 1


# ---------------------------------------------------------------------------
# The health gate is wired in (GOAL W6)
# ---------------------------------------------------------------------------


class TestHealthGate:
    def test_gate_runs_after_the_steps_and_sees_this_run(self, tmp_path: pathlib.Path) -> None:
        run = _run_sync(tmp_path)
        assert run.called("check_pipeline_health.py")
        assert run.calls[-1].startswith("check_pipeline_health.py")
        argv = run.argv_for("check_pipeline_health.py")
        assert "--allow-prod" in argv
        # The status file must be on disk before the gate reads it.
        assert "--status-file" in argv

    def test_gate_breach_fails_the_run(self, tmp_path: pathlib.Path) -> None:
        run = _run_sync(tmp_path, check_pipeline_health=1)
        assert run.returncode == 1
        assert "Health gate reported a breach" in run.stdout
        assert run.status()["health_exit_code"] == 1

    def test_gate_result_recorded_on_a_healthy_run(self, tmp_path: pathlib.Path) -> None:
        run = _run_sync(tmp_path)
        assert run.status()["health_exit_code"] == 0


# ---------------------------------------------------------------------------
# (ii) log noise — the -v flags that only enabled third-party DEBUG
# ---------------------------------------------------------------------------


class TestLogVerbosity:
    @pytest.mark.parametrize(
        "script", ["harvest_daily.py", "ingest.py", "backfill_recent_from_ads.py"]
    )
    def test_no_verbose_flag_on_http_heavy_steps(self, tmp_path: pathlib.Path, script: str) -> None:
        """These scripts emit no DEBUG of their own; -v only switched on
        urllib3's per-request dumps."""
        run = _run_sync(tmp_path)
        assert "-v" not in run.argv_for(script)

    def test_embed_keeps_verbose(self, tmp_path: pathlib.Path) -> None:
        """scix.embed does emit useful DEBUG; the noise is clamped per-logger
        instead (see TestQuietNoisyHttpLoggers)."""
        run = _run_sync(tmp_path)
        assert "-v" in run.argv_for("embed.py")


class TestQuietNoisyHttpLoggers:
    """scix.embed.quiet_noisy_http_loggers — the per-logger clamp that keeps
    scix DEBUG while dropping httpcore/urllib3/HuggingFace frame narration."""

    def _restore(self, names: list[str]) -> dict[str, int]:
        return {name: logging.getLogger(name).level for name in names}

    def test_clamps_the_http_stack_to_warning(self) -> None:
        from scix.embed import _NOISY_HTTP_LOGGERS, quiet_noisy_http_loggers

        saved = self._restore(list(_NOISY_HTTP_LOGGERS))
        try:
            for name in _NOISY_HTTP_LOGGERS:
                logging.getLogger(name).setLevel(logging.DEBUG)
            quiet_noisy_http_loggers()
            for name in _NOISY_HTTP_LOGGERS:
                logger = logging.getLogger(name)
                assert logger.level == logging.WARNING
                assert not logger.isEnabledFor(logging.DEBUG)
        finally:
            for name, level in saved.items():
                logging.getLogger(name).setLevel(level)

    def test_httpcore_is_covered(self) -> None:
        """A11 names httpcore specifically — it is the qdrant_client/httpx lane."""
        from scix.embed import _NOISY_HTTP_LOGGERS

        for expected in ("httpcore", "httpx", "urllib3", "huggingface_hub"):
            assert expected in _NOISY_HTTP_LOGGERS

    def test_scix_logger_keeps_its_level(self) -> None:
        from scix.embed import quiet_noisy_http_loggers

        scix_logger = logging.getLogger("scix.embed")
        saved = scix_logger.level
        try:
            scix_logger.setLevel(logging.DEBUG)
            quiet_noisy_http_loggers()
            assert scix_logger.level == logging.DEBUG
        finally:
            scix_logger.setLevel(saved)

    def test_pipeline_applies_the_clamp(self) -> None:
        """A11 is only satisfied if run_embedding_pipeline *calls* the clamp.

        Asserted behaviourally (observed logger levels), not by spying on the
        symbol: deleting the call from run_embedding_pipeline must turn this
        test red. The pipeline is driven only far enough to reach the clamp —
        the model_name guard raises immediately after it, before any DB, Qdrant
        or model IO.
        """
        from scix.embed import _NOISY_HTTP_LOGGERS, run_embedding_pipeline

        saved = self._restore(list(_NOISY_HTTP_LOGGERS))
        try:
            for name in _NOISY_HTTP_LOGGERS:
                logging.getLogger(name).setLevel(logging.DEBUG)
            with pytest.raises(ValueError, match="indus"):
                run_embedding_pipeline(dsn="dbname=unreachable", model_name="not-indus")
            for name in _NOISY_HTTP_LOGGERS:
                assert logging.getLogger(name).level == logging.WARNING, (
                    f"{name} was left at DEBUG — run_embedding_pipeline no longer "
                    "calls quiet_noisy_http_loggers()"
                )
        finally:
            for name, level in saved.items():
                logging.getLogger(name).setLevel(level)
