"""Tests for scripts/check_pipeline_health.py (GOAL W6, bead tdl).

The three assertions are pure functions over already-fetched facts, so the
healthy and breached cases are tested without a database. The SQL itself is
exercised against SCIX_TEST_DSN (schema only, no data) so a typo or a renamed
column is caught, and the production-DSN refusal is tested because the gate
runs from cron against prod every day.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import sys

import psycopg
import pytest

from tests.helpers import get_test_dsn

# scripts/ is not a package — make it importable like the other script tests.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_pipeline_health as cph  # noqa: E402

NOW = _dt.datetime(2026, 7, 27, 12, 0, 0, tzinfo=_dt.timezone.utc)


def _status(
    *,
    finished_at: str = "2026-07-27T06:20:00Z",
    steps: dict[str, str] | None = None,
    failed: list[int] | None = None,
) -> dict:
    return {
        "script": "daily_sync.sh",
        "started_at": "2026-07-27T06:15:00Z",
        "finished_at": finished_at,
        "total_steps": 6,
        "steps": steps if steps is not None else {str(n): "ok" for n in range(1, 7)},
        "failed_steps": failed or [],
        "exit_code": 0,
    }


# ---------------------------------------------------------------------------
# Check 1 — last run completed every step
# ---------------------------------------------------------------------------


class TestCheckLastRun:
    def _run(self, status: dict | None, *, max_age_hours: float = 36.0) -> cph.CheckResult:
        return cph.check_last_run(
            status,
            now=NOW,
            max_age_hours=max_age_hours,
            status_path=pathlib.Path("/nonexistent/daily_sync_status.json"),
        )

    def test_all_six_steps_ok_passes(self) -> None:
        result = self._run(_status())
        assert result.ok is True
        assert "6 steps complete" in result.detail

    def test_skipped_counts_as_complete(self) -> None:
        # Step 2 legitimately skips on a day with no new records.
        steps = {str(n): "ok" for n in range(1, 7)}
        steps["2"] = "skipped"
        steps["5"] = "skipped"
        assert self._run(_status(steps=steps)).ok is True

    def test_failed_step_breaches(self) -> None:
        steps = {str(n): "ok" for n in range(1, 7)}
        steps["5"] = "failed"
        result = self._run(_status(steps=steps, failed=[5]))
        assert result.ok is False
        assert "5=failed" in result.detail

    def test_missing_step_breaches(self) -> None:
        # An aborted run leaves later steps unrecorded.
        steps = {"1": "ok", "2": "ok", "3": "ok"}
        result = self._run(_status(steps=steps))
        assert result.ok is False
        assert "4=missing" in result.detail
        assert "6=missing" in result.detail

    def test_stale_run_breaches(self) -> None:
        # This is the 2026-07-15..27 outage: the last recorded run is old.
        result = self._run(_status(finished_at="2026-07-15T06:33:00Z"))
        assert result.ok is False
        assert "not running" in result.detail

    def test_age_threshold_is_honoured(self) -> None:
        status = _status(finished_at="2026-07-26T06:20:00Z")  # 29.7 h before NOW
        assert self._run(status, max_age_hours=36.0).ok is True
        assert self._run(status, max_age_hours=24.0).ok is False

    def test_absent_status_file_breaches(self) -> None:
        result = self._run(None)
        assert result.ok is False
        assert "no status file" in result.detail

    def test_unparseable_timestamp_breaches(self) -> None:
        assert self._run(_status(finished_at="not-a-timestamp")).ok is False

    def test_missing_steps_object_breaches(self) -> None:
        status = _status()
        del status["steps"]
        assert self._run(status).ok is False


class TestLoadStatus:
    def test_absent_file_returns_none(self, tmp_path: pathlib.Path) -> None:
        assert cph.load_status(tmp_path / "nope.json") is None

    def test_roundtrip(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "daily_sync_status.json"
        path.write_text(json.dumps(_status()), encoding="utf-8")
        loaded = cph.load_status(path)
        assert loaded is not None
        assert loaded["steps"]["6"] == "ok"


# ---------------------------------------------------------------------------
# Check 2 — dense-lane gap
# ---------------------------------------------------------------------------


class TestCheckDenseGap:
    def test_under_threshold_passes(self) -> None:
        result = cph.check_dense_gap(37, threshold=500)
        assert result.ok is True
        assert "37 papers behind" in result.detail

    def test_at_threshold_passes(self) -> None:
        assert cph.check_dense_gap(500, threshold=500).ok is True

    def test_over_threshold_breaches(self) -> None:
        # 9052 is the gap left by the 12-day GPU outage.
        result = cph.check_dense_gap(9052, threshold=500)
        assert result.ok is False
        assert "9052" in result.detail
        assert "Step 5" in result.detail


# ---------------------------------------------------------------------------
# Check 3 — v_claim_edges freshness
# ---------------------------------------------------------------------------


class TestCheckViewFreshness:
    def test_fresh_passes(self) -> None:
        refreshed = NOW - _dt.timedelta(hours=6)
        result = cph.check_view_freshness(refreshed, now=NOW, max_age_days=2.0)
        assert result.ok is True

    def test_one_missed_day_still_passes(self) -> None:
        refreshed = NOW - _dt.timedelta(days=1, hours=12)
        assert cph.check_view_freshness(refreshed, now=NOW, max_age_days=2.0).ok is True

    def test_stale_breaches(self) -> None:
        refreshed = _dt.datetime(2026, 7, 15, 6, 33, tzinfo=_dt.timezone.utc)
        result = cph.check_view_freshness(refreshed, now=NOW, max_age_days=2.0)
        assert result.ok is False
        assert "12.2d ago" in result.detail

    def test_never_refreshed_breaches(self) -> None:
        result = cph.check_view_freshness(None, now=NOW, max_age_days=2.0)
        assert result.ok is False
        assert "no refresh recorded" in result.detail

    def test_naive_timestamp_treated_as_utc(self) -> None:
        naive = _dt.datetime(2026, 7, 27, 6, 33)
        assert cph.check_view_freshness(naive, now=NOW, max_age_days=2.0).ok is True

    def test_recent_but_failed_refresh_breaches_and_keeps_the_age(self) -> None:
        """refresh_v_claim_edges.py upserts on `filename`, so a failed refresh
        overwrites the last successful row. A fresh timestamp with status
        'failed' must breach, and must still report how old it is — filtering
        the failure out would report 'never refreshed' and lose the age."""
        refreshed = NOW - _dt.timedelta(hours=6)
        result = cph.check_view_freshness(refreshed, now=NOW, max_age_days=2.0, status="failed")
        assert result.ok is False
        assert "'failed'" in result.detail
        assert "0.2d ago" in result.detail


# ---------------------------------------------------------------------------
# Reporting + orchestration
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self._row: tuple | None = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: dict | None = None) -> None:
        self._row = self._rows.pop(0)

    def fetchone(self) -> tuple | None:
        return self._row


class _FakeConn:
    """Returns the queued rows in run_checks' query order: refresh, then gap.

    run_checks fetches the refresh row before the gap so the refresh status and
    timestamp arrive together; keep this order in step with it.
    """

    def __init__(
        self,
        gap: int,
        refreshed_at: _dt.datetime | None,
        status: str | None = "complete",
    ) -> None:
        self._rows = [(refreshed_at, status), (gap,)]

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)


class TestRunChecks:
    def _paths(self, tmp_path: pathlib.Path, status: dict | None) -> pathlib.Path:
        path = tmp_path / "daily_sync_status.json"
        if status is not None:
            path.write_text(json.dumps(status), encoding="utf-8")
        return path

    def test_healthy_pipeline_passes_all_three(self, tmp_path: pathlib.Path) -> None:
        results = cph.run_checks(
            _FakeConn(12, NOW - _dt.timedelta(hours=6)),
            status_path=self._paths(tmp_path, _status()),
            now=NOW,
            max_run_age_hours=36.0,
            max_dense_gap=500,
            max_view_age_days=2.0,
        )
        assert [r.name for r in results] == [
            "last_run_complete",
            "dense_lane_gap",
            "v_claim_edges_fresh",
        ]
        assert all(r.ok for r in results)
        assert "all 3 checks passed" in cph.render(results)

    def test_gpu_outage_shape_breaches_all_three(self, tmp_path: pathlib.Path) -> None:
        """The 2026-07-15..27 state: stale run, 9052-paper gap, 12-day-old view."""
        stale = _status(
            finished_at="2026-07-15T06:33:00Z",
            steps={"1": "ok", "2": "ok", "3": "ok", "4": "ok", "5": "failed"},
            failed=[5],
        )
        results = cph.run_checks(
            _FakeConn(9052, _dt.datetime(2026, 7, 15, 6, 33, tzinfo=_dt.timezone.utc)),
            status_path=self._paths(tmp_path, stale),
            now=NOW,
            max_run_age_hours=36.0,
            max_dense_gap=500,
            max_view_age_days=2.0,
        )
        assert [r.ok for r in results] == [False, False, False]
        report = cph.render(results)
        assert "3/3 checks FAILED" in report
        assert report.count("FAIL ") == 3


class TestRender:
    def test_marks_each_check_and_summarises(self) -> None:
        results = [
            cph.CheckResult("a", True, "fine"),
            cph.CheckResult("b", False, "broken"),
        ]
        out = cph.render(results)
        assert "PASS a: fine" in out
        assert "FAIL b: broken" in out
        assert "1/2 checks FAILED (b)" in out


# ---------------------------------------------------------------------------
# CLI safety + live SQL
# ---------------------------------------------------------------------------


class TestProductionGuard:
    def test_prod_dsn_without_allow_prod_refuses(self) -> None:
        # Exit 2 = refusal, and it must happen before any connection attempt.
        assert cph.main(["--dsn", "dbname=scix"]) == 2

    def test_require_batch_scope_without_scope_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SYSTEMD_SCOPE", raising=False)
        assert cph.main(["--dsn", "dbname=scix", "--allow-prod", "--require-batch-scope"]) == 2


@pytest.mark.integration
class TestQueriesAgainstSchema:
    """Run the real SQL so a renamed column or table cannot pass review.

    Skips are scoped to *whole tables the test schema does not have* and are
    decided by an explicit ``to_regclass`` probe before the query runs. Catching
    ``UndefinedTable`` around the query itself would have swallowed a renamed
    column too, so a broken query would ship green.
    """

    def _conn(self) -> psycopg.Connection:
        dsn = get_test_dsn()
        if dsn is None:
            pytest.skip("SCIX_TEST_DSN not set")
        return psycopg.connect(dsn)

    @staticmethod
    def _require_tables(conn: psycopg.Connection, *names: str) -> None:
        for name in names:
            row = conn.execute("select to_regclass(%s)", (f"public.{name}",)).fetchone()
            if row is None or row[0] is None:
                pytest.skip(f"{name} is absent from the test schema (migration not applied there)")

    def test_dense_gap_query_executes(self) -> None:
        with self._conn() as conn:
            self._require_tables(conn, "papers", "indus_qdrant_synced")
            # No try/except: past this point any error is a real defect in the
            # query (renamed column, bad join), not an environment gap.
            assert cph.query_dense_gap(conn) >= 0

    def test_view_refresh_query_executes(self) -> None:
        with self._conn() as conn:
            self._require_tables(conn, "ingest_log")
            refreshed, status = cph.query_view_refresh(conn)
            # Schema-only database: no refresh has ever been recorded.
            assert refreshed is None or isinstance(refreshed, _dt.datetime)
            assert status is None or isinstance(status, str)
