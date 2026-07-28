"""The health gate's alert sink (bead scix_experiments-p0h).

The gate runs from the last line of daily_sync.sh, so the failure it exists to
catch — the script dying early, or cron never firing — is exactly the one that
stops it running. `--notify` plus an out-of-band cron invocation closes that.

The channel is the bead store rather than cron mail: this host has no MTA, so a
MAILTO line would deliver nothing while looking like monitoring.

These tests drive `notify()` through a fake `bd` runner, so they assert the
lifecycle contract (exactly one open bead; opened on breach, closed on
recovery) without touching the real issue tracker.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_pipeline_health as cph  # noqa: E402

NOW = _dt.datetime(2026, 7, 28, 7, 45, tzinfo=_dt.timezone.utc)

PASS = [
    cph.CheckResult("last_run_complete", True, "all 6 steps complete, finished 1.5h ago"),
    cph.CheckResult("dense_lane_gap", True, "0 papers behind, limit 500"),
    cph.CheckResult("v_claim_edges_fresh", True, "refreshed 0.1d ago, limit 2d"),
]
BREACH = [
    cph.CheckResult("last_run_complete", False, "last run finished 49.2h ago, limit 36h"),
    cph.CheckResult("dense_lane_gap", False, "9052 titled papers missing, limit 500"),
    cph.CheckResult("v_claim_edges_fresh", True, "refreshed 0.1d ago, limit 2d"),
]


class FakeBd:
    """Records bd invocations and replays canned issue state."""

    def __init__(self, issues: list[dict] | None = None, *, fail_on: str | None = None) -> None:
        self.issues = issues or []
        self.fail_on = fail_on
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if self.fail_on and argv[0] == self.fail_on:
            return subprocess.CompletedProcess(argv, 1, "", "bd: database is locked")
        if argv[0] == "list":
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.issues), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def call_verbs(self) -> list[str]:
        return [c[0] for c in self.calls]


def _issue(id_: str = "scix_experiments-abc", created: str = "2026-07-26T07:45:00+00:00") -> dict:
    return {"id": id_, "title": cph.NOTIFY_TITLE, "created_at": created, "status": "open"}


class TestBreachOpensExactlyOneBead:
    def test_creates_a_bead_when_none_is_open(self) -> None:
        bd = FakeBd([])
        assert cph.notify(BREACH, now=NOW, runner=bd) == "created"
        assert bd.call_verbs() == ["list", "create"]

    def test_created_bead_carries_the_label_the_lookup_uses(self) -> None:
        """Label is the key for find-existing; a mismatch would file a new bead daily."""
        bd = FakeBd([])
        cph.notify(BREACH, now=NOW, runner=bd)
        create = next(c for c in bd.calls if c[0] == "create")
        assert "-l" in create
        assert create[create.index("-l") + 1] == cph.NOTIFY_LABEL
        lookup = next(c for c in bd.calls if c[0] == "list")
        assert cph.NOTIFY_LABEL in lookup

    def test_updates_instead_of_creating_when_one_is_already_open(self) -> None:
        bd = FakeBd([_issue()])
        assert cph.notify(BREACH, now=NOW, runner=bd) == "updated"
        assert "create" not in bd.call_verbs()
        assert bd.call_verbs() == ["list", "update"]

    def test_repeated_breaches_never_accumulate_beads(self) -> None:
        """Thirty runs of one outage must leave one bead, not thirty."""
        issues = [_issue()]
        for _ in range(30):
            bd = FakeBd(issues)
            assert cph.notify(BREACH, now=NOW, runner=bd) == "updated"
            assert "create" not in bd.call_verbs()

    def test_oldest_bead_wins_when_several_are_open(self) -> None:
        """Keeps 'first seen' truthful if a race or a human left a duplicate."""
        bd = FakeBd(
            [
                _issue("scix_experiments-new", "2026-07-28T07:45:00+00:00"),
                _issue("scix_experiments-old", "2026-07-20T07:45:00+00:00"),
            ]
        )
        cph.notify(BREACH, now=NOW, runner=bd)
        update = next(c for c in bd.calls if c[0] == "update")
        assert update[1] == "scix_experiments-old"


class TestRecoveryClosesIt:
    def test_closes_the_open_bead_when_everything_passes(self) -> None:
        bd = FakeBd([_issue()])
        assert cph.notify(PASS, now=NOW, runner=bd) == "closed"
        assert bd.call_verbs() == ["list", "close"]

    def test_close_reason_records_when_it_recovered(self) -> None:
        bd = FakeBd([_issue()])
        cph.notify(PASS, now=NOW, runner=bd)
        close = next(c for c in bd.calls if c[0] == "close")
        assert NOW.isoformat() in close[close.index("--reason") + 1]

    def test_healthy_with_no_open_bead_touches_nothing(self) -> None:
        bd = FakeBd([])
        assert cph.notify(PASS, now=NOW, runner=bd) == "noop"
        assert bd.call_verbs() == ["list"]


class TestBreachBody:
    def test_names_every_failing_check_with_its_detail(self) -> None:
        body = cph.breach_body(BREACH, now=NOW, first_seen=None)
        assert "last_run_complete" in body
        assert "49.2h ago" in body
        assert "9052 titled papers missing" in body

    def test_reports_passing_checks_too(self) -> None:
        """Which checks still pass is how a reader narrows the cause."""
        body = cph.breach_body(BREACH, now=NOW, first_seen=None)
        assert "v_claim_edges_fresh" in body

    def test_preserves_first_seen_so_outage_length_is_visible(self) -> None:
        body = cph.breach_body(BREACH, now=NOW, first_seen="2026-07-15T07:45:00+00:00")
        assert "2026-07-15T07:45:00+00:00" in body
        assert NOW.isoformat() in body

    def test_carries_the_reproduce_command(self) -> None:
        body = cph.breach_body(BREACH, now=NOW, first_seen=None)
        assert "check_pipeline_health.py --allow-prod" in body

    def test_says_it_closes_itself(self) -> None:
        """Without this a reader closes it by hand and loses the signal."""
        body = cph.breach_body(BREACH, now=NOW, first_seen=None)
        assert "closed automatically" in body.lower()

    def test_check_names_carry_no_markdown_emphasis(self) -> None:
        """Check names contain underscores; the bead renderer reads ** as emphasis
        and mangled "**last_run_complete**" into "**last_****run_****complete**"."""
        body = cph.breach_body(BREACH, now=NOW, first_seen=None)
        assert "**" not in body

    def test_failing_and_passing_are_distinguishable_without_markup(self) -> None:
        body = cph.breach_body(BREACH, now=NOW, first_seen=None)
        assert "- FAIL last_run_complete" in body
        assert "- pass v_claim_edges_fresh" in body


class TestChannelFailureIsNotSilent:
    def test_lookup_failure_raises(self) -> None:
        bd = FakeBd([], fail_on="list")
        with pytest.raises(cph.NotifyError, match="bd list"):
            cph.notify(BREACH, now=NOW, runner=bd)

    def test_create_failure_raises(self) -> None:
        bd = FakeBd([], fail_on="create")
        with pytest.raises(cph.NotifyError, match="bd create"):
            cph.notify(BREACH, now=NOW, runner=bd)

    def test_close_failure_raises(self) -> None:
        bd = FakeBd([_issue()], fail_on="close")
        with pytest.raises(cph.NotifyError, match="bd close"):
            cph.notify(PASS, now=NOW, runner=bd)

    def test_unparseable_output_raises_rather_than_filing_a_duplicate(self) -> None:
        class Garbage(FakeBd):
            def __call__(self, argv):
                self.calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "not json", "")

        with pytest.raises(cph.NotifyError, match="unparseable"):
            cph.notify(BREACH, now=NOW, runner=Garbage([]))

    def test_empty_output_is_treated_as_no_open_bead(self) -> None:
        class Empty(FakeBd):
            def __call__(self, argv):
                self.calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "", "")

        assert cph.notify(BREACH, now=NOW, runner=Empty([])) == "created"


class TestCLIWiring:
    def test_notify_flag_is_documented(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "check_pipeline_health.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "--notify" in result.stdout

    def test_notify_is_off_by_default(self) -> None:
        """The in-pipeline invocation must not file beads; only the cron one does."""
        assert cph.build_parser().parse_args(["--dsn", "dbname=scix_test"]).notify is False

    def test_notify_parses_when_given(self) -> None:
        args = cph.build_parser().parse_args(["--dsn", "dbname=scix_test", "--notify"])
        assert args.notify is True

    def test_daily_sync_does_not_pass_notify(self) -> None:
        """The gate is invoked twice; only the out-of-band cron run may file beads.

        If daily_sync.sh passed --notify, a run that died before the last line
        would file nothing at all — the exact blind spot this closes.
        """
        script = (REPO_ROOT / "scripts" / "daily_sync.sh").read_text()
        health_lines = [ln for ln in script.splitlines() if "check_pipeline_health" in ln]
        assert health_lines, "daily_sync.sh no longer invokes the health gate"
        assert not any("--notify" in ln for ln in health_lines)
