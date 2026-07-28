#!/usr/bin/env python3
"""Post-run health gate for the daily ADS pipeline (GOAL W6, bead tdl).

The daily pipeline failed on every run from 2026-07-15 to 2026-07-27 and
nothing noticed, because a silent failure is indistinguishable from a success.
This script closes that gap: it asserts three independent properties and exits
non-zero with a one-line reason for each breach.

    1. run completeness  — the last ``daily_sync.sh`` run recorded all six
       steps as ok/skipped, and that run is recent.
    2. dense-lane gap    — papers with a title that are missing from the INDUS
       Qdrant serving lane (``indus_qdrant_synced``, migration 072) are under
       a threshold.
    3. view freshness    — ``v_claim_edges`` was refreshed within N days.

Mechanism for (1): ``daily_sync.sh`` writes a status file
(``logs/daily_sync_status.json``) from an EXIT trap, so an aborted run leaves a
truthful record of which steps ran rather than no record at all. Parsing the
log was the alternative and was rejected: the log is append-only across runs,
rotated by size (not by run), and its format is prose meant for humans, so a
parser would silently drift the first time a step's echo line is reworded.

Running it only from daily_sync.sh leaves the gate blind to its own headline
failure: if the script dies before the last line, or cron never fires it, the
check never runs — and "never ran" is indistinguishable from "passed". The
out-of-band invocation plus ``--notify`` is what closes that:

    45 7 * * * cd /home/ds/projects/scix_experiments && \
        .venv/bin/python scripts/check_pipeline_health.py --allow-prod --notify

Usage:
    python scripts/check_pipeline_health.py --allow-prod
    python scripts/check_pipeline_health.py --allow-prod --notify
    python scripts/check_pipeline_health.py --max-dense-gap 100 --dsn "dbname=scix_test"

Exit codes:
    0  all checks passed
    1  at least one check breached
    2  refused to run (production DSN without --allow-prod, or missing scope)
    3  checks ran but --notify could not reach the bead store (the alerting
       channel itself is broken — distinct from 1 so it cannot be mistaken for
       an ordinary breach)

Production safety: read-only (three SELECTs), but gated behind ``--allow-prod``
like every other production script here. ``--require-batch-scope`` is available
for parity with the batch convention; it is off by default because this script
is read-only and allocates nothing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import psycopg

# Add src/ to path for direct script execution.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger("check_pipeline_health")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_STATUS_FILE = REPO_ROOT / "logs" / "daily_sync_status.json"

# daily_sync.sh numbers its steps 1..6; see the header of that script.
EXPECTED_STEPS = ("1", "2", "3", "4", "5", "6")
# A step that ran (ok) or was legitimately not needed (skipped) is complete.
# "failed" and "missing" are breaches.
COMPLETE_STATUSES = frozenset({"ok", "skipped"})

# Defaults, all overridable by flag.
#
# max_run_age_hours=36: cron runs at 06:15 UTC daily. 36 h means "the last run
# is at most one cadence old plus half a day of slack" — a single skipped cron
# fires the check without flapping on a run that started late or ran long.
DEFAULT_MAX_RUN_AGE_HOURS = 36.0
# max_dense_gap=500: after a healthy run the gap is ~0, because Step 5 embeds
# everything Steps 2/4 ingested. Daily harvest volume is ~750-2500 papers, so
# one missed embed day pushes the gap past 500 while a partial batch or a
# handful of stubborn papers does not. (The gap sat at 9052 after the 12-day
# GPU outage — ~750/day.)
DEFAULT_MAX_DENSE_GAP = 500
# max_view_age_days=2: v_claim_edges is refreshed by Step 6 every day, so 2
# days tolerates exactly one missed run.
DEFAULT_MAX_VIEW_AGE_DAYS = 2.0

# ingest_log key written by scripts/refresh_v_claim_edges.py on every refresh.
VIEW_REFRESH_LOG_KEY = "refresh::v_claim_edges"

DENSE_GAP_SQL = """
    SELECT count(*)
      FROM papers p
      LEFT JOIN indus_qdrant_synced s USING (bibcode)
     WHERE s.bibcode IS NULL
       AND p.title IS NOT NULL
"""

# No status filter: refresh_v_claim_edges.py upserts on a unique `filename`, so a
# failed refresh OVERWRITES the last successful row. Filtering on
# status='complete' would therefore turn a failed refresh into "never refreshed"
# and throw away the age, which is the signal an operator most needs then. Read
# the row whatever its status and let the checker judge it.
VIEW_REFRESH_SQL = """
    SELECT finished_at, status
      FROM ingest_log
     WHERE filename = %(key)s
"""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one assertion. ``detail`` is printed verbatim, so it must
    carry the observed value and the threshold it was judged against."""

    name: str
    ok: bool
    detail: str


# ---------------------------------------------------------------------------
# Check 1 — did the last run complete every step?
# ---------------------------------------------------------------------------


def load_status(path: pathlib.Path) -> dict | None:
    """Read the daily_sync status file. Returns None if it is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ts(raw: object) -> _dt.datetime | None:
    """Parse an ISO-8601 UTC timestamp as written by daily_sync.sh's ts()."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def check_last_run(
    status: dict | None,
    *,
    now: _dt.datetime,
    max_age_hours: float,
    status_path: pathlib.Path,
) -> CheckResult:
    """Assert the most recent daily_sync run recorded all six steps, recently."""
    name = "last_run_complete"
    if status is None:
        return CheckResult(
            name,
            False,
            f"no status file at {status_path} — daily_sync.sh has not run since the health gate was added",
        )

    finished_at = _parse_ts(status.get("finished_at"))
    if finished_at is None:
        return CheckResult(name, False, f"status file {status_path} has no parseable finished_at")

    age_h = (now - finished_at).total_seconds() / 3600.0
    if age_h > max_age_hours:
        return CheckResult(
            name,
            False,
            f"last run finished {age_h:.1f}h ago ({status.get('finished_at')}), "
            f"limit {max_age_hours:.0f}h — the daily job is not running",
        )

    steps = status.get("steps")
    if not isinstance(steps, dict):
        return CheckResult(name, False, f"status file {status_path} has no steps object")

    bad = [
        f"{n}={steps.get(n, 'missing')}"
        for n in EXPECTED_STEPS
        if steps.get(n) not in COMPLETE_STATUSES
    ]
    if bad:
        return CheckResult(
            name,
            False,
            f"last run ({status.get('finished_at')}) did not complete every step: {', '.join(bad)}",
        )

    return CheckResult(
        name,
        True,
        f"all {len(EXPECTED_STEPS)} steps complete, finished {age_h:.1f}h ago",
    )


# ---------------------------------------------------------------------------
# Check 2 — is the INDUS dense lane keeping up?
# ---------------------------------------------------------------------------


def query_dense_gap(conn: psycopg.Connection) -> int:
    """Count titled papers absent from the INDUS Qdrant serving lane."""
    with conn.cursor() as cur:
        cur.execute(DENSE_GAP_SQL)
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("dense-gap query returned no row")
    return int(row[0])


def check_dense_gap(gap: int, *, threshold: int) -> CheckResult:
    name = "dense_lane_gap"
    if gap > threshold:
        return CheckResult(
            name,
            False,
            f"{gap} titled papers missing from the INDUS dense lane, limit {threshold} "
            "— Step 5 (embed) is not keeping up",
        )
    return CheckResult(name, True, f"{gap} papers behind, limit {threshold}")


# ---------------------------------------------------------------------------
# Check 3 — is v_claim_edges fresh?
# ---------------------------------------------------------------------------


def query_view_refresh(conn: psycopg.Connection) -> tuple[_dt.datetime | None, str | None]:
    """Return ``(finished_at, status)`` for the last v_claim_edges refresh attempt.

    Source is the ingest_log row that scripts/refresh_v_claim_edges.py upserts;
    PostgreSQL does not record materialized-view refresh times itself. The row
    is the *last attempt*, not the last success — see VIEW_REFRESH_SQL.
    """
    with conn.cursor() as cur:
        cur.execute(VIEW_REFRESH_SQL, {"key": VIEW_REFRESH_LOG_KEY})
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None, None
    refreshed: _dt.datetime = row[0]
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=_dt.timezone.utc)
    return refreshed, row[1]


def check_view_freshness(
    refreshed_at: _dt.datetime | None,
    *,
    now: _dt.datetime,
    max_age_days: float,
    status: str | None = "complete",
) -> CheckResult:
    name = "v_claim_edges_fresh"
    if refreshed_at is None:
        return CheckResult(
            name,
            False,
            f"no refresh recorded in ingest_log under {VIEW_REFRESH_LOG_KEY!r}",
        )
    age_days = (now - refreshed_at.astimezone(_dt.timezone.utc)).total_seconds() / 86400.0
    if status != "complete":
        return CheckResult(
            name,
            False,
            f"last v_claim_edges refresh attempt {age_days:.1f}d ago "
            f"({refreshed_at.isoformat()}) ended status={status!r}, not 'complete'",
        )
    if age_days > max_age_days:
        return CheckResult(
            name,
            False,
            f"v_claim_edges last refreshed {age_days:.1f}d ago "
            f"({refreshed_at.isoformat()}), limit {max_age_days:g}d",
        )
    return CheckResult(name, True, f"refreshed {age_days:.1f}d ago, limit {max_age_days:g}d")


# ---------------------------------------------------------------------------
# Notification — file the breach where it will actually be seen
# ---------------------------------------------------------------------------
#
# The gate's blind spot is that it runs from the last line of daily_sync.sh, so
# the failure it exists to catch (the script dying early, or cron not firing) is
# exactly the one that stops it running. Fixing that needs the gate invoked
# out-of-band AND a way to reach a human that does not depend on the pipeline.
#
# That channel is the bead store, not email: this host has no MTA at all (no
# sendmail/mail/mailx, postfix and exim4 inactive), so a MAILTO line in the
# crontab would deliver nothing while looking like monitoring — the same silent
# failure this whole gate exists to remove. `bd ready` is read every session.
#
# Exactly one open bead is maintained, keyed by label: a breach opens or updates
# it, a healthy run closes it. Filing a fresh bead per run would produce thirty
# for one outage and train the reader to ignore them.

NOTIFY_LABEL = "pipeline-health"
NOTIFY_TITLE = "daily_sync pipeline health breach"
NOTIFY_PRIORITY = "1"
NOTIFY_TYPE = "bug"
NOTIFY_TIMEOUT_S = 30


class NotifyError(RuntimeError):
    """The notification channel itself failed."""


def _run_bd(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Default runner: invoke the `bd` CLI. Injected in tests."""
    return subprocess.run(
        ["bd", *argv],
        capture_output=True,
        text=True,
        timeout=NOTIFY_TIMEOUT_S,
        cwd=str(REPO_ROOT),
    )


def _checked(runner, argv: list[str]) -> str:
    result = runner(argv)
    if result.returncode != 0:
        raise NotifyError(f"bd {' '.join(argv)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def find_open_notification(runner) -> dict | None:
    """Return the open pipeline-health bead, or None.

    More than one open bead means a previous run raced or a human filed one by
    hand; the oldest is kept as the canonical record so `first seen` stays true.
    """
    raw = _checked(runner, ["list", "--status=open", "--label", NOTIFY_LABEL, "--json", "-n", "0"])
    try:
        issues = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        raise NotifyError(f"bd list returned unparseable JSON: {exc}") from exc
    if not issues:
        return None
    return sorted(issues, key=lambda i: i.get("created_at") or "")[0]


def breach_body(results: list[CheckResult], *, now: _dt.datetime, first_seen: str | None) -> str:
    """Render the bead description. Pure — the current state, not an append log.

    Deliberately overwritten each run rather than appended: the reader needs
    'what is broken now', and an append log of 30 identical breaches buries it.
    """
    breaches = [r for r in results if not r.ok]
    lines = [
        f"The daily ADS pipeline health gate is failing {len(breaches)} of {len(results)} checks.",
        "",
        f"Last checked: {now.isoformat()}",
    ]
    if first_seen:
        lines.append(f"First seen:   {first_seen}")
    # No markdown emphasis on check names: they contain underscores, and the
    # bead renderer reads those as emphasis markers — "**last_run_complete**"
    # displays as "**last_****run_****complete**". Plain FAIL/pass markers stay
    # readable and greppable.
    lines += ["", "## Failing", ""]
    lines += [f"- FAIL {r.name} — {r.detail}" for r in breaches]
    passing = [r for r in results if r.ok]
    if passing:
        lines += ["", "## Passing", ""]
        lines += [f"- pass {r.name} — {r.detail}" for r in passing]
    lines += [
        "",
        "## What to do",
        "",
        "Reproduce with:",
        "",
        "    .venv/bin/python scripts/check_pipeline_health.py --allow-prod",
        "",
        "This bead is maintained by that gate: it is updated while the breach",
        "persists and closed automatically on the first healthy run. Closing it",
        "by hand while the pipeline is still broken will simply re-open it.",
    ]
    return "\n".join(lines)


def notify(results: list[CheckResult], *, now: _dt.datetime, runner=_run_bd) -> str:
    """Sync the pipeline-health bead to ``results``. Returns the action taken."""
    existing = find_open_notification(runner)
    breached = [r for r in results if not r.ok]

    if not breached:
        if existing is None:
            return "noop"
        _checked(
            runner,
            [
                "close",
                existing["id"],
                "--reason",
                f"Pipeline healthy again as of {now.isoformat()}: "
                f"all {len(results)} checks pass. Closed automatically by "
                "scripts/check_pipeline_health.py --notify.",
            ],
        )
        return "closed"

    if existing is None:
        body = breach_body(results, now=now, first_seen=now.isoformat())
        _checked(
            runner,
            [
                "create",
                NOTIFY_TITLE,
                "-t",
                NOTIFY_TYPE,
                "-p",
                NOTIFY_PRIORITY,
                "-l",
                NOTIFY_LABEL,
                "-d",
                body,
            ],
        )
        return "created"

    body = breach_body(results, now=now, first_seen=existing.get("created_at"))
    _checked(runner, ["update", existing["id"], "-d", body])
    return "updated"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(results: list[CheckResult]) -> str:
    """One line per check, breaches first-class and greppable."""
    lines = []
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        lines.append(f"{marker} {result.name}: {result.detail}")
    breaches = [r for r in results if not r.ok]
    if breaches:
        lines.append(
            f"pipeline health: {len(breaches)}/{len(results)} checks FAILED "
            f"({', '.join(r.name for r in breaches)})"
        )
    else:
        lines.append(f"pipeline health: all {len(results)} checks passed")
    return "\n".join(lines)


def run_checks(
    conn: psycopg.Connection,
    *,
    status_path: pathlib.Path,
    now: _dt.datetime,
    max_run_age_hours: float,
    max_dense_gap: int,
    max_view_age_days: float,
) -> list[CheckResult]:
    """Run all three assertions. IO here, judgement in the pure checkers."""
    refreshed_at, refresh_status = query_view_refresh(conn)
    return [
        check_last_run(
            load_status(status_path),
            now=now,
            max_age_hours=max_run_age_hours,
            status_path=status_path,
        ),
        check_dense_gap(query_dense_gap(conn), threshold=max_dense_gap),
        check_view_freshness(
            refreshed_at,
            now=now,
            max_age_days=max_view_age_days,
            status=refresh_status,
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-run health gate for daily_sync.sh")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN (read-only)")
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against a production DSN (read-only, but gated).",
    )
    parser.add_argument(
        "--require-batch-scope",
        action="store_true",
        help="Refuse to run unless invoked under a systemd-run scope "
        "(SYSTEMD_SCOPE in env); per CLAUDE.md convention. Optional — this "
        "script is read-only and lightweight.",
    )
    parser.add_argument(
        "--status-file",
        type=pathlib.Path,
        default=DEFAULT_STATUS_FILE,
        help=f"daily_sync.sh run-status file (default: {DEFAULT_STATUS_FILE})",
    )
    parser.add_argument(
        "--max-run-age-hours",
        type=float,
        default=DEFAULT_MAX_RUN_AGE_HOURS,
        help=f"Fail if the last run finished longer ago (default: {DEFAULT_MAX_RUN_AGE_HOURS:g})",
    )
    parser.add_argument(
        "--max-dense-gap",
        type=int,
        default=DEFAULT_MAX_DENSE_GAP,
        help=f"Fail above this many unembedded titled papers (default: {DEFAULT_MAX_DENSE_GAP})",
    )
    parser.add_argument(
        "--max-view-age-days",
        type=float,
        default=DEFAULT_MAX_VIEW_AGE_DAYS,
        help=f"Fail if v_claim_edges is staler (default: {DEFAULT_MAX_VIEW_AGE_DAYS:g})",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Maintain a single 'pipeline-health' bead: open/update it on breach, "
        "close it on recovery. Intended for the out-of-band cron invocation, "
        "which is the only one that can catch daily_sync.sh never running.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod (read-only).",
            redact_dsn(args.dsn),
        )
        return 2

    if args.require_batch_scope and "SYSTEMD_SCOPE" not in os.environ:
        sys.stderr.write(
            "ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.\n"
            "       Run via: scix-batch python scripts/check_pipeline_health.py ...\n"
        )
        return 2

    now = _dt.datetime.now(_dt.timezone.utc)
    with psycopg.connect(args.dsn) as conn:
        results = run_checks(
            conn,
            status_path=args.status_file,
            now=now,
            max_run_age_hours=args.max_run_age_hours,
            max_dense_gap=args.max_dense_gap,
            max_view_age_days=args.max_view_age_days,
        )

    print(render(results))
    breached = any(not r.ok for r in results)

    if args.notify:
        try:
            action = notify(results, now=now)
        except (NotifyError, OSError, subprocess.SubprocessError) as exc:
            # The alerting channel is down. Say so loudly and distinctly: a
            # breach nobody can be told about is the failure mode this gate
            # exists to remove, so it must not hide behind the health result.
            logger.error("notification failed: %s", exc)
            return 3
        logger.info("notification: %s", action)

    return 1 if breached else 0


if __name__ == "__main__":
    sys.exit(main())
