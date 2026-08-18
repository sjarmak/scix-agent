#!/usr/bin/env python3
"""Hold one idle connection to the scix production database, forever.

Why this exists
---------------
On 2026-08-10 an agent working in another repo classified `scix` as a dead
pre-Qdrant benchmark artifact and dropped it, costing a 193 GB restore and a
20-hour GIN rebuild. Its reasoning was not careless: it verified the archive
table by table before acting. The premise it never tested was whether
anything still used the database, and the evidence it accepted for "nothing
does" was `ss` and pg_stat_activity showing zero connections.

That evidence was worthless. The MCP server is a stdio process started per
session and holds no socket while no session is open; daily_sync connects
once a day. Idle looked identical to dead.

This process makes the two distinguishable, and does it through the same
signal an agent already checks:

  1. `DROP DATABASE scix` fails outright while any session is connected.
     Verified: dropdb returns rc=1 with "database is being accessed by other
     users". A superuser cannot talk its way past this the way it can past
     ownership -- ownership was tested first and does NOT gate a superuser
     (`ds` dropped a postgres-owned database with rc=0).
  2. The connection names itself. An agent running the standard preflight
     sees application_name='scix-production-sentinel' in pg_stat_activity
     rather than a bare count, so the refusal arrives with its own
     explanation instead of looking like a transient lock.

`dropdb --force` still gets through, and that is deliberate. The goal is not
to make the database undroppable; it is to make dropping it an explicit act
that cannot be reached by an agent that believes the thing is already dead.

Design constraints
------------------
The connection must be genuinely idle between heartbeats, never
idle-in-transaction and never sitting inside a long-running statement -- both
of those pin a snapshot and would block vacuum on a 587 GB database. So:
autocommit is on, the heartbeat is a bare `SELECT 1`, and the wait happens
client-side.

Run via deploy/systemd/scix-production-sentinel.service.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

import psycopg

DSN = os.environ.get("SCIX_SENTINEL_DSN", "dbname=scix")
APP_NAME = "scix-production-sentinel"
HEARTBEAT_SECONDS = 60
RECONNECT_BASE_SECONDS = 5
RECONNECT_MAX_SECONDS = 300

log = logging.getLogger("scix.sentinel")

_stop = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _stop
    _stop = True
    log.info("received signal %d, shutting down", signum)


def _hold_connection() -> None:
    """Open one connection and heartbeat it until told to stop or it dies."""
    with psycopg.connect(DSN, application_name=APP_NAME, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            row = cur.fetchone()
            pid = row[0] if row else None
        log.info("holding connection to %s (backend pid %s)", DSN, pid)

        while not _stop:
            # Sleep in small slices so a stop signal is honoured promptly
            # rather than after a full heartbeat interval.
            waited = 0.0
            while waited < HEARTBEAT_SECONDS and not _stop:
                time.sleep(1.0)
                waited += 1.0
            if _stop:
                return
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    backoff = RECONNECT_BASE_SECONDS
    while not _stop:
        try:
            _hold_connection()
            backoff = RECONNECT_BASE_SECONDS
        except Exception as exc:
            # A dropped connection means Postgres restarted or was killed.
            # That is exactly when the guard must come back, so retry rather
            # than exit -- but never in a tight loop against a dead server.
            log.warning("connection lost (%s); retrying in %ds", exc, backoff)
            slept = 0.0
            while slept < backoff and not _stop:
                time.sleep(1.0)
                slept += 1.0
            backoff = min(backoff * 2, RECONNECT_MAX_SECONDS)

    log.info("sentinel stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
