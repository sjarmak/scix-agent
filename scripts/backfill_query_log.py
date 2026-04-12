#!/usr/bin/env python3
"""Backfill query_log from structured MCP session log files (M3.5.0).

Each line of the source file is a JSON object with at least:

    {"tool": "search_dual", "query": "...", "result_count": 12,
     "session_id": "abc", "ts": "2026-04-12T00:00:00Z", "is_test": false}

``ts`` is optional — if absent the row is inserted with ``now()`` via the
column default. ``session_id``/``is_test`` are also optional.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import psycopg

# Allow running as ``python scripts/backfill_query_log.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scix.db import DEFAULT_DSN, get_connection  # noqa: E402

logger = logging.getLogger(__name__)


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s:%d — %s", path, line_num, exc)


def run_backfill(
    source: Path,
    conn: psycopg.Connection,
    *,
    default_is_test: bool = False,
) -> int:
    """Replay ``source`` into ``query_log``. Returns rows inserted.

    Uses the extended INSERT directly (not ``log_query``) so that an explicit
    ``ts`` from the source log is preserved instead of being clobbered by the
    column default.
    """
    inserted = 0
    with conn.cursor() as cur:
        for rec in _iter_records(source):
            tool = rec.get("tool")
            query = rec.get("query", "")
            result_count = rec.get("result_count", 0)
            session_id = rec.get("session_id")
            ts = rec.get("ts")
            is_test = bool(rec.get("is_test", default_is_test))

            if tool is None:
                logger.warning("Skipping record without 'tool': %r", rec)
                continue

            if ts is None:
                cur.execute(
                    """
                    INSERT INTO query_log (
                        tool_name, success, tool, query,
                        result_count, session_id, is_test
                    ) VALUES (%s, TRUE, %s, %s, %s, %s, %s)
                    """,
                    (tool, tool, query, int(result_count), session_id, is_test),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO query_log (
                        tool_name, success, ts, tool, query,
                        result_count, session_id, is_test
                    ) VALUES (%s, TRUE, %s, %s, %s, %s, %s, %s)
                    """,
                    (tool, ts, tool, query, int(result_count), session_id, is_test),
                )
            inserted += 1
    conn.commit()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to JSONL log file")
    parser.add_argument(
        "--dsn", default=None, help="DB DSN (defaults to SCIX_TEST_DSN or SCIX_DSN)"
    )
    parser.add_argument("--is-test", action="store_true", help="Tag inserted rows as test traffic")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.source.exists():
        print(f"ERROR: source file does not exist: {args.source}", file=sys.stderr)
        return 2

    dsn = args.dsn or os.environ.get("SCIX_TEST_DSN") or DEFAULT_DSN
    conn = get_connection(dsn)
    try:
        n = run_backfill(args.source, conn, default_is_test=args.is_test)
        print(f"Backfilled {n} rows from {args.source}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
