#!/usr/bin/env python3
"""Verify post-driver ``papers_fulltext`` populated state.

Part of PRD ``structural-fulltext-parsing`` W4 (D3 verify portion). The
fulltext driver ingests parsed papers into ``papers_fulltext`` tagged by
``source`` (one of ``ar5iv``, ``arxiv_local``, ``s2orc``, ``ads_body``,
``docling``, ``abstract``). After a driver run on ``scix_test``, we gate
promotion to production by asserting each expected tier has >= 1 row.

Checks performed
----------------

For each tier name supplied via ``--require-tiers`` (default
``ar5iv,ads_body``):

* ``SELECT count(*) FROM papers_fulltext WHERE source = %s``
* Emit ``PASS: tier {name} has {N} rows`` if N >= 1.
* Emit ``FAIL: tier {name} has 0 rows`` otherwise.

Exit codes
----------

* ``0`` — every required tier has >= 1 row.
* ``1`` — at least one required tier has 0 rows.

Usage
-----

::

    # Against the test DB after a driver run:
    python scripts/verify_fulltext_populated.py \\
        --dsn "dbname=scix_test" \\
        --require-tiers "ar5iv,ads_body"

The script never writes to the database. It is safe to run against
production, though the intended use is as a pre-promotion gate on
``scix_test``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import DEFAULT_DSN, redact_dsn  # noqa: E402

logger = logging.getLogger("verify_fulltext")

DEFAULT_REQUIRE_TIERS = "ar5iv,ads_body"


def _count_tier(conn: psycopg.Connection, tier: str) -> int:
    """Return count of ``papers_fulltext`` rows whose ``source`` equals ``tier``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM papers_fulltext WHERE source = %s",
            (tier,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def verify(conn: psycopg.Connection, tiers: Sequence[str]) -> tuple[int, list[str]]:
    """Count rows per tier; return ``(exit_code, messages)``.

    ``exit_code`` is 0 when every tier has >= 1 row, else 1. ``messages``
    contains one ``PASS`` or ``FAIL`` line per tier, in input order.
    """
    messages: list[str] = []
    failed = False
    for tier in tiers:
        count = _count_tier(conn, tier)
        if count >= 1:
            messages.append(f"PASS: tier {tier} has {count} rows")
        else:
            messages.append(f"FAIL: tier {tier} has 0 rows")
            failed = True
    return (1 if failed else 0), messages


def _parse_tiers(raw: str) -> list[str]:
    """Split a comma-separated tier list, drop empties, strip whitespace."""
    return [t.strip() for t in raw.split(",") if t.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify papers_fulltext has >= 1 row per required tier source. "
            "Exit 0 on success, 1 if any tier is empty."
        )
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help=(
            "PostgreSQL DSN (default: $SCIX_DSN or dbname=scix). Read-only. "
            "Intended target is scix_test for pre-promotion gating."
        ),
    )
    parser.add_argument(
        "--require-tiers",
        default=DEFAULT_REQUIRE_TIERS,
        help=(
            "Comma-separated list of expected tier source names "
            f"(default: {DEFAULT_REQUIRE_TIERS!r})."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging (PASS/FAIL lines still emitted).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tiers = _parse_tiers(args.require_tiers)
    if not tiers:
        print("FAIL: --require-tiers produced an empty tier list")
        return 1

    dsn_redacted = redact_dsn(args.dsn)
    logger.info("verifying papers_fulltext tiers against %s", dsn_redacted)

    with psycopg.connect(args.dsn) as conn:
        conn.autocommit = True
        exit_code, messages = verify(conn, tiers)

    for msg in messages:
        print(msg)

    if exit_code == 0:
        logger.info("all %d tier(s) passed", len(tiers))
    else:
        failed = [m for m in messages if m.startswith("FAIL")]
        logger.warning("%d tier(s) failed", len(failed))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
