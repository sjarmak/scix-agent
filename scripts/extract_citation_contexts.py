#!/usr/bin/env python3
"""Run the citation context extraction pipeline.

Reads papers with body text and references from the database, extracts
~250-word context windows around [N] citation markers, resolves markers
to target bibcodes, and stores results in the citation_contexts table.

Production safety: refuses to run against the production DSN unless
``--allow-prod`` is passed; ``--allow-prod`` itself requires a systemd
scope (set via ``scix-batch``) so oomd can enforce a memory ceiling
without collateral-killing the gascity supervisor — see CLAUDE.md
§Memory isolation.

Usage:
    python scripts/extract_citation_contexts.py
    python scripts/extract_citation_contexts.py --limit 1000 --batch-size 500
    python scripts/extract_citation_contexts.py --dsn "dbname=scix_test"

    # Sharded production run (PRD 79n.1):
    scix-batch python scripts/extract_citation_contexts.py \\
        --allow-prod --shard 0/4 --batch-size 1000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping

from scix.citation_context import run_pipeline
from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn

logger = logging.getLogger(__name__)

INGEST_LOG_BASE_FILENAME = "citctx_full_backfill_2026"


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse a ``"i/n"`` shard spec.

    Validates ``0 <= i < n`` and ``n > 0`` so an out-of-range value fails
    fast at argument parse rather than producing a silently-empty result
    set in the SELECT.
    """
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"shard must be of form 'i/n' (got {spec!r})")
    try:
        index = int(parts[0])
        total = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"shard parts must be integers (got {spec!r})") from exc
    if total <= 0:
        raise ValueError(f"shard total must be > 0 (got {total})")
    if index < 0 or index >= total:
        raise ValueError(f"shard index must satisfy 0 <= i < n (got {index}/{total})")
    return index, total


def enforce_prod_guard(
    *,
    dsn: str,
    allow_prod: bool,
    env: Mapping[str, str],
) -> None:
    """Refuse to run against prod unless ``--allow-prod`` AND systemd scope.

    Mirrors ``scripts/backfill_part_of_inheritance.py::enforce_prod_guard``.
    Raises :class:`SystemExit` (code 2) on policy violation so the script
    surfaces a non-zero exit to cron / scix-batch wrappers.
    """
    if is_production_dsn(dsn) and not allow_prod:
        logger.error(
            "Refusing to write to production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        raise SystemExit(2)

    if allow_prod and not env.get("INVOCATION_ID"):
        logger.error(
            "Refusing to run --allow-prod outside a systemd scope. "
            "Invoke via: scix-batch python %s <args...>",
            os.path.basename(sys.argv[0] or __file__),
        )
        raise SystemExit(2)


def ingest_log_filename_for_shard(shard: tuple[int, int] | None) -> str:
    """Derive the ``ingest_log`` filename used to track this run.

    Sharded runs use a per-worker filename so independent processes don't
    overwrite each other's progress rows; an admin step rolls them up
    into the canonical bare filename when the full backfill lands.
    """
    if shard is None:
        return INGEST_LOG_BASE_FILENAME
    index, total = shard
    return f"{INGEST_LOG_BASE_FILENAME}_shard_{index}_of_{total}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract citation contexts from paper body text.",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=None,
        help="Database connection string (default: from SCIX_DSN or dbname=scix)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of context rows to accumulate before flushing via COPY (default: 1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: all)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="i/n",
        help=(
            "Shard spec 'i/n'. Restricts to papers where mod(hashtext(bibcode), n) = i; "
            "use to run N concurrent workers without locking. Required for the full "
            "backfill — see docs/prd/citation_contexts_throughput.md."
        ),
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against production DSN (dbname=scix).",
    )
    parser.add_argument(
        "--no-ingest-log",
        action="store_true",
        help=(
            "Skip ingest_log bookkeeping (default: write to ingest_log under "
            f"{INGEST_LOG_BASE_FILENAME!r}, or a per-shard variant when --shard is set)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    shard: tuple[int, int] | None = None
    if args.shard is not None:
        try:
            shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))

    dsn = args.dsn if args.dsn else DEFAULT_DSN
    enforce_prod_guard(dsn=dsn, allow_prod=args.allow_prod, env=os.environ)

    ingest_log_filename = None if args.no_ingest_log else ingest_log_filename_for_shard(shard)

    total = run_pipeline(
        dsn=args.dsn,
        batch_size=args.batch_size,
        limit=args.limit,
        shard=shard,
        ingest_log_filename=ingest_log_filename,
    )

    print(f"Inserted {total} citation context rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
