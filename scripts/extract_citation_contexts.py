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
import shutil
import sys
from collections.abc import Mapping

from scix.citation_context import run_pipeline
from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn

logger = logging.getLogger(__name__)

INGEST_LOG_BASE_FILENAME = "citctx_full_backfill_2026"

# Free-disk floor for the citation_contexts shard backfill (bead 6hr7 AC d).
# Full population projection is ~232 GB; a single shard adds ~58 GB. The
# 50 GB floor leaves headroom for WAL + autovacuum bursts during the run.
DEFAULT_MIN_FREE_GB = 50
DEFAULT_FREE_DISK_PATH = "/"


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


def enforce_free_disk_guard(
    *,
    path: str,
    min_free_gb: int,
) -> None:
    """Refuse to run if free disk on *path* is below *min_free_gb*.

    Citation_contexts shard runs add ~58 GB each and the full backfill
    projects to ~232 GB; a 50 GB floor leaves headroom for WAL +
    autovacuum bursts. Raises :class:`SystemExit` (code 3) so the script
    surfaces a non-zero exit distinct from the prod-guard's exit code 2.
    """
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        logger.error(
            "Refusing to run: free disk on %s is %.1f GB, below the %d GB floor. "
            "Free space, lower SCIX_MIN_FREE_GB, or pass --min-free-gb to override.",
            path,
            free_gb,
            min_free_gb,
        )
        raise SystemExit(3)
    logger.info(
        "Free-disk guard: %.1f GB free on %s (floor %d GB).",
        free_gb,
        path,
        min_free_gb,
    )


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
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help=(
            "Process closed-access papers as well as OA/preprints. Default "
            "is OA-only (bead 8584 publisher-policy gate). Use only with "
            "explicit operator approval."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=int,
        default=int(os.environ.get("SCIX_MIN_FREE_GB", DEFAULT_MIN_FREE_GB)),
        help=(
            "Minimum free disk in GB on --free-disk-path before the run is "
            f"allowed to start (default: {DEFAULT_MIN_FREE_GB}, env: "
            "SCIX_MIN_FREE_GB). Enforced regardless of --allow-prod."
        ),
    )
    parser.add_argument(
        "--free-disk-path",
        type=str,
        default=os.environ.get("SCIX_FREE_DISK_PATH", DEFAULT_FREE_DISK_PATH),
        help=(
            f"Filesystem path to check for free space (default: "
            f"{DEFAULT_FREE_DISK_PATH!r}, env: SCIX_FREE_DISK_PATH). "
            "Should resolve to the partition holding the Postgres data dir."
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
    enforce_free_disk_guard(path=args.free_disk_path, min_free_gb=args.min_free_gb)

    ingest_log_filename = None if args.no_ingest_log else ingest_log_filename_for_shard(shard)

    logger.info(
        "Running citation-context extraction on %s "
        "(shard=%s, batch_size=%d, limit=%s, oa_only=%s)",
        redact_dsn(dsn),
        args.shard,
        args.batch_size,
        args.limit,
        not args.include_closed,
    )
    if args.include_closed:
        logger.warning(
            "--include-closed is ACTIVE: citation-context extraction will run "
            "on closed-access papers as well as OA/preprints. Confirm operator "
            "approval and publisher-agreement (Wiley/Springer TDM) clearance "
            "before each run."
        )

    total = run_pipeline(
        dsn=args.dsn,
        batch_size=args.batch_size,
        limit=args.limit,
        shard=shard,
        ingest_log_filename=ingest_log_filename,
        oa_only=not args.include_closed,
    )

    print(f"Inserted {total} citation context rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
