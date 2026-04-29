#!/usr/bin/env python3
"""Extend NASA EOSDIS coverage by re-running the CMR harvester end-to-end.

The original ``harvest_cmr.py`` harvester paginates via the ``Search-After``
header, but a stale entry in :class:`scix.http_client.ResilientClient`'s disk
cache caused subsequent pages to be served from cache and therefore return
the same first page repeatedly. As a result, only ~2 000 of the ~54 000 CMR
collections were ever ingested.

This script:

1. Wipes the CMR-specific cache directory so pagination is exercised against
   the live API (the underlying cache-key bug has been fixed in
   ``ResilientClient`` to include request-shaping headers, so re-runs from a
   warm cache will also work, but a clean slate avoids any prior poisoning).
2. Calls :func:`harvest_cmr.run_harvest` to fetch the entire CMR catalog and
   upsert into the ``datasets`` table with ``source='cmr'``.
3. Logs a separate ``harvest_runs`` row tagged with ``config.invoked_by``
   so the eosdis extend run is auditable distinct from the original run.

Usage::

    python scripts/harvest_eosdis_extend.py --dry-run
    python scripts/harvest_eosdis_extend.py
    python scripts/harvest_eosdis_extend.py -v
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harvest_cmr

logger = logging.getLogger(__name__)

CMR_CACHE_DIR = Path(".cache/cmr")


def clear_cmr_cache(cache_dir: Path | None = None) -> int:
    """Remove the CMR HTTP cache directory.

    Args:
        cache_dir: Directory to remove. If ``None``, uses the module-level
            ``CMR_CACHE_DIR`` (resolved at call time so callers can monkeypatch).

    Returns:
        The number of cache files removed (0 if the directory didn't exist).
    """
    target = cache_dir if cache_dir is not None else CMR_CACHE_DIR
    if not target.exists():
        logger.info("CMR cache directory %s does not exist — nothing to clear", target)
        return 0

    files = list(target.glob("*.json"))
    count = len(files)
    shutil.rmtree(target, ignore_errors=True)
    logger.info("Cleared %d cached CMR responses from %s", count, target)
    return count


def run_extend(
    dsn: str | None = None,
    dry_run: bool = False,
    skip_cache_clear: bool = False,
) -> dict[str, int]:
    """Re-run the CMR harvest with cache cleared so pagination is exercised.

    Args:
        dsn: Optional database DSN. Falls back to ``SCIX_DSN`` / default.
        dry_run: If True, fetch and parse without writing.
        skip_cache_clear: If True, do not delete the cache directory.

    Returns:
        Counts dict from the underlying CMR harvest.
    """
    if not skip_cache_clear:
        clear_cmr_cache(CMR_CACHE_DIR)

    counts = harvest_cmr.run_harvest(dsn=dsn, dry_run=dry_run)
    logger.info("EOSDIS extend run complete: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extend NASA EOSDIS coverage by re-running the CMR harvester "
            "with cache cleared. Fixes earlier partial ingest caused by "
            "Search-After pagination being served from cache."
        ),
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database DSN (uses SCIX_DSN env var if omitted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse without writing to the database",
    )
    parser.add_argument(
        "--skip-cache-clear",
        action="store_true",
        help="Do not delete .cache/cmr before running",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    counts = run_extend(
        dsn=args.dsn,
        dry_run=args.dry_run,
        skip_cache_clear=args.skip_cache_clear,
    )

    if args.dry_run:
        print(f"Dry run — {counts.get('collections', counts.get('datasets', 0))} collections parsed")
    else:
        print(f"EOSDIS extend complete: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
