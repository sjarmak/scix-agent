#!/usr/bin/env python3
"""CLI entry point for the ADS body loader.

Loads the `body` field from an ADS-harvested JSONL file into papers_ads_body
and flips papers_external_ids.has_ads_body. See src/scix/ads_body.py for the
module-level implementation and safety guarantees.

Usage:
    SCIX_TEST_DSN="dbname=scix_test" python scripts/ingest_ads_body.py \
        --dsn "dbname=scix_test" \
        --jsonl fixtures/ads_body.jsonl \
        --batch-size 10000

To target production, --yes-production must be passed explicitly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution (mirrors scripts/ingest.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.ads_body import (  # noqa: E402
    AdsBodyLoader,
    LoaderConfig,
    ProductionGuardError,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ADS body field from JSONL into papers_ads_body.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix'). "
        "Targeting production requires --yes-production.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to the JSONL file (plain, .gz, or .xz).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Records per COPY batch (default: 10000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + stage records but do not commit any rows.",
    )
    parser.add_argument(
        "--yes-production",
        action="store_true",
        help="Explicitly allow targeting a production DSN. "
        "Without this flag the loader refuses if dbname=scix.",
    )
    parser.add_argument(
        "--drop-indexes",
        action="store_true",
        help="Drop papers_ads_body GIN index before load and recreate after "
        "(bulk-load performance flag; unused for small smoke loads).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = LoaderConfig(
        dsn=args.dsn,
        jsonl_path=args.jsonl,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        yes_production=args.yes_production,
        drop_indexes=args.drop_indexes,
    )

    try:
        stats = AdsBodyLoader(cfg).run()
    except ProductionGuardError as exc:
        logging.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    logging.info(
        "Done: loaded=%d skipped=%d dry_run=%s already_complete=%s elapsed=%.2fs",
        stats.records_loaded,
        stats.records_skipped,
        stats.dry_run,
        stats.already_complete,
        stats.elapsed_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
