#!/usr/bin/env python3
"""CLI entry point for the SciX JSONL ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.ingest import IngestPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ADS JSONL metadata into PostgreSQL"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ads_metadata_by_year_picard"),
        help="Directory containing JSONL files (default: ads_metadata_by_year_picard/)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Records per batch (default: 10000)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Ingest a single file instead of the entire data directory",
    )
    parser.add_argument(
        "--no-drop-indexes",
        action="store_true",
        help="Skip dropping/recreating indexes (slower for bulk, useful for incremental)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = IngestPipeline(
        data_dir=args.data_dir,
        dsn=args.dsn,
        batch_size=args.batch_size,
    )
    pipeline.run(
        drop_indexes=not args.no_drop_indexes,
        single_file=args.file,
    )


if __name__ == "__main__":
    main()
