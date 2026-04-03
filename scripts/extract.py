#!/usr/bin/env python3
"""CLI entry point for the SciX entity extraction pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.extract import run_extraction_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract entities from ADS paper abstracts via Anthropic Batches API"
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=10_000,
        help="Number of papers to extract (by citation_count DESC; default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Papers per batch submission (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extractions",
        help="Directory for JSONL checkpoint files (default: data/extractions)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model ID (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--extraction-version",
        default="v1",
        help="Version tag for extractions (default: v1)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between batch status polls (default: 60)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    total = run_extraction_pipeline(
        dsn=args.dsn,
        pilot_size=args.pilot_size,
        model=args.model,
        output_dir=args.output_dir,
        extraction_version=args.extraction_version,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
    )
    logger = logging.getLogger(__name__)
    logger.info("Done. Loaded %d extraction rows.", total)


if __name__ == "__main__":
    main()
