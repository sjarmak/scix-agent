#!/usr/bin/env python3
"""CLI entry point for the SciX embedding pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.embed import NIGHTLY_YEAR_LOOKBACK, default_year_floor, run_embedding_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate INDUS embeddings for ADS papers and upsert to the Qdrant dense lane"
    )
    parser.add_argument(
        "--model",
        default="indus",
        help="Embedding model name (default: indus)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Papers per embedding batch (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max papers to embed (useful for testing; default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Scan the whole corpus instead of recent years. Required to pick up "
            "backfilled older papers, since the default bound is on publication "
            "year, not ingest date. Costs a full seq scan of papers (~530 s "
            "before the first row on a cold cache) — do not use for the nightly run."
        ),
    )
    parser.add_argument(
        "--year-floor",
        type=int,
        default=None,
        help=(
            "Only embed papers with year >= this value "
            f"(default: current year - {NIGHTLY_YEAR_LOOKBACK}). Ignored with --full."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.full and args.year_floor is not None:
        parser.error("--full and --year-floor are mutually exclusive")
    year_floor = None if args.full else (args.year_floor or default_year_floor())

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    total = run_embedding_pipeline(
        dsn=args.dsn,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        year_floor=year_floor,
    )
    logger = logging.getLogger(__name__)
    logger.info("Done. Embedded %d papers.", total)


if __name__ == "__main__":
    main()
