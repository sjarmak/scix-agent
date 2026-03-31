#!/usr/bin/env python3
"""CLI entry point for the SciX embedding pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.embed import run_embedding_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for ADS papers and store in PostgreSQL"
    )
    parser.add_argument(
        "--model",
        default="specter2",
        help="Embedding model name (default: specter2)",
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
        "--no-copy",
        action="store_true",
        help="Use individual INSERTs instead of COPY (slower, useful for debugging)",
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

    total = run_embedding_pipeline(
        dsn=args.dsn,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        use_copy=not args.no_copy,
    )
    logger = logging.getLogger(__name__)
    logger.info("Done. Embedded %d papers.", total)


if __name__ == "__main__":
    main()
