#!/usr/bin/env python3
"""CLI for loading UAT concept hierarchy into the SciX database."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.uat import run_pipeline


def main() -> None:
    """Parse arguments and run the UAT loading pipeline."""
    parser = argparse.ArgumentParser(
        description="Load UAT concept hierarchy into SciX database",
    )
    parser.add_argument(
        "--skos",
        type=Path,
        default=None,
        help="Path to SKOS RDF/XML file (downloads if not provided)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
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
    )

    run_pipeline(skos_path=args.skos, dsn=args.dsn)


if __name__ == "__main__":
    main()
