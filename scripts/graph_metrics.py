#!/usr/bin/env python3
"""CLI for graph metrics computation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.graph_metrics import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute graph metrics on SciX citation graph")
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip resolution calibration, use fixed values",
    )
    parser.add_argument(
        "--res-coarse",
        type=float,
        default=5.0,
        help="Leiden resolution for coarse communities (default: 5.0)",
    )
    parser.add_argument(
        "--res-medium",
        type=float,
        default=1.0,
        help="Leiden resolution for medium communities (default: 1.0)",
    )
    parser.add_argument(
        "--res-fine",
        type=float,
        default=0.1,
        help="Leiden resolution for fine communities (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Leiden (default: 42)",
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

    run_pipeline(
        dsn=args.dsn,
        calibrate=not args.no_calibrate,
        res_coarse=args.res_coarse,
        res_medium=args.res_medium,
        res_fine=args.res_fine,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
