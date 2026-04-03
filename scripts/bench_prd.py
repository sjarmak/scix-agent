#!/usr/bin/env python3
"""CLI runner for SciX PRD benchmark validation.

Usage:
    python scripts/bench_prd.py
    python scripts/bench_prd.py --iterations 10 --output report.md
    python scripts/bench_prd.py --skip-semantic  # skip model-dependent benchmarks
    python scripts/bench_prd.py --device auto     # use GPU if available
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.benchmark import generate_markdown_report, run_prd_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="SciX PRD benchmark validation")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        choices=range(1, 1001),
        metavar="N",
        help="Number of iterations per benchmark (1-1000, default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for markdown report (default: stdout)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for SPECTER2 model (default: cpu)",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip benchmarks that require SPECTER2 model loading (1 and 4)",
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

    report = run_prd_benchmarks(
        args.dsn,
        iterations=args.iterations,
        device=args.device,
        skip_semantic=args.skip_semantic,
    )

    markdown = generate_markdown_report(report)

    if args.output:
        Path(args.output).write_text(markdown)
        logging.getLogger(__name__).info("Report written to %s", args.output)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
