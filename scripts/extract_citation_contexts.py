#!/usr/bin/env python3
"""Run the citation context extraction pipeline.

Reads papers with body text and references from the database, extracts
~250-word context windows around [N] citation markers, resolves markers
to target bibcodes, and stores results in the citation_contexts table.

Usage:
    python scripts/extract_citation_contexts.py
    python scripts/extract_citation_contexts.py --limit 1000 --batch-size 500
    python scripts/extract_citation_contexts.py --dsn "dbname=scix_test"
"""

from __future__ import annotations

import argparse
import logging
import sys

from scix.citation_context import run_pipeline


def main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    total = run_pipeline(
        dsn=args.dsn,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    print(f"Inserted {total} citation context rows.")


if __name__ == "__main__":
    main()
