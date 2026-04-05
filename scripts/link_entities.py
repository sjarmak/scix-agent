#!/usr/bin/env python3
"""CLI entrypoint for the document-entity linking pipeline.

Usage:
    python scripts/link_entities.py --db-url "dbname=scix"
    python scripts/link_entities.py --batch-size 500 --resume
    python scripts/link_entities.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

from scix.db import get_connection
from scix.link_entities import get_linking_progress, link_entities_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Link extracted entity mentions to canonical entities."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of bibcodes per commit batch (default: 1000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip bibcodes already linked in document_entities.",
    )
    parser.add_argument(
        "--extraction-type",
        type=str,
        default="entity_extraction_v3",
        help="Extraction type to process (default: entity_extraction_v3).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Database connection string (default: SCIX_DSN env or 'dbname=scix').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Count mentions without writing to database.",
    )

    args = parser.parse_args(argv)

    conn = get_connection(args.db_url)
    try:
        if args.dry_run:
            logger.info("DRY RUN — no rows will be written")

        summary = link_entities_batch(
            conn,
            batch_size=args.batch_size,
            resume=args.resume,
            extraction_type=args.extraction_type,
            dry_run=args.dry_run,
        )

        print(f"\nLinking summary:")
        print(f"  Bibcodes processed: {summary['bibcodes_processed']}")
        print(f"  Links created:      {summary['links_created']}")
        print(f"  Skipped (no match): {summary['skipped_no_match']}")

        progress = get_linking_progress(conn)
        print(f"\nOverall progress:")
        print(f"  Total bibcodes:   {progress['total_bibcodes']}")
        print(f"  Linked bibcodes:  {progress['linked_bibcodes']}")
        print(f"  Pending bibcodes: {progress['pending_bibcodes']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
