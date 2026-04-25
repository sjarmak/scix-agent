#!/usr/bin/env python3
"""Run the negative-results detector over papers (PRD M3).

ALWAYS wrap in scix-batch (CLAUDE.md memory rule on systemd-oomd):

    scix-batch python scripts/run_negative_results.py --max-papers 1000

Examples
--------
Sample run (no DB writes):

    python scripts/run_negative_results.py --max-papers 100 --dry-run

Resume from a watermark bibcode:

    scix-batch python scripts/run_negative_results.py \\
        --since-bibcode 2020ApJ...900....1S

Full pass against production (must explicitly opt in):

    scix-batch python scripts/run_negative_results.py --allow-prod

The script refuses to write to a production DSN unless ``--allow-prod`` is
passed. ``--dry-run`` skips writes entirely (still iterates papers and
reports counts).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src/ importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    is_production_dsn,
    redact_dsn,
)
from scix.negative_results import (  # noqa: E402
    EXTRACTION_TYPE,
    EXTRACTION_VERSION,
    SOURCE,
    detect_negative_results,
    insert_extractions,
)

logger = logging.getLogger(__name__)


def _iter_papers(
    conn,
    *,
    since_bibcode: str | None,
    max_papers: int | None,
    batch_size: int,
):
    """Yield ``(bibcode, body)`` pairs in bibcode order, body NOT NULL.

    Uses a server-side named cursor so the result set is streamed in
    chunks of ``batch_size`` instead of materialised in memory — bodies
    are large.
    """
    sql_parts = ["SELECT bibcode, body FROM papers WHERE body IS NOT NULL"]
    params: list[object] = []
    if since_bibcode is not None:
        sql_parts.append("AND bibcode > %s")
        params.append(since_bibcode)
    sql_parts.append("ORDER BY bibcode")
    if max_papers is not None:
        sql_parts.append("LIMIT %s")
        params.append(max_papers)

    with conn.cursor(name="neg_results_iter") as cur:
        cur.itersize = batch_size
        cur.execute(" ".join(sql_parts), params)
        for row in cur:
            yield row[0], row[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--dsn",
        default=None,
        help="Database DSN; defaults to SCIX_DSN.",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for sample / dev runs).",
    )
    p.add_argument(
        "--since-bibcode",
        default=None,
        help="Resume watermark — only process bibcodes strictly greater than this.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Server-side cursor itersize (default: 500).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection but skip DB writes (for sample / quality checks).",
    )
    p.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to write against production DSN. Off by default for safety.",
    )
    p.add_argument(
        "--require-batch-scope",
        action="store_true",
        help="Refuse to run unless invoked under systemd-run scope (CLAUDE.md rule).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.require_batch_scope and "SYSTEMD_SCOPE" not in os.environ:
        sys.stderr.write(
            "ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.\n"
            "       Run via: scix-batch python scripts/run_negative_results.py ...\n"
        )
        return 2

    dsn = args.dsn or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod and not args.dry_run:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override "
            "(or --dry-run for a no-write inspection).",
            redact_dsn(dsn),
        )
        return 2

    if args.dry_run:
        logger.info("DRY RUN — DB writes suppressed.")
    logger.info(
        "neg-results pipeline: dsn=%s extraction_type=%s version=%s source=%s",
        redact_dsn(dsn),
        EXTRACTION_TYPE,
        EXTRACTION_VERSION,
        SOURCE,
    )

    conn = get_connection(dsn)
    n_papers = 0
    n_papers_with_spans = 0
    n_spans = 0
    try:
        for bibcode, body in _iter_papers(
            conn,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
            batch_size=args.batch_size,
        ):
            spans = detect_negative_results(body)
            n_papers += 1
            if spans:
                n_papers_with_spans += 1
                n_spans += len(spans)

            if args.dry_run:
                if spans and args.verbose:
                    for s in spans:
                        sys.stdout.write(
                            f"{bibcode}\t{s.section}\t{s.pattern_id}\t"
                            f"{s.confidence_label}\t{s.match_text}\n"
                        )
                continue

            insert_extractions(conn, bibcode, spans)
            # Commit per-paper: keeps interrupted runs idempotent and bounded.
            conn.commit()

            if n_papers % args.batch_size == 0:
                logger.info(
                    "checkpoint: papers=%d papers_with_spans=%d spans=%d last=%s",
                    n_papers,
                    n_papers_with_spans,
                    n_spans,
                    bibcode,
                )

        logger.info(
            "TOTAL: papers=%d papers_with_spans=%d spans=%d",
            n_papers,
            n_papers_with_spans,
            n_spans,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
