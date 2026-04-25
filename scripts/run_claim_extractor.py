#!/usr/bin/env python3
"""Run the regex-first quantitative-claim extractor across paper bodies (M4).

Always wrap heavy production runs in scix-batch (see CLAUDE.md memory rule
on systemd-oomd):

    scix-batch python scripts/run_claim_extractor.py --target body \\
        --max-papers 1000 --dry-run

    scix-batch --mem-high 4G --mem-max 8G \\
        python scripts/run_claim_extractor.py --target body --allow-prod

The pipeline is resumable: each batch is processed in bibcode order and
``--since-bibcode`` lets you continue from a checkpoint.

Writes go to ``staging.extractions`` with ``extraction_type='quant_claim'``
and an aggregated JSONB ``payload`` of the per-paper claim spans. The
unique key ``(bibcode, extraction_type, extraction_version)`` lets the
script use ``ON CONFLICT DO UPDATE`` so a re-run is idempotent.

The ``--allow-prod`` guard mirrors ``scripts/run_ner_pass.py`` /
``scripts/refresh_fusion_mv.py``: any DSN whose dbname is in the
production set (currently ``{'scix'}``) refuses to run without it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

# Make src/ importable when running from a worktree without an editable
# install — same pattern as scripts/run_ner_pass.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.claim_extractor import (  # noqa: E402  (post-sys.path)
    EXTRACTION_SOURCE,
    EXTRACTION_TYPE,
    EXTRACTION_VERSION,
    ClaimSpan,
    extract_claims,
    to_payload,
)
from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    is_production_dsn,
    redact_dsn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Iteration over papers
# ---------------------------------------------------------------------------


def iter_paper_batches(
    conn,
    *,
    target: str = "body",
    batch_size: int = 200,
    since_bibcode: str | None = None,
    max_papers: int | None = None,
) -> Iterable[list[tuple[str, str]]]:
    """Yield ``(bibcode, text)`` batches from ``papers.<target>``.

    ``target`` is ``'body'`` or ``'abstract'``. The cursor walks bibcodes
    in lexicographic order so ``--since-bibcode`` is a stable resume key.
    Rows with NULL or empty target text are skipped.
    """
    if target not in {"body", "abstract"}:
        raise ValueError(f"unsupported target: {target!r}")

    n_yielded = 0
    last_bibcode = since_bibcode

    while True:
        if max_papers is not None and n_yielded >= max_papers:
            return
        remaining = (
            None if max_papers is None else max_papers - n_yielded
        )
        chunk = batch_size if remaining is None else min(batch_size, remaining)

        sql_text = (
            f"SELECT bibcode, {target} FROM papers "
            f"WHERE {target} IS NOT NULL AND {target} <> '' "
            "AND (%s::text IS NULL OR bibcode > %s::text) "
            "ORDER BY bibcode LIMIT %s"
        )
        with conn.cursor() as cur:
            cur.execute(sql_text, (last_bibcode, last_bibcode, chunk))
            rows = cur.fetchall()

        if not rows:
            return

        batch = [(r[0], r[1]) for r in rows]
        last_bibcode = batch[-1][0]
        n_yielded += len(batch)
        yield batch


# ---------------------------------------------------------------------------
# Insert into staging.extractions
# ---------------------------------------------------------------------------


_INSERT_SQL: str = (
    "INSERT INTO staging.extractions "
    "(bibcode, extraction_type, extraction_version, payload, source, confidence_tier) "
    "VALUES (%s, %s, %s, %s::jsonb, %s, %s) "
    "ON CONFLICT (bibcode, extraction_type, extraction_version) "
    "DO UPDATE SET payload = EXCLUDED.payload, "
    "             source = EXCLUDED.source, "
    "             confidence_tier = EXCLUDED.confidence_tier, "
    "             created_at = now()"
)


def _best_confidence_tier(claims: list[ClaimSpan]) -> int | None:
    """Return the most-confident tier across a paper's claims (1=high)."""
    if not claims:
        return None
    return min(c.confidence_tier for c in claims)


def insert_claims(conn, bibcode: str, claims: list[ClaimSpan]) -> int:
    """Insert (or upsert) one row into staging.extractions for a paper.

    Returns 1 if a row was written, 0 if no claims were extracted (we do
    not write empty payloads — they would inflate the table without
    informational content).
    """
    if not claims:
        return 0
    payload = json.dumps(to_payload(claims))
    tier = _best_confidence_tier(claims)
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_SQL,
            (
                bibcode,
                EXTRACTION_TYPE,
                EXTRACTION_VERSION,
                payload,
                EXTRACTION_SOURCE,
                tier,
            ),
        )
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regex-first quantitative-claim extractor (M4).",
    )
    p.add_argument(
        "--target",
        choices=("body", "abstract"),
        default="body",
        help="Which papers column to scan (default: body).",
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
        default=200,
        help="Papers per cursor batch (default: 200).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction but skip DB writes (sample / quality checks).",
    )
    p.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against the production DSN (mirrors run_ner_pass.py).",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help="Database DSN; defaults to SCIX_DSN.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.dsn or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    logger.info(
        "Running claim extractor on %s (target=%s, dry_run=%s, since=%s, max=%s)",
        redact_dsn(dsn),
        args.target,
        args.dry_run,
        args.since_bibcode,
        args.max_papers,
    )

    conn = get_connection(dsn)
    try:
        n_papers = 0
        n_papers_with_claims = 0
        n_claims = 0
        n_inserted = 0

        for batch in iter_paper_batches(
            conn,
            target=args.target,
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
        ):
            for bibcode, text in batch:
                n_papers += 1
                claims = extract_claims(text)
                if claims:
                    n_papers_with_claims += 1
                    n_claims += len(claims)
                if args.dry_run:
                    for c in claims:
                        sys.stdout.write(
                            f"{bibcode}\t{c.quantity}\t{c.value}\t"
                            f"{c.uncertainty}\t{c.unit}\t{c.span}\n"
                        )
                else:
                    n_inserted += insert_claims(conn, bibcode, claims)
            if not args.dry_run:
                conn.commit()
            logger.info(
                "progress: papers=%d with_claims=%d claims=%d inserted=%d",
                n_papers,
                n_papers_with_claims,
                n_claims,
                n_inserted,
            )

        logger.info(
            "DONE: papers=%d with_claims=%d claims=%d inserted=%d",
            n_papers,
            n_papers_with_claims,
            n_claims,
            n_inserted,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
