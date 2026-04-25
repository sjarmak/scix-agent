#!/usr/bin/env python3
"""Run the GLiNER NER backfill pass over papers (dbl.3).

ALWAYS wrap in scix-batch (see CLAUDE.md memory rule on systemd-oomd):

    scix-batch --mem-high 8G --mem-max 12G \\
        python scripts/run_ner_pass.py --target abstract

Examples:
    # Sample run: first 1000 abstracts in bibcode order, dry-run (no DB writes)
    python scripts/run_ner_pass.py --target abstract --max-papers 1000 --dry-run

    # Resume an interrupted full pass from a watermark bibcode
    scix-batch python scripts/run_ner_pass.py --target abstract \\
        --since-bibcode 2020ApJ...900....1S

    # Phase 1 production run — full 23 M abstracts (4 days at 100 docs/s)
    scix-batch --mem-high 12G --mem-max 16G \\
        python scripts/run_ner_pass.py --target abstract --batch-size 1000

The pipeline is resumable: every batch is checkpointed in ``ingest_log``,
so a killed run picks up at the next un-checkpointed batch on rerun.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.extract.ner_pass import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_INFERENCE_BATCH,
    DEFAULT_MODEL_NAME,
    NER_SOURCE_VERSION,
    GlinerExtractor,
    run,
)

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--target",
        choices=("abstract", "body"),
        default="abstract",
        help="Which papers column to extract from (default: abstract).",
    )
    p.add_argument("--batch-size", type=int, default=1000, help="Papers per checkpointed batch.")
    p.add_argument(
        "--since-bibcode",
        default=None,
        help="Resume watermark — only process bibcodes strictly greater than this.",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for sample / dev runs).",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"GLiNER mention confidence floor (default: {DEFAULT_CONFIDENCE}).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model id.")
    p.add_argument(
        "--source-version",
        default=NER_SOURCE_VERSION,
        help="Stamp written to entities.source_version. Bump for new label set or model.",
    )
    p.add_argument(
        "--inference-batch",
        type=int,
        default=DEFAULT_INFERENCE_BATCH,
        help="GLiNER batch_predict batch size.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but skip DB writes (for sample / quality checks).",
    )
    p.add_argument(
        "--require-batch-scope",
        action="store_true",
        help="Refuse to run unless invoked under systemd-run scope (CLAUDE.md rule).",
    )
    p.add_argument("--dsn", default=None, help="Database DSN; defaults to SCIX_DSN.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.require_batch_scope and "SYSTEMD_SCOPE" not in os.environ:
        sys.stderr.write(
            "ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.\n"
            "       Run via: scix-batch python scripts/run_ner_pass.py ...\n"
        )
        return 2

    extractor = GlinerExtractor(
        model_name=args.model,
        confidence=args.confidence,
        inference_batch=args.inference_batch,
    )

    conn = get_connection(args.dsn)
    try:
        if args.dry_run:
            logger.info("DRY RUN — DB writes suppressed.")
            from scix.extract.ner_pass import iter_paper_batches

            n_papers = 0
            n_mentions = 0
            for batch in iter_paper_batches(
                conn,
                target=args.target,
                batch_size=args.batch_size,
                since_bibcode=args.since_bibcode,
                max_papers=args.max_papers,
            ):
                per_doc = extractor.predict(batch)
                n_papers += len(batch)
                for paper, mentions in zip(batch, per_doc, strict=True):
                    for m in mentions:
                        n_mentions += 1
                        sys.stdout.write(
                            f"{paper.bibcode}\t{m.entity_type}\t{m.confidence:.2f}\t"
                            f"{m.canonical_name}\t{m.surface_text}\n"
                        )
            logger.info("dry-run done: %d papers, %d mentions", n_papers, n_mentions)
            return 0

        totals = run(
            conn,
            extractor,
            target=args.target,
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
            source_version=args.source_version,
        )
        logger.info(
            "TOTAL: papers=%d papers_with_mentions=%d mentions=%d "
            "new_entities=%d doc_entities=%d "
            "infer=%.1fs db=%.1fs",
            totals.papers_seen,
            totals.papers_with_mentions,
            totals.mentions_kept,
            totals.new_entities,
            totals.upserted_doc_entities,
            totals.elapsed_inference_s,
            totals.elapsed_db_s,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
