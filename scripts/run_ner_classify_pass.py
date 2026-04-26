#!/usr/bin/env python3
"""Run the INDUS post-classifier over GLiNER document_entities (dbl.3 quality lift).

Reads pending mentions (match_method='gliner', no agreement verdict yet),
classifies each via INDUS anchor-centroid cosine, writes result back to
document_entities.evidence as a jsonb merge.

ALWAYS wrap in scix-batch (CLAUDE.md memory rule):

    scix-batch --mem-high 8G --mem-max 12G \\
        python scripts/run_ner_classify_pass.py

Examples:
    # Sample on the most recent 50K mentions (no commit if --dry-run)
    python scripts/run_ner_classify_pass.py --max-rows 50000

    # Full pass, conservative batch size
    scix-batch python scripts/run_ner_classify_pass.py --batch-size 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.extract.ner_classifier import DEFAULT_ANCHORS_PATH, NerClassifier  # noqa: E402
from scix.extract.ner_classify_pass import run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--since-bibcode", default=None)
    p.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS_PATH)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--dsn", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    classifier = NerClassifier(anchors_path=args.anchors, device=args.device)

    conn = get_connection(args.dsn)
    try:
        totals = run(
            conn,
            classifier,
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_rows=args.max_rows,
        )
        n_judged = totals.agreements + totals.disagreements
        agree_pct = (totals.agreements / n_judged * 100) if n_judged else 0.0
        logging.info(
            "TOTAL: rows=%d judged=%d agree=%d (%.1f%%) skipped_no_text=%d " "infer=%.1fs db=%.1fs",
            totals.rows_seen,
            n_judged,
            totals.agreements,
            agree_pct,
            totals.skipped_no_text,
            totals.elapsed_inference_s,
            totals.elapsed_db_s,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
