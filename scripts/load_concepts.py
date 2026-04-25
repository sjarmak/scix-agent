#!/usr/bin/env python3
"""Load cross-discipline vocabularies into the ``concepts`` substrate.

Drives the per-vocabulary loaders in ``scix.concept_loaders``. Each loader
is idempotent (TEMP-stage + COPY + ON CONFLICT upsert), so re-running the
script refreshes labels in place. Bead: scix_experiments-dbl.1.

Examples:
    # Load all five vocabularies
    python scripts/load_concepts.py

    # Load just one
    python scripts/load_concepts.py --vocab openalex

    # Load multiple
    python scripts/load_concepts.py --vocab acm_ccs --vocab msc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.concept_loaders import (
    acm_ccs,
    chebi,
    gcmd,
    gene_ontology,
    mesh,
    msc,
    ncbi_taxonomy,
    openalex,
    physh,
)
from scix.db import get_connection

logger = logging.getLogger(__name__)

LOADERS = {
    "openalex": openalex,
    "acm_ccs": acm_ccs,
    "msc": msc,
    "physh": physh,
    "gcmd": gcmd,
    # dbl.2 — biomed vocabularies
    "mesh": mesh,
    "ncbi_tax": ncbi_taxonomy,
    "chebi": chebi,
    "gene_ontology": gene_ontology,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--vocab",
        action="append",
        choices=sorted(LOADERS.keys()),
        help="Vocabulary to load (repeatable). Defaults to all.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database DSN (defaults to SCIX_DSN env var).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    selected = args.vocab or sorted(LOADERS.keys())
    conn = get_connection(args.dsn)
    try:
        totals = {"concepts": 0, "relationships": 0}
        for name in selected:
            loader = LOADERS[name]
            logger.info("== %s ==", name)
            result = loader.load(conn)
            totals["concepts"] += result["concepts"]
            totals["relationships"] += result["relationships"]
            logger.info("%s: %s", name, result)
        logger.info("TOTAL: %s", totals)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
