#!/usr/bin/env python3
"""Aggregate citation_edges into a community-to-community heatmap.

One row per (src_community, tgt_community) at the chosen resolution.
Runs a single GROUP BY pass over citation_edges + paper_metrics joins.
On 299M edges × 32M papers this takes ~1-3 minutes at coarse resolution.

Output schema:
    {
        "resolution": "coarse",
        "n_communities": 20,
        "cells": [
            {"src": 0, "tgt": 1, "n": 12345},
            ...
        ],
        "row_totals": [...],   # indexed by src community
        "col_totals": [...],   # indexed by tgt community
        "grand_total": N,
    }

Usage:
    .venv/bin/python scripts/viz/build_citation_heatmap.py \
        --resolution coarse --output data/viz/citation_heatmap.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import psycopg

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn

logger = logging.getLogger("build_citation_heatmap")

COMMUNITY_COLUMNS = {
    "coarse": "community_semantic_coarse",
    "medium": "community_semantic_medium",
    "fine": "community_semantic_fine",
}


def run(dsn: str, resolution: str, output: Path) -> dict:
    column = COMMUNITY_COLUMNS[resolution]
    logger.info(
        "building citation heatmap dsn=%s resolution=%s column=%s",
        redact_dsn(dsn),
        resolution,
        column,
    )
    if is_production_dsn(dsn):
        logger.info("DSN points at production (read-only query, proceeding)")

    sql = f"""
        SELECT
            src.{column} AS src_c,
            tgt.{column} AS tgt_c,
            count(*) AS n
        FROM citation_edges ce
        JOIN paper_metrics src ON ce.source_bibcode = src.bibcode
        JOIN paper_metrics tgt ON ce.target_bibcode = tgt.bibcode
        WHERE src.{column} IS NOT NULL
          AND tgt.{column} IS NOT NULL
        GROUP BY src.{column}, tgt.{column}
    """

    t0 = time.time()
    with psycopg.connect(dsn) as conn:
        conn.set_read_only(True)
        with conn.cursor() as cur:
            # Boost work_mem so the 13M+ paper_metrics hash joins don't spill
            # to base/pgsql_tmp. Medium/fine grouping fan out to 40K/400K cells
            # respectively, and the default 256MB work_mem triggers disk spill
            # on busy hosts. This is a session-scoped SET — no global impact.
            cur.execute("SET work_mem = '8GB'")
            logger.info("executing aggregation (~1-3 min on 299M edges)")
            cur.execute(sql)
            rows = cur.fetchall()
    dt = time.time() - t0
    logger.info("aggregation returned %d cells in %.1fs", len(rows), dt)

    communities = sorted({int(r[0]) for r in rows} | {int(r[1]) for r in rows})
    n_communities = len(communities)
    cells = [{"src": int(s), "tgt": int(t), "n": int(n)} for s, t, n in rows]
    # Row/col totals indexed by community id order.
    row_totals = {c: 0 for c in communities}
    col_totals = {c: 0 for c in communities}
    grand = 0
    for s, t, n in rows:
        row_totals[int(s)] += int(n)
        col_totals[int(t)] += int(n)
        grand += int(n)
    result = {
        "resolution": resolution,
        "n_communities": n_communities,
        "communities": communities,
        "cells": cells,
        "row_totals": [row_totals[c] for c in communities],
        "col_totals": [col_totals[c] for c in communities],
        "grand_total": grand,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    logger.info(
        "wrote %s (%d cells, %d communities, %d total edges)",
        output,
        len(cells),
        n_communities,
        grand,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--resolution", choices=list(COMMUNITY_COLUMNS), default="coarse")
    parser.add_argument("--output", type=Path, default=Path("data/viz/citation_heatmap.json"))
    ns = parser.parse_args(argv)
    run(ns.dsn, ns.resolution, ns.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
