#!/usr/bin/env python3
"""One-time seed of the ``indus_qdrant_synced`` watermark from Qdrant (bead s7cy).

After ``paper_embeddings`` was dropped, the source of truth for "which papers
already have an INDUS vector" is the Qdrant serving collection itself. This
scrolls ``scix_indus_v2_papers_s1`` and records every point's bibcode into the
watermark table (migration 072), so the embed pipeline's anti-join only targets
the post-drop gap instead of re-embedding the entire 32 M corpus.

Idempotent (``ON CONFLICT DO NOTHING``); safe to re-run. Read-only against
Qdrant. Recommended under scix-batch given the full-collection scroll.

    scix-batch python scripts/seed_indus_qdrant_synced.py --allow-prod
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg  # noqa: E402

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402
from scix.qdrant_dense import INDUS_COLLECTION  # noqa: E402

logger = logging.getLogger("seed_indus_qdrant_synced")

DEFAULT_QDRANT_URL = "http://127.0.0.1:6633"


def bibcodes_from_points(points: Iterable[Any]) -> list[str]:
    """Extract bibcodes from scrolled Qdrant points, skipping any without one."""
    out: list[str] = []
    for p in points:
        bibcode = (p.payload or {}).get("bibcode")
        if bibcode:
            out.append(bibcode)
    return out


def insert_synced(conn: psycopg.Connection, bibcodes: list[str]) -> int:
    """Insert bibcodes into the watermark, ignoring ones already present."""
    if not bibcodes:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO indus_qdrant_synced (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
            [(b,) for b in bibcodes],
        )
    conn.commit()
    return len(bibcodes)


def seed(conn: psycopg.Connection, client: Any, collection: str, batch_size: int) -> int:
    """Scroll the whole collection into the watermark. Returns points seen."""
    seen = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        insert_synced(conn, bibcodes_from_points(points))
        seen += len(points)
        if seen % (batch_size * 20) == 0:
            logger.info("seeded %d points...", seen)
        if offset is None:
            break
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL (default: env or local)")
    parser.add_argument("--collection", default=INDUS_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--allow-prod", action="store_true", help="Required for the production DSN")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(args.dsn),
        )
        return 2

    import os

    from qdrant_client import QdrantClient

    url = args.qdrant_url or os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL
    client = QdrantClient(url=url, timeout=60)

    t0 = time.monotonic()
    with psycopg.connect(args.dsn) as conn:
        seen = seed(conn, client, args.collection, args.batch_size)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM indus_qdrant_synced")
            total = cur.fetchone()[0]
    logger.info(
        "seed complete: %d points scrolled, %d watermark rows, %.1fs",
        seen,
        total,
        time.monotonic() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
