"""Backfill is_retracted on the existing scix_papers_v1 Qdrant pilot collection.

Reference implementation for the field-addition discipline in
docs/ADR/008_qdrant_payload_schema.md. Given a bool field that doesn't
exist on the live points, we:

  1. Create the payload index (idempotent — Qdrant skips if it exists).
  2. Pull retracted bibcodes from Postgres (papers.retracted_at IS NOT NULL).
  3. Set is_retracted=True on the matching points by stable point_id hash.

Non-retracted points are left without the field (absence-treated-as-false
semantics — see ADR-008). This keeps the backfill at O(retracted) writes
(~2.5K) instead of O(collection) writes (~400K) and matches the
must_not(match_value=true) filter pattern documented in the ADR.

Idempotent: re-runs only re-set the same true values on the same points;
no duplication. Safe to run before, during, or after the upserter.

Usage:
    python scripts/backfill_qdrant_is_retracted.py
    python scripts/backfill_qdrant_is_retracted.py --collection scix_papers_v1 \
        --batch 200 --dry-run

Env:
    QDRANT_URL   default http://127.0.0.1:6333
    SCIX_DSN     Postgres DSN
"""
from __future__ import annotations

import argparse
import os
import struct
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from scix.db import get_connection


def bibcode_to_point_id(bibcode: str) -> int:
    """Stable 63-bit int derived from bibcode.

    MUST stay byte-identical to the function in
    scripts/qdrant_upsert_pilot.py — both scripts target the same
    point_ids in the same collection. Any change here must be made
    there too, plus a rebackfill of every payload field.
    """
    import hashlib

    h = hashlib.blake2b(bibcode.encode("utf-8"), digest_size=8).digest()
    return struct.unpack(">Q", h)[0] >> 1  # keep positive


def fetch_retracted_bibcodes(conn) -> list[str]:
    """All bibcodes currently flagged retracted in Postgres."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode FROM papers WHERE retracted_at IS NOT NULL ORDER BY bibcode"
        )
        return [row[0] for row in cur.fetchall()]


def ensure_index(client: QdrantClient, collection: str) -> None:
    """Create the is_retracted payload index. Idempotent — Qdrant skips if it exists."""
    try:
        client.create_payload_index(
            collection,
            field_name="is_retracted",
            field_schema=qm.PayloadSchemaType.BOOL,
        )
        print(f"created is_retracted index on {collection}")
    except Exception as exc:
        # Qdrant raises when the index already exists; treat that as success
        # and re-raise everything else.
        msg = str(exc).lower()
        if "already" in msg or "exists" in msg:
            print(f"is_retracted index already exists on {collection}")
            return
        raise


def set_is_retracted(
    client: QdrantClient,
    collection: str,
    bibcodes: list[str],
    *,
    batch: int,
    dry_run: bool,
) -> int:
    """Apply is_retracted=True to points whose payload bibcode is in ``bibcodes``.

    Filter-based update so points outside the pilot (the pilot only carries the
    top 400K by PageRank, while papers.retracted_at IS NOT NULL covers the full
    32M corpus) are silently skipped instead of raising 404. Qdrant
    short-circuits when the filter matches no points in a batch, so over-asking
    is cheap.

    Returns the number of bibcodes attempted (not the number of points actually
    matched — that's not exposed by the set_payload API).
    """
    attempted = 0
    for i in range(0, len(bibcodes), batch):
        chunk = bibcodes[i : i + batch]
        if dry_run:
            print(f"[dry-run] would attempt is_retracted=True on {len(chunk)} bibcodes")
            attempted += len(chunk)
            continue
        client.set_payload(
            collection_name=collection,
            payload={"is_retracted": True},
            points=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="bibcode", match=qm.MatchAny(any=chunk)
                    )
                ]
            ),
            wait=True,
        )
        attempted += len(chunk)
        if i % (batch * 5) == 0:
            print(f"  wrote up to {attempted}/{len(bibcodes)} (filter-based)")
    return attempted


def count_is_retracted_true(client: QdrantClient, collection: str) -> int:
    """Count points with is_retracted=True. Used for verification."""
    return client.count(
        collection_name=collection,
        count_filter=qm.Filter(
            must=[qm.FieldCondition(key="is_retracted", match=qm.MatchValue(value=True))]
        ),
        exact=True,
    ).count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="scix_papers_v1")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    client = QdrantClient(url=url, timeout=60)

    # Refuse to run against a collection that doesn't exist — better a
    # loud failure than silently inserting points by id into a new
    # auto-created collection.
    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        raise SystemExit(
            f"collection {args.collection!r} not found in Qdrant — "
            f"available: {sorted(existing)}"
        )

    ensure_index(client, args.collection)

    t0 = time.time()
    with get_connection() as conn:
        retracted = fetch_retracted_bibcodes(conn)
    print(f"{len(retracted)} retracted bibcodes in Postgres")

    if not retracted:
        print("nothing to backfill")
        return

    attempted = set_is_retracted(
        client,
        args.collection,
        retracted,
        batch=args.batch,
        dry_run=args.dry_run,
    )
    elapsed = time.time() - t0

    if args.dry_run:
        print(
            f"done — would attempt {attempted} bibcodes in {elapsed:.1f}s (dry-run)"
        )
        return

    matched = count_is_retracted_true(client, args.collection)
    print(
        f"done — attempted {attempted} bibcodes, "
        f"{matched} points now flagged is_retracted=True in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
