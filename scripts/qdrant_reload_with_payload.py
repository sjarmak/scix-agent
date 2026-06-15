#!/usr/bin/env python3
"""Re-ingest the INDUS collection into a fresh Qdrant collection WITH the full
ADR-008 payload carried in each point — the fix for bead scix_experiments-tqg4.

Why this exists: per-point ``set_payload`` on ``scix_indus_v2_papers_s1`` costs
~47-63 s/op to APPLY (intrinsic to its ~1.3M-point on_disk segments; proven by
single-threaded WAL recovery replay — see results/qdrant_writepath/). Carrying
the payload at upsert time instead applies as a normal point write and leaves a
clean WAL (no restart poisoning).

Vector source: ``paper_embeddings`` was DROPPED 2026-06-14 (vectors now live only
in Qdrant), so this scrolls vectors out of the EXISTING source collection
(read-only) and enriches each point with payload looked up from Postgres
(``papers`` + ``paper_metrics``) by bibcode, then upserts into a NEW collection.
Payload indexes are built AFTER the bulk load. The source collection is never
mutated.

Reuses the already-unit-tested ``_build_payload`` / ``_bibcode_to_point_id`` /
``INDEXED_FIELDS`` from ``backfill_qdrant_filter_fields``.

SAFETY: writes only to ``--target-collection`` (defaults to a NON-prod ``*_payload``
name); reads the source read-only. Operator-gated, heavy: scrolling 32.4M vectors
is RAM- and IO-intensive — run under scix-batch in a RAM-headroom window, then
swap the alias.

Usage:
    # dry-run plan (1 scroll page from source + PG enrich, no writes):
    scix-batch python scripts/qdrant_reload_with_payload.py --dry-run --limit 256

    # build the payload-carrying collection, then swap:
    scix-batch python scripts/qdrant_reload_with_payload.py \
        --source-collection scix_indus_v2_papers_s1 \
        --target-collection scix_indus_v2_papers_s1_payload

Env:
    QDRANT_URL   default http://127.0.0.1:6633
    SCIX_DSN     Postgres DSN (falls back to scix.db.DEFAULT_DSN)
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import shutil
import threading
import time
from typing import Any

import psycopg

# Reuse the tested payload + point-id helpers (keeps them under
# tests/test_backfill_qdrant_filter_fields.py coverage — do NOT duplicate).
from backfill_qdrant_filter_fields import (  # noqa: E402 — sibling script import
    INDEXED_FIELDS,
    _build_payload,
)
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from scix.db import DEFAULT_DSN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("qdrant_reload_with_payload")

PROD_COLLECTION = "scix_indus_v2_papers_s1"
DIM = 768
SCROLL_BATCH = 1024
WORKERS = 4
DISK_FLOOR_GB = 10

# Payload-field lookup for a batch of bibcodes (no aggregation → no work_mem use).
_ENRICH_SQL = """
    SELECT
        p.bibcode,
        p.year, p.doctype, p.arxiv_class, p.bibstem,
        p.title, p.first_author, p.citation_count,
        (p.retracted_at IS NOT NULL) AS is_retracted,
        pm.community_semantic_coarse,
        pm.community_semantic_medium,
        pm.pagerank
    FROM papers p
    LEFT JOIN paper_metrics pm USING (bibcode)
    WHERE p.bibcode = ANY(%s)
"""


def disk_free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def _enrich(conn: psycopg.Connection, bibcodes: list[str]) -> dict[str, dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ENRICH_SQL, (bibcodes,))
        return {r["bibcode"]: r for r in cur.fetchall()}


def _points_from_page(page: list, by_bibcode: dict[str, dict[str, Any]]) -> list:
    """Build upsert points from a scroll page + PG enrichment, preserving id+vector."""
    out = []
    for rec in page:
        bibcode = (rec.payload or {}).get("bibcode")
        row = by_bibcode.get(bibcode) if bibcode else None
        payload = {"bibcode": bibcode} if bibcode else {}
        if row:
            payload.update(_build_payload(row))
        out.append(qm.PointStruct(id=rec.id, vector=rec.vector, payload=payload))
    return out


def _upsert_worker(client: QdrantClient, collection: str, q: "queue.Queue", errors: list) -> None:
    while True:
        pts = q.get()
        if pts is None:
            q.task_done()
            return
        try:
            client.upsert(collection, points=pts, wait=False)
        except Exception as e:  # noqa: BLE001 — record and stop the load
            errors.append(e)
        finally:
            q.task_done()


def _create_collection(client: QdrantClient, collection: str) -> None:
    """Target config byte-identical to prod (qdrant_full_load.py); indexes deferred."""
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(
            size=DIM, distance=qm.Distance.COSINE, datatype=qm.Datatype.FLOAT16, on_disk=True
        ),
        hnsw_config=qm.HnswConfigDiff(m=32, on_disk=False),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, quantile=0.99, always_ram=False
            )
        ),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=8),
    )
    log.info(
        "created %s (m=32, f16 on_disk, SQ-INT8 on_disk; payload indexes deferred)", collection
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-ingest INDUS collection with full ADR-008 payload")
    ap.add_argument("--source-collection", default=PROD_COLLECTION, help="read-only vector source")
    ap.add_argument(
        "--target-collection",
        default=f"{PROD_COLLECTION}_payload",
        help="Collection to build. Defaults to a NON-prod *_payload name.",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = full collection (cap for pilots)")
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6633"))
    ap.add_argument("--grpc-port", type=int, default=6634)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.target_collection == args.source_collection:
        raise SystemExit("target must differ from source (this builds a fresh collection)")

    client = QdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True, timeout=300)
    dsn = os.environ.get("SCIX_DSN", DEFAULT_DSN)

    if args.dry_run:
        page, _ = client.scroll(
            args.source_collection,
            limit=min(args.limit or SCROLL_BATCH, SCROLL_BATCH),
            with_vectors=True,
            with_payload=["bibcode"],
        )
        with psycopg.connect(dsn) as pg:
            bibs = [(r.payload or {}).get("bibcode") for r in page]
            by_bibcode = _enrich(pg, [b for b in bibs if b])
        pts = _points_from_page(page, by_bibcode)
        log.info(
            "[dry-run] scrolled %d pts from %s; %d enriched from PG; sample payload keys:",
            len(page),
            args.source_collection,
            len(by_bibcode),
        )
        for p in pts[:3]:
            log.info(
                "[dry-run]   id=%s vec_dim=%s payload_keys=%s",
                p.id,
                len(p.vector) if p.vector else 0,
                sorted(p.payload),
            )
        log.info(
            "[dry-run] would build %s + %d indexes after load",
            args.target_collection,
            len(INDEXED_FIELDS),
        )
        return

    if client.collection_exists(args.target_collection):
        raise SystemExit(
            f"{args.target_collection!r} exists — drop it or pick a new name (builds fresh for a clean WAL)"
        )
    _create_collection(client, args.target_collection)

    q: "queue.Queue" = queue.Queue(maxsize=WORKERS * 4)
    errors: list = []
    threads = [
        threading.Thread(
            target=_upsert_worker, args=(client, args.target_collection, q, errors), daemon=True
        )
        for _ in range(WORKERS)
    ]
    for t in threads:
        t.start()

    t0 = time.monotonic()
    n = 0
    offset = None
    last_disk_check = 0.0
    with psycopg.connect(dsn) as pg:
        while True:
            page_size = SCROLL_BATCH if not args.limit else min(SCROLL_BATCH, args.limit - n)
            if page_size <= 0:
                break
            page, offset = client.scroll(
                args.source_collection,
                limit=page_size,
                offset=offset,
                with_vectors=True,
                with_payload=["bibcode"],
            )
            if not page:
                break
            bibs = [(r.payload or {}).get("bibcode") for r in page]
            by_bibcode = _enrich(pg, [b for b in bibs if b])
            if errors:
                raise RuntimeError(f"upsert worker failed: {errors[0]}")
            q.put(_points_from_page(page, by_bibcode))
            n += len(page)
            now = time.monotonic()
            if now - last_disk_check > 30:
                last_disk_check = now
                if disk_free_gb() < DISK_FLOOR_GB:
                    raise RuntimeError(f"DISK FLOOR: {disk_free_gb():.1f}G free — aborting")
            if n % 256_000 == 0:
                log.info(
                    "copied %d (%.0f pts/s, disk_free=%.0fG)", n, n / (now - t0), disk_free_gb()
                )
            if offset is None:
                break

    for _ in threads:
        q.put(None)
    q.join()
    if errors:
        raise RuntimeError(f"upsert worker failed: {errors[0]}")
    t_load = time.monotonic() - t0
    log.info("COPY DONE: %d points in %.0f s (%.0f pts/s)", n, t_load, n / max(t_load, 1e-9))

    for field_name, schema_type in INDEXED_FIELDS:
        client.create_payload_index(
            args.target_collection, field_name=field_name, field_schema=schema_type
        )
        log.info("created payload index %s (%s)", field_name, schema_type)

    log.info(
        "DONE — %s built with payload + %d indexes. Operator: verify, then swap the alias.",
        args.target_collection,
        len(INDEXED_FIELDS),
    )


if __name__ == "__main__":
    main()
