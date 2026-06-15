#!/usr/bin/env python3
"""Full-corpus INDUS load into Qdrant (CPU-built HNSW per A/B verdict in
bead pkcd; GPU builds produce disconnected nodes on this stack).

Fast path: binary pgvector fetch (~15k rows/s) + parallel gRPC upserts.
Safety: disk watchdog aborts below a floor so a near-full NVMe never
takes Postgres down (WAL) mid-load.

Usage:
    python scripts/qdrant_full_load.py [--limit N] [--url http://127.0.0.1:6633]
"""

import argparse
import logging
import queue
import shutil
import threading
import time
import uuid

import psycopg
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient
from qdrant_client import models as qm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("qdrant_full_load")

COLLECTION = "scix_indus_v2_papers_s1"
DSN = "dbname=scix"
DIM = 768
BATCH = 2048
WORKERS = 4
DISK_FLOOR_GB = 10
TOTAL_EXPECTED = 32_383_535


def disk_free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def upsert_worker(client: QdrantClient, q: "queue.Queue", errors: list) -> None:
    while True:
        pts = q.get()
        if pts is None:
            q.task_done()
            return
        try:
            client.upsert(COLLECTION, points=pts, wait=False)
        except Exception as e:  # noqa: BLE001 — record and stop the load
            errors.append(e)
        finally:
            q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--url", default="http://127.0.0.1:6633")
    ap.add_argument("--grpc-port", type=int, default=6634)
    args = ap.parse_args()

    client = QdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True, timeout=300)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(
                size=DIM,
                distance=qm.Distance.COSINE,
                datatype=qm.Datatype.FLOAT16,
                on_disk=True,
            ),
            hnsw_config=qm.HnswConfigDiff(m=32, on_disk=False),
            # always_ram=False: ~25GB quantized layer stays mmap'd so Qdrant
            # coexists with PG(16G)+gascity on the 62G host; revisit post-cutover.
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8, quantile=0.99, always_ram=False
                )
            ),
            optimizers_config=qm.OptimizersConfigDiff(default_segment_number=8),
        )
        log.info("created %s (m=32, f16 on_disk, SQ-INT8 on_disk)", COLLECTION)
    else:
        log.info("collection %s exists — resuming/idempotent upsert", COLLECTION)

    sql = (
        "SELECT bibcode, embedding FROM paper_embeddings "
        "WHERE model_name='indus' AND embedding IS NOT NULL"
        + (f" LIMIT {args.limit}" if args.limit else "")
    )

    q: "queue.Queue" = queue.Queue(maxsize=WORKERS * 4)
    errors: list = []
    threads = [
        threading.Thread(target=upsert_worker, args=(client, q, errors), daemon=True)
        for _ in range(WORKERS)
    ]
    for t in threads:
        t.start()

    t0 = time.monotonic()
    n = 0
    last_disk_check = 0.0
    with psycopg.connect(DSN) as pg:
        register_vector(pg)
        with pg.cursor(name="full_load_cursor", binary=True) as cur:
            cur.itersize = BATCH * 4
            cur.execute(sql)
            pts = []
            for bibcode, vec in cur:
                pts.append(
                    qm.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode)),
                        vector=vec.tolist(),
                        payload={"bibcode": bibcode},
                    )
                )
                if len(pts) >= BATCH:
                    if errors:
                        raise RuntimeError(f"upsert worker failed: {errors[0]}")
                    q.put(pts)
                    n += len(pts)
                    pts = []
                    now = time.monotonic()
                    if now - last_disk_check > 30:
                        last_disk_check = now
                        free = disk_free_gb()
                        if free < DISK_FLOOR_GB:
                            raise RuntimeError(f"DISK FLOOR: only {free:.1f}G free — aborting load")
                    if n % 512_000 == 0:
                        el = now - t0
                        log.info(
                            "queued %d (%.0f pts/s, disk_free=%.0fG)", n, n / el, disk_free_gb()
                        )
            if pts:
                q.put(pts)
                n += len(pts)
    for _ in threads:
        q.put(None)
    q.join()
    if errors:
        raise RuntimeError(f"upsert worker failed: {errors[0]}")
    t_load = time.monotonic() - t0
    log.info("LOAD DONE: %d points in %.0f s (%.0f pts/s)", n, t_load, n / t_load)

    t1 = time.monotonic()
    while True:
        info = client.get_collection(COLLECTION)
        if "green" in str(info.status).lower():
            break
        time.sleep(30)
    log.info(
        "INDEX DONE: points=%s indexed=%s | load=%.0fs index_tail=%.0fs total=%.1f min | disk_free=%.0fG",
        info.points_count,
        info.indexed_vectors_count,
        t_load,
        time.monotonic() - t1,
        (time.monotonic() - t0) / 60,
        disk_free_gb(),
    )


if __name__ == "__main__":
    main()
