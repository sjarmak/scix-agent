#!/usr/bin/env python3
"""MH-13-style capacity/throughput pilot: stream INDUS vectors from
paper_embeddings into a GPU-indexing Qdrant instance and measure
load throughput + GPU HNSW index-build time.

Collection config is the PRD converge-amended candidate
(qdrant_nas_migration.md): m=32, SQ-INT8 always-in-RAM, float16
originals on disk. Collection name follows the PRD scheme
scix_{model}_{ver}_{granularity}_{schema_ver}.

Usage:
    python scripts/qdrant_gpu_pilot.py --limit 1000000 [--url http://127.0.0.1:6433]
"""

import argparse
import logging
import time
import uuid

import psycopg
from qdrant_client import QdrantClient
from qdrant_client import models as qm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qdrant_gpu_pilot")

COLLECTION = "scix_indus_v2_papers_s1"
DSN = "dbname=scix"
DIM = 768
BATCH = 1024


def parse_vec(s: str) -> list[float]:
    return [float(x) for x in s[1:-1].split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1_000_000)
    ap.add_argument("--url", default="http://127.0.0.1:6433")
    ap.add_argument("--grpc-port", type=int, default=6434)
    ap.add_argument("--recreate", action="store_true", default=True)
    args = ap.parse_args()

    client = QdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True, timeout=120)

    if args.recreate and client.collection_exists(COLLECTION):
        log.info("dropping existing collection %s", COLLECTION)
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(
            size=DIM,
            distance=qm.Distance.COSINE,
            datatype=qm.Datatype.FLOAT16,
            on_disk=True,
        ),
        hnsw_config=qm.HnswConfigDiff(m=32, on_disk=False),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        ),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=8),
    )
    log.info("created %s (m=32, f16 on_disk, SQ-INT8 always_ram)", COLLECTION)

    t0 = time.monotonic()
    n = 0
    with psycopg.connect(DSN) as pg:
        with pg.cursor(name="pilot_cursor") as cur:
            cur.itersize = BATCH * 4
            cur.execute(
                "SELECT bibcode, (embedding)::vector(768)::text "
                "FROM paper_embeddings WHERE model_name='indus' LIMIT %s",
                (args.limit,),
            )
            points: list[qm.PointStruct] = []
            for bibcode, vec_text in cur:
                points.append(
                    qm.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode)),
                        vector=parse_vec(vec_text),
                        payload={"bibcode": bibcode},
                    )
                )
                if len(points) >= BATCH:
                    client.upsert(COLLECTION, points=points, wait=False)
                    n += len(points)
                    points = []
                    if n % 102_400 == 0:
                        el = time.monotonic() - t0
                        log.info("upserted %d (%.0f pts/s)", n, n / el)
            if points:
                client.upsert(COLLECTION, points=points, wait=True)
                n += len(points)
    t_load = time.monotonic() - t0
    log.info("LOAD DONE: %d points in %.1f s (%.0f pts/s)", n, t_load, n / t_load)

    # wait for indexing (GPU HNSW builds run in the optimizer)
    t1 = time.monotonic()
    while True:
        info = client.get_collection(COLLECTION)
        status = str(info.status)
        if "green" in status.lower():
            break
        time.sleep(10)
    t_index_tail = time.monotonic() - t1
    t_total = time.monotonic() - t0

    info = client.get_collection(COLLECTION)
    log.info("INDEX DONE: status=%s points=%s indexed_vectors=%s", info.status, info.points_count, info.indexed_vectors_count)
    log.info(
        "TIMINGS: load=%.1fs  index_tail_after_load=%.1fs  total=%.1fs  (%.0f pts/s end-to-end)",
        t_load, t_index_tail, t_total, n / t_total,
    )
    proj = 32_383_535 / n * t_total / 3600
    log.info("PROJECTION to 32.38M at this end-to-end rate: %.1f h", proj)


if __name__ == "__main__":
    main()
