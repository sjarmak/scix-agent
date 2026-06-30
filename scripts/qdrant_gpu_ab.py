#!/usr/bin/env python3
"""A/B: GPU-built vs CPU-built HNSW graph quality on identical data/config.

Same qdrant 1.18.2 image; only difference is GPU device access.
Loads a deterministic 250k INDUS slice into both, then measures
self-recall@10 (disconnected-node detector) and recall@10 vs exact.
"""

import logging
import random
import statistics as st
import time
import uuid

import psycopg
from qdrant_client import QdrantClient
from qdrant_client import models as qm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ab")

TARGETS = {
    "gpu": dict(url="http://127.0.0.1:6433", grpc_port=6434),
    "cpu": dict(url="http://127.0.0.1:6533", grpc_port=6534),
}
COLL = "ab_indus_250k"
N = 250_000
BATCH = 1024
SQL = (
    "SELECT bibcode, (embedding)::vector(768)::text FROM paper_embeddings "
    "WHERE model_name='indus' AND bibcode >= '2015' ORDER BY bibcode LIMIT %s"
)


def load(client: QdrantClient) -> float:
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(
        collection_name=COLL,
        vectors_config=qm.VectorParams(
            size=768, distance=qm.Distance.COSINE, datatype=qm.Datatype.FLOAT16, on_disk=True
        ),
        hnsw_config=qm.HnswConfigDiff(m=32, on_disk=False),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        ),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=4),
    )
    t0 = time.monotonic()
    pts = []
    with psycopg.connect("dbname=scix") as pg, pg.cursor(name="ab_cur") as cur:
        cur.itersize = BATCH * 4
        cur.execute(SQL, (N,))
        for bib, vt in cur:
            pts.append(
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, bib)),
                    vector=[float(x) for x in vt[1:-1].split(",")],
                    payload={"bibcode": bib},
                )
            )
            if len(pts) >= BATCH:
                client.upsert(COLL, points=pts, wait=False)
                pts = []
        if pts:
            client.upsert(COLL, points=pts, wait=True)
    while "green" not in str(client.get_collection(COLL).status).lower():
        time.sleep(5)
    return time.monotonic() - t0


def measure(client: QdrantClient) -> dict:
    random.seed(7)
    batch, _ = client.scroll(COLL, limit=5000, with_vectors=True)
    sample = random.sample(batch, 400)
    self_miss = 0
    r10 = []
    for i, p in enumerate(sample):
        nq = client.query_points(
            COLL,
            query=p.vector,
            limit=10,
            search_params=qm.SearchParams(quantization=qm.QuantizationSearchParams(ignore=True)),
        ).points
        if not any(x.id == p.id for x in nq):
            self_miss += 1
        if i < 200:  # exact GT is the slow part; 200 is plenty
            exact = client.query_points(
                COLL, query=p.vector, limit=10, search_params=qm.SearchParams(exact=True)
            ).points
            truth = set(x.id for x in exact)
            dflt = client.query_points(COLL, query=p.vector, limit=10).points
            r10.append(len(truth & set(x.id for x in dflt)) / 10)
    return dict(self_miss=self_miss, n_self=len(sample), r10_mean=st.mean(r10), r10_min=min(r10))


def main() -> None:
    results = {}
    for name, kw in TARGETS.items():
        client = QdrantClient(prefer_grpc=True, timeout=120, **kw)
        t = load(client)
        log.info("[%s] loaded+indexed %d in %.1f s", name, N, t)
        results[name] = dict(build_s=t, **measure(client))
        log.info("[%s] %s", name, results[name])
    print("\n=== A/B RESULT ===")
    for name, r in results.items():
        print(
            f"{name}: build={r['build_s']:.0f}s  self-miss={r['self_miss']}/{r['n_self']}  "
            f"recall@10 mean={r['r10_mean']:.3f} min={r['r10_min']:.2f}"
        )


if __name__ == "__main__":
    main()
