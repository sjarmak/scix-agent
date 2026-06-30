#!/usr/bin/env python3
"""Controlled before/after benchmark of Qdrant payload write paths (bead tqg4).

Reproduces, on an ISOLATED scratch collection, the comparison behind
results/qdrant_writepath/root_cause_and_fix.md:

    A) per-point  set_payload(wait=False)   — the current backfill path
    B) batched    batch_update_points       — fewer HTTP round-trips, same op cost
    C) upsert     PointStruct(vec+payload)  — the recommended fix

SAFETY (this host is RAM-walled; oomd kills the gascity supervisor at pressure):
  - Targets the empty, idle ``scix-qdrant-gpu`` instance (:6433) by DEFAULT, NOT
    the live serving :6633. Refuses :6633's prod collection name outright.
  - Refuses to run if MemAvailable is below ``--mem-floor-mb`` (default 4096).
  - Uses on_disk vectors + a modest default N to bound resident memory.
  - Drops the scratch collection on exit (``--keep`` to retain for inspection).

The pathology (47-63 s/op) is segment-size-dependent and only manifests near
prod's ~1.3M-pt segments; ``--force-large-segment`` packs all N points into one
segment to surface it at small N. Without it you measure the relative ordering
(C >> A) on byte-identical prod config, which is the actionable result.

Usage:
    python scripts/qdrant_writepath_bench.py --n 50000
    python scripts/qdrant_writepath_bench.py --n 200000 --force-large-segment
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("qdrant_writepath_bench")

DIM = 768
PROD_COLLECTION = "scix_indus_v2_papers_s1"


def mem_available_mb() -> int:
    """MemAvailable from /proc/meminfo, in MiB (Linux)."""
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _vec(i: int) -> list[float]:
    """Deterministic pseudo-random unit-ish vector (no RNG: Date/random barred,
    and content is irrelevant to payload-write timing)."""
    return [((i * 2654435761 + j * 40503) % 1000) / 1000.0 for j in range(DIM)]


def _payload(i: int) -> dict:
    return {
        "bibcode": f"scratch{i:016d}",
        "year": 1990 + (i % 35),
        "doctype": ("article", "eprint", "inproceedings")[i % 3],
        "arxiv_class": [("astro-ph.GA", "astro-ph.CO", "physics.space-ph")[i % 3]],
        "bibstem": [("ApJ", "MNRAS", "A&A")[i % 3]],
        "community_semantic_coarse": i % 64,
        "community_semantic_medium": i % 512,
        "citation_count": i % 1000,
        "pagerank": (i % 997) / 1e6,
    }


def _ids(start: int, count: int) -> list[str]:
    return [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"scratch{i:016d}")) for i in range(start, start + count)
    ]


def _create(client: QdrantClient, name: str, *, force_large_segment: bool) -> None:
    opt = qm.OptimizersConfigDiff(
        default_segment_number=1 if force_large_segment else 8,
        max_segment_size=(10_000_000 if force_large_segment else None),
    )
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=DIM, distance=qm.Distance.COSINE, datatype=qm.Datatype.FLOAT16, on_disk=True
        ),
        hnsw_config=qm.HnswConfigDiff(m=32, on_disk=False),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, quantile=0.99, always_ram=False
            )
        ),
        optimizers_config=opt,
    )
    for f, t in (
        ("year", qm.PayloadSchemaType.INTEGER),
        ("doctype", qm.PayloadSchemaType.KEYWORD),
        ("arxiv_class", qm.PayloadSchemaType.KEYWORD),
        ("bibstem", qm.PayloadSchemaType.KEYWORD),
        ("community_semantic_coarse", qm.PayloadSchemaType.INTEGER),
        ("community_semantic_medium", qm.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(name, field_name=f, field_schema=t)


def _seed_vectors(client: QdrantClient, name: str, n: int, batch: int) -> None:
    """Load N points carrying ONLY a vector (the 'before' starting state — payload
    is added by the benchmarked path)."""
    for s in range(0, n, batch):
        c = min(batch, n - s)
        pts = [qm.PointStruct(id=i, vector=_vec(s + j)) for j, i in enumerate(_ids(s, c))]
        client.upsert(name, points=pts, wait=False)
    client.upsert(name, points=[qm.PointStruct(id=_ids(0, 1)[0], vector=_vec(0))], wait=True)


def bench_set_payload(client: QdrantClient, name: str, n: int) -> float:
    t0 = time.monotonic()
    for j, pid in enumerate(_ids(0, n)):
        client.set_payload(
            name, payload=_payload(j), points=qm.PointIdsList(points=[pid]), wait=False
        )
    client.set_payload(
        name, payload=_payload(0), points=qm.PointIdsList(points=[_ids(0, 1)[0]]), wait=True
    )
    return n / (time.monotonic() - t0)


def bench_batch_update(client: QdrantClient, name: str, n: int, batch: int) -> float:
    t0 = time.monotonic()
    ids = _ids(0, n)
    for s in range(0, n, batch):
        ops = [
            qm.SetPayloadOperation(set_payload=qm.SetPayload(payload=_payload(s + j), points=[pid]))
            for j, pid in enumerate(ids[s : s + batch])
        ]
        client.batch_update_points(name, update_operations=ops, wait=False)
    client.set_payload(
        name, payload=_payload(0), points=qm.PointIdsList(points=[ids[0]]), wait=True
    )
    return n / (time.monotonic() - t0)


def bench_upsert_payload(client: QdrantClient, name: str, n: int, batch: int) -> float:
    t0 = time.monotonic()
    for s in range(0, n, batch):
        c = min(batch, n - s)
        pts = [
            qm.PointStruct(id=i, vector=_vec(s + j), payload=_payload(s + j))
            for j, i in enumerate(_ids(s, c))
        ]
        client.upsert(name, points=pts, wait=False)
    client.upsert(
        name,
        points=[qm.PointStruct(id=_ids(0, 1)[0], vector=_vec(0), payload=_payload(0))],
        wait=True,
    )
    return n / (time.monotonic() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Qdrant payload write-path before/after benchmark")
    ap.add_argument(
        "--url", default="http://127.0.0.1:6433", help="default: isolated gpu instance :6433"
    )
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mem-floor-mb", type=int, default=4096)
    ap.add_argument("--force-large-segment", action="store_true")
    ap.add_argument("--keep", action="store_true", help="don't drop scratch collections")
    args = ap.parse_args()

    if "6633" in args.url:
        raise SystemExit(
            "refusing :6633 (live serving instance). Use the isolated :6433 gpu instance."
        )
    avail = mem_available_mb()
    if avail < args.mem_floor_mb:
        raise SystemExit(
            f"MemAvailable {avail} MiB < floor {args.mem_floor_mb} MiB — refusing to load "
            "(would risk oomd killing the gascity supervisor). Retry in a RAM-headroom window."
        )
    log.info("MemAvailable %d MiB (floor %d) — ok", avail, args.mem_floor_mb)

    client = QdrantClient(url=args.url, timeout=300)
    results: dict[str, float] = {}
    # Each path gets its own fresh collection so they don't interfere.
    cases = [
        ("A_set_payload_per_point", bench_set_payload, False),
        ("B_batch_update_points", bench_batch_update, True),
        ("C_upsert_with_payload", bench_upsert_payload, True),
    ]
    for label, fn, needs_batch in cases:
        name = f"scratch_writepath_{label.lower()}"
        if client.collection_exists(name):
            client.delete_collection(name)
        _create(client, name, force_large_segment=args.force_large_segment)
        if fn is not bench_upsert_payload:
            _seed_vectors(client, name, args.n, args.batch)  # upsert path seeds itself
        rate = fn(client, name, args.n, args.batch) if needs_batch else fn(client, name, args.n)
        results[label] = rate
        log.info("%-26s %8.0f pts/s", label, rate)
        if not args.keep:
            client.delete_collection(name)

    print("\n=== write-path throughput (pts/s submission; wait=True flush at end) ===")
    print(f"  N={args.n:,}  large_segment={args.force_large_segment}  url={args.url}")
    for label, rate in results.items():
        print(f"  {label:28s} {rate:10.0f} pts/s")
    if results.get("A_set_payload_per_point"):
        print(
            f"\n  fix speedup (C/A): {results['C_upsert_with_payload'] / results['A_set_payload_per_point']:.1f}x"
        )


if __name__ == "__main__":
    main()
