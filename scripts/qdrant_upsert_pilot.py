"""Upsert the top-N papers by PageRank into Qdrant with rich payload.

Named vector: indus (768d, cosine).
Payload fields indexed for filtering: year, doctype, community_id_coarse,
community_id_medium, arxiv_class.

Usage:
    python scripts/qdrant_upsert_pilot.py --limit 500000 --batch 1000

Env:
    QDRANT_URL   default http://127.0.0.1:6333
    SCIX_DSN     Postgres DSN
"""
from __future__ import annotations

import argparse
import os
import struct
import time
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from scix.db import get_connection

COLLECTION = "scix_papers_v1"
VECTOR_NAME = "indus"
VECTOR_DIM = 768


def bibcode_to_point_id(bibcode: str) -> int:
    """Stable 63-bit int derived from bibcode (Qdrant accepts int or UUID)."""
    import hashlib
    h = hashlib.blake2b(bibcode.encode("utf-8"), digest_size=8).digest()
    return struct.unpack(">Q", h)[0] >> 1  # keep positive


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"collection {COLLECTION} already exists")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            VECTOR_NAME: qm.VectorParams(
                size=VECTOR_DIM,
                distance=qm.Distance.COSINE,
                on_disk=False,
            )
        },
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
    )
    # We index the semantic-community signal (k-means on INDUS embeddings) and
    # the taxonomic signal (arxiv_class). Citation-Leiden communities in
    # paper_metrics are currently all -1/NULL — do not index them.
    for field, schema in [
        ("year", qm.PayloadSchemaType.INTEGER),
        ("doctype", qm.PayloadSchemaType.KEYWORD),
        ("community_semantic_coarse", qm.PayloadSchemaType.INTEGER),
        ("community_semantic_medium", qm.PayloadSchemaType.INTEGER),
        ("arxiv_class", qm.PayloadSchemaType.KEYWORD),
        ("bibstem", qm.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
    print(f"created collection {COLLECTION} + payload indexes")


def stream_rows(conn, limit: int, batch: int) -> Iterator[list[dict]]:
    """Stream rows from Postgres in server-side cursor batches.

    We join top-PageRank papers with their INDUS embedding and their
    community assignments. Server-side named cursor avoids loading all 500K
    rows into memory at once.
    """
    # Select papers that have both a semantic community and arxiv_class — the
    # two payload fields that make filtered-recommendation demos meaningful.
    # Ordered by pagerank DESC to pick the most influential N from that pool.
    # Citation-Leiden communities are currently all -1/NULL so we use the
    # semantic (INDUS k-means) signal instead.
    sql = f"""
        WITH candidates AS (
            SELECT m.bibcode,
                   m.pagerank,
                   m.community_semantic_coarse,
                   m.community_semantic_medium
            FROM paper_metrics m
            WHERE m.community_semantic_coarse IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM papers p
                  WHERE p.bibcode = m.bibcode
                    AND array_length(p.arxiv_class, 1) >= 1
              )
        ),
        top_n AS (
            SELECT bibcode, pagerank, community_semantic_coarse, community_semantic_medium
            FROM candidates
            ORDER BY pagerank DESC NULLS LAST
            LIMIT {int(limit)}
        )
        SELECT p.bibcode,
               p.title,
               p.year,
               p.doctype,
               p.first_author,
               p.arxiv_class,
               p.bibstem,
               p.citation_count,
               t.pagerank,
               t.community_semantic_coarse,
               t.community_semantic_medium,
               e.embedding::text AS vec
        FROM top_n t
        JOIN papers p ON p.bibcode = t.bibcode
        JOIN paper_embeddings e
          ON e.bibcode = t.bibcode AND e.model_name = 'indus'
    """
    with conn.cursor(name="qdrant_upsert_cursor") as cur:
        cur.itersize = batch
        cur.execute(sql)
        buf: list[dict] = []
        for row in cur:
            (bibcode, title, year, doctype, first_author, arxiv_class,
             bibstem, citation_count, pagerank, sem_c, sem_m, vec_text) = row
            # pgvector `::text` serializes as "[0.1,0.2,...]"
            vec = [float(x) for x in vec_text.strip("[]").split(",")]
            if len(vec) != VECTOR_DIM:
                continue
            buf.append({
                "bibcode": bibcode,
                "title": title,
                "year": int(year) if year is not None else None,
                "doctype": doctype,
                "first_author": first_author,
                "arxiv_class": list(arxiv_class) if arxiv_class else [],
                "bibstem": list(bibstem) if bibstem else [],
                "citation_count": int(citation_count) if citation_count is not None else 0,
                "pagerank": float(pagerank) if pagerank is not None else None,
                "community_semantic_coarse": int(sem_c) if sem_c is not None else None,
                "community_semantic_medium": int(sem_m) if sem_m is not None else None,
                "_vector": vec,
            })
            if len(buf) >= batch:
                yield buf
                buf = []
        if buf:
            yield buf


def rows_to_points(rows: list[dict]) -> list[qm.PointStruct]:
    points = []
    for r in rows:
        vec = r.pop("_vector")
        payload = {k: v for k, v in r.items() if v is not None}
        points.append(qm.PointStruct(
            id=bibcode_to_point_id(r["bibcode"]),
            vector={VECTOR_NAME: vec},
            payload=payload,
        ))
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500_000)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    client = QdrantClient(url=url, timeout=60)
    ensure_collection(client)

    total = 0
    t0 = time.time()
    with get_connection() as conn:
        for batch_rows in stream_rows(conn, args.limit, args.batch):
            points = rows_to_points(batch_rows)
            client.upsert(collection_name=COLLECTION, points=points, wait=False)
            total += len(points)
            if total % (args.batch * 10) == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed else 0
                print(f"  upserted {total:,} ({rate:,.0f} pts/s)")
    dt = time.time() - t0
    print(f"done: {total:,} points in {dt:.1f}s ({total/dt:,.0f} pts/s)")
    info = client.get_collection(COLLECTION)
    print(f"collection status: {info.status}, points: {info.points_count}")


if __name__ == "__main__":
    main()
