#!/usr/bin/env python3
"""ADR-014 pilot: build a Qdrant sparse BM25 lexical collection for the 50q
head-to-head against the Postgres ``scix_english`` tsvector lane.

Design (see docs/ADR/014_qdrant_sparse_lexical_lane.md):

* Universe = union of each 50q query's top-100 Postgres lexical candidates
  ∪ all ``gold_bibcodes`` ∪ a ~50k ``TABLESAMPLE`` draw. The sample makes
  collection-level IDF approximate the 32M-corpus IDF instead of a
  topically-collapsed subset; both systems are later scored on this same
  universe (the eval restricts Postgres to pilot membership), so the
  comparison is fair.
* Index = FastEmbed ``Qdrant/bm25`` (true Okapi BM25, client-side
  tokenization) over ``title + ' ' + abstract + ' ' + keywords`` — matching
  the Postgres lane's fields, no field weighting (the conservative case).
  The collection uses ``Modifier.IDF`` so Qdrant supplies IDF at query time.
* Point IDs = UUID-v5 from bibcode (same scheme as the dense v2 collection),
  payload ``{"bibcode": ...}``.

Local-only: ``Qdrant/bm25`` is a ~10 MB CPU tokenizer, no neural net, no
paid API (``feedback_no_paid_apis``). Run the build under ``scix-batch``.

Usage::

    scix-batch python scripts/qdrant_sparse_pilot.py \
        --url http://127.0.0.1:6633 --sample 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import psycopg
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.search import SearchFilters, lexical_search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("qdrant_sparse_pilot")

COLLECTION = "scix_sparse_pilot_v1"
DSN = "dbname=scix"
BM25_MODEL = "Qdrant/bm25"
QUERIES_PATH = _REPO_ROOT / "eval" / "retrieval_50q.jsonl"
UNIVERSE_PATH = _REPO_ROOT / "results" / "sparse_pilot_universe.json"
_NAMESPACE = uuid.NAMESPACE_URL
BATCH = 1000
CAND_PER_QUERY = 100


def _point_id(bibcode: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, bibcode))


def build_universe(conn: psycopg.Connection, queries: list[dict]) -> set[str]:
    """Postgres top-100 per query ∪ gold ∪ TABLESAMPLE draw."""
    universe: set[str] = set()
    for q in queries:
        res = lexical_search(conn, q["query"], filters=SearchFilters(), limit=CAND_PER_QUERY)
        universe.update(p["bibcode"] for p in res.papers if p.get("bibcode"))
        universe.update(q.get("gold_bibcodes", []))
    log.info("candidates ∪ gold: %d bibcodes", len(universe))
    return universe


def add_sample(conn: psycopg.Connection, universe: set[str], n: int) -> None:
    """Add a block sample so IDF statistics resemble the full corpus."""
    # SYSTEM samples by heap block; 0.25% of ~32M rows ≫ n, then LIMIT.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode FROM papers TABLESAMPLE SYSTEM (0.25) "
            "WHERE title IS NOT NULL LIMIT %s",
            (n,),
        )
        before = len(universe)
        universe.update(r[0] for r in cur.fetchall())
    log.info("added %d sampled bibcodes (universe now %d)", len(universe) - before, len(universe))


def fetch_texts(conn: psycopg.Connection, bibcodes: list[str]) -> list[tuple[str, str]]:
    """Return (bibcode, 'title abstract keywords') for indexable papers."""
    rows: list[tuple[str, str]] = []
    with conn.cursor() as cur:
        for i in range(0, len(bibcodes), 5000):
            chunk = bibcodes[i : i + 5000]
            cur.execute(
                "SELECT bibcode, "
                "  coalesce(title,'') || ' ' || coalesce(abstract,'') || ' ' "
                "  || coalesce(array_to_string(keywords,' '),'') AS txt "
                "FROM papers WHERE bibcode = ANY(%s)",
                (chunk,),
            )
            for bib, txt in cur.fetchall():
                txt = (txt or "").strip()
                if txt:
                    rows.append((bib, txt))
    log.info("fetched text for %d papers", len(rows))
    return rows


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        log.info("dropping existing %s", COLLECTION)
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={},  # sparse-only pilot
        sparse_vectors_config={
            "bm25": qm.SparseVectorParams(
                modifier=qm.Modifier.IDF,
                index=qm.SparseIndexParams(on_disk=True),
            )
        },
    )
    log.info("created collection %s (sparse 'bm25', Modifier.IDF, on_disk)", COLLECTION)


def index(client: QdrantClient, model: SparseTextEmbedding, rows: list[tuple[str, str]]) -> None:
    total = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        embeddings = list(model.embed([t for _, t in chunk]))
        points = [
            qm.PointStruct(
                id=_point_id(bib),
                payload={"bibcode": bib},
                vector={
                    "bm25": qm.SparseVector(
                        indices=emb.indices.tolist(), values=emb.values.tolist()
                    )
                },
            )
            for (bib, _), emb in zip(chunk, embeddings)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        if total % 10000 == 0 or total == len(rows):
            log.info("upserted %d/%d", total, len(rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:6633")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--dsn", default=DSN)
    args = ap.parse_args(argv)

    queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]
    log.info("loaded %d queries", len(queries))

    conn = psycopg.connect(args.dsn)
    try:
        universe = build_universe(conn, queries)
        add_sample(conn, universe, args.sample)
        rows = fetch_texts(conn, sorted(universe))
    finally:
        conn.close()

    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_PATH.write_text(json.dumps({"bibcodes": sorted(b for b, _ in rows)}))
    log.info("wrote universe membership -> %s", UNIVERSE_PATH)

    log.info("loading FastEmbed %s ...", BM25_MODEL)
    model = SparseTextEmbedding(model_name=BM25_MODEL)

    client = QdrantClient(url=args.url, prefer_grpc=False, timeout=120)
    ensure_collection(client)
    index(client, model, rows)

    count = client.count(COLLECTION).count
    log.info("done: %s holds %d points", COLLECTION, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
