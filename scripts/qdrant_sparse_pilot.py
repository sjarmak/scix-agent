#!/usr/bin/env python3
"""ADR-014: build a Qdrant sparse BM25 lexical collection — either the bounded
50q **pilot** universe (i5oa) or the **full 32M corpus** (qajc Phase 2).

Design (see docs/ADR/014_qdrant_sparse_lexical_lane.md):

* Pilot universe (default) = union of each 50q query's top-100 Postgres lexical
  candidates ∪ all ``gold_bibcodes`` ∪ a ~50k ``TABLESAMPLE`` draw. The sample
  makes collection-level IDF approximate the 32M-corpus IDF instead of a
  topically-collapsed subset; both systems are later scored on this same
  universe (the eval restricts Postgres to pilot membership), so the
  comparison is fair.
* Full corpus (``--full-corpus``) = stream every indexable paper
  (``title IS NOT NULL``) through a server-side cursor, so the BM25 lane also
  searches 32M and the haystack confound is retired for good. This is a
  CPU-only throwaway eval collection — ``on_disk``, NVMe only, dropped after
  the eval; do **not** migrate production until the numbers hold.
* Index = FastEmbed ``Qdrant/bm25`` (true Okapi BM25, client-side
  tokenization) over ``title + ' ' + abstract + ' ' + keywords`` — matching
  the Postgres lane's fields, no field weighting (the conservative case).
  The collection uses ``Modifier.IDF`` so Qdrant supplies IDF at query time.
  Tokenizer config is shared with the eval via :mod:`scripts._sparse_bm25`;
  ``--disable-stemmer`` mirrors ``scix_english`` simple_nostem to recover the
  ``title_matchable`` regression.
* Point IDs = UUID-v5 from bibcode (same scheme as the dense v2 collection),
  payload ``{"bibcode": ...}``.

Local-only: ``Qdrant/bm25`` is a ~10 MB CPU tokenizer, no neural net, no
paid API (``feedback_no_paid_apis``). Run the build under ``scix-batch``.

DISK PRECONDITION (qajc dispatch guard): a full-corpus build writes tens of GB
of on-disk index. Verify free NVMe headroom before ``--full-corpus`` — do not
fill a near-full disk. Never point the collection at NAS.

Usage::

    # pilot (52k universe)
    scix-batch python scripts/qdrant_sparse_pilot.py \
        --url http://127.0.0.1:6633 --sample 50000
    # full 32M corpus, no-stem tokenizer (throwaway eval collection)
    scix-batch python scripts/qdrant_sparse_pilot.py --full-corpus \
        --collection scix_sparse_full_v1 --disable-stemmer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
from qdrant_client import QdrantClient
from qdrant_client import models as qm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scix.search import SearchFilters, lexical_search  # noqa: E402
from scripts._sparse_bm25 import (  # noqa: E402
    Bm25TokenizerConfig,
    add_bm25_tokenizer_args,
    build_model,
    config_from_args,
    record_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("qdrant_sparse_pilot")

PILOT_COLLECTION = "scix_sparse_pilot_v1"
DSN = "dbname=scix"
QUERIES_PATH = _REPO_ROOT / "eval" / "retrieval_50q.jsonl"
UNIVERSE_PATH = _REPO_ROOT / "results" / "sparse_pilot_universe.json"
_NAMESPACE = uuid.NAMESPACE_URL
BATCH = 1000
CAND_PER_QUERY = 100
# Server-side cursor page size for the full-corpus stream.
STREAM_ITERSIZE = 5000

# The text projection both modes index — title + abstract + keywords, matching
# the Postgres scix_english lane's fields.
_TEXT_SQL = (
    "coalesce(title,'') || ' ' || coalesce(abstract,'') || ' ' "
    "|| coalesce(array_to_string(keywords,' '),'')"
)


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


def fetch_texts(conn: psycopg.Connection, bibcodes: list[str]) -> Iterator[tuple[str, str]]:
    """Yield (bibcode, 'title abstract keywords') for the given indexable papers
    (pilot path — the universe is already bounded, so a client-side IN works)."""
    n = 0
    with conn.cursor() as cur:
        for i in range(0, len(bibcodes), 5000):
            chunk = bibcodes[i : i + 5000]
            cur.execute(
                f"SELECT bibcode, {_TEXT_SQL} AS txt FROM papers WHERE bibcode = ANY(%s)",
                (chunk,),
            )
            for bib, txt in cur.fetchall():
                txt = (txt or "").strip()
                if txt:
                    n += 1
                    yield bib, txt
    log.info("fetched text for %d papers", n)


def stream_full_corpus(conn: psycopg.Connection) -> Iterator[tuple[str, str]]:
    """Yield (bibcode, text) for every indexable paper via a server-side cursor,
    so the full 32M corpus streams without materializing in memory."""
    with conn.cursor(name="sparse_full_stream") as cur:
        cur.itersize = STREAM_ITERSIZE
        cur.execute(f"SELECT bibcode, {_TEXT_SQL} AS txt FROM papers WHERE title IS NOT NULL")
        for bib, txt in cur:
            txt = (txt or "").strip()
            if txt:
                yield bib, txt


def ensure_collection(client: QdrantClient, collection: str) -> None:
    if client.collection_exists(collection):
        log.info("dropping existing %s", collection)
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={},  # sparse-only
        sparse_vectors_config={
            "bm25": qm.SparseVectorParams(
                modifier=qm.Modifier.IDF,
                index=qm.SparseIndexParams(on_disk=True),
            )
        },
    )
    log.info("created collection %s (sparse 'bm25', Modifier.IDF, on_disk)", collection)


def index(client: QdrantClient, model, collection: str, rows: Iterator[tuple[str, str]]) -> int:
    """Embed and upsert ``rows`` in fixed batches. Streams the iterator, so the
    full-corpus path holds only one batch in memory at a time. Returns the count."""
    total = 0
    batch: list[tuple[str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= BATCH:
            total += _upsert_batch(client, model, collection, batch)
            batch = []
            if total % 100000 == 0:
                log.info("upserted %d", total)
    if batch:
        total += _upsert_batch(client, model, collection, batch)
    log.info("upserted %d total", total)
    return total


def _upsert_batch(
    client: QdrantClient, model, collection: str, batch: list[tuple[str, str]]
) -> int:
    embeddings = list(model.embed([t for _, t in batch]))
    points = [
        qm.PointStruct(
            id=_point_id(bib),
            payload={"bibcode": bib},
            vector={
                "bm25": qm.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            },
        )
        for (bib, _), emb in zip(batch, embeddings)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def smoke_test(client: QdrantClient, model, collection: str) -> None:
    """ADR-013 rule: don't trust an index until one query has returned from it."""
    sparse = next(iter(model.query_embed("dark matter halo")))
    hits = client.query_points(
        collection_name=collection,
        query=qm.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
        using="bm25",
        limit=3,
    ).points
    if not hits:
        raise RuntimeError(f"smoke test returned 0 hits from {collection} — index not trustworthy")
    log.info("smoke test OK: %d hits, top bibcode=%s", len(hits), hits[0].payload.get("bibcode"))


def _build_pilot(conn: psycopg.Connection, client, model, collection: str, sample: int) -> None:
    """Bounded 50q universe path. Materializes membership so the eval can
    restrict all lanes to the same haystack (fair A/B)."""
    queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]
    log.info("loaded %d queries", len(queries))
    universe = build_universe(conn, queries)
    add_sample(conn, universe, sample)
    rows = list(fetch_texts(conn, sorted(universe)))

    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_PATH.write_text(json.dumps({"bibcodes": sorted(b for b, _ in rows)}))
    log.info("wrote universe membership -> %s", UNIVERSE_PATH)
    index(client, model, collection, iter(rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:6633")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument(
        "--full-corpus",
        action="store_true",
        help="Stream every indexable paper (32M) instead of the 50q universe. "
        "Writes tens of GB on disk — verify NVMe headroom first (qajc guard).",
    )
    ap.add_argument(
        "--collection",
        default=None,
        help=f"Collection name (default {PILOT_COLLECTION}, "
        "or scix_sparse_full_v1 with --full-corpus).",
    )
    add_bm25_tokenizer_args(ap)
    args = ap.parse_args(argv)

    collection = args.collection or (
        "scix_sparse_full_v1" if args.full_corpus else PILOT_COLLECTION
    )
    config: Bm25TokenizerConfig = config_from_args(args)
    log.info(
        "collection=%s mode=%s tokenizer=%s",
        collection,
        "full" if args.full_corpus else "pilot",
        config.signature(),
    )

    log.info("loading FastEmbed BM25 ...")
    model = build_model(config)
    client = QdrantClient(url=args.url, prefer_grpc=False, timeout=120)

    # Connect to Postgres before dropping/recreating the collection, so a DB
    # failure doesn't leave an empty collection behind.
    conn = psycopg.connect(args.dsn)
    try:
        ensure_collection(client, collection)
        if args.full_corpus:
            index(client, model, collection, stream_full_corpus(conn))
        else:
            _build_pilot(conn, client, model, collection, args.sample)
    finally:
        conn.close()

    record_config(collection, config)
    smoke_test(client, model, collection)
    count = client.count(collection).count
    log.info("done: %s holds %d points", collection, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
