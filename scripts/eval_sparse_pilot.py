#!/usr/bin/env python3
"""ADR-014 pilot eval: Qdrant sparse BM25 vs Postgres ``scix_english``
tsvector, on the 50q gold set, scored with the *existing*
``scripts/eval_retrieval_50q.py`` metric functions (nDCG@10 / MRR@10 /
Recall@50), overall and per bucket.

Fair comparison: both systems rank within the pilot universe written by
``scripts/qdrant_sparse_pilot.py``. The Postgres lane runs over the full
corpus, then its ranking is restricted to pilot membership before scoring
(its own top-100 candidates are in the universe by construction, so this is
near-lossless and removes the universe-asymmetry confound).

Run *after* ``qdrant_sparse_pilot.py``. Writes
``results/sparse_pilot_eval.{json,md}``.

Usage::

    python scripts/eval_sparse_pilot.py --url http://127.0.0.1:6633
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import psycopg
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.search import SearchFilters, lexical_search  # noqa: E402

# Reuse the canonical metric functions rather than re-deriving them.
_spec = importlib.util.spec_from_file_location(
    "eval_retrieval_50q", _REPO_ROOT / "scripts" / "eval_retrieval_50q.py"
)
_e = importlib.util.module_from_spec(_spec)
sys.modules["eval_retrieval_50q"] = _e  # dataclass introspection needs this
_spec.loader.exec_module(_e)  # type: ignore[union-attr]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("eval_sparse_pilot")

COLLECTION = "scix_sparse_pilot_v1"
DSN = "dbname=scix"
BM25_MODEL = "Qdrant/bm25"
QUERIES_PATH = _REPO_ROOT / "eval" / "retrieval_50q.jsonl"
UNIVERSE_PATH = _REPO_ROOT / "results" / "sparse_pilot_universe.json"
OUT_JSON = _REPO_ROOT / "results" / "sparse_pilot_eval.json"
OUT_MD = _REPO_ROOT / "results" / "sparse_pilot_eval.md"
TOP_K = 50
METRICS = ("ndcg_at_10", "mrr_at_10", "recall_at_50")


def postgres_rank(conn, query: str, universe: set[str]) -> list[str]:
    res = lexical_search(conn, query, filters=SearchFilters(), limit=200)
    return [p["bibcode"] for p in res.papers if p.get("bibcode") in universe][:TOP_K]


def qdrant_rank(client, model, query: str) -> list[str]:
    sparse = next(iter(model.query_embed(query)))
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qm.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
        using="bm25",
        limit=TOP_K,
        with_payload=True,
    ).points
    return [h.payload["bibcode"] for h in hits]


def aggregate(per_query: list[dict], key: str) -> dict:
    """Mean of each metric over queries that had a defined score."""
    out: dict[str, float | int] = {}
    for m in METRICS:
        vals = [r[key][m] for r in per_query if r[key][m] is not None]
        out[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["n_scored"] = sum(1 for r in per_query if r[key]["ndcg_at_10"] is not None)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:6633")
    ap.add_argument("--dsn", default=DSN)
    args = ap.parse_args(argv)

    queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]
    universe = set(json.loads(UNIVERSE_PATH.read_text())["bibcodes"])
    log.info("loaded %d queries; universe = %d bibcodes", len(queries), len(universe))

    model = SparseTextEmbedding(model_name=BM25_MODEL)
    client = QdrantClient(url=args.url, prefer_grpc=False, timeout=120)
    conn = psycopg.connect(args.dsn)

    per_query: list[dict] = []
    try:
        for q in queries:
            gold = list(q.get("gold_bibcodes", []))
            if not gold:
                continue
            pg = postgres_rank(conn, q["query"], universe)
            qd = qdrant_rank(client, model, q["query"])
            per_query.append(
                {
                    "query": q["query"],
                    "bucket": q.get("bucket", "?"),
                    "gold_n": len(gold),
                    "postgres": _e.score_query(pg, gold, k=10),
                    "qdrant_bm25": _e.score_query(qd, gold, k=10),
                }
            )
    finally:
        conn.close()

    overall = {
        "postgres": aggregate(per_query, "postgres"),
        "qdrant_bm25": aggregate(per_query, "qdrant_bm25"),
    }
    by_bucket: dict[str, dict] = {}
    buckets = defaultdict(list)
    for r in per_query:
        buckets[r["bucket"]].append(r)
    for b, rows in sorted(buckets.items()):
        by_bucket[b] = {
            "n": len(rows),
            "postgres": aggregate(rows, "postgres"),
            "qdrant_bm25": aggregate(rows, "qdrant_bm25"),
        }

    payload = {
        "collection": COLLECTION,
        "universe_size": len(universe),
        "n_scored": len(per_query),
        "top_k": TOP_K,
        "overall": overall,
        "by_bucket": by_bucket,
        "per_query": per_query,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_MD.write_text(render_md(payload))
    log.info("wrote %s and %s", OUT_JSON, OUT_MD)

    d = overall["qdrant_bm25"]["ndcg_at_10"] - overall["postgres"]["ndcg_at_10"]
    tb = by_bucket.get("title_matchable", {})
    dt = (
        tb.get("qdrant_bm25", {}).get("ndcg_at_10", 0.0)
        - tb.get("postgres", {}).get("ndcg_at_10", 0.0)
        if tb
        else 0.0
    )
    log.info("GATE: Δ overall nDCG@10 = %+.4f | Δ title_matchable = %+.4f", d, dt)
    log.info("GATE: %s", "PASS" if (d >= -0.02 and dt >= -0.03) else "FAIL")
    return 0


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['ndcg_at_10']:.4f} | {m['mrr_at_10']:.4f} "
        f"| {m['recall_at_50']:.4f} | {m['n_scored']} |"
    )


def render_md(p: dict) -> str:
    o = p["overall"]
    lines = [
        "# ADR-014 sparse-BM25 pilot — Qdrant BM25 vs Postgres scix_english",
        "",
        f"- Collection: `{p['collection']}` (sparse `bm25`, Modifier.IDF)",
        f"- Universe: {p['universe_size']} papers (per-query top-100 ∪ gold ∪ ~50k sample)",
        f"- Scored queries (non-empty gold): {p['n_scored']} / 50; top-{p['top_k']}",
        "",
        "## Overall",
        "",
        "| lane | nDCG@10 | MRR@10 | Recall@50 | n |",
        "|---|---|---|---|---|",
        _row("Postgres scix_english", o["postgres"]),
        _row("Qdrant BM25", o["qdrant_bm25"]),
        "",
        f"**Δ nDCG@10 (Qdrant − Postgres) = "
        f"{o['qdrant_bm25']['ndcg_at_10'] - o['postgres']['ndcg_at_10']:+.4f}**",
        "",
        "## By bucket",
        "",
        "| bucket | n | lane | nDCG@10 | MRR@10 | Recall@50 |",
        "|---|---|---|---|---|---|",
    ]
    for b, bb in p["by_bucket"].items():
        lines.append(
            f"| {b} | {bb['n']} | Postgres | {bb['postgres']['ndcg_at_10']:.4f} "
            f"| {bb['postgres']['mrr_at_10']:.4f} | {bb['postgres']['recall_at_50']:.4f} |"
        )
        lines.append(
            f"| | | Qdrant BM25 | {bb['qdrant_bm25']['ndcg_at_10']:.4f} "
            f"| {bb['qdrant_bm25']['mrr_at_10']:.4f} | {bb['qdrant_bm25']['recall_at_50']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
