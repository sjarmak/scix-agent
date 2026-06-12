#!/usr/bin/env python3
"""ADR-014 pilot, system-level: does swapping the lexical lane from Postgres
``scix_english`` tsvector to Qdrant BM25 help the *fused hybrid*?

The first pilot (``eval_sparse_pilot.py``) compared the lexical lanes in
isolation — but the lexical lane never runs alone in production; it is
RRF-fused (k=60) with the dense INDUS lane. This harness measures the lanes
as they actually serve: it computes the lanes per query and fuses several
arms, holding the dense lane FIXED so the delta isolates the lexical swap.

Lanes (each top-50, ranked bibcodes):
  lex_pg     — Postgres ``scix_english`` tsvector, AND  (search.lexical_search)
  lex_pg_or  — same lane, OR semantics                  (tsquery_mode=plain_or)
  lex_qd     — Qdrant BM25 sparse                       (--collection)
  body       — Postgres body BM25                       (search.lexical_search_body)
  dense      — Qdrant INDUS dense                        (search.vector_search)

Arms (RRF k=60 over the lanes shown):
  dense_only         : [dense]
  pg_lex+dense       : [lex_pg, dense]          AND parsing + ts_rank_cd
  pg_or+dense        : [lex_pg_or, dense]       OR parsing + ts_rank_cd
  bm25+dense         : [lex_qd, dense]          OR parsing + BM25 scoring
  pg_lex+body+dense  : [lex_pg, body, dense]    current production hybrid
  pg_or+body+dense   : [lex_pg_or, body, dense]
  bm25+body+dense    : [lex_qd, body, dense]    production hybrid w/ lexical swapped

Headline pair = pg_lex+dense vs bm25+dense (both two-lane, dense identical).
Attribution chain (ADR-014 confound): pg_lex -> pg_or isolates the AND->OR
query-parsing effect; pg_or -> bm25 isolates the BM25-scoring effect (both OR).

Reuses the canonical INDUS encoder, RRF, and metric functions from
``eval_retrieval_50q.py``. Local-only (INDUS + FastEmbed BM25, no paid API).
Run under ``scix-batch`` (loads the INDUS model). Needs ``QDRANT_URL`` so the
dense lane routes to Qdrant (defaulted here if unset).

Usage::

    scix-batch python scripts/eval_sparse_hybrid_pilot.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg
from qdrant_client import QdrantClient
from qdrant_client import models as qm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6633")
# Rank the FULL Postgres lexical match set, not the 30k production TID-ordered
# cap. The cap clips a larger fraction of the OR arm's (much larger) match set
# than the AND arm's, which would make the pg_or vs pg_lex attribution partly
# TID-bias rather than pure AND->OR semantics. Qdrant BM25 already returns exact
# ranked top-k with no cap, so uncapping the pg lanes is also the fair A/B. This
# is the eval-harness usage lexical_search documents (offline, slower is fine).
os.environ.setdefault("SCIX_LEXICAL_POOL", "INF")

from scix.search import (  # noqa: E402
    SearchFilters,
    lexical_search,
    lexical_search_body,
    vector_search,
)
from scripts._sparse_bm25 import (  # noqa: E402
    add_bm25_tokenizer_args,
    build_model,
    config_from_args,
    load_recorded_config,
)

_spec = importlib.util.spec_from_file_location(
    "eval_retrieval_50q", _REPO_ROOT / "scripts" / "eval_retrieval_50q.py"
)
_e = importlib.util.module_from_spec(_spec)
sys.modules["eval_retrieval_50q"] = _e
_spec.loader.exec_module(_e)  # type: ignore[union-attr]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("eval_sparse_hybrid_pilot")

PILOT_COLLECTION = "scix_sparse_pilot_v1"
DSN = "dbname=scix"
QUERIES_PATH = _REPO_ROOT / "eval" / "retrieval_50q.jsonl"
UNIVERSE_PATH = _REPO_ROOT / "results" / "sparse_pilot_universe.json"
OUT_JSON = _REPO_ROOT / "results" / "sparse_hybrid_pilot_eval.json"
OUT_MD = _REPO_ROOT / "results" / "sparse_hybrid_pilot_eval.md"
K = 60
TOP_K = 50
METRICS = ("ndcg_at_10", "mrr_at_10", "recall_at_50")

ARMS = {
    "dense_only": ("dense",),
    "pg_lex+dense": ("lex_pg", "dense"),
    "pg_or+dense": ("lex_pg_or", "dense"),
    "bm25+dense": ("lex_qd", "dense"),
    "pg_lex+body+dense": ("lex_pg", "body", "dense"),
    "pg_or+body+dense": ("lex_pg_or", "body", "dense"),
    "bm25+body+dense": ("lex_qd", "body", "dense"),
}


def _bibs(result) -> list[str]:
    # No truncation here — the caller's keep() filters to the universe (when
    # restricting) and then slices to TOP_K.
    return [p["bibcode"] for p in result.papers if p.get("bibcode")]


def lanes_for(
    conn, client, model, collection: str, query: str, universe: set[str] | None
) -> dict[str, list[str]]:
    vec = _e._indus_encode(query)
    sparse = next(iter(model.query_embed(query)))
    # Over-fetch the corpus-wide lanes when restricting, so the post-filter to
    # the pilot universe still yields a full top-K.
    fetch = TOP_K if universe is None else 500
    qd = client.query_points(
        collection_name=collection,
        query=qm.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
        using="bm25",
        limit=TOP_K,
        with_payload=True,
    ).points

    def keep(bibs: list[str]) -> list[str]:
        if universe is not None:
            bibs = [b for b in bibs if b in universe]
        return bibs[:TOP_K]

    sf = SearchFilters()
    return {
        "lex_pg": keep(_bibs(lexical_search(conn, query, filters=sf, limit=fetch))),
        "lex_pg_or": keep(
            _bibs(lexical_search(conn, query, filters=sf, limit=fetch, tsquery_mode="plain_or"))
        ),
        "lex_qd": keep([h.payload["bibcode"] for h in qd]),
        "body": keep(_bibs(lexical_search_body(conn, query, filters=sf, limit=fetch))),
        "dense": keep(_bibs(vector_search(conn, vec, model_name="indus", limit=fetch))),
    }


def aggregate(per_query: list[dict], arm: str) -> dict:
    out: dict[str, float | int] = {}
    for m in METRICS:
        vals = [r["arms"][arm][m] for r in per_query if r["arms"][arm][m] is not None]
        out[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ["QDRANT_URL"])
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument(
        "--collection",
        default=PILOT_COLLECTION,
        help=f"Qdrant BM25 collection for the lex_qd lane (default {PILOT_COLLECTION}; "
        "use scix_sparse_full_v1 for the Phase-2 full-corpus run).",
    )
    ap.add_argument(
        "--restrict-universe",
        action="store_true",
        help="Restrict ALL lanes to the 52k pilot universe so the BM25 lane's "
        "small-haystack advantage is removed (fair A/B; dense/body lose their "
        "32M reach). Only valid for the pilot collection. Writes *_fair.{json,md}.",
    )
    add_bm25_tokenizer_args(ap)
    args = ap.parse_args(argv)

    if args.restrict_universe and args.collection != PILOT_COLLECTION:
        ap.error(
            "--restrict-universe only applies to the pilot collection "
            f"({PILOT_COLLECTION}); the universe file is the pilot's membership."
        )

    queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]
    universe = (
        set(json.loads(UNIVERSE_PATH.read_text())["bibcodes"]) if args.restrict_universe else None
    )
    suffix = "_fair" if args.restrict_universe else ""
    # Query tokenizer MUST match what built the collection. Prefer the config
    # the build recorded; fall back to flags (the pilot predates the sidecar,
    # so it defaults to the original Qdrant/bm25 params — backward compatible).
    config = load_recorded_config(args.collection) or config_from_args(args)
    log.info(
        "collection=%s restrict_universe=%s tokenizer=%s",
        args.collection,
        args.restrict_universe,
        config.signature(),
    )
    model = build_model(config)
    client = QdrantClient(url=args.url, prefer_grpc=False, timeout=120)
    conn = psycopg.connect(args.dsn)
    # qajc operator condition: same memory bounds as the build session
    # (host OOM'd postgres 2026-06-11/12 on parallel hash + large work_mem).
    conn.execute("SET work_mem = '256MB'")
    conn.execute("SET max_parallel_workers_per_gather = 0")

    per_query: list[dict] = []
    try:
        for q in queries:
            gold = list(q.get("gold_bibcodes", []))
            if not gold:
                continue
            lanes = lanes_for(conn, client, model, args.collection, q["query"], universe)
            arms = {}
            for name, lane_keys in ARMS.items():
                fused = _e.rrf_fuse_bibcodes([lanes[k] for k in lane_keys], k_rrf=K)[:TOP_K]
                arms[name] = _e.score_query(fused, gold, k=10)
            per_query.append({"query": q["query"], "bucket": q.get("bucket", "?"), "arms": arms})
            log.info("scored %d/%d", len(per_query), len(queries))
    finally:
        conn.close()

    overall = {arm: aggregate(per_query, arm) for arm in ARMS}
    buckets = defaultdict(list)
    for r in per_query:
        buckets[r["bucket"]].append(r)
    by_bucket = {
        b: {arm: aggregate(rows, arm) for arm in ARMS} for b, rows in sorted(buckets.items())
    }

    payload = {
        "collection": args.collection,
        "tokenizer": config.signature(),
        "n_scored": len(per_query),
        "rrf_k": K,
        "top_k": TOP_K,
        "restrict_universe": args.restrict_universe,
        "overall": overall,
        "by_bucket": by_bucket,
        "per_query": per_query,
    }
    out_json = OUT_JSON.with_name(OUT_JSON.stem + suffix + OUT_JSON.suffix)
    out_md = OUT_MD.with_name(OUT_MD.stem + suffix + OUT_MD.suffix)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text(render_md(payload))
    log.info("wrote %s and %s", out_json, out_md)

    a = overall["pg_lex+dense"]["ndcg_at_10"]
    o = overall["pg_or+dense"]["ndcg_at_10"]
    b = overall["bm25+dense"]["ndcg_at_10"]
    c = overall["pg_lex+body+dense"]["ndcg_at_10"]
    d = overall["bm25+body+dense"]["ndcg_at_10"]
    log.info("HEADLINE 2-lane: bm25+dense %.4f vs pg_lex+dense %.4f (Δ %+.4f)", b, a, b - a)
    log.info("3-lane: bm25+body+dense %.4f vs pg_lex+body+dense %.4f (Δ %+.4f)", d, c, d - c)
    log.info(
        "ATTRIBUTION: AND→OR parsing %+.4f (pg_or−pg_lex); BM25 scoring %+.4f (bm25−pg_or)",
        o - a,
        b - o,
    )
    return 0


def render_md(p: dict) -> str:
    haystack = "52k pilot universe (fair A/B)" if p.get("restrict_universe") else "full 32M corpus"
    lines = [
        "# ADR-014 sparse-BM25 pilot — system-level (fused hybrid)",
        "",
        f"- Collection: `{p['collection']}`; dense/body lanes search the {haystack}.",
        f"- BM25 tokenizer: `{p.get('tokenizer', 'bm25[default]')}`.",
        f"- Scored queries: {p['n_scored']} / 50; RRF k={p['rrf_k']}; top-{p['top_k']}",
        "- Dense lane is identical in every arm, so each Δ isolates the lexical swap.",
        "",
        "## Overall",
        "",
        "| arm | nDCG@10 | MRR@10 | Recall@50 |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        m = p["overall"][arm]
        lines.append(
            f"| {arm} | {m['ndcg_at_10']:.4f} | {m['mrr_at_10']:.4f} | {m['recall_at_50']:.4f} |"
        )
    a, b = p["overall"]["pg_lex+dense"], p["overall"]["bm25+dense"]
    o = p["overall"]["pg_or+dense"]
    c, d = p["overall"]["pg_lex+body+dense"], p["overall"]["bm25+body+dense"]
    lines += [
        "",
        f"**Headline (2-lane, dense+lexical): bm25+dense − pg_lex+dense = "
        f"{b['ndcg_at_10'] - a['ndcg_at_10']:+.4f} nDCG@10**",
        f"**3-lane (+body): bm25+body+dense − pg_lex+body+dense = "
        f"{d['ndcg_at_10'] - c['ndcg_at_10']:+.4f} nDCG@10**",
        "",
        "### Attribution (2-lane): scoring vs query parsing",
        "",
        f"- AND→OR query parsing: pg_or+dense − pg_lex+dense = "
        f"{o['ndcg_at_10'] - a['ndcg_at_10']:+.4f} nDCG@10",
        f"- BM25 scoring (both OR): bm25+dense − pg_or+dense = "
        f"{b['ndcg_at_10'] - o['ndcg_at_10']:+.4f} nDCG@10",
        "",
        "## By bucket (nDCG@10)",
        "",
        "| bucket | " + " | ".join(ARMS) + " |",
        "|---|" + "|".join("---" for _ in ARMS) + "|",
    ]
    for b_name, bb in p["by_bucket"].items():
        cells = " | ".join(f"{bb[arm]['ndcg_at_10']:.4f}" for arm in ARMS)
        lines.append(f"| {b_name} | {cells} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
