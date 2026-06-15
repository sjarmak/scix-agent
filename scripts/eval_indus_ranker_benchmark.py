#!/usr/bin/env python3
"""M2 wiring sanity check: INDUS ranker on its home benchmark.

Confirms that ``nasa-impact/nasa-smd-ibm-ranker`` — wired through
:class:`scix.search.CrossEncoderReranker` — improves ranking quality over a
first-stage BM25 retrieval on ``nasa-impact/nasa-smd-IR-benchmark`` (BEIR
format). This is a cheap integration gate that proves OUR reranker plumbing
scores correctly before spending corpus-eval time on the 50q harness (M3).

If the reranker cannot beat BM25 on its own benchmark, the wiring is wrong —
stop and debug rather than running M3.

BM25 is a compact, auditable Okapi implementation (eval scaffolding only, not
production code) so this script adds no new runtime dependency.

Usage (wrap in scix-batch per CLAUDE.md memory-isolation rules):

    scix-batch .venv/bin/python scripts/eval_indus_ranker_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_indus_ranker_benchmark")

BENCHMARK_REPO = "nasa-impact/nasa-smd-IR-benchmark"
BENCHMARK_SHA = "1c79e339a2e2ef8f0f2b6218aba66b9531573908"
INDUS_RANKER_MODEL = "nasa-impact/nasa-smd-ibm-ranker"

FIRST_STAGE_K = 100  # BM25 candidate-pool depth fed to the reranker
K_METRIC = 10

DEFAULT_OUT_JSON = Path("results/indus_ranker_benchmark_m2.json")

# Okapi BM25 hyperparameters (textbook defaults).
BM25_K1 = 1.5
BM25_B = 0.75


# ---------------------------------------------------------------------------
# Compact Okapi BM25 over an in-memory corpus (eval scaffolding only)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization — deterministic, dependency-free."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


@dataclass
class BM25Index:
    """Inverted-index Okapi BM25. Scores only docs sharing a query term."""

    doc_ids: list[str]
    doc_len: list[int]
    avgdl: float
    postings: dict[str, list[tuple[int, int]]]  # term -> [(doc_idx, tf), ...]
    idf: dict[str, float]
    n_docs: int

    @classmethod
    def build(cls, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len: list[int] = []
        df: dict[str, int] = defaultdict(int)
        for idx, text in enumerate(texts):
            toks = _tokenize(text)
            doc_len.append(len(toks))
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, c in tf.items():
                postings[t].append((idx, c))
                df[t] += 1
        n_docs = len(doc_ids)
        avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0
        # BM25 idf with the +1 inside the log to keep it non-negative.
        idf = {t: math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5)) for t, dfi in df.items()}
        return cls(doc_ids, doc_len, avgdl, dict(postings), idf, n_docs)

    def search(self, query: str, top_k: int) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(_tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl)
                scores[doc_idx] += idf * (tf * (BM25_K1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self.doc_ids[idx] for idx, _ in ranked]


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(BENCHMARK_REPO, filename, repo_type="dataset", revision=BENCHMARK_SHA)


def load_benchmark(
    qrels_split: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, set[str]]]:
    """Return (corpus by id, queries by id, qrels: query_id -> {relevant doc ids})."""
    corpus: dict[str, dict[str, str]] = {}
    for line in open(_download("corpus.jsonl")):
        rec = json.loads(line)
        corpus[rec["_id"]] = {"title": rec.get("title") or "", "text": rec.get("text") or ""}

    queries: dict[str, str] = {}
    for line in open(_download("queries.jsonl")):
        rec = json.loads(line)
        queries[rec["_id"]] = rec.get("text") or ""

    qrels: dict[str, set[str]] = defaultdict(set)
    qrels_lines = open(_download(f"qrels/{qrels_split}.tsv")).read().splitlines()
    for line in qrels_lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qid, doc_id, score = parts[0], parts[1], parts[2]
        if int(score) > 0:
            qrels[qid].add(doc_id)
    return corpus, queries, dict(qrels)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    label: str
    n_queries: int
    ndcg_10: float
    mrr_10: float
    recall_pool: float  # recall@FIRST_STAGE_K of the BM25 pool (rerank ceiling)


def _mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(qrels_split: str, max_queries: int | None, out_json: Path) -> int:
    from scix.ir_metrics import ndcg_at_k, recall_at_k
    from scix.search import CrossEncoderReranker

    logger.info("Loading benchmark (%s split) ...", qrels_split)
    corpus, queries, qrels = load_benchmark(qrels_split)
    doc_ids = list(corpus.keys())
    logger.info("corpus=%d queries=%d judged_queries=%d", len(corpus), len(queries), len(qrels))

    logger.info("Building BM25 index over %d docs ...", len(doc_ids))
    t0 = time.perf_counter()
    texts = [f"{corpus[d]['title']} {corpus[d]['text']}".strip() for d in doc_ids]
    bm25 = BM25Index.build(doc_ids, texts)
    logger.info("BM25 index built in %.1fs", time.perf_counter() - t0)

    reranker = CrossEncoderReranker(model_name=INDUS_RANKER_MODEL)
    reranker("warmup", [{"bibcode": "_w", "title": "warm", "abstract_snippet": ""}])

    judged = [q for q in qrels if q in queries]
    if max_queries is not None:
        judged = judged[:max_queries]

    bm25_ndcg, bm25_mrr, pool_recall = [], [], []
    rr_ndcg, rr_mrr = [], []
    rerank_latencies: list[float] = []

    for qid in judged:
        relevant = qrels[qid]
        relevance_map = {d: 1.0 for d in relevant}
        pool = bm25.search(queries[qid], FIRST_STAGE_K)
        if not pool:
            continue

        bm25_ndcg.append(ndcg_at_k(pool, relevance_map, k=K_METRIC))
        bm25_mrr.append(_mrr_at_k(pool, relevant, K_METRIC))
        pool_recall.append(recall_at_k(pool, relevant, k=FIRST_STAGE_K))

        candidates = [
            {
                "bibcode": d,
                "title": corpus[d]["title"],
                "abstract_snippet": corpus[d]["text"][:1000],
            }
            for d in pool
        ]
        t_r = time.perf_counter()
        ranked = reranker(queries[qid], candidates)
        rerank_latencies.append((time.perf_counter() - t_r) * 1000.0)
        ranked_ids = [p["bibcode"] for p in ranked]

        rr_ndcg.append(ndcg_at_k(ranked_ids, relevance_map, k=K_METRIC))
        rr_mrr.append(_mrr_at_k(ranked_ids, relevant, K_METRIC))

    n = len(bm25_ndcg)
    if n == 0:
        logger.error("No evaluable queries — aborting.")
        return 1

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    bm25_m = StageMetrics(
        "bm25", n, round(mean(bm25_ndcg), 4), round(mean(bm25_mrr), 4), round(mean(pool_recall), 4)
    )
    rr_m = StageMetrics(
        "bm25+indus_ranker",
        n,
        round(mean(rr_ndcg), 4),
        round(mean(rr_mrr), 4),
        round(mean(pool_recall), 4),
    )

    sorted_lat = sorted(rerank_latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    p95 = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))] if sorted_lat else 0.0

    ndcg_delta = round(rr_m.ndcg_10 - bm25_m.ndcg_10, 4)
    mrr_delta = round(rr_m.mrr_10 - bm25_m.mrr_10, 4)
    passed = ndcg_delta > 0 and mrr_delta > 0

    logger.info("=== M2 sanity check (%s, n=%d) ===", qrels_split, n)
    logger.info(
        "BM25            nDCG@10=%.4f MRR@10=%.4f recall@%d=%.4f",
        bm25_m.ndcg_10,
        bm25_m.mrr_10,
        FIRST_STAGE_K,
        bm25_m.recall_pool,
    )
    logger.info("BM25+INDUS rank nDCG@10=%.4f MRR@10=%.4f", rr_m.ndcg_10, rr_m.mrr_10)
    logger.info(
        "Δ nDCG@10=%+.4f  Δ MRR@10=%+.4f  rerank p50=%.0fms p95=%.0fms",
        ndcg_delta,
        mrr_delta,
        p50,
        p95,
    )
    logger.info("M2 GATE: %s", "PASS — reranker beats BM25" if passed else "FAIL — wiring suspect")

    payload = {
        "benchmark": BENCHMARK_REPO,
        "benchmark_revision": BENCHMARK_SHA,
        "model": INDUS_RANKER_MODEL,
        "qrels_split": qrels_split,
        "first_stage_k": FIRST_STAGE_K,
        "k_metric": K_METRIC,
        "n_queries": n,
        "bm25": bm25_m.__dict__,
        "bm25_plus_indus_ranker": rr_m.__dict__,
        "ndcg_delta": ndcg_delta,
        "mrr_delta": mrr_delta,
        "rerank_latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1)},
        "m2_pass": passed,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote %s", out_json)
    return 0 if passed else 2


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qrels-split", default="test", choices=["test", "dev"])
    p.add_argument(
        "--max-queries", type=int, default=None, help="Cap evaluable queries (default: all)."
    )
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return evaluate(args.qrels_split, args.max_queries, args.out_json)


if __name__ == "__main__":
    raise SystemExit(main())
