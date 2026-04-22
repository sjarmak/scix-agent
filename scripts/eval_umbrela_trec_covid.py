#!/usr/bin/env python3
"""TREC-COVID cross-domain anchor for the UMBRELA judge (bead xz4.1.32 leg 2).

Pulls BEIR's TREC-COVID queries + corpus + official NIST qrels (3-level: 0
not-relevant, 1 partial, 2 highly-relevant), samples a balanced slice of
query-passage pairs, dispatches the ``umbrela_judge`` subagent, and reports
how well the judge agrees with real human NIST assessors.

Rationale: the 30-row ordinal spot-check was meant to validate the judge's
0-3 gradations against a human. Replacing it with TREC-COVID swaps
sjarmak's labeling time for published NIST human qrels in a scientific
(biomedical) domain. Still not astro-specific, but it's the best public
graded-relevance anchor we have against *actual* humans rather than
click-model proxies.

Metrics:
  - Binary kappa on collapsed scale (UMBRELA >= 2 <=> NIST >= 1)
  - AUROC with UMBRELA score as probability, NIST >= 1 as label
  - Spearman rho on ordinal scales (NIST 0/1/2 vs UMBRELA 0-3)
  - Score distribution per NIST bucket

Success criteria: binary kappa >= 0.6 AND Spearman rho >= 0.6.

Usage::

    python scripts/eval_umbrela_trec_covid.py

    # smaller / larger pool
    python scripts/eval_umbrela_trec_covid.py --n-per-bucket 20

    # smoke test
    python scripts/eval_umbrela_trec_covid.py --stub --n-per-bucket 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.eval.persona_judge import (  # noqa: E402
    DEFAULT_MAX_CONCURRENCY,
    ClaudeSubprocessDispatcher,
    Dispatcher,
    JudgeTriple,
    PersonaJudge,
    StubDispatcher,
    quadratic_weighted_kappa,
    spearman_rho,
)

logger = logging.getLogger(__name__)

TREC_COVID_ID = "BeIR/trec-covid"
TREC_COVID_QRELS_ID = "BeIR/trec-covid-qrels"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "trec_covid_anchor.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "trec_covid_anchor.json"
RELEVANT_THRESHOLD_UMBRELA = 2
RELEVANT_THRESHOLD_NIST = 1
KAPPA_SUCCESS = 0.6
RHO_SUCCESS = 0.6


@dataclass(frozen=True)
class Pair:
    query_id: str
    corpus_id: str
    query_text: str
    passage_text: str
    nist_score: int  # 0, 1, or 2


def load_trec_covid() -> tuple[dict[str, str], dict[str, tuple[str, str]], list[tuple[str, str, int]]]:
    """Pull queries, corpus (as {id: (title, text)}), and qrels rows."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    queries_pq = hf_hub_download(
        TREC_COVID_ID, "queries/queries-00000-of-00001.parquet", repo_type="dataset"
    )
    corpus_pq = hf_hub_download(
        TREC_COVID_ID, "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"
    )
    qrels_tsv = hf_hub_download(TREC_COVID_QRELS_ID, "test.tsv", repo_type="dataset")

    # TREC-COVID queries use 'text' field (the question); 'title' is a topic label.
    q_df = pd.read_parquet(queries_pq)
    queries = {str(row["_id"]): str(row["text"]) for _, row in q_df.iterrows()}

    c_df = pd.read_parquet(corpus_pq)
    corpus: dict[str, tuple[str, str]] = {}
    for _, row in c_df.iterrows():
        corpus[str(row["_id"])] = (str(row.get("title") or ""), str(row.get("text") or ""))

    qrels: list[tuple[str, str, int]] = []
    with open(qrels_tsv) as f:
        header = f.readline()  # skip header
        _ = header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                score = int(parts[2])
            except ValueError:
                continue
            if score not in (0, 1, 2):
                continue  # drop the stray -1 entries
            qrels.append((parts[0], parts[1], score))

    return queries, corpus, qrels


def sample_pairs(
    *,
    queries: dict[str, str],
    corpus: dict[str, tuple[str, str]],
    qrels: list[tuple[str, str, int]],
    n_per_bucket: int,
    seed: int,
) -> list[Pair]:
    """Sample up to n_per_bucket pairs at each NIST score level.

    Filters out pairs where either the query or the passage is missing
    / empty. Shuffles the final output so the judge sees mixed labels.
    """
    rng = random.Random(seed)
    by_score: dict[int, list[tuple[str, str]]] = {0: [], 1: [], 2: []}
    for qid, cid, score in qrels:
        if qid not in queries or not queries[qid].strip():
            continue
        if cid not in corpus:
            continue
        title, text = corpus[cid]
        passage = (title + "\n\n" + text).strip() if title else text.strip()
        if not passage:
            continue
        by_score[score].append((qid, cid))

    pairs: list[Pair] = []
    for score in (0, 1, 2):
        pool = by_score[score]
        rng.shuffle(pool)
        for qid, cid in pool[:n_per_bucket]:
            title, text = corpus[cid]
            passage = (title + "\n\n" + text).strip() if title else text.strip()
            pairs.append(
                Pair(
                    query_id=qid,
                    corpus_id=cid,
                    query_text=queries[qid],
                    passage_text=passage,
                    nist_score=score,
                )
            )
        if len(pool) < n_per_bucket:
            logger.warning(
                "only %d/%d pairs available at NIST score=%d", len(pool), n_per_bucket, score
            )

    rng.shuffle(pairs)
    return pairs


def _cap(text: str, max_chars: int = 2000) -> str:
    t = text.strip()
    return t[:max_chars] + "..." if len(t) > max_chars else t


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def confusion_2x2(human_bin: list[int], judge_bin: list[int]) -> dict[str, int]:
    tp = sum(1 for h, j in zip(human_bin, judge_bin) if h == 1 and j == 1)
    tn = sum(1 for h, j in zip(human_bin, judge_bin) if h == 0 and j == 0)
    fp = sum(1 for h, j in zip(human_bin, judge_bin) if h == 0 and j == 1)
    fn = sum(1 for h, j in zip(human_bin, judge_bin) if h == 1 and j == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


async def run(
    *, pairs: list[Pair], dispatcher: Dispatcher, max_concurrency: int
) -> tuple[list[tuple[Pair, int, int | None, str]], dict]:
    triples = [
        JudgeTriple(query=p.query_text, bibcode=p.corpus_id, snippet=_cap(p.passage_text))
        for p in pairs
    ]
    judge = PersonaJudge(dispatcher=dispatcher, max_concurrency=max_concurrency)
    scores = await judge.run(triples)

    per_row: list[tuple[Pair, int, int | None, str]] = []
    umbrela_raw: list[int] = []
    nist_raw: list[int] = []
    failed = 0
    for p, s in zip(pairs, scores):
        per_row.append((p, s.score, s.needs_human_review, s.reason))
        if s.score < 0:
            failed += 1
            continue
        umbrela_raw.append(s.score)
        nist_raw.append(p.nist_score)

    if not umbrela_raw:
        raise RuntimeError("every pair failed — cannot compute metrics")

    nist_bin = [1 if s >= RELEVANT_THRESHOLD_NIST else 0 for s in nist_raw]
    umbrela_bin = [1 if s >= RELEVANT_THRESHOLD_UMBRELA else 0 for s in umbrela_raw]

    # Per-bucket mean UMBRELA score — sanity check on monotonic ordering.
    dist: dict[int, dict[int, int]] = {0: {}, 1: {}, 2: {}}
    means: dict[int, float] = {}
    for bucket in (0, 1, 2):
        scores_in_bucket = [u for u, n in zip(umbrela_raw, nist_raw) if n == bucket]
        hist = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in scores_in_bucket:
            if 0 <= s <= 3:
                hist[s] += 1
        dist[bucket] = hist
        means[bucket] = sum(scores_in_bucket) / len(scores_in_bucket) if scores_in_bucket else float("nan")

    metrics: dict = {
        "n_pairs_scored": len(umbrela_raw),
        "n_pairs_failed": failed,
        "binary_kappa": quadratic_weighted_kappa(nist_bin, umbrela_bin, min_score=0, max_score=1),
        "spearman_rho": spearman_rho([float(x) for x in nist_raw], [float(x) for x in umbrela_raw]),
        "auroc": auroc([float(x) for x in umbrela_raw], nist_bin),
        "mean_umbrela_by_nist": means,
        "score_dist_by_nist": dist,
        "confusion_matrix": confusion_2x2(nist_bin, umbrela_bin),
    }
    metrics["passes_kappa"] = metrics["binary_kappa"] >= KAPPA_SUCCESS
    metrics["passes_rho"] = metrics["spearman_rho"] >= RHO_SUCCESS
    metrics["passes"] = metrics["passes_kappa"] and metrics["passes_rho"]
    return per_row, metrics


def write_csv(path: Path, rows: list[tuple[Pair, int, int | None, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["query_id", "corpus_id", "nist_score", "umbrela_score", "needs_human_review",
             "query_text", "passage_preview", "reason"]
        )
        for p, u, needs, reason in rows:
            w.writerow([
                p.query_id,
                p.corpus_id,
                p.nist_score,
                u,
                "" if needs is None else ("true" if needs else "false"),
                p.query_text,
                p.passage_text[:300].replace("\n", " "),
                reason.replace("\n", " ")[:500] if reason else "",
            ])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-bucket", type=int, default=40, help="Pairs at each NIST score level.")
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument("--claude-binary", default="claude")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("loading TREC-COVID from HuggingFace...")
    queries, corpus, qrels = load_trec_covid()
    logger.info("%d queries, %d corpus, %d qrels", len(queries), len(corpus), len(qrels))

    pairs = sample_pairs(
        queries=queries, corpus=corpus, qrels=qrels, n_per_bucket=args.n_per_bucket, seed=args.seed
    )
    logger.info("sampled %d pairs (balanced across NIST 0/1/2)", len(pairs))

    dispatcher: Dispatcher
    if args.stub:
        dispatcher = StubDispatcher(fixed_score=2, reason="stub")
    else:
        dispatcher = ClaudeSubprocessDispatcher(claude_binary=args.claude_binary)

    rows, metrics = asyncio.run(run(pairs=pairs, dispatcher=dispatcher, max_concurrency=args.concurrency))

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nTREC-COVID anchor")
    print(f"  n_scored / n_failed    : {metrics['n_pairs_scored']} / {metrics['n_pairs_failed']}")
    print(f"  mean UMBRELA by NIST   : {metrics['mean_umbrela_by_nist']}")
    print(f"  score dist by NIST     : {metrics['score_dist_by_nist']}")
    print(f"  binary kappa           : {metrics['binary_kappa']:+.3f}  (>= {KAPPA_SUCCESS}; passes={metrics['passes_kappa']})")
    print(f"  spearman rho           : {metrics['spearman_rho']:+.3f}  (>= {RHO_SUCCESS}; passes={metrics['passes_rho']})")
    print(f"  AUROC                  : {metrics['auroc']:+.3f}")
    cm = metrics["confusion_matrix"]
    print(f"  confusion (NIST/UMB)   : TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}")
    print(f"\n  csv : {args.output_csv}")
    print(f"  json: {args.output_json}")
    return 0 if metrics["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
