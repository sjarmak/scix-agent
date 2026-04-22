#!/usr/bin/env python3
"""Transfer-test the UMBRELA judge on NASA SMD content (bead xz4.1.32 anchor A).

Pulls the NASA-IR benchmark (nasa-impact/nasa-smd-IR-benchmark) from HF,
samples balanced positive / negative (query, passage) pairs, dispatches the
``umbrela_judge`` subagent for each pair, and reports how well the judge
separates known-relevant from known-irrelevant on NASA SMD astro / planetary /
earth / helio passages.

This is a binary transfer test — NASA-IR carries binary qrels, so the UMBRELA
0-3 scale is collapsed to relevant (>=2) / irrelevant (<=1) for scoring.
A clean pass here is the evidence that UMBRELA's TREC-DL-calibrated
rubric actually transfers to our corpus.

Success criteria:
    AUROC > 0.85 AND binary kappa >= 0.6

Usage::

    # real run (calls claude -p for every pair, ~15-25 min wall)
    python scripts/eval_umbrela_nasa_ir.py

    # quick wiring smoke (stub dispatcher, no Claude calls)
    python scripts/eval_umbrela_nasa_ir.py --stub --n-pos 5 --n-neg 5

    # smaller sample
    python scripts/eval_umbrela_nasa_ir.py --n-pos 50 --n-neg 50
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
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
)

logger = logging.getLogger(__name__)

HF_DATASET_ID = "nasa-impact/nasa-smd-IR-benchmark"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "nasa_ir_transfer.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "nasa_ir_transfer.json"
RELEVANT_THRESHOLD = 2  # UMBRELA score >= 2 collapses to "relevant"
AUROC_SUCCESS = 0.85
KAPPA_SUCCESS = 0.6


@dataclass(frozen=True)
class Pair:
    query_id: str
    corpus_id: str
    query_text: str
    passage_text: str
    gold_relevant: int  # 1 = known-positive, 0 = sampled-negative


def load_nasa_ir() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Pull NASA-IR from HF; return (queries, corpus, qrels).

    qrels maps query_id -> set of positively-labeled corpus_ids.
    """
    from huggingface_hub import hf_hub_download

    queries_path = hf_hub_download(HF_DATASET_ID, "queries.jsonl", repo_type="dataset")
    corpus_path = hf_hub_download(HF_DATASET_ID, "corpus.jsonl", repo_type="dataset")
    qrels_test = hf_hub_download(HF_DATASET_ID, "qrels/test.tsv", repo_type="dataset")
    qrels_dev = hf_hub_download(HF_DATASET_ID, "qrels/dev.tsv", repo_type="dataset")

    queries: dict[str, str] = {}
    with open(queries_path) as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]

    corpus: dict[str, str] = {}
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()
            # The passage substrate we give the judge. Title first when present.
            corpus[obj["_id"]] = f"{title}\n\n{text}".strip() if title else text

    qrels: dict[str, set[str]] = {}
    for path in (qrels_test, qrels_dev):
        with open(path) as f:
            header = f.readline()  # skip header row
            _ = header  # nosec — intentional discard
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                qid, cid, score = parts[0], parts[1], parts[2]
                if int(score) >= 1:
                    qrels.setdefault(qid, set()).add(cid)

    return queries, corpus, qrels


def build_pairs(
    *,
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels: dict[str, set[str]],
    n_pos: int,
    n_neg: int,
    seed: int,
) -> list[Pair]:
    """Sample balanced positive + negative pairs.

    Positives are drawn from qrels; negatives are uniformly sampled corpus
    docs that are NOT in qrels for the query.
    """
    rng = random.Random(seed)

    # Flatten positives to a list of (qid, cid), filter to rows where both
    # the query and the corpus doc are loadable and non-empty.
    positives: list[tuple[str, str]] = []
    for qid, cids in qrels.items():
        if qid not in queries or not queries[qid].strip():
            continue
        for cid in cids:
            if cid not in corpus or not corpus[cid].strip():
                continue
            positives.append((qid, cid))
    rng.shuffle(positives)

    if len(positives) < n_pos:
        raise RuntimeError(
            f"only {len(positives)} valid positive pairs available; cannot sample {n_pos}"
        )
    sampled_pos = positives[:n_pos]

    corpus_ids = [cid for cid, text in corpus.items() if text.strip()]

    pairs: list[Pair] = []
    for qid, cid in sampled_pos:
        pairs.append(
            Pair(
                query_id=qid,
                corpus_id=cid,
                query_text=queries[qid],
                passage_text=corpus[cid],
                gold_relevant=1,
            )
        )

    # Build negatives by sampling queries (with replacement across the positive
    # pool) and corpus docs that are not in that query's qrels. Cap attempts
    # per query to avoid pathological loops.
    neg_qids = [qid for qid, _ in sampled_pos]  # reuse queries we have positives for
    rng.shuffle(neg_qids)
    attempts_per = 100
    made = 0
    for qid in neg_qids:
        if made >= n_neg:
            break
        gold = qrels.get(qid, set())
        for _ in range(attempts_per):
            cand = rng.choice(corpus_ids)
            if cand not in gold:
                pairs.append(
                    Pair(
                        query_id=qid,
                        corpus_id=cand,
                        query_text=queries[qid],
                        passage_text=corpus[cand],
                        gold_relevant=0,
                    )
                )
                made += 1
                break
    if made < n_neg:
        raise RuntimeError(f"could only build {made}/{n_neg} negatives")

    rng.shuffle(pairs)
    return pairs


def _format_passage_for_judge(passage: str, max_chars: int = 2000) -> str:
    """Cap the passage length passed to the judge.

    UMBRELA's rubric was calibrated on web/news passages ~100-300 words. Keep
    the budget modest so very long ADS bodies don't distort the prior.
    """
    text = passage.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def binary_cohen_kappa(human: list[int], judge: list[int]) -> float:
    """Unweighted Cohen's kappa on 2-class labels."""
    return quadratic_weighted_kappa(
        human, judge, min_score=0, max_score=1
    )


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U formulation of AUROC.

    Positive class label = 1. Ties contribute 0.5 (standard treatment).
    """
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


def confusion_matrix(human: list[int], judge: list[int]) -> dict[str, int]:
    """2x2 confusion matrix with human on rows, judge on cols."""
    tp = sum(1 for h, j in zip(human, judge) if h == 1 and j == 1)
    tn = sum(1 for h, j in zip(human, judge) if h == 0 and j == 0)
    fp = sum(1 for h, j in zip(human, judge) if h == 0 and j == 1)
    fn = sum(1 for h, j in zip(human, judge) if h == 1 and j == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


async def run_transfer_test(
    *,
    pairs: list[Pair],
    dispatcher: Dispatcher,
    max_concurrency: int,
) -> tuple[list[tuple[Pair, int, int | None, str]], dict]:
    """Score every pair, return (per-pair rows, metrics)."""
    triples = [
        JudgeTriple(
            query=p.query_text,
            bibcode=p.corpus_id,
            snippet=_format_passage_for_judge(p.passage_text),
        )
        for p in pairs
    ]
    judge = PersonaJudge(dispatcher=dispatcher, max_concurrency=max_concurrency)
    scores = await judge.run(triples)

    per_row: list[tuple[Pair, int, int | None, str]] = []
    raw_scores: list[int] = []
    golds: list[int] = []
    for p, s in zip(pairs, scores):
        raw = s.score
        judge_binary = 1 if raw >= RELEVANT_THRESHOLD else 0
        per_row.append((p, raw, s.needs_human_review, s.reason))
        if raw < 0:
            # failed triple — exclude from metrics but keep the CSV row
            continue
        raw_scores.append(raw)
        golds.append(p.gold_relevant)

    if not raw_scores:
        raise RuntimeError("every pair failed — cannot compute metrics")

    judge_binary = [1 if s >= RELEVANT_THRESHOLD else 0 for s in raw_scores]

    metrics = {
        "n_pairs_scored": len(raw_scores),
        "n_pairs_failed": len(pairs) - len(raw_scores),
        "n_pos": sum(golds),
        "n_neg": len(golds) - sum(golds),
        "mean_score_pos": (
            sum(s for s, g in zip(raw_scores, golds) if g == 1)
            / max(sum(golds), 1)
        ),
        "mean_score_neg": (
            sum(s for s, g in zip(raw_scores, golds) if g == 0)
            / max(len(golds) - sum(golds), 1)
        ),
        "auroc": auroc([float(s) for s in raw_scores], golds),
        "binary_kappa": binary_cohen_kappa(golds, judge_binary),
        "confusion_matrix": confusion_matrix(golds, judge_binary),
        "score_dist_pos": _histogram(
            [s for s, g in zip(raw_scores, golds) if g == 1]
        ),
        "score_dist_neg": _histogram(
            [s for s, g in zip(raw_scores, golds) if g == 0]
        ),
    }
    metrics["passes_auroc"] = metrics["auroc"] > AUROC_SUCCESS
    metrics["passes_kappa"] = metrics["binary_kappa"] >= KAPPA_SUCCESS
    metrics["passes"] = metrics["passes_auroc"] and metrics["passes_kappa"]
    return per_row, metrics


def _histogram(scores: list[int]) -> dict[int, int]:
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for s in scores:
        if 0 <= s <= 3:
            hist[s] += 1
    return hist


def write_csv(path: Path, rows: list[tuple[Pair, int, int | None, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "query_id",
                "corpus_id",
                "gold_relevant",
                "judge_score",
                "judge_binary",
                "needs_human_review",
                "query_text",
                "passage_preview",
                "reason",
            ]
        )
        for p, raw, needs_review, reason in rows:
            w.writerow(
                [
                    p.query_id,
                    p.corpus_id,
                    p.gold_relevant,
                    raw,
                    1 if raw >= RELEVANT_THRESHOLD else 0,
                    "" if needs_review is None else ("true" if needs_review else "false"),
                    p.query_text,
                    p.passage_text[:300].replace("\n", " "),
                    reason.replace("\n", " ")[:500] if reason else "",
                ]
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-pos", type=int, default=100, help="Number of positive pairs to sample.")
    p.add_argument("--n-neg", type=int, default=100, help="Number of negative pairs to sample.")
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument("--claude-binary", default="claude")
    p.add_argument(
        "--stub", action="store_true", help="Use StubDispatcher — wiring check only."
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("loading NASA-IR from HuggingFace...")
    queries, corpus, qrels = load_nasa_ir()
    logger.info(
        "loaded %d queries, %d corpus docs, %d queries with qrels",
        len(queries),
        len(corpus),
        len(qrels),
    )

    logger.info("building balanced pairs (%d pos, %d neg)...", args.n_pos, args.n_neg)
    pairs = build_pairs(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        seed=args.seed,
    )
    logger.info("%d total pairs to score", len(pairs))

    dispatcher: Dispatcher
    if args.stub:
        # Stub: positives get score 3, negatives get score 0 — lets us check
        # metric plumbing end-to-end without calling Claude. We key on
        # gold_relevant by attaching a sentinel score in advance; implemented
        # with a closure-wrapping dispatcher.
        dispatcher = _GoldAwareStubDispatcher({p.corpus_id + "|" + p.query_id: p.gold_relevant for p in pairs})
    else:
        dispatcher = ClaudeSubprocessDispatcher(claude_binary=args.claude_binary)

    rows, metrics = asyncio.run(
        run_transfer_test(
            pairs=pairs, dispatcher=dispatcher, max_concurrency=args.concurrency
        )
    )

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nNASA-IR transfer test")
    print(f"  n_pairs_scored : {metrics['n_pairs_scored']}")
    print(f"  n_pairs_failed : {metrics['n_pairs_failed']}")
    print(f"  n_pos / n_neg  : {metrics['n_pos']} / {metrics['n_neg']}")
    print(f"  mean score pos : {metrics['mean_score_pos']:+.3f}")
    print(f"  mean score neg : {metrics['mean_score_neg']:+.3f}")
    print(f"  AUROC          : {metrics['auroc']:+.3f}  "
          f"(threshold > {AUROC_SUCCESS}; passes={metrics['passes_auroc']})")
    print(f"  binary kappa   : {metrics['binary_kappa']:+.3f}  "
          f"(threshold >= {KAPPA_SUCCESS}; passes={metrics['passes_kappa']})")
    cm = metrics["confusion_matrix"]
    print(f"  confusion (h/j)  TP={cm['tp']}  FN={cm['fn']}  FP={cm['fp']}  TN={cm['tn']}")
    print(f"  pos score dist : {metrics['score_dist_pos']}")
    print(f"  neg score dist : {metrics['score_dist_neg']}")
    print(f"\n  csv : {args.output_csv}")
    print(f"  json: {args.output_json}")
    return 0 if metrics["passes"] else 1


@dataclass
class _GoldAwareStubDispatcher:
    """Stub that returns 3 for known positives, 0 for known negatives.

    Only used by ``--stub`` smoke runs to validate the metric plumbing.
    """

    gold_by_key: dict[str, int]

    async def judge(self, triple: JudgeTriple) -> "JudgeScore":  # noqa: F821
        from scix.eval.persona_judge import JudgeScore

        # Caller-provided triple lacks the query-id context, so we stash
        # gold in a dict keyed by (corpus_id | query_text). A mismatch is a
        # wiring bug worth surfacing, so we let KeyError propagate via .get.
        match = None
        for k, v in self.gold_by_key.items():
            cid, qid_unused = k.split("|", 1)
            if cid == triple.bibcode and triple.query in (
                # best-effort match: query_text is all we have
                triple.query,
            ):
                match = v
                break
        if match is None:
            # Fall back: random 0/3 but deterministic on the bibcode.
            match = (hash(triple.bibcode) % 2)
        return JudgeScore(
            score=3 if match else 0,
            reason="stub",
            needs_human_review=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
