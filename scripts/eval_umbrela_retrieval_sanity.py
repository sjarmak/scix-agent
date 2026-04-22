#!/usr/bin/env python3
"""Own-corpus retrieval-quality sanity for the UMBRELA judge (bead xz4.1.32 leg 3).

Runs our own ``hybrid_search`` on a stratified slice of the entity-enrichment
gold queries (``data/eval/entity_value_props/*.yaml``), then asks the UMBRELA
judge to score papers at four rank positions:

  - rank 1   : top hit for the query
  - rank 3   : adjacent high-rank
  - rank 10  : borderline rank
  - random   : uniformly-sampled corpus paper (should be irrelevant)

If the UMBRELA judge tracks retrieval quality (not just lexical overlap), the
mean UMBRELA score should decline monotonically from rank 1 -> 3 -> 10 -> random.
Non-monotonic means either the retrieval is very noisy or the judge is reading
the wrong signal; either way it's evidence we need before trusting the
published 0.92/3.0 value-props baseline.

Metrics:
  - Mean UMBRELA score per rank position
  - Monotonicity check: mean[r1] > mean[r3] > mean[r10] > mean[random]
  - Spearman rho between (rank_numeric, umbrela_score)
    (rank_numeric = 1, 3, 10, 100 for random)

Success criteria:
  - All three monotonic inequalities hold (with a small tolerance)
  - Spearman rho <= -0.4 (negative, since higher rank = lower relevance)

Usage::

    python scripts/eval_umbrela_retrieval_sanity.py

    # fewer queries per lane
    python scripts/eval_umbrela_retrieval_sanity.py --queries-per-lane 2

    # smoke
    python scripts/eval_umbrela_retrieval_sanity.py --stub --queries-per-lane 2
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
from typing import Protocol

import yaml

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
    build_snippet,
    spearman_rho,
)

logger = logging.getLogger(__name__)

VALUE_PROPS_DIR = REPO_ROOT / "data" / "eval" / "entity_value_props"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "retrieval_sanity.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "retrieval_sanity.json"

RANK_SLOTS: tuple[int, ...] = (1, 3, 10)  # 0-indexed internally; 1-indexed in output
RANDOM_SENTINEL = 100  # used as rank for Spearman computation
MONOTONIC_TOLERANCE = 0.10  # 0.1 point of slack (0-3 scale) between adjacent slots
RHO_SUCCESS = -0.4


@dataclass(frozen=True)
class LaneQuery:
    lane: str
    query_id: str
    query: str


@dataclass(frozen=True)
class ScoredPair:
    lane: str
    query_id: str
    query: str
    bibcode: str
    rank_slot: str  # "r1", "r3", "r10", or "random"
    rank_numeric: int  # 1, 3, 10, or RANDOM_SENTINEL


class Searcher(Protocol):
    def topk(self, query: str, k: int) -> list[str]: ...
    def random_bibcode(self) -> str: ...
    def snippet_for(self, bibcode: str) -> str | None: ...


def load_queries(value_props_dir: Path, *, queries_per_lane: int, seed: int) -> list[LaneQuery]:
    rng = random.Random(seed)
    out: list[LaneQuery] = []
    for yml in sorted(value_props_dir.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text())
        lane = data.get("prop", yml.stem)
        qs = data.get("queries", [])
        rng.shuffle(qs)
        for q in qs[:queries_per_lane]:
            out.append(
                LaneQuery(
                    lane=str(lane),
                    query_id=str(q["id"]),
                    query=str(q["query"]),
                )
            )
    return out


class HybridSearcher:
    """Wraps ``scix.search.hybrid_search`` + ``papers`` table lookups.

    When ``bm25_only`` is True, skips the dense-vector leg and runs
    lexical-only retrieval (``query_embedding=None``). Used when the local
    DB is missing halfvec shadow columns (pre-migration-053 state) or when
    testing the judge against lexical-only rankings specifically.
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        model_name: str = "indus",
        bm25_only: bool = False,
    ) -> None:
        self._dsn = dsn
        self._model_name = model_name
        self._bm25_only = bm25_only
        self._model = None
        self._tokenizer = None
        self._all_bibcodes: list[str] | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from scix.embed import load_model

        self._model, self._tokenizer = load_model(self._model_name, device="cpu")

    def _embed(self, query: str) -> list[float]:
        from scix.embed import embed_batch

        self._ensure_model()
        pooling = "mean" if self._model_name == "indus" else "cls"
        return list(embed_batch(self._model, self._tokenizer, [query], pooling=pooling)[0])

    def topk(self, query: str, k: int) -> list[str]:
        from scix.db import get_connection
        from scix.search import hybrid_search

        vec = None if self._bm25_only else self._embed(query)
        with get_connection(dsn=self._dsn) as conn:
            result = hybrid_search(
                conn, query_text=query, query_embedding=vec, model_name=self._model_name, top_n=k
            )
        return [p["bibcode"] for p in result.papers[:k]]

    def random_bibcode(self) -> str:
        """Pick a uniform-random bibcode from papers table (cached)."""
        from scix.db import get_connection

        if self._all_bibcodes is None:
            with get_connection(dsn=self._dsn) as conn, conn.cursor() as cur:
                # Sample cheap: TABLESAMPLE SYSTEM on the papers table. 10k rows
                # is enough for uniformly-random sampling of negatives.
                cur.execute("SELECT bibcode FROM papers TABLESAMPLE SYSTEM (0.05) LIMIT 10000")
                self._all_bibcodes = [r[0] for r in cur.fetchall()]
        return random.choice(self._all_bibcodes)

    def snippet_for(self, bibcode: str) -> str | None:
        from scix.db import get_connection

        with get_connection(dsn=self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT title, abstract, body FROM papers WHERE bibcode = %s", (bibcode,)
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            return build_snippet(title=row[0], abstract=row[1], body=row[2])


@dataclass
class StubSearcher:
    """Deterministic stub: returns synthetic bibcodes for rank slots + negatives."""

    def topk(self, query: str, k: int) -> list[str]:
        return [f"STUB_{hash(query) % 100000}_{i}" for i in range(k)]

    def random_bibcode(self) -> str:
        return f"STUB_RANDOM_{random.randint(0, 100000)}"

    def snippet_for(self, bibcode: str) -> str | None:
        return f"Title: Stub passage for {bibcode}\n\nAbstract: Synthetic content."


def build_pairs(
    *, lane_queries: list[LaneQuery], searcher: Searcher, seed: int
) -> tuple[list[ScoredPair], dict[tuple[str, str], str]]:
    """Return pair list + {(qid, bibcode): snippet}."""
    rng = random.Random(seed)
    pairs: list[ScoredPair] = []
    snippets: dict[tuple[str, str], str] = {}

    for lq in lane_queries:
        top10 = searcher.topk(lq.query, 10)
        # Grab the three rank slots (1-indexed).
        for rank in RANK_SLOTS:
            if len(top10) < rank:
                logger.warning(
                    "query %s only returned %d results; skipping rank %d",
                    lq.query_id, len(top10), rank,
                )
                continue
            bib = top10[rank - 1]
            snip = searcher.snippet_for(bib)
            if not snip:
                logger.warning("no snippet for %s at rank %d of %s", bib, rank, lq.query_id)
                continue
            pairs.append(
                ScoredPair(
                    lane=lq.lane,
                    query_id=lq.query_id,
                    query=lq.query,
                    bibcode=bib,
                    rank_slot=f"r{rank}",
                    rank_numeric=rank,
                )
            )
            snippets[(lq.query_id, bib)] = snip

        # Random-corpus negative: retry up to 10 times to find one with snippet.
        for _ in range(10):
            bib = searcher.random_bibcode()
            if bib in top10:
                continue  # don't use a top-10 result as a random negative
            snip = searcher.snippet_for(bib)
            if snip:
                pairs.append(
                    ScoredPair(
                        lane=lq.lane,
                        query_id=lq.query_id,
                        query=lq.query,
                        bibcode=bib,
                        rank_slot="random",
                        rank_numeric=RANDOM_SENTINEL,
                    )
                )
                snippets[(lq.query_id, bib)] = snip
                break
    _ = rng  # seeded via searcher.random_bibcode already
    return pairs, snippets


async def run(
    *,
    pairs: list[ScoredPair],
    snippets: dict[tuple[str, str], str],
    dispatcher: Dispatcher,
    max_concurrency: int,
) -> tuple[list[tuple[ScoredPair, int, str]], dict]:
    triples = [
        JudgeTriple(
            query=p.query,
            bibcode=p.bibcode,
            snippet=snippets[(p.query_id, p.bibcode)],
        )
        for p in pairs
    ]
    judge = PersonaJudge(dispatcher=dispatcher, max_concurrency=max_concurrency)
    scores = await judge.run(triples)

    per_row: list[tuple[ScoredPair, int, str]] = []
    ranks: list[int] = []
    ums: list[int] = []
    failed = 0
    for p, s in zip(pairs, scores):
        per_row.append((p, s.score, s.reason))
        if s.score < 0:
            failed += 1
            continue
        ranks.append(p.rank_numeric)
        ums.append(s.score)

    if not ums:
        raise RuntimeError("all pairs failed")

    # Mean per slot
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for slot, rank_numeric in (("r1", 1), ("r3", 3), ("r10", 10), ("random", RANDOM_SENTINEL)):
        vals = [u for r, u in zip(ranks, ums) if r == rank_numeric]
        counts[slot] = len(vals)
        means[slot] = sum(vals) / len(vals) if vals else float("nan")

    # Monotonicity checks
    mono_r1_r3 = means["r1"] >= means["r3"] - MONOTONIC_TOLERANCE
    mono_r3_r10 = means["r3"] >= means["r10"] - MONOTONIC_TOLERANCE
    mono_r10_random = means["r10"] >= means["random"] - MONOTONIC_TOLERANCE
    all_monotonic = mono_r1_r3 and mono_r3_r10 and mono_r10_random

    rho = spearman_rho([float(r) for r in ranks], [float(u) for u in ums])

    metrics: dict = {
        "n_scored": len(ums),
        "n_failed": failed,
        "mean_by_slot": means,
        "count_by_slot": counts,
        "monotonic_r1_r3": mono_r1_r3,
        "monotonic_r3_r10": mono_r3_r10,
        "monotonic_r10_random": mono_r10_random,
        "all_monotonic": all_monotonic,
        "spearman_rho_rank_vs_score": rho,
        "passes_rho": rho <= RHO_SUCCESS,
        "passes": all_monotonic and rho <= RHO_SUCCESS,
    }
    return per_row, metrics


def write_csv(path: Path, rows: list[tuple[ScoredPair, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["lane", "query_id", "query", "rank_slot", "bibcode", "umbrela_score", "reason"]
        )
        for p, u, r in rows:
            w.writerow(
                [p.lane, p.query_id, p.query, p.rank_slot, p.bibcode, u, (r or "").replace("\n", " ")[:400]]
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries-per-lane", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument("--claude-binary", default="claude")
    p.add_argument("--dsn", default=None)
    p.add_argument("--model-name", default="indus")
    p.add_argument(
        "--bm25-only",
        action="store_true",
        help="Skip dense-vector leg (use when paper_embeddings.embedding_hv is "
        "missing, i.e. pre-migration-053 DB state).",
    )
    p.add_argument("--stub", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    random.seed(args.seed)

    lane_queries = load_queries(VALUE_PROPS_DIR, queries_per_lane=args.queries_per_lane, seed=args.seed)
    logger.info(
        "%d lane-queries across %d lanes", len(lane_queries), len({q.lane for q in lane_queries})
    )

    searcher: Searcher
    if args.stub:
        searcher = StubSearcher()
    else:
        searcher = HybridSearcher(
            dsn=args.dsn, model_name=args.model_name, bm25_only=args.bm25_only
        )

    pairs, snippets = build_pairs(lane_queries=lane_queries, searcher=searcher, seed=args.seed)
    logger.info("built %d pairs (4 slots * %d queries)", len(pairs), len(lane_queries))

    dispatcher: Dispatcher
    if args.stub:
        dispatcher = StubDispatcher(fixed_score=2, reason="stub")
    else:
        dispatcher = ClaudeSubprocessDispatcher(claude_binary=args.claude_binary)

    rows, metrics = asyncio.run(
        run(pairs=pairs, snippets=snippets, dispatcher=dispatcher, max_concurrency=args.concurrency)
    )

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nOwn-corpus retrieval-quality sanity")
    print(f"  n_scored / n_failed : {metrics['n_scored']} / {metrics['n_failed']}")
    print(f"  mean UMBRELA per slot:")
    for slot in ("r1", "r3", "r10", "random"):
        print(f"    {slot:7s}: {metrics['mean_by_slot'][slot]:+.3f} (n={metrics['count_by_slot'][slot]})")
    print(f"  monotonic r1>=r3>=r10>=random : {metrics['all_monotonic']}")
    print(f"    r1 >= r3    : {metrics['monotonic_r1_r3']}")
    print(f"    r3 >= r10   : {metrics['monotonic_r3_r10']}")
    print(f"    r10 >= rand : {metrics['monotonic_r10_random']}")
    print(f"  Spearman rho  : {metrics['spearman_rho_rank_vs_score']:+.3f}  "
          f"(<= {RHO_SUCCESS}; passes={metrics['passes_rho']})")
    print(f"\n  csv : {args.output_csv}")
    print(f"  json: {args.output_json}")
    return 0 if metrics["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
