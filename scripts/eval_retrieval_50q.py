#!/usr/bin/env python3
"""50-query retrieval evaluation against a hand-curated gold set.

Drives three retrieval modes against ``eval/retrieval_50q.jsonl``:

1. ``baseline``  — ``scix.search.hybrid_search`` (INDUS dense + title/abstract
   BM25 + body BM25) fused via Reciprocal Rank Fusion (k=60).
2. ``section``   — the new ``section_retrieval`` MCP tool (nomic dense over
   ``section_embeddings`` + ``papers_fulltext.sections_tsv`` BM25, fused via
   RRF k=60).
3. ``fused``     — ``baseline`` and ``section`` ranked lists fused again via
   RRF k=60. Re-uses the per-query baseline ranking from mode (1) so we
   don't pay the cost twice.

Metrics per query: nDCG@10, MRR@10, Recall@50. Each is averaged overall and
per ``bucket`` (``title_matchable``, ``concept``, ``method``, ``author_specific``).
Queries with empty ``gold_bibcodes`` are excluded from each metric average.

Local-only: query encoding for both modes uses local open-weight models
(NASA INDUS for baseline, nomic-embed-text-v1.5 for section). No paid API
SDKs are imported anywhere in this script — see project memory note
``feedback_no_paid_apis``.

Upstream-blocked corpus encode
------------------------------
The full-corpus section embedding pass is blocked on the parser PRD shipping.
Until then, ``--dry-run`` skips DB and model entirely and writes a fixed-shape
stub JSON so downstream tooling can wire against the schema. When run live
against an empty ``section_embeddings`` table, ``section`` and ``fused`` modes
emit a zero-stub with ``skipped_reason = "section_embeddings_empty"`` and
``baseline`` continues normally.

Usage
-----
::

    # Schema-only (no DB, no model)
    python scripts/eval_retrieval_50q.py --dry-run

    # Live, all three modes (default)
    python scripts/eval_retrieval_50q.py

    # Live, baseline only — useful while section_embeddings is being populated
    python scripts/eval_retrieval_50q.py --modes baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Make ``src`` importable when invoked from a checkout root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


logger = logging.getLogger("eval_retrieval_50q")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RRF_K: int = 60
DEFAULT_QUERIES: str = "eval/retrieval_50q.jsonl"
DEFAULT_OUTPUT: str = "docs/eval/retrieval_50q_2026-04.json"
DEFAULT_MODES: tuple[str, ...] = ("baseline", "section", "fused")
DEFAULT_K: int = 10
RECALL_K: int = 50
BUCKETS: tuple[str, ...] = ("title_matchable", "concept", "method", "author_specific")
VALID_MODES: tuple[str, ...] = ("baseline", "section", "fused")


# ---------------------------------------------------------------------------
# Pure metric primitives — no DB, no model, easily testable.
# ``None`` return signals "exclude this query from the average".
# ---------------------------------------------------------------------------


def ndcg_at_10(retrieved: Sequence[str], gold: Sequence[str]) -> float | None:
    """Binary-relevance nDCG@10 with log2(rank+1) discount.

    Returns ``None`` when ``gold`` is empty (the spec excludes those queries
    from the per-mode average rather than scoring them as zero).
    """
    if not gold:
        return None
    gold_set = set(gold)
    k = 10
    # Actual DCG using rank+1 = i+2 (so the rank-1 doc contributes 1/log2(2)).
    dcg = 0.0
    for i, bib in enumerate(retrieved[:k]):
        if bib in gold_set:
            dcg += 1.0 / math.log2(i + 2)
    # Ideal DCG: as many 1's at the top as we have gold items, but no more than k.
    n_ideal = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr_at_10(retrieved: Sequence[str], gold: Sequence[str]) -> float | None:
    """Mean Reciprocal Rank at 10. ``None`` when ``gold`` is empty."""
    if not gold:
        return None
    gold_set = set(gold)
    for i, bib in enumerate(retrieved[:10]):
        if bib in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    """Recall@k: |gold ∩ top-k| / |gold|. ``None`` when ``gold`` is empty."""
    if not gold:
        return None
    gold_set = set(gold)
    found = sum(1 for bib in retrieved[:k] if bib in gold_set)
    return found / len(gold_set)


def score_query(retrieved: Sequence[str], gold: Sequence[str], k: int) -> dict[str, float | None]:
    """All three metrics for one query.

    ``k`` is the nDCG/MRR cutoff (always 10 in the spec, kept as a parameter
    so tests can sanity-check non-default values).
    """
    return {
        "ndcg_at_10": ndcg_at_10(retrieved, gold) if k == 10 else _ndcg_at_k(retrieved, gold, k),
        "mrr_at_10": mrr_at_10(retrieved, gold) if k == 10 else _mrr_at_k(retrieved, gold, k),
        "recall_at_50": recall_at_k(retrieved, gold, RECALL_K),
    }


def _ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    """Generalized nDCG@k for tests that probe non-default cutoffs."""
    if not gold:
        return None
    gold_set = set(gold)
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, bib in enumerate(retrieved[:k]) if bib in gold_set
    )
    n_ideal = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return 0.0 if idcg == 0.0 else dcg / idcg


def _mrr_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    if not gold:
        return None
    gold_set = set(gold)
    for i, bib in enumerate(retrieved[:k]):
        if bib in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_metrics(per_query: list[dict[str, float | None]]) -> dict[str, Any]:
    """Average metrics across a list of per-query results, ignoring ``None``.

    Returns a dict with the three metric means plus diagnostics on how many
    queries actually contributed (so an empty bucket shows ``n_scored=0``
    rather than NaN).
    """
    metric_names = ("ndcg_at_10", "mrr_at_10", "recall_at_50")
    sums: dict[str, float] = {m: 0.0 for m in metric_names}
    counts: dict[str, int] = {m: 0 for m in metric_names}
    for row in per_query:
        for m in metric_names:
            v = row.get(m)
            if v is None:
                continue
            sums[m] += float(v)
            counts[m] += 1
    out: dict[str, Any] = {}
    for m in metric_names:
        out[m] = (sums[m] / counts[m]) if counts[m] > 0 else 0.0
    out["n_queries"] = len(per_query)
    out["n_scored_ndcg"] = counts["ndcg_at_10"]
    out["n_scored_mrr"] = counts["mrr_at_10"]
    out["n_scored_recall"] = counts["recall_at_50"]
    return out


# ---------------------------------------------------------------------------
# RRF over bibcode rankings
# ---------------------------------------------------------------------------


def rrf_fuse_bibcodes(rankings: Sequence[Sequence[str]], k_rrf: int = RRF_K) -> list[str]:
    """Fuse multiple bibcode rankings with Reciprocal Rank Fusion.

    score(d) = sum_i 1 / (k_rrf + rank_i(d))   (rank starts at 1)

    Stable secondary sort on bibcode keeps the output deterministic when
    two documents tie, which matters for repeatable eval numbers.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, bib in enumerate(ranking, start=1):
            scores[bib] += 1.0 / (k_rrf + rank)
    # Sort by descending score, then ascending bibcode for tiebreaks.
    return [bib for bib, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    query: str
    bucket: str
    discipline: str
    gold_bibcodes: tuple[str, ...]
    notes: str = ""


def load_queries(path: Path) -> list[EvalQuery]:
    """Read JSONL gold set. Skips blank/comment lines.

    Schema-validates each row against the four required fields; missing
    ``notes`` defaults to empty string.
    """
    if not path.exists():
        raise FileNotFoundError(f"queries file not found: {path}")
    out: list[EvalQuery] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
        for field in ("query", "bucket", "discipline", "gold_bibcodes"):
            if field not in row:
                raise ValueError(f"line {line_no}: missing required field {field!r}")
        if not isinstance(row["gold_bibcodes"], list):
            raise ValueError(f"line {line_no}: gold_bibcodes must be a list")
        out.append(
            EvalQuery(
                query=str(row["query"]),
                bucket=str(row["bucket"]),
                discipline=str(row["discipline"]),
                gold_bibcodes=tuple(str(b) for b in row["gold_bibcodes"]),
                notes=str(row.get("notes", "")),
            )
        )
    return out


def write_output(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Retrieval drivers — imports are kept local so --dry-run avoids torch.
# ---------------------------------------------------------------------------


_indus_state: dict[str, Any] = {"loaded": False, "model": None, "tokenizer": None}


def _indus_encode(query: str) -> list[float]:
    """Encode a query string with the local INDUS model. Lazy-loads on first call."""
    if not _indus_state["loaded"]:
        from scix.embed import embed_batch, load_model

        model, tokenizer = load_model("indus", device="auto")
        _indus_state.update(loaded=True, model=model, tokenizer=tokenizer, embed_batch=embed_batch)
        logger.info("Loaded INDUS model for baseline query encoding")
    embed_batch = _indus_state["embed_batch"]
    vectors = embed_batch(
        _indus_state["model"], _indus_state["tokenizer"], [query], pooling="mean"
    )
    if not vectors:
        raise RuntimeError("INDUS embed_batch returned no vectors")
    return vectors[0]


def baseline_search(conn: Any, query: str, top_n: int = RECALL_K) -> list[str]:
    """Run the baseline hybrid_search (INDUS dense + title/abstract BM25 + body BM25)."""
    from scix.search import hybrid_search

    vec = _indus_encode(query)
    result = hybrid_search(
        conn,
        query_text=query,
        query_embedding=vec,
        model_name="indus",
        top_n=top_n,
        rrf_k=RRF_K,
        include_body=True,
    )
    return _dedupe_preserving_order(p["bibcode"] for p in result.papers if "bibcode" in p)


def section_search(conn: Any, query: str, k: int = RECALL_K) -> list[str]:
    """Call the section_retrieval MCP handler directly and dedupe by bibcode.

    Multiple sections from the same paper collapse to one bibcode at the
    earliest hit, which matches how baseline returns one row per paper.
    """
    from scix.mcp_server import _handle_section_retrieval

    raw = _handle_section_retrieval(conn, {"query": query, "k": k})
    payload = json.loads(raw)
    if "error" in payload:
        raise RuntimeError(f"section_retrieval error: {payload}")
    bibs = (item.get("bibcode") for item in payload.get("results", []))
    return _dedupe_preserving_order(bibs)


def _dedupe_preserving_order(items: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x is None or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def section_embeddings_available(conn: Any) -> bool:
    """Probe ``section_embeddings`` for at least one row.

    Used to short-circuit ``section`` and ``fused`` modes when the upstream
    encode pipeline hasn't run yet.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM section_embeddings LIMIT 1")
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def _empty_metrics_block(n_queries: int) -> dict[str, Any]:
    return {
        "ndcg_at_10": 0.0,
        "mrr_at_10": 0.0,
        "recall_at_50": 0.0,
        "n_queries": n_queries,
        "n_scored_ndcg": 0,
        "n_scored_mrr": 0,
        "n_scored_recall": 0,
    }


def _empty_mode_block(queries: list[EvalQuery], skipped_reason: str | None = None) -> dict[str, Any]:
    by_bucket = {b: _empty_metrics_block(sum(1 for q in queries if q.bucket == b)) for b in BUCKETS}
    overall = _empty_metrics_block(len(queries))
    if skipped_reason:
        overall["skipped_reason"] = skipped_reason
    return {"overall": overall, "by_bucket": by_bucket}


def _aggregate_mode(
    per_query: list[tuple[EvalQuery, dict[str, float | None]]],
) -> dict[str, Any]:
    overall = aggregate_metrics([row for _, row in per_query])
    by_bucket: dict[str, Any] = {}
    for bucket in BUCKETS:
        rows = [row for q, row in per_query if q.bucket == bucket]
        by_bucket[bucket] = aggregate_metrics(rows)
    return {"overall": overall, "by_bucket": by_bucket}


def run_mode(
    mode: str,
    queries: list[EvalQuery],
    conn: Any | None,
    k: int,
    *,
    dry_run: bool,
    section_available: bool,
    baseline_cache: dict[str, list[str]] | None = None,
    section_cache: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Run one retrieval mode across all queries and aggregate metrics.

    ``baseline_cache`` and ``section_cache`` are passed through so the
    ``fused`` mode can reuse the rankings already computed for the two
    constituent modes — running a 4th encode pass would double cost without
    changing scores.
    """
    if dry_run:
        return _empty_mode_block(queries, skipped_reason="dry_run")

    if mode in ("section", "fused") and not section_available:
        logger.info("section_embeddings empty — skipping mode=%s with zero stub", mode)
        return _empty_mode_block(queries, skipped_reason="section_embeddings_empty")

    if conn is None:
        # Defensive — should be unreachable when not dry_run.
        return _empty_mode_block(queries, skipped_reason="no_db_connection")

    per_query: list[tuple[EvalQuery, dict[str, float | None]]] = []
    for q in queries:
        try:
            if mode == "baseline":
                retrieved = (
                    baseline_cache[q.query]
                    if baseline_cache is not None and q.query in baseline_cache
                    else baseline_search(conn, q.query, top_n=RECALL_K)
                )
                if baseline_cache is not None:
                    baseline_cache[q.query] = retrieved
            elif mode == "section":
                retrieved = (
                    section_cache[q.query]
                    if section_cache is not None and q.query in section_cache
                    else section_search(conn, q.query, k=RECALL_K)
                )
                if section_cache is not None:
                    section_cache[q.query] = retrieved
            elif mode == "fused":
                base = (
                    baseline_cache.get(q.query)
                    if baseline_cache is not None
                    else None
                )
                if base is None:
                    base = baseline_search(conn, q.query, top_n=RECALL_K)
                    if baseline_cache is not None:
                        baseline_cache[q.query] = base
                sect = (
                    section_cache.get(q.query)
                    if section_cache is not None
                    else None
                )
                if sect is None:
                    sect = section_search(conn, q.query, k=RECALL_K)
                    if section_cache is not None:
                        section_cache[q.query] = sect
                retrieved = rrf_fuse_bibcodes([base, sect], k_rrf=RRF_K)
            else:
                raise ValueError(f"unknown mode {mode!r}")
        except Exception:
            logger.exception("query failed in mode=%s: %r — scoring as empty", mode, q.query)
            retrieved = []
        scores = score_query(retrieved, list(q.gold_bibcodes), k=k)
        per_query.append((q, scores))
    return _aggregate_mode(per_query)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_modes(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--modes requires at least one mode")
    bad = [p for p in parts if p not in VALID_MODES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown mode(s): {bad}. Valid modes: {VALID_MODES}"
        )
    # Preserve insertion order, dedupe.
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="50-query retrieval eval (baseline / section / fused)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=Path(DEFAULT_QUERIES),
        help="Path to JSONL gold set",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Path to write JSON results",
    )
    p.add_argument(
        "--modes",
        type=_parse_modes,
        default=list(DEFAULT_MODES),
        help="Comma-separated subset of {baseline,section,fused}",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Cutoff for nDCG/MRR (Recall is always at 50)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip DB and model entirely; emit a fixed-shape stub JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)

    # Load queries — needed even in --dry-run so the stub knows bucket sizes.
    if args.queries.exists():
        try:
            queries = load_queries(args.queries)
        except Exception:
            logger.exception("failed to load queries from %s", args.queries)
            queries = []
    else:
        if not args.dry_run:
            logger.error("queries file %s not found", args.queries)
            return 2
        logger.warning("queries file %s missing — emitting empty stub", args.queries)
        queries = []

    payload: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "k": int(args.k),
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "modes": {},
    }

    if args.dry_run:
        for mode in args.modes:
            payload["modes"][mode] = _empty_mode_block(queries, skipped_reason="dry_run")
        write_output(args.output, payload)
        logger.info("dry-run output written to %s", args.output)
        return 0

    # Live run — connect once, share connection across modes.
    try:
        from scix.db import get_connection
    except ImportError:
        logger.exception("scix.db is unavailable; cannot run live eval")
        return 1

    conn = None
    try:
        conn = get_connection()
        section_available = False
        if any(m in ("section", "fused") for m in args.modes):
            try:
                section_available = section_embeddings_available(conn)
                if not section_available:
                    logger.info(
                        "section_embeddings is empty; section/fused modes will emit zero-stubs."
                    )
            except Exception:
                logger.exception(
                    "section_embeddings probe failed; assuming unavailable"
                )
                section_available = False

        baseline_cache: dict[str, list[str]] = {}
        section_cache: dict[str, list[str]] = {}

        # Run baseline first if requested OR if fused requested (fused needs it).
        run_order = list(args.modes)
        if "fused" in run_order:
            # Ensure baseline+section run before fused so caches populate.
            if "baseline" not in run_order:
                run_order = ["baseline", *run_order]
            if "section" not in run_order:
                run_order.insert(run_order.index("fused"), "section")
            else:
                # Move 'fused' to the end.
                run_order = [m for m in run_order if m != "fused"] + ["fused"]

        # We compute results for every mode in run_order, but only emit those
        # the user actually asked for in args.modes.
        results: dict[str, dict[str, Any]] = {}
        for mode in run_order:
            results[mode] = run_mode(
                mode,
                queries,
                conn,
                k=args.k,
                dry_run=False,
                section_available=section_available,
                baseline_cache=baseline_cache,
                section_cache=section_cache,
            )

        for mode in args.modes:
            payload["modes"][mode] = results[mode]

        write_output(args.output, payload)
        logger.info("live eval output written to %s", args.output)
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
