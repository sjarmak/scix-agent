#!/usr/bin/env python3
"""Generate the calibration seed CSV with UMBRELA draft scores.

Pipeline for bead xz4.1.28.1:

1. Load the query seed YAML (default: src/scix/eval/prompts/calibration_queries.yaml).
2. For each query, embed it with INDUS and run hybrid_search against the
   configured DSN (read-only) to pull top-K candidate bibcodes.
3. Build a title+abstract+body snippet for each candidate.
4. Dispatch the umbrela_judge Claude Code subagent to score each pair
   using the verbatim UMBRELA rubric. Scores carry a self-reported
   needs_human_review flag.
5. Write data/eval/calibration_seed_draft.csv atomically (tmp + rename).

Output columns (NOT human labels — see --help for why):
    query_id, lane, query, bibcode, title,
    draft_score, needs_human_review, snippet_preview

The script reads production data read-only; it writes no DB rows. The
draft CSV is a *starting point* for Stephanie to correct into human_score
via the follow-up bead — do not use it as ground truth for kappa
calibration directly.

Usage::

    # full 50-query run (calls Claude for every pair)
    python scripts/generate_calibration_seed.py

    # small smoke run (5 queries, stubbed scorer — no Claude calls)
    python scripts/generate_calibration_seed.py --max-queries 5 --stub

    # custom output path
    python scripts/generate_calibration_seed.py --output /tmp/seed.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
)

logger = logging.getLogger(__name__)


DEFAULT_QUERIES_YAML = REPO_ROOT / "src" / "scix" / "eval" / "prompts" / "calibration_queries.yaml"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "eval" / "calibration_seed_draft.csv"
SNIPPET_PREVIEW_CHARS = 240

CSV_COLUMNS: tuple[str, ...] = (
    "query_id",
    "lane",
    "query",
    "bibcode",
    "title",
    "draft_score",
    "needs_human_review",
    "snippet_preview",
)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySpec:
    id: str
    lane: str
    query: str


@dataclass(frozen=True)
class Candidate:
    bibcode: str
    title: str
    snippet: str


@dataclass(frozen=True)
class SeedRow:
    query_id: str
    lane: str
    query: str
    bibcode: str
    title: str
    draft_score: int
    needs_human_review: bool | None
    snippet_preview: str


@dataclass
class GenerationStats:
    n_queries: int = 0
    n_candidates: int = 0
    n_scored: int = 0
    n_failed: int = 0
    lanes: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_query_specs(path: Path) -> tuple[list[QuerySpec], int]:
    """Load query specs and default top-K from the YAML seed file."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(f"{path}: invalid YAML; expected a 'queries' list")

    top_k = int(data.get("default_top_k", 5))
    if top_k < 1:
        raise ValueError(f"default_top_k must be >= 1, got {top_k}")

    specs: list[QuerySpec] = []
    seen_ids: set[str] = set()
    for raw in data["queries"]:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: query entry is not a mapping: {raw!r}")
        for required in ("id", "lane", "query"):
            if required not in raw:
                raise ValueError(f"{path}: query missing {required!r}: {raw}")
        qid = str(raw["id"])
        if qid in seen_ids:
            raise ValueError(f"{path}: duplicate query id {qid!r}")
        seen_ids.add(qid)
        specs.append(QuerySpec(id=qid, lane=str(raw["lane"]), query=str(raw["query"])))
    return specs, top_k


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------


class CandidateSource:
    """Retrieves top-K candidate (bibcode, title, snippet) for a query.

    The real implementation uses ``scix.search.hybrid_search`` against the
    configured DSN. Tests inject a stub via :class:`StubCandidateSource`.
    """

    def __init__(self, *, dsn: str | None = None, model_name: str = "indus") -> None:
        self._dsn = dsn
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from scix.embed import load_model

        self._model, self._tokenizer = load_model(self._model_name, device="cpu")

    def _embed_query(self, query: str) -> list[float]:
        from scix.embed import embed_batch

        self._ensure_model()
        # embed_batch signature is (model, tokenizer, texts, ..., pooling).
        # INDUS trains with mean pooling; SPECTER2 uses CLS.
        pooling = "mean" if self._model_name == "indus" else "cls"
        vectors = embed_batch(self._model, self._tokenizer, [query], pooling=pooling)
        return list(vectors[0])

    def fetch(self, spec: QuerySpec, top_k: int) -> list[Candidate]:
        """Run hybrid_search and pull title/abstract/body for the top-K hits."""
        from scix.db import get_connection
        from scix.search import hybrid_search

        vec = self._embed_query(spec.query)
        with get_connection(dsn=self._dsn) as conn:
            result = hybrid_search(
                conn,
                query_text=spec.query,
                query_embedding=vec,
                model_name=self._model_name,
                top_n=top_k,
            )
            bibcodes = [p["bibcode"] for p in result.papers[:top_k]]
            if not bibcodes:
                return []

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT bibcode, title, abstract, body " "FROM papers WHERE bibcode = ANY(%s)",
                    (bibcodes,),
                )
                rows = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}

        candidates: list[Candidate] = []
        for bib in bibcodes:
            if bib not in rows:
                logger.warning("bibcode %s returned by search but missing from papers", bib)
                continue
            title, abstract, body = rows[bib]
            if not title:
                continue
            snippet = build_snippet(title=title, abstract=abstract, body=body)
            candidates.append(Candidate(bibcode=bib, title=title, snippet=snippet))
        return candidates


@dataclass
class StubCandidateSource:
    """Deterministic in-memory candidate source for tests and smoke runs."""

    per_query: int = 3

    def fetch(self, spec: QuerySpec, top_k: int) -> list[Candidate]:
        n = min(self.per_query, top_k)
        return [
            Candidate(
                bibcode=f"STUB{spec.id}.{i}",
                title=f"Stub paper {i} for {spec.id}",
                snippet=(
                    f"Title: Stub paper {i} for {spec.id}\n\n"
                    f"Abstract: Synthetic abstract for {spec.query!r}."
                ),
            )
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


async def generate_seed_rows(
    *,
    specs: list[QuerySpec],
    top_k: int,
    candidate_source: Any,
    dispatcher: Dispatcher,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> tuple[list[SeedRow], GenerationStats]:
    """Score every (query, candidate) pair and return rows + stats.

    Rows with dispatcher failures are skipped; they're counted in
    ``stats.n_failed`` so the caller can surface the miss-rate.
    """
    triples: list[JudgeTriple] = []
    metadata: list[tuple[QuerySpec, Candidate]] = []
    stats = GenerationStats()

    for spec in specs:
        stats.n_queries += 1
        stats.lanes[spec.lane] = stats.lanes.get(spec.lane, 0) + 1
        candidates = candidate_source.fetch(spec, top_k)
        stats.n_candidates += len(candidates)
        for cand in candidates:
            triples.append(
                JudgeTriple(query=spec.query, bibcode=cand.bibcode, snippet=cand.snippet)
            )
            metadata.append((spec, cand))

    if not triples:
        return [], stats

    judge = PersonaJudge(dispatcher=dispatcher, max_concurrency=max_concurrency)
    scores = await judge.run(triples)

    rows: list[SeedRow] = []
    for (spec, cand), score in zip(metadata, scores):
        if score.score < 0:
            stats.n_failed += 1
            continue
        stats.n_scored += 1
        rows.append(
            SeedRow(
                query_id=spec.id,
                lane=spec.lane,
                query=spec.query,
                bibcode=cand.bibcode,
                title=cand.title,
                draft_score=score.score,
                needs_human_review=score.needs_human_review,
                snippet_preview=cand.snippet[:SNIPPET_PREVIEW_CHARS].replace("\n", " "),
            )
        )
    return rows, stats


# ---------------------------------------------------------------------------
# Atomic CSV write
# ---------------------------------------------------------------------------


def write_seed_csv(path: Path, rows: list[SeedRow]) -> None:
    """Write ``rows`` to ``path`` atomically (tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(CSV_COLUMNS),
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "query_id": row.query_id,
                        "lane": row.lane,
                        "query": row.query,
                        "bibcode": row.bibcode,
                        "title": row.title,
                        "draft_score": row.draft_score,
                        "needs_human_review": (
                            ""
                            if row.needs_human_review is None
                            else "true" if row.needs_human_review else "false"
                        ),
                        "snippet_preview": row.snippet_preview,
                    }
                )
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_YAML)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit to the first N queries (for smoke runs).",
    )
    p.add_argument("--top-k", type=int, default=None, help="Override default_top_k from YAML.")
    p.add_argument("--dsn", default=None, help="PostgreSQL DSN (defaults to scix.db).")
    p.add_argument("--model-name", default="indus")
    p.add_argument("--claude-binary", default="claude")
    p.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument(
        "--stub",
        action="store_true",
        help="Use StubDispatcher + StubCandidateSource — skips Claude and DB.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    specs, default_top_k = load_query_specs(args.queries)
    top_k = args.top_k if args.top_k is not None else default_top_k
    if args.max_queries is not None:
        specs = specs[: args.max_queries]

    if args.stub:
        candidate_source: Any = StubCandidateSource()
        dispatcher: Dispatcher = StubDispatcher(fixed_score=2, reason="stub")
    else:
        candidate_source = CandidateSource(dsn=args.dsn, model_name=args.model_name)
        dispatcher = ClaudeSubprocessDispatcher(claude_binary=args.claude_binary)

    logger.info(
        "generating seed: %d queries, top_k=%d, stub=%s, output=%s",
        len(specs),
        top_k,
        args.stub,
        args.output,
    )

    rows, stats = asyncio.run(
        generate_seed_rows(
            specs=specs,
            top_k=top_k,
            candidate_source=candidate_source,
            dispatcher=dispatcher,
            max_concurrency=args.concurrency,
        )
    )

    write_seed_csv(args.output, rows)

    print("Calibration seed generation complete")
    print(f"  queries     : {stats.n_queries}")
    print(f"  candidates  : {stats.n_candidates}")
    print(f"  scored      : {stats.n_scored}")
    print(f"  failed      : {stats.n_failed}")
    print(f"  by lane     : {stats.lanes}")
    print(f"  output      : {args.output}")
    if stats.n_failed:
        print(f"  WARNING: {stats.n_failed} rows failed scoring and were dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
