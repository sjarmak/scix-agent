#!/usr/bin/env python3
"""Dispatch the single-persona relevance judge over a JSONL of triples.

Input JSONL rows: ``{"query": str, "bibcode": str, "paper_snippet": str}``.

Output JSONL rows:
    ``{"query": str, "bibcode": str, "score": int, "reason": str}``.

Dispatch uses ``claude -p`` as a subprocess (OAuth, no paid API). Parallel
with a bounded semaphore (default 4) plus exponential-backoff retry.

Usage::

    # Production run
    python scripts/judge_triples.py \\
        --in triples.jsonl \\
        --out scores.jsonl \\
        --concurrency 4

    # Stub run (no subagent calls — for wiring checks)
    python scripts/judge_triples.py \\
        --in triples.jsonl \\
        --out scores.jsonl \\
        --stub
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

# Ensure we can import scix when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.eval.persona_judge import (  # noqa: E402
    ClaudeSubprocessDispatcher,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    Dispatcher,
    JudgeScore,
    JudgeTriple,
    PersonaJudge,
    StubDispatcher,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS: tuple[str, ...] = ("query", "bibcode", "paper_snippet")


def read_triples(path: Path) -> list[JudgeTriple]:
    """Read a JSONL file of ``{query, bibcode, paper_snippet}`` triples.

    Blank lines are skipped. Missing fields raise :class:`ValueError`.
    """
    triples: list[JudgeTriple] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            for field in _REQUIRED_FIELDS:
                if field not in rec:
                    raise ValueError(f"{path}:{line_no}: missing required field {field!r}")
            triples.append(
                JudgeTriple(
                    query=str(rec["query"]),
                    bibcode=str(rec["bibcode"]),
                    snippet=str(rec["paper_snippet"]),
                )
            )
    return triples


def write_scores(path: Path, scores: Iterable[JudgeScore]) -> None:
    """Write JSONL ``{query, bibcode, score, reason}`` records to ``path``.

    Creates the parent directory if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for score in scores:
            if score.triple is None:
                raise ValueError(
                    "JudgeScore missing triple back-reference — "
                    "PersonaJudge.run should always attach it"
                )
            rec = {
                "query": score.triple.query,
                "bibcode": score.triple.bibcode,
                "score": score.score,
                "reason": score.reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run(
    *,
    input_path: Path,
    output_path: Path,
    dispatcher: Dispatcher,
    max_concurrency: int,
    max_retries: int,
) -> None:
    """Read triples, judge in parallel, write scores."""
    triples = read_triples(input_path)
    logger.info("read %d triples from %s", len(triples), input_path)

    judge = PersonaJudge(
        dispatcher=dispatcher,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
    )
    scores = asyncio.run(judge.run(triples))
    write_scores(output_path, scores)
    logger.info("wrote %d scores to %s", len(scores), output_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score (query, paper) triples via the in-domain researcher persona."
    )
    parser.add_argument(
        "--in", dest="input_path", type=Path, required=True,
        help="Input JSONL with {query, bibcode, paper_snippet} rows.",
    )
    parser.add_argument(
        "--out", dest="output_path", type=Path, required=True,
        help="Output JSONL with {query, bibcode, score, reason} rows.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
        help=f"Max parallel subagent calls (default {DEFAULT_MAX_CONCURRENCY}).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"Retry budget per triple (default {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Use the deterministic StubDispatcher (no subagent calls).",
    )
    parser.add_argument(
        "--claude-binary", default="claude",
        help="Path to the claude CLI (default 'claude').",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dispatcher: Dispatcher
    if args.stub:
        dispatcher = StubDispatcher(fixed_score=2, reason="stub")
    else:
        dispatcher = ClaudeSubprocessDispatcher(claude_binary=args.claude_binary)

    run(
        input_path=args.input_path,
        output_path=args.output_path,
        dispatcher=dispatcher,
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
