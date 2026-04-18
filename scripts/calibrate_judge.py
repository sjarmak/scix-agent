#!/usr/bin/env python3
"""Calibrate the in-domain-researcher judge against a human-labeled seed.

Input: CSV with columns ``query, bibcode, human_score`` (human_score 0-3).
The script pulls snippets for each bibcode from the ``papers`` table,
runs the persona judge, and reports:

    - Spearman rho (rank correlation, scale-free)
    - Quadratic-weighted Cohen's kappa (ordinal agreement)
    - ``trustworthy`` flag (kappa >= 0.6 — Landis-Koch "substantial")

Every calibration run appends a drift-watch entry to
``results/judge_calibration_log.jsonl`` so we can catch quality drift over
prompt revisions.

Usage::

    python scripts/calibrate_judge.py \\
        --seed seed.csv \\
        --log results/judge_calibration_log.jsonl \\
        --prompt-version in_domain_researcher-v1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.eval.persona_judge import (  # noqa: E402
    ClaudeSubprocessDispatcher,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PROMPT_VERSION,
    Dispatcher,
    JudgeTriple,
    PersonaJudge,
    StubDispatcher,
    build_snippet,
    quadratic_weighted_kappa,
    spearman_rho,
)

logger = logging.getLogger(__name__)


TRUSTWORTHY_KAPPA_THRESHOLD: float = 0.6
"""Landis-Koch: kappa >= 0.6 is 'substantial' agreement."""


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRow:
    query: str
    bibcode: str
    human_score: int


@dataclass(frozen=True)
class CalibrationReport:
    spearman: float
    kappa: float
    n_triples: int
    trustworthy: bool


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


_REQUIRED_COLS: tuple[str, ...] = ("query", "bibcode", "human_score")


def read_seed_csv(path: Path) -> list[SeedRow]:
    """Read a calibration seed CSV.

    Required columns: ``query, bibcode, human_score`` (scores 0-3).
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        missing = [c for c in _REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")

        rows: list[SeedRow] = []
        for line_no, raw in enumerate(reader, start=2):
            try:
                score = int(raw["human_score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: human_score must be int, got {raw['human_score']!r}"
                ) from exc
            if not (0 <= score <= 3):
                raise ValueError(f"{path}:{line_no}: human_score {score} out of range [0, 3]")
            rows.append(
                SeedRow(
                    query=str(raw["query"]),
                    bibcode=str(raw["bibcode"]),
                    human_score=score,
                )
            )
    return rows


def append_drift_entry(
    *,
    log_path: Path,
    prompt_version: str,
    kappa: float,
    spearman: float,
    n_triples: int,
) -> None:
    """Append one JSONL line to the drift-watch log.

    Creates parent directories if missing. Entries carry an ISO-8601 UTC
    timestamp so successive prompt revisions can be plotted over time.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "prompt_version": prompt_version,
        "kappa": round(kappa, 4),
        "spearman": round(spearman, 4),
        "n_triples": n_triples,
        "run_date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Snippet fetcher (DB-backed) and protocol
# ---------------------------------------------------------------------------


class SnippetFetcher(Protocol):
    def __call__(self, bibcodes: list[str]) -> dict[str, str]: ...


def fetch_snippets_from_db(bibcodes: list[str], *, dsn: str | None = None) -> dict[str, str]:
    """Fetch ``{bibcode: snippet}`` from the ``papers`` table.

    Uses :func:`build_snippet` with the default 500-char body budget.
    Returns an empty mapping when ``bibcodes`` is empty. Unknown bibcodes
    are omitted from the result (caller handles missing).

    Args:
        bibcodes: List of ADS bibcodes to look up.
        dsn: PostgreSQL DSN; defaults to :data:`scix.db.DEFAULT_DSN`.
    """
    if not bibcodes:
        return {}

    # Import here so the module is importable without psycopg installed
    # (the unit tests patch fetch_snippets_from_db via dependency injection).
    from scix.db import get_connection  # type: ignore[import-not-found]

    out: dict[str, str] = {}
    with get_connection(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, title, abstract, body "
            "FROM papers WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
        for bibcode, title, abstract, body in cur.fetchall():
            if not title:
                continue
            out[bibcode] = build_snippet(title=title, abstract=abstract, body=body)
    return out


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------


def run_calibration(
    *,
    seed_path: Path,
    log_path: Path,
    snippet_fetcher: SnippetFetcher,
    dispatcher: Dispatcher,
    prompt_version: str,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> CalibrationReport:
    """Run the persona judge over the seed and report calibration metrics.

    The drift-watch log is appended on every run, regardless of trustworthy
    status — tracking bad runs is part of catching drift.
    """
    seed_rows = read_seed_csv(seed_path)
    if not seed_rows:
        raise ValueError(f"{seed_path}: no rows to calibrate")

    snippets = snippet_fetcher([r.bibcode for r in seed_rows])

    triples: list[JudgeTriple] = []
    humans: list[int] = []
    missing: list[str] = []
    for row in seed_rows:
        snippet = snippets.get(row.bibcode)
        if snippet is None:
            missing.append(row.bibcode)
            continue
        triples.append(JudgeTriple(query=row.query, bibcode=row.bibcode, snippet=snippet))
        humans.append(row.human_score)

    if missing:
        logger.warning(
            "skipping %d seed rows with missing snippets: %s",
            len(missing),
            missing[:5] + (["..."] if len(missing) > 5 else []),
        )

    if not triples:
        raise RuntimeError("no triples with usable snippets — seed CSV cannot be calibrated")

    judge = PersonaJudge(
        dispatcher=dispatcher,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
    )
    scores = asyncio.run(judge.run(triples))

    judge_scores: list[int] = []
    aligned_humans: list[int] = []
    for human, score in zip(humans, scores):
        if score.score < 0:  # error sentinel — drop from calibration
            logger.warning("dropping failed triple %s from calibration", score.triple and score.triple.bibcode)
            continue
        judge_scores.append(score.score)
        aligned_humans.append(human)

    if not judge_scores:
        raise RuntimeError("every triple failed — cannot compute calibration metrics")

    rho = spearman_rho(aligned_humans, judge_scores)
    kappa = quadratic_weighted_kappa(aligned_humans, judge_scores)
    trustworthy = kappa >= TRUSTWORTHY_KAPPA_THRESHOLD

    append_drift_entry(
        log_path=log_path,
        prompt_version=prompt_version,
        kappa=kappa,
        spearman=rho,
        n_triples=len(judge_scores),
    )

    return CalibrationReport(
        spearman=rho,
        kappa=kappa,
        n_triples=len(judge_scores),
        trustworthy=trustworthy,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the in-domain-researcher judge against human labels."
    )
    parser.add_argument("--seed", type=Path, required=True, help="CSV seed file.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/judge_calibration_log.jsonl"),
        help="Drift-watch log (default results/judge_calibration_log.jsonl).",
    )
    parser.add_argument(
        "--prompt-version", default=DEFAULT_PROMPT_VERSION,
        help="Prompt version tag recorded in the drift log.",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--stub", action="store_true",
        help="Use StubDispatcher — wiring check, does not call Claude.",
    )
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN (defaults to scix.db).")
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

    def snippet_fetcher(bibs: list[str]) -> dict[str, str]:
        return fetch_snippets_from_db(bibs, dsn=args.dsn)

    report = run_calibration(
        seed_path=args.seed,
        log_path=args.log,
        snippet_fetcher=snippet_fetcher,
        dispatcher=dispatcher,
        prompt_version=args.prompt_version,
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
    )

    print("Calibration report")
    print(f"  n_triples:   {report.n_triples}")
    print(f"  spearman ρ:  {report.spearman:+.3f}")
    print(f"  kappa (qw):  {report.kappa:+.3f}")
    print(f"  trustworthy: {report.trustworthy} "
          f"(threshold kappa >= {TRUSTWORTHY_KAPPA_THRESHOLD})")
    if not report.trustworthy:
        print("  WARNING: kappa below threshold — iterate on the prompt before shipping.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
