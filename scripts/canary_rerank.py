#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Daily rerank canary — drift detection for the MiniLM cross-encoder.
#
# Background (from the M1 ablation, results/retrieval_eval_50q_rerank_local.md):
# both candidate rerankers (cross-encoder/ms-marco-MiniLM-L-12-v2 and
# BAAI/bge-reranker-large) regressed nDCG@10 vs the unranked INDUS-hybrid
# baseline at p<0.025 (Bonferroni-corrected). The M4 rollout decision left
# SCIX_RERANK_DEFAULT_MODEL='off'. This canary is forward-looking: it
# (a) catches model behaviour drift in the MiniLM checkpoint if/when an
#     operator flips rerank on for an experiment, and
# (b) acts as a smoke test that the CrossEncoderReranker code path still
#     loads + scores, so when a follow-up reranker (e.g. domain-tuned) lands
#     we have continuity on the integration surface.
#
# Why MiniLM specifically (not bge-reranker-large): MiniLM is ~80 MB and
# scores 20 pairs in well under a second on CPU; bge-reranker-large takes
# ~570 ms/run on GPU and 4+s on CPU and is too heavy for a daily canary on
# the shared host.
#
# Scheduling (comment only; not auto-installed — operators add their own):
#
# Cron:
#   0 6 * * * cd /home/ds/projects/scix_experiments && \
#     .venv/bin/python scripts/canary_rerank.py >> logs/canary_rerank/cron.log 2>&1
#
# systemd user timer (~/.config/systemd/user/canary-rerank.timer):
#   [Unit]
#   Description=Daily rerank-score drift check
#
#   [Timer]
#   OnCalendar=*-*-* 06:00:00
#   Persistent=true
#
#   [Install]
#   WantedBy=timers.target
#
# Companion service unit (~/.config/systemd/user/canary-rerank.service):
#   [Unit]
#   Description=Rerank-score drift check
#
#   [Service]
#   Type=oneshot
#   WorkingDirectory=/home/ds/projects/scix_experiments
#   ExecStart=/home/ds/projects/scix_experiments/.venv/bin/python \
#     scripts/canary_rerank.py
#
# Enable with:  systemctl --user enable --now canary-rerank.timer
#
# Exit codes:
#   0 — every pair within drift threshold (default 5%).
#   2 — at least one pair drifted beyond threshold (alerting signal).
#   3 — fixture or baseline file invalid / missing required schema fields.
#   4 — model load / scoring failed (sentence-transformers, weights, etc.).
#   5 — first-run baseline build failed (DB unreachable, no usable seeds).
#
# Memory: MiniLM is ~80 MB; the full canary peaks well under 1 GB RSS.
# scix-batch is NOT required.
# ---------------------------------------------------------------------------
"""Daily MiniLM cross-encoder drift canary.

Loads 20 fixed (query, paper) pairs from a checked-in fixture
(``data/canary/rerank_baseline.json``), scores them with the same model
the M1 ablation used (``cross-encoder/ms-marco-MiniLM-L-12-v2``), and
compares each new score to the baseline_score persisted in the fixture.
Exits non-zero if any pair drifts more than 5% (relative, |Δ| / |base|).

First-run bootstrap: if the baseline fixture does not yet exist, the
script builds it from the live database. It draws the first 20 seed
bibcodes from ``results/retrieval_eval_50q.json``, pairs each with the
top-1 cited paper (``citation_edges``, deterministic by lexicographic
``target_bibcode``), and writes the fixture with the freshly-computed
MiniLM scores serving as the baseline. After the first run the fixture
is the single source of truth and the DB is no longer needed.

Per-run output: a timestamped JSON log under ``logs/canary_rerank/``
containing per-pair scores, deltas, and pass/fail. A single human-
readable summary line is also written to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Pin to the same MiniLM checkpoint the M1 ablation used so drift is
# measured against a stable reference. If a future operator wants to
# canary a different reranker (e.g. a domain-tuned variant), build a new
# fixture from scratch with that model rather than mixing scores.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

DEFAULT_BASELINE = Path("data/canary/rerank_baseline.json")
DEFAULT_LOG_DIR = Path("logs/canary_rerank")
DEFAULT_GOLD_PATH = Path("results/retrieval_eval_50q.json")

DEFAULT_DRIFT_THRESHOLD = 0.05  # relative, |new - base| / |base|
DEFAULT_N_PAIRS = 20

# Trim abstract feeds to the cross-encoder to 1000 chars — matches the
# convention in eval_rerank_local_ab.py so the canary scores the same
# document text the M1 ablation did.
ABSTRACT_SNIPPET_CHARS = 1000

REQUIRED_BASELINE_TOP_KEYS = ("schema_version", "model_name", "pairs")
REQUIRED_PAIR_KEYS = (
    "query_id",
    "paper_id",
    "query",
    "paper_title",
    "paper_abstract_snippet",
    "baseline_score",
)

_WS_RE = re.compile(r"\s+")


class FileFormatError(ValueError):
    """Raised when fixture / baseline JSON is missing required schema."""


# ---------------------------------------------------------------------------
# Data containers (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One (query, paper) pair plus its baseline rerank score."""

    query_id: str
    paper_id: str
    query: str
    paper_title: str
    paper_abstract_snippet: str
    baseline_score: float


@dataclass(frozen=True)
class PairResult:
    """Per-pair canary outcome."""

    query_id: str
    paper_id: str
    baseline_score: float
    new_score: float
    abs_delta: float
    rel_delta: float
    drifted: bool


@dataclass(frozen=True)
class CanaryReport:
    """Aggregated canary outcome for one run."""

    generated_at: str
    model_name: str
    drift_threshold: float
    n_pairs: int
    n_drifted: int
    pass_: bool
    pairs: tuple[PairResult, ...]


# ---------------------------------------------------------------------------
# Fixture / baseline IO
# ---------------------------------------------------------------------------


def build_query_text(title: str | None, abstract: str | None, *, max_words: int = 50) -> str:
    """Title + first ``max_words`` words of abstract — mirrors the M1 eval."""
    title_clean = _WS_RE.sub(" ", (title or "")).strip()
    abstract_clean = _WS_RE.sub(" ", (abstract or "")).strip()
    if abstract_clean:
        abstract_clean = " ".join(abstract_clean.split()[:max_words])
    if title_clean and abstract_clean:
        return f"{title_clean}. {abstract_clean}"
    return title_clean or abstract_clean


def load_baseline(path: Path) -> tuple[Pair, ...]:
    """Load and validate the baseline fixture.

    Raises FileFormatError if the file is malformed. Caller is expected
    to handle FileNotFoundError separately (it's a signal to bootstrap).
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline fixture not found at {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FileFormatError(f"Baseline at {path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise FileFormatError(
            f"Baseline at {path} must be a JSON object, got {type(payload).__name__}"
        )

    missing_top = [k for k in REQUIRED_BASELINE_TOP_KEYS if k not in payload]
    if missing_top:
        raise FileFormatError(
            f"Baseline at {path} missing required keys: {missing_top}. "
            f"Expected schema: {list(REQUIRED_BASELINE_TOP_KEYS)}."
        )

    if payload.get("model_name") != MODEL_NAME:
        raise FileFormatError(
            f"Baseline at {path} declares model_name="
            f"{payload.get('model_name')!r}, but this canary scores with "
            f"{MODEL_NAME!r}. Refusing to compare scores across models."
        )

    pairs_raw = payload.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise FileFormatError(
            f"Baseline at {path} 'pairs' must be a non-empty list."
        )

    pairs: list[Pair] = []
    for i, row in enumerate(pairs_raw):
        if not isinstance(row, dict):
            raise FileFormatError(
                f"Baseline at {path} pairs[{i}] must be an object, got "
                f"{type(row).__name__}."
            )
        missing = [k for k in REQUIRED_PAIR_KEYS if k not in row]
        if missing:
            raise FileFormatError(
                f"Baseline at {path} pairs[{i}] missing keys: {missing}. "
                f"Expected: {list(REQUIRED_PAIR_KEYS)}."
            )
        try:
            pairs.append(
                Pair(
                    query_id=str(row["query_id"]),
                    paper_id=str(row["paper_id"]),
                    query=str(row["query"]),
                    paper_title=str(row["paper_title"]),
                    paper_abstract_snippet=str(row["paper_abstract_snippet"]),
                    baseline_score=float(row["baseline_score"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise FileFormatError(
                f"Baseline at {path} pairs[{i}] failed type coercion: {exc}"
            ) from exc

    return tuple(pairs)


def write_baseline(
    path: Path,
    pairs: Iterable[Pair],
    *,
    drift_threshold: float,
    gold_path: Path,
) -> None:
    """Persist the baseline fixture (used on first-run bootstrap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": MODEL_NAME,
        "drift_threshold": drift_threshold,
        "gold_path": str(gold_path),
        "abstract_snippet_chars": ABSTRACT_SNIPPET_CHARS,
        "pairs": [asdict(p) for p in pairs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote baseline fixture: %s", path)


# ---------------------------------------------------------------------------
# Reranker bridge
# ---------------------------------------------------------------------------


def _build_reranker() -> Any:
    """Load the CrossEncoderReranker from scix.search.

    Deferred import — sentence-transformers + torch are heavy and the
    fixture-load codepath of this script doesn't need them.
    """
    # Repo bootstrap so this script runs from anywhere.
    repo_root = Path(__file__).resolve().parent.parent
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from scix.search import CrossEncoderReranker  # type: ignore[import-not-found]

    return CrossEncoderReranker(model_name=MODEL_NAME)


def score_pairs(reranker: Any, pairs: Iterable[Pair]) -> dict[str, float]:
    """Score every pair, keyed by ``f"{query_id}::{paper_id}"``.

    The CrossEncoderReranker API takes one query + many candidate papers
    per call. Each pair has its OWN query, so we issue one call per pair.
    That's slightly slower than batching but keeps the scoring path
    identical to single-pair use and produces scores that aren't sensitive
    to within-batch normalization.
    """
    scores: dict[str, float] = {}
    for pair in pairs:
        candidate = {
            "bibcode": pair.paper_id,
            "title": pair.paper_title,
            "abstract_snippet": pair.paper_abstract_snippet,
        }
        ranked = reranker(pair.query, [candidate])
        if not ranked:
            raise RuntimeError(
                f"Reranker returned no scored candidates for pair "
                f"{pair.query_id}::{pair.paper_id}"
            )
        scores[f"{pair.query_id}::{pair.paper_id}"] = float(ranked[0]["rerank_score"])
    return scores


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


def compute_drift(
    pairs: Iterable[Pair],
    new_scores: dict[str, float],
    *,
    threshold: float,
) -> CanaryReport:
    """Build the per-pair drift report.

    Drift rule: |new - base| / |base| > threshold. We use absolute-value
    in the denominator so signed scores (cross-encoder logits can go
    negative) are handled cleanly. For base==0 we fall back to absolute
    delta against threshold so the rule is still well-defined.
    """
    results: list[PairResult] = []
    n_drifted = 0
    for pair in pairs:
        key = f"{pair.query_id}::{pair.paper_id}"
        if key not in new_scores:
            raise KeyError(f"Missing new score for pair {key}")
        new = new_scores[key]
        base = pair.baseline_score
        abs_delta = abs(new - base)
        if abs(base) > 0:
            rel_delta = abs_delta / abs(base)
            drifted = rel_delta > threshold
        else:
            # Degenerate baseline — compare absolute delta vs threshold
            # directly so the rule remains well-defined.
            rel_delta = abs_delta
            drifted = abs_delta > threshold
        if drifted:
            n_drifted += 1
        results.append(
            PairResult(
                query_id=pair.query_id,
                paper_id=pair.paper_id,
                baseline_score=base,
                new_score=new,
                abs_delta=abs_delta,
                rel_delta=rel_delta,
                drifted=drifted,
            )
        )
    return CanaryReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_name=MODEL_NAME,
        drift_threshold=threshold,
        n_pairs=len(results),
        n_drifted=n_drifted,
        pass_=(n_drifted == 0),
        pairs=tuple(results),
    )


def write_log(log_dir: Path, report: CanaryReport) -> Path:
    """Persist the per-run JSON log under ``log_dir`` with a UTC stamp."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = log_dir / f"{stamp}.json"
    payload = {
        "generated_at": report.generated_at,
        "model_name": report.model_name,
        "drift_threshold": report.drift_threshold,
        "n_pairs": report.n_pairs,
        "n_drifted": report.n_drifted,
        "pass": report.pass_,
        "pairs": [asdict(p) for p in report.pairs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# First-run bootstrap (DB-backed)
# ---------------------------------------------------------------------------


def _load_first_n_seeds(gold_path: Path, n: int) -> list[str]:
    """Pull the first ``n`` unique seed bibcodes from the gold-set JSON."""
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold-set file not found at {gold_path}; cannot bootstrap baseline."
        )
    payload = json.loads(gold_path.read_text())
    per_query = payload.get("per_query")
    if not isinstance(per_query, list) or not per_query:
        raise FileFormatError(
            f"Gold-set {gold_path} missing non-empty 'per_query' list."
        )
    seen: set[str] = set()
    seeds: list[str] = []
    for entry in per_query:
        bib = entry.get("seed_bibcode") if isinstance(entry, dict) else None
        if not bib or bib in seen:
            continue
        seen.add(bib)
        seeds.append(bib)
        if len(seeds) >= n:
            break
    if len(seeds) < n:
        raise FileFormatError(
            f"Gold-set {gold_path} has only {len(seeds)} unique seeds; need {n}."
        )
    return seeds


def _bootstrap_pairs_from_db(seed_bibcodes: list[str]) -> list[Pair]:
    """Resolve (query_text, top-1 cited paper) for each seed from the DB.

    ``top-1 cited paper`` is the lexicographically smallest
    ``target_bibcode`` from ``citation_edges WHERE source_bibcode = seed``.
    Lex order is deterministic, reproducible, and doesn't require any
    quality signal — this fixture is about model behaviour drift, not
    relevance. Seeds with no outbound citations are skipped.
    """
    # Repo bootstrap so this script runs from anywhere.
    repo_root = Path(__file__).resolve().parent.parent
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from psycopg.rows import dict_row  # type: ignore[import-not-found]
    from scix.db import get_connection  # type: ignore[import-not-found]

    pairs: list[Pair] = []
    conn = get_connection()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            for bib in seed_bibcodes:
                # Seed paper text → query.
                cur.execute(
                    "SELECT bibcode, title, abstract FROM papers WHERE bibcode = %s",
                    [bib],
                )
                seed_row = cur.fetchone()
                if seed_row is None:
                    logger.warning("Skipping seed %s: not in papers table", bib)
                    continue
                query_text = build_query_text(seed_row.get("title"), seed_row.get("abstract"))
                if not query_text:
                    logger.warning("Skipping seed %s: empty title and abstract", bib)
                    continue

                # Top-1 cited paper — deterministic lex order.
                cur.execute(
                    """
                    SELECT p.bibcode, p.title, p.abstract
                    FROM citation_edges ce
                    JOIN papers p ON p.bibcode = ce.target_bibcode
                    WHERE ce.source_bibcode = %s
                    ORDER BY ce.target_bibcode ASC
                    LIMIT 1
                    """,
                    [bib],
                )
                paper_row = cur.fetchone()
                if paper_row is None:
                    logger.warning("Skipping seed %s: no outbound citations", bib)
                    continue

                paper_title = paper_row.get("title") or ""
                paper_abs = paper_row.get("abstract") or ""
                pairs.append(
                    Pair(
                        query_id=bib,
                        paper_id=paper_row["bibcode"],
                        query=query_text,
                        paper_title=paper_title,
                        paper_abstract_snippet=paper_abs[:ABSTRACT_SNIPPET_CHARS],
                        baseline_score=0.0,  # filled in below after scoring
                    )
                )
    finally:
        conn.close()
    return pairs


def _score_and_attach(pairs: list[Pair], reranker: Any) -> list[Pair]:
    """Score each bootstrap pair and return new Pairs with baseline_score set."""
    scores = score_pairs(reranker, pairs)
    out: list[Pair] = []
    for p in pairs:
        key = f"{p.query_id}::{p.paper_id}"
        out.append(
            Pair(
                query_id=p.query_id,
                paper_id=p.paper_id,
                query=p.query,
                paper_title=p.paper_title,
                paper_abstract_snippet=p.paper_abstract_snippet,
                baseline_score=scores[key],
            )
        )
    return out


def bootstrap_baseline(
    baseline_path: Path,
    gold_path: Path,
    *,
    n_pairs: int,
    drift_threshold: float,
    reranker: Any,
) -> tuple[Pair, ...]:
    """Build, score, and persist the baseline fixture from the live DB.

    Caller invokes this when ``baseline_path`` does not exist. Returns
    the same tuple of Pairs the canary main loop would have loaded from
    the file, so the bootstrap run can immediately verify (and trivially
    pass) on the freshly-computed scores.
    """
    seeds = _load_first_n_seeds(gold_path, n_pairs)
    raw_pairs = _bootstrap_pairs_from_db(seeds)
    if len(raw_pairs) < n_pairs:
        raise RuntimeError(
            f"Bootstrap produced only {len(raw_pairs)} pairs from {len(seeds)} "
            f"seeds (need {n_pairs}). Some seeds had no outbound citations or "
            f"missing paper rows."
        )
    raw_pairs = raw_pairs[:n_pairs]
    scored = _score_and_attach(raw_pairs, reranker)
    write_baseline(
        baseline_path,
        scored,
        drift_threshold=drift_threshold,
        gold_path=gold_path,
    )
    return tuple(scored)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="canary_rerank",
        description=(
            "Daily MiniLM cross-encoder drift canary. Scores 20 fixed "
            "(query, paper) pairs and exits non-zero on >5% drift vs "
            "baseline. First run bootstraps the baseline fixture from "
            "the live DB."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Baseline fixture path (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help="Relative drift threshold |new-base|/|base| (default: %(default)s).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-run JSON logs (default: %(default)s).",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=(
            "Gold-set JSON used only on first-run bootstrap to pick seed "
            "bibcodes (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=DEFAULT_N_PAIRS,
        help="Number of pairs in the canary fixture (default: %(default)s).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)

    # Phase A: load or bootstrap the baseline fixture.
    try:
        pairs = load_baseline(args.baseline)
        bootstrapped = False
    except FileNotFoundError:
        logger.info(
            "Baseline fixture %s not found — bootstrapping from DB and "
            "gold set %s",
            args.baseline,
            args.gold_path,
        )
        try:
            reranker = _build_reranker()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load CrossEncoderReranker: %s", exc)
            return 4
        try:
            pairs = bootstrap_baseline(
                args.baseline,
                args.gold_path,
                n_pairs=args.n_pairs,
                drift_threshold=args.threshold,
                reranker=reranker,
            )
        except (FileNotFoundError, FileFormatError, RuntimeError) as exc:
            logger.error("Bootstrap failed: %s", exc)
            return 5
        bootstrapped = True
    except FileFormatError as exc:
        logger.error("Baseline fixture invalid: %s", exc)
        return 3

    # Phase B: score the pairs with the current MiniLM checkpoint.
    if not bootstrapped:
        try:
            reranker = _build_reranker()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load CrossEncoderReranker: %s", exc)
            return 4
    try:
        t0 = time.perf_counter()
        new_scores = score_pairs(reranker, pairs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        logger.error("Scoring failed: %s", exc)
        return 4

    # Phase C: drift comparison + log.
    report = compute_drift(pairs, new_scores, threshold=args.threshold)
    log_path = write_log(args.log_dir, report)

    status = "PASS" if report.pass_ else "FAIL"
    print(
        f"canary_rerank {status} model={MODEL_NAME} pairs={report.n_pairs} "
        f"drifted={report.n_drifted} threshold={args.threshold:.3f} "
        f"score_ms={elapsed_ms:.1f} log={log_path}"
        f"{' [bootstrap]' if bootstrapped else ''}"
    )

    return 0 if report.pass_ else 2


if __name__ == "__main__":
    raise SystemExit(main())
