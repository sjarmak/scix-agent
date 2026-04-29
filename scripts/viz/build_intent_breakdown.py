#!/usr/bin/env python3
"""Build the V9 citation-intent breakdown dataset.

Aggregates ``citation_contexts.intent`` (method / background /
result_comparison) into the four panels needed by ``web/viz/intent.html``:

1. ``totals`` — overall intent split with the corpus-coverage caveat.
2. ``top_method`` and ``top_background`` — papers that are most often cited
   for their method or as background, joined to ``papers`` for title/year.
3. ``by_year`` — intent mix by source-paper publication year. The corpus
   currently has classified intents only for citing papers from 2001-2014,
   so a per-decade stream collapses to one or two buckets; per-year is the
   honest unit. The page can collapse to decade for display if desired.
4. ``communities_method`` — medium-resolution communities ranked by
   method-cite ratio, gated on a minimum-volume threshold so 1-paper
   communities don't dominate.

Read-only. The query plan touches ``citation_contexts`` (~825K rows),
``papers``, and ``paper_metrics``; with ``work_mem=4GB`` the full pipeline
finishes well under five minutes.

Usage::

    .venv/bin/python scripts/viz/build_intent_breakdown.py \\
        --output data/viz/citation_intent.json

    .venv/bin/python scripts/viz/build_intent_breakdown.py \\
        --synthetic --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger("build_intent_breakdown")

INTENTS: tuple[str, ...] = ("method", "background", "result_comparison")
TOTAL_CITATION_EDGES = 299_397_468  # paper_metrics-anchored corpus reference
DEFAULT_TOP_PAPERS = 25
DEFAULT_TOP_COMMUNITIES = 20
DEFAULT_MIN_COMMUNITY_VOLUME = 200
DEFAULT_OUTPUT_REL = Path("data/viz/citation_intent.json")
DEFAULT_RESOLUTION = "medium"

COMMUNITY_COLUMNS: dict[str, str] = {
    "coarse": "community_semantic_coarse",
    "medium": "community_semantic_medium",
    "fine": "community_semantic_fine",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopPaper:
    bibcode: str
    title: Optional[str]
    year: Optional[int]
    n: int


@dataclass(frozen=True)
class YearRow:
    year: int
    method: int
    background: int
    result_comparison: int

    @property
    def total(self) -> int:
        return self.method + self.background + self.result_comparison


@dataclass(frozen=True)
class CommunityRow:
    community_id: int
    method: int
    background: int
    result_comparison: int
    method_ratio: float
    terms: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.method + self.background + self.result_comparison


@dataclass(frozen=True)
class IntentDataset:
    totals: dict[str, int]
    coverage: dict[str, int | float]
    top_method: tuple[TopPaper, ...]
    top_background: tuple[TopPaper, ...]
    by_year: tuple[YearRow, ...]
    communities_method: tuple[CommunityRow, ...]
    resolution: str
    min_community_volume: int


@dataclass(frozen=True)
class Config:
    output: Path
    dsn: str
    resolution: str
    top_papers: int
    top_communities: int
    min_community_volume: int
    dry_run: bool
    synthetic: bool
    labels_path: Optional[Path] = field(default=None)


# ---------------------------------------------------------------------------
# Pure aggregation helpers (testable, no DB)
# ---------------------------------------------------------------------------


def coverage_dict(totals: dict[str, int]) -> dict[str, int | float]:
    """Return a serializable coverage block for the ``citation_intent.json`` payload."""
    classified = sum(totals.values())
    pct = (classified / TOTAL_CITATION_EDGES) if TOTAL_CITATION_EDGES > 0 else 0.0
    return {
        "classified_edges": classified,
        "total_edges": TOTAL_CITATION_EDGES,
        "pct_classified": round(pct, 6),
    }


def rank_communities(
    rows: Iterable[tuple[int, str, int]],
    top_n: int,
    min_volume: int,
    labels: Optional[dict[int, tuple[str, ...]]] = None,
) -> tuple[CommunityRow, ...]:
    """Aggregate (community_id, intent, n) rows into ranked CommunityRow tuples.

    Communities below ``min_volume`` total classified citations are dropped so
    the ranking is not dominated by single-paper communities. Ties are broken
    by total volume desc, then community_id asc, both deterministic.
    """
    if top_n < 0:
        raise ValueError(f"top_n must be non-negative, got {top_n}")
    if min_volume < 0:
        raise ValueError(f"min_volume must be non-negative, got {min_volume}")

    buckets: dict[int, dict[str, int]] = {}
    for community_id, intent, n in rows:
        if intent not in INTENTS:
            continue
        b = buckets.setdefault(int(community_id), {k: 0 for k in INTENTS})
        b[intent] += int(n)

    out: list[CommunityRow] = []
    for community_id, b in buckets.items():
        total = sum(b.values())
        if total < min_volume:
            continue
        ratio = b["method"] / total if total > 0 else 0.0
        terms = tuple(labels.get(community_id, ())) if labels else ()
        out.append(
            CommunityRow(
                community_id=community_id,
                method=b["method"],
                background=b["background"],
                result_comparison=b["result_comparison"],
                method_ratio=round(ratio, 6),
                terms=terms,
            )
        )

    out.sort(key=lambda r: (-r.method_ratio, -r.total, r.community_id))
    return tuple(out[:top_n])


def aggregate_by_year(
    rows: Iterable[tuple[int, str, int]],
) -> tuple[YearRow, ...]:
    """Aggregate (year, intent, n) rows into per-year ``YearRow`` tuples, sorted ascending."""
    buckets: dict[int, dict[str, int]] = {}
    for year, intent, n in rows:
        if intent not in INTENTS:
            continue
        b = buckets.setdefault(int(year), {k: 0 for k in INTENTS})
        b[intent] += int(n)

    out = [
        YearRow(
            year=year,
            method=b["method"],
            background=b["background"],
            result_comparison=b["result_comparison"],
        )
        for year, b in buckets.items()
    ]
    out.sort(key=lambda r: r.year)
    return tuple(out)


def load_community_labels(path: Optional[Path]) -> Optional[dict[int, tuple[str, ...]]]:
    """Load a ``community_labels_*.json`` file as ``{community_id: (term, ...)}``.

    Returns ``None`` if ``path`` is ``None`` or the file does not exist; the
    labels file is non-critical decoration on the community-ratio panel.
    """
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, tuple[str, ...]] = {}
    for entry in raw.get("communities", []):
        cid = entry.get("community_id")
        terms = entry.get("terms") or []
        if cid is None:
            continue
        out[int(cid)] = tuple(str(t) for t in terms)
    return out


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------


def _query_totals(cur) -> dict[str, int]:
    cur.execute(
        "SELECT intent, count(*) FROM citation_contexts " "WHERE intent IS NOT NULL GROUP BY intent"
    )
    out = {k: 0 for k in INTENTS}
    for intent, n in cur.fetchall():
        if intent in out:
            out[intent] = int(n)
    return out


def _query_top_papers(cur, intent: str, limit: int) -> tuple[TopPaper, ...]:
    if intent not in INTENTS:
        raise ValueError(f"unknown intent {intent!r}")
    cur.execute(
        """
        WITH tops AS (
            SELECT target_bibcode, count(*) AS n
            FROM citation_contexts
            WHERE intent = %s
            GROUP BY target_bibcode
            ORDER BY n DESC
            LIMIT %s
        )
        SELECT t.target_bibcode, t.n, p.title, p.year
        FROM tops t LEFT JOIN papers p ON t.target_bibcode = p.bibcode
        ORDER BY t.n DESC, t.target_bibcode ASC
        """,
        (intent, limit),
    )
    out: list[TopPaper] = []
    for bibcode, n, title, year in cur.fetchall():
        out.append(
            TopPaper(
                bibcode=str(bibcode),
                title=str(title) if title is not None else None,
                year=int(year) if year is not None else None,
                n=int(n),
            )
        )
    return tuple(out)


def _query_by_year(cur) -> tuple[YearRow, ...]:
    cur.execute("""
        SELECT p.year, cc.intent, count(*) AS n
        FROM citation_contexts cc
        JOIN papers p ON cc.source_bibcode = p.bibcode
        WHERE p.year IS NOT NULL AND cc.intent IS NOT NULL
        GROUP BY p.year, cc.intent
        """)
    return aggregate_by_year(((y, i, n) for y, i, n in cur.fetchall()))


def _query_community_intent(
    cur,
    column: str,
) -> list[tuple[int, str, int]]:
    cur.execute(f"""
        SELECT pm.{column}, cc.intent, count(*) AS n
        FROM citation_contexts cc
        JOIN paper_metrics pm ON cc.source_bibcode = pm.bibcode
        WHERE pm.{column} IS NOT NULL AND cc.intent IS NOT NULL
        GROUP BY pm.{column}, cc.intent
        """)
    return [(int(c), str(i), int(n)) for c, i, n in cur.fetchall()]


# ---------------------------------------------------------------------------
# Synthetic dataset (deterministic, no DB)
# ---------------------------------------------------------------------------


def _synthetic_dataset(config: Config) -> IntentDataset:
    totals = {"method": 600, "background": 300, "result_comparison": 100}
    top_method = tuple(
        TopPaper(
            bibcode=f"synth-method-{i:02d}",
            title=f"Synthetic Method Paper {i}",
            year=2000 + i,
            n=100 - i * 4,
        )
        for i in range(min(config.top_papers, 5))
    )
    top_background = tuple(
        TopPaper(
            bibcode=f"synth-bg-{i:02d}",
            title=f"Synthetic Background Paper {i}",
            year=1990 + i,
            n=80 - i * 3,
        )
        for i in range(min(config.top_papers, 5))
    )
    by_year = tuple(
        YearRow(year=y, method=100 - i * 5, background=50 - i * 2, result_comparison=20 - i)
        for i, y in enumerate(range(2001, 2005))
    )
    communities_method = rank_communities(
        rows=[
            (0, "method", 400),
            (0, "background", 80),
            (0, "result_comparison", 20),
            (1, "method", 200),
            (1, "background", 250),
            (1, "result_comparison", 50),
            (2, "method", 50),
            (2, "background", 5),
            (2, "result_comparison", 5),
        ],
        top_n=config.top_communities,
        min_volume=config.min_community_volume,
    )
    return IntentDataset(
        totals=totals,
        coverage=coverage_dict(totals),
        top_method=top_method,
        top_background=top_background,
        by_year=by_year,
        communities_method=communities_method,
        resolution=config.resolution,
        min_community_volume=config.min_community_volume,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(config: Config) -> IntentDataset:
    if config.synthetic:
        logger.info("synthetic mode — no DB")
        return _synthetic_dataset(config)

    if config.resolution not in COMMUNITY_COLUMNS:
        raise ValueError(
            f"unknown resolution {config.resolution!r}; must be one of "
            f"{tuple(COMMUNITY_COLUMNS)}"
        )
    column = COMMUNITY_COLUMNS[config.resolution]

    logger.info(
        "querying DB dsn=%s resolution=%s column=%s",
        redact_dsn(config.dsn),
        config.resolution,
        column,
    )
    if is_production_dsn(config.dsn):
        logger.info("DSN appears to point at production (read-only, proceeding)")

    import psycopg

    labels = load_community_labels(config.labels_path)
    if labels is not None:
        logger.info("loaded %d community labels from %s", len(labels), config.labels_path)

    t0 = time.time()
    with psycopg.connect(config.dsn) as conn:
        conn.set_read_only(True)
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '4GB'")

            t = time.time()
            totals = _query_totals(cur)
            logger.info("totals %s (%.1fs)", totals, time.time() - t)

            t = time.time()
            top_method = _query_top_papers(cur, "method", config.top_papers)
            logger.info("top_method n=%d (%.1fs)", len(top_method), time.time() - t)

            t = time.time()
            top_background = _query_top_papers(cur, "background", config.top_papers)
            logger.info("top_background n=%d (%.1fs)", len(top_background), time.time() - t)

            t = time.time()
            by_year = _query_by_year(cur)
            logger.info("by_year n=%d (%.1fs)", len(by_year), time.time() - t)

            t = time.time()
            community_rows = _query_community_intent(cur, column)
            logger.info(
                "community_intent rows=%d (%.1fs)",
                len(community_rows),
                time.time() - t,
            )

    communities_method = rank_communities(
        rows=community_rows,
        top_n=config.top_communities,
        min_volume=config.min_community_volume,
        labels=labels,
    )
    logger.info(
        "ranked %d communities (kept top %d) in %.1fs total",
        len(communities_method),
        config.top_communities,
        time.time() - t0,
    )

    return IntentDataset(
        totals=totals,
        coverage=coverage_dict(totals),
        top_method=top_method,
        top_background=top_background,
        by_year=by_year,
        communities_method=communities_method,
        resolution=config.resolution,
        min_community_volume=config.min_community_volume,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _top_paper_to_dict(p: TopPaper) -> dict:
    return {"bibcode": p.bibcode, "title": p.title, "year": p.year, "n": p.n}


def _year_row_to_dict(r: YearRow) -> dict:
    return {
        "year": r.year,
        "method": r.method,
        "background": r.background,
        "result_comparison": r.result_comparison,
        "total": r.total,
    }


def _community_row_to_dict(r: CommunityRow) -> dict:
    return {
        "community_id": r.community_id,
        "method": r.method,
        "background": r.background,
        "result_comparison": r.result_comparison,
        "total": r.total,
        "method_ratio": r.method_ratio,
        "terms": list(r.terms),
    }


def to_payload(ds: IntentDataset) -> dict:
    return {
        "totals": ds.totals,
        "coverage": ds.coverage,
        "top_method": [_top_paper_to_dict(p) for p in ds.top_method],
        "top_background": [_top_paper_to_dict(p) for p in ds.top_background],
        "by_year": [_year_row_to_dict(r) for r in ds.by_year],
        "communities_method": [_community_row_to_dict(r) for r in ds.communities_method],
        "resolution": ds.resolution,
        "min_community_volume": ds.min_community_volume,
    }


def serialize_to_json(ds: IntentDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_payload(ds), indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_REL),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_REL}).",
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: scix.db.DEFAULT_DSN).",
    )
    parser.add_argument(
        "--resolution",
        choices=tuple(COMMUNITY_COLUMNS),
        default=DEFAULT_RESOLUTION,
        help=f"Community resolution for the per-community panel (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--top-papers",
        type=int,
        default=DEFAULT_TOP_PAPERS,
        help=f"Top-N papers per intent panel (default: {DEFAULT_TOP_PAPERS}).",
    )
    parser.add_argument(
        "--top-communities",
        type=int,
        default=DEFAULT_TOP_COMMUNITIES,
        help=f"Top-N communities by method-ratio (default: {DEFAULT_TOP_COMMUNITIES}).",
    )
    parser.add_argument(
        "--min-community-volume",
        type=int,
        default=DEFAULT_MIN_COMMUNITY_VOLUME,
        help=(
            f"Minimum classified-citations per community to be considered "
            f"(default: {DEFAULT_MIN_COMMUNITY_VOLUME})."
        ),
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Optional path to a community_labels_*.json for term-list decoration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but skip writing the output file.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a deterministic synthetic dataset; no DB required.",
    )
    return parser.parse_args(argv)


def _resolve_path(raw: str | None) -> Optional[Path]:
    if raw is None:
        return None
    p = Path(raw)
    return p if p.is_absolute() else _REPO_ROOT / p


def _config_from_args(args: argparse.Namespace) -> Config:
    labels_raw = args.labels_path
    if labels_raw is None and not args.synthetic:
        # Default to the medium-resolution labels file when --resolution=medium.
        candidate = _REPO_ROOT / "data" / "viz" / f"community_labels_{args.resolution}.json"
        if candidate.exists():
            labels_raw = str(candidate)
    return Config(
        output=_resolve_path(args.output) or _REPO_ROOT / DEFAULT_OUTPUT_REL,
        dsn=args.dsn,
        resolution=args.resolution,
        top_papers=int(args.top_papers),
        top_communities=int(args.top_communities),
        min_community_volume=int(args.min_community_volume),
        dry_run=bool(args.dry_run),
        synthetic=bool(args.synthetic),
        labels_path=_resolve_path(labels_raw),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    config = _config_from_args(args)
    ds = run(config)

    logger.info(
        "summary: totals=%s coverage_pct=%.4f top_method=%d top_bg=%d years=%d communities=%d",
        ds.totals,
        ds.coverage["pct_classified"],
        len(ds.top_method),
        len(ds.top_background),
        len(ds.by_year),
        len(ds.communities_method),
    )

    if config.dry_run:
        logger.info("--dry-run set, skipping write of %s", config.output)
        return 0

    serialize_to_json(ds, config.output)
    logger.info("wrote %s", config.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
