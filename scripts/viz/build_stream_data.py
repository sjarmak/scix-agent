#!/usr/bin/env python3
"""Aggregate (year, community_id, paper_count) for the streamgraph view.

Read-only tool: groups ``papers.year`` by ``paper_metrics.community_semantic_*``
at the requested resolution and serializes the result as a JSON payload sized
for D3's stack layout. Defaults to the 2005-2024 window so the trailing two
years (which are incomplete in our snapshot) don't fake a downturn.

Usage
-----

Smoke-test mode (no DB)::

    python scripts/viz/build_stream_data.py \\
        --synthetic --resolution coarse \\
        --output /tmp/stream_test.json

Production::

    python scripts/viz/build_stream_data.py \\
        --resolution medium --year-min 2005 --year-max 2024 \\
        --output data/viz/stream.medium.json

Output JSON schema::

    {
      "resolution": "medium",
      "year_min": 2005,
      "year_max": 2024,
      "years": [2005, 2006, ..., 2024],
      "communities": [
        {
          "community_id": 17,
          "label": "exoplanet atmospheres",
          "total": 12345,
          "counts": [10, 14, ..., 1500]   # aligned with `years`
        },
        ...
      ]
    }

``counts`` is dense (zero-filled per year), aligned with ``years``, and
sorted within ``communities`` by ``total`` descending so D3 stack layouts
get the largest bands first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_stream_data")


# Allowlist mapping --resolution -> paper_metrics column. Used as a
# defence-in-depth check before the column name is interpolated into SQL.
COMMUNITY_COLUMNS: dict[str, str] = {
    "coarse": "community_semantic_coarse",
    "medium": "community_semantic_medium",
    "fine": "community_semantic_fine",
}
RESOLUTIONS: tuple[str, ...] = tuple(COMMUNITY_COLUMNS.keys())

# The corpus snapshot stretches 1800-2026 but the last two years are
# incomplete and pre-2000 papers are sparse (NASA ADS coverage thins out).
# These defaults give a 20-year window where every bin holds ~1M papers.
DEFAULT_YEAR_MIN = 2005
DEFAULT_YEAR_MAX = 2024
DEFAULT_RESOLUTION = "coarse"


@dataclass(frozen=True)
class Cell:
    """One (community_id, year, count) tuple from the GROUP BY."""

    community_id: int
    year: int
    count: int


@dataclass(frozen=True)
class Config:
    resolution: str
    year_min: int
    year_max: int
    dsn: str
    output: Path
    synthetic: bool
    labels_path: Optional[Path]


def _build_sql(resolution: str) -> str:
    """Return parametrised SQL for the (community, year, count) aggregation.

    Column name comes from the hard-coded allowlist; the year window comes
    in as bound parameters. The query relies on ``ix_papers_year`` and the
    BTree on ``paper_metrics.community_semantic_<res>`` to keep the scan
    in the seconds-not-minutes range on 32M rows.
    """
    if resolution not in COMMUNITY_COLUMNS:
        raise ValueError(
            f"unknown resolution {resolution!r}; must be one of {RESOLUTIONS}"
        )
    column = COMMUNITY_COLUMNS[resolution]
    return (
        f"SELECT pm.{column} AS community_id, p.year AS year, COUNT(*) AS n\n"
        "FROM papers p\n"
        "JOIN paper_metrics pm USING (bibcode)\n"
        f"WHERE pm.{column} IS NOT NULL\n"
        "  AND p.year BETWEEN %s AND %s\n"
        f"GROUP BY pm.{column}, p.year\n"
        f"ORDER BY pm.{column}, p.year"
    )


def load_cells_from_db(
    dsn: str, resolution: str, year_min: int, year_max: int
) -> list[Cell]:
    """Stream the (community, year, count) rows from the database."""
    if year_min > year_max:
        raise ValueError(
            f"year_min {year_min} > year_max {year_max}; check the CLI args"
        )

    import psycopg  # lazy so synthetic / tests don't need a live DB

    sql = _build_sql(resolution)
    cells: list[Cell] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (year_min, year_max))
            for community_id, year, count in cur.fetchall():
                if community_id is None or year is None:
                    continue
                cells.append(
                    Cell(
                        community_id=int(community_id),
                        year=int(year),
                        count=int(count),
                    )
                )
    return cells


def load_cells_synthetic(
    resolution: str, year_min: int, year_max: int, seed: int = 42
) -> list[Cell]:
    """Generate deterministic synthetic cells for offline tests.

    Each community gets a mild Poisson-like growth curve so the resulting
    streamgraph has shape rather than flat slabs.
    """
    import random

    rng = random.Random(seed)
    k = {"coarse": 20, "medium": 200, "fine": 2000}.get(resolution, 20)
    cells: list[Cell] = []
    for cid in range(k):
        base = 50 + rng.randint(0, 200)
        growth = rng.uniform(0.92, 1.08)
        cur = float(base)
        for year in range(year_min, year_max + 1):
            count = int(max(1, cur * (1 + rng.uniform(-0.15, 0.15))))
            cells.append(Cell(community_id=cid, year=year, count=count))
            cur *= growth
    return cells


def load_labels(labels_path: Optional[Path]) -> dict[int, str]:
    """Read a community_labels JSON and return ``{community_id: label}``.

    Accepts the schema produced by ``scripts/viz/compute_community_labels.py``::

        {"resolution": "...", "communities": [{"community_id": N, "terms": [...]}]}

    Returns an empty dict if ``labels_path`` is ``None`` or the file is
    missing — the streamgraph viz falls back to ``"community {id}"``.
    """
    if labels_path is None or not labels_path.exists():
        return {}
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("failed to parse %s; ignoring", labels_path)
        return {}
    out: dict[int, str] = {}
    for entry in payload.get("communities") or []:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("community_id")
        terms = entry.get("terms") or []
        if cid is None or not isinstance(terms, list) or not terms:
            continue
        # First three keyword tokens, joined — matches umap_browser legend style.
        out[int(cid)] = " / ".join(str(t) for t in terms[:3])
    return out


def build_payload(
    cells: Iterable[Cell],
    resolution: str,
    year_min: int,
    year_max: int,
    labels: Optional[dict[int, str]] = None,
) -> dict:
    """Reshape the flat cell list into a stream-friendly nested payload.

    Output is sorted by community total descending so D3 stack layouts get
    the biggest bands first. ``counts`` is dense (zero-filled per year) and
    aligned positionally with ``years``.
    """
    if labels is None:
        labels = {}
    if year_min > year_max:
        raise ValueError("year_min > year_max")

    years = list(range(year_min, year_max + 1))
    year_index = {y: i for i, y in enumerate(years)}

    by_cid: dict[int, list[int]] = {}
    for cell in cells:
        if cell.year not in year_index:
            continue
        bucket = by_cid.setdefault(cell.community_id, [0] * len(years))
        bucket[year_index[cell.year]] += cell.count

    communities = []
    for cid, counts in by_cid.items():
        total = sum(counts)
        if total == 0:
            continue
        communities.append(
            {
                "community_id": cid,
                "label": labels.get(cid, f"community {cid}"),
                "total": total,
                "counts": counts,
            }
        )
    communities.sort(key=lambda c: c["total"], reverse=True)

    return {
        "resolution": resolution,
        "year_min": year_min,
        "year_max": year_max,
        "years": years,
        "communities": communities,
    }


def serialize(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=RESOLUTIONS,
        default=DEFAULT_RESOLUTION,
        help=f"Semantic-community resolution (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=DEFAULT_YEAR_MIN,
        help=f"Lowest year to include (default: {DEFAULT_YEAR_MIN}).",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=DEFAULT_YEAR_MAX,
        help=f"Highest year to include (default: {DEFAULT_YEAR_MAX}).",
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: scix.db.DEFAULT_DSN).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Default: data/viz/stream.<resolution>.json "
            "(relative to repo root)."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional community labels JSON path. Default: try "
            "data/viz/community_labels.<resolution>.json then web/viz/."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate deterministic synthetic data instead of querying the DB.",
    )
    return parser.parse_args(argv)


def _resolve_output(raw: Optional[str], resolution: str) -> Path:
    if raw is not None:
        p = Path(raw)
        return p if p.is_absolute() else _REPO_ROOT / p
    return _REPO_ROOT / "data" / "viz" / f"stream.{resolution}.json"


def _resolve_labels(raw: Optional[str], resolution: str) -> Optional[Path]:
    if raw is not None:
        p = Path(raw)
        return p if p.is_absolute() else _REPO_ROOT / p
    candidates = [
        _REPO_ROOT / "data" / "viz" / f"community_labels.{resolution}.json",
        _REPO_ROOT / "web" / "viz" / f"community_labels.{resolution}.json",
        _REPO_ROOT / "web" / "viz" / f"community_labels_{resolution}.json",
        _REPO_ROOT / "web" / "viz" / "community_labels.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        resolution=args.resolution,
        year_min=int(args.year_min),
        year_max=int(args.year_max),
        dsn=args.dsn,
        output=_resolve_output(args.output, args.resolution),
        synthetic=bool(args.synthetic),
        labels_path=_resolve_labels(args.labels, args.resolution),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)

    if config.resolution not in COMMUNITY_COLUMNS:
        logger.error("resolution %r not in allowlist", config.resolution)
        return 2

    if config.synthetic:
        logger.info(
            "synthetic mode: resolution=%s year_min=%d year_max=%d",
            config.resolution,
            config.year_min,
            config.year_max,
        )
        cells = load_cells_synthetic(config.resolution, config.year_min, config.year_max)
    else:
        logger.info(
            "loading from DB dsn=%s resolution=%s year_min=%d year_max=%d",
            redact_dsn(config.dsn),
            config.resolution,
            config.year_min,
            config.year_max,
        )
        cells = load_cells_from_db(
            config.dsn, config.resolution, config.year_min, config.year_max
        )

    labels = load_labels(config.labels_path)
    if labels:
        logger.info("loaded %d community labels from %s", len(labels), config.labels_path)

    payload = build_payload(
        cells,
        resolution=config.resolution,
        year_min=config.year_min,
        year_max=config.year_max,
        labels=labels,
    )
    serialize(payload, config.output)
    logger.info(
        "wrote %s (%d communities, %d years)",
        config.output,
        len(payload["communities"]),
        len(payload["years"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
