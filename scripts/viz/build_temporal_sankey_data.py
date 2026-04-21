#!/usr/bin/env python3
"""Build a temporal-community Sankey dataset from ``paper_metrics``.

Read-only tool that joins ``papers`` to ``paper_metrics`` on ``bibcode``,
buckets each paper into a decade (``year // 10 * 10``), and produces a
Sankey-compatible JSON file describing:

* **nodes**: one per ``(decade, community_id)`` with a paper count;
* **links**: one per ``(decade_d, c) -> (decade_{d+1}, c)`` transition for
  every community that persists across adjacent decades, weighted by the
  community size in the later decade.

The number of links is capped by ``--top-flows`` (default 500), keeping the
largest-valued transitions for rendering clarity.

Usage
-----

Smoke-test mode (no DB required) — generates a deterministic synthetic
dataset and prints a summary without writing::

    python scripts/viz/build_temporal_sankey_data.py --dry-run --synthetic

Production::

    python scripts/viz/build_temporal_sankey_data.py \\
        --resolution medium \\
        --top-flows 500 \\
        --output data/viz/sankey.json

Output schema::

    {
      "nodes": [
        {"id": "1990-3", "decade": 1990, "community_id": 3, "paper_count": 42},
        ...
      ],
      "links": [
        {"source": "1990-3", "target": "2000-3", "value": 51},
        ...
      ]
    }

The script is strictly read-only. DB-path mode uses a server-side named
cursor (``fetchmany``) so the entire result set never lands in memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_temporal_sankey_data")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


RESOLUTIONS: tuple[str, ...] = ("coarse", "medium", "fine")

# Allowlist mapping --resolution -> paper_metrics column name. Used as a
# defence-in-depth check before interpolating the column into SQL.
COMMUNITY_COLUMNS: dict[str, str] = {
    "coarse": "community_semantic_coarse",
    "medium": "community_semantic_medium",
    "fine": "community_semantic_fine",
}

DEFAULT_OUTPUT_REL = Path("data/viz/sankey.json")
DEFAULT_TOP_FLOWS = 500
DEFAULT_RESOLUTION = "medium"

# Server-side cursor batch size. 10k rows per fetch keeps peak RSS modest
# even on the full 32M-paper corpus.
_DB_FETCH_BATCH = 10_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class PaperRow(NamedTuple):
    """A single (bibcode, year, community_id) record fed to ``aggregate``."""

    bibcode: str
    year: int
    community_id: int


@dataclass(frozen=True)
class Node:
    """A Sankey node — one ``(decade, community_id)`` bucket."""

    id: str
    decade: int
    community_id: int
    paper_count: int


@dataclass(frozen=True)
class Link:
    """A Sankey link — a flow from one node to another."""

    source: str
    target: str
    value: int


@dataclass(frozen=True)
class SankeyData:
    """Top-level Sankey dataset. Nodes and links are immutable tuples."""

    nodes: tuple[Node, ...]
    links: tuple[Link, ...]


@dataclass(frozen=True)
class Config:
    """Resolved CLI configuration."""

    resolution: str
    top_flows: int
    output: Path
    dsn: str
    dry_run: bool
    synthetic: bool


# ---------------------------------------------------------------------------
# Pure helpers — decade bucketing & node identity
# ---------------------------------------------------------------------------


def decade_of(year: int) -> int:
    """Return the decade bucket for ``year``.

    Uses floor division: 1999 -> 1990, 2000 -> 2000, 2025 -> 2020.
    """
    return (year // 10) * 10


def node_id(decade: int, community_id: int) -> str:
    """Deterministic Sankey-node identifier.

    Format: ``"{decade}-{community_id}"`` — unique across ``(decade, community)``.
    """
    return f"{decade}-{community_id}"


# ---------------------------------------------------------------------------
# Aggregation — the pure, testable core
# ---------------------------------------------------------------------------


def aggregate(rows: Iterable[PaperRow], top_flows: int) -> SankeyData:
    """Bucket rows into (decade, community) nodes and compute decade-transition links.

    A *link* connects ``(d, c) -> (d+1, c)`` whenever the same community ``c``
    has at least one paper in both decade ``d`` and decade ``d+1``. The link
    value is the community's paper count in the later decade (the receiver).

    Links are sorted by value descending, with ``(source, target)`` as the
    deterministic tiebreaker, and truncated to ``top_flows``.

    Nodes in the returned dataset include every ``(decade, community)`` pair
    that has at least one paper — both nodes that participate in a kept link
    AND isolated nodes (no successor community) — so the Sankey renderer can
    still show terminal buckets.
    """
    if top_flows < 0:
        raise ValueError(f"top_flows must be non-negative, got {top_flows}")

    # Pass 1: count papers per (decade, community).
    bucket_counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        bucket_counts[(decade_of(row.year), row.community_id)] += 1

    # Index by decade for the transition pass. Sort communities for determinism.
    by_decade: dict[int, dict[int, int]] = {}
    for (decade, community_id), count in bucket_counts.items():
        by_decade.setdefault(decade, {})[community_id] = count

    # Pass 2: build candidate links for every (d, c) -> (d+1, c) where c
    # persists. Value = paper_count in the later decade.
    decades_sorted = sorted(by_decade.keys())
    candidates: list[Link] = []
    for i, d in enumerate(decades_sorted[:-1]):
        next_d = decades_sorted[i + 1]
        # Only emit a transition between *adjacent* decade buckets (d and
        # next_d). Gaps (e.g. no papers in 1990 but some in 1980 and 2000)
        # are intentionally skipped — the visual would be misleading.
        if next_d != d + 10:
            continue
        src_communities = by_decade[d]
        dst_communities = by_decade[next_d]
        persistent = sorted(set(src_communities).intersection(dst_communities))
        for c in persistent:
            candidates.append(
                Link(
                    source=node_id(d, c),
                    target=node_id(next_d, c),
                    value=dst_communities[c],
                )
            )

    # Sort by value desc, then by (source, target) asc for deterministic ties.
    candidates.sort(key=lambda lk: (-lk.value, lk.source, lk.target))
    kept_links = tuple(candidates[:top_flows])

    # Build nodes: every bucket that has papers. Sort by (decade, community_id)
    # for stable output.
    nodes = tuple(
        Node(
            id=node_id(decade, community_id),
            decade=decade,
            community_id=community_id,
            paper_count=count,
        )
        for (decade, community_id), count in sorted(bucket_counts.items())
    )

    return SankeyData(nodes=nodes, links=kept_links)


# ---------------------------------------------------------------------------
# Synthetic data — deterministic, no DB
# ---------------------------------------------------------------------------


def load_synthetic(n: int = 10_000, seed: int = 42) -> Iterator[PaperRow]:
    """Yield ``n`` deterministic synthetic papers.

    Community ids drawn uniformly from ``range(20)``; years drawn uniformly
    from ``[1990, 2025]``. Same ``(n, seed)`` always produces the same rows.
    """
    rng = random.Random(seed)
    for i in range(n):
        yield PaperRow(
            bibcode=f"synthetic-{i:06d}",
            year=rng.randint(1990, 2025),
            community_id=rng.randint(0, 19),
        )


# ---------------------------------------------------------------------------
# DB loader — server-side cursor, yields one row at a time
# ---------------------------------------------------------------------------


def load_from_db(dsn: str, resolution: str) -> Iterator[PaperRow]:
    """Stream ``PaperRow`` tuples from the database.

    Joins ``papers`` to ``paper_metrics`` on ``bibcode`` and filters rows with
    NULL year or NULL community_id. Uses a server-side named cursor so the
    result set is streamed in ``_DB_FETCH_BATCH``-sized chunks.
    """
    if resolution not in COMMUNITY_COLUMNS:
        raise ValueError(
            f"unknown resolution {resolution!r}; must be one of {RESOLUTIONS}"
        )
    column = COMMUNITY_COLUMNS[resolution]

    # Import psycopg lazily so tests and --synthetic don't require a live DB.
    import psycopg  # noqa: WPS433

    query = (
        "SELECT p.bibcode, p.year, pm."
        + column
        + " "
        + "FROM papers p "
        + "JOIN paper_metrics pm USING (bibcode) "
        + "WHERE p.year IS NOT NULL AND pm."
        + column
        + " IS NOT NULL"
    )

    logger.info("DB query: %s", query)
    with psycopg.connect(dsn) as conn:
        # Named cursors require an explicit transaction; default (autocommit
        # off) gives us one. Read-only — no COMMIT needed.
        with conn.cursor(name="sankey_cursor") as cur:
            cur.execute(query)
            while True:
                batch = cur.fetchmany(_DB_FETCH_BATCH)
                if not batch:
                    break
                for bibcode, year, community_id in batch:
                    # year is SMALLINT; normalise to int defensively.
                    yield PaperRow(
                        bibcode=str(bibcode),
                        year=int(year),
                        community_id=int(community_id),
                    )


# ---------------------------------------------------------------------------
# Serialization & validation
# ---------------------------------------------------------------------------


def _node_to_dict(n: Node) -> dict[str, object]:
    return {
        "id": n.id,
        "decade": n.decade,
        "community_id": n.community_id,
        "paper_count": n.paper_count,
    }


def _link_to_dict(lk: Link) -> dict[str, object]:
    return {"source": lk.source, "target": lk.target, "value": lk.value}


def serialize_to_json(sd: SankeyData, path: Path) -> None:
    """Write ``sd`` to ``path`` as pretty-printed JSON (UTF-8, trailing newline)."""
    payload = {
        "nodes": [_node_to_dict(n) for n in sd.nodes],
        "links": [_link_to_dict(lk) for lk in sd.links],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_sankey_data(obj: dict) -> SankeyData:
    """Type-check a raw dict against the Sankey schema and return a ``SankeyData``.

    Raises ``ValueError`` with a descriptive message on any structural or
    type mismatch. The returned dataclass is frozen.
    """
    _require(isinstance(obj, dict), "top-level must be a dict")
    _require("nodes" in obj, "missing required key 'nodes'")
    _require("links" in obj, "missing required key 'links'")
    _require(isinstance(obj["nodes"], list), "'nodes' must be a list")
    _require(isinstance(obj["links"], list), "'links' must be a list")

    nodes: list[Node] = []
    for i, raw_node in enumerate(obj["nodes"]):
        _require(isinstance(raw_node, dict), f"nodes[{i}] must be a dict")
        for key in ("id", "decade", "community_id", "paper_count"):
            _require(key in raw_node, f"nodes[{i}] missing key '{key}'")
        _require(isinstance(raw_node["id"], str), f"nodes[{i}].id must be str")
        _require(
            isinstance(raw_node["decade"], int) and not isinstance(raw_node["decade"], bool),
            f"nodes[{i}].decade must be int",
        )
        _require(
            isinstance(raw_node["community_id"], int)
            and not isinstance(raw_node["community_id"], bool),
            f"nodes[{i}].community_id must be int",
        )
        _require(
            isinstance(raw_node["paper_count"], int)
            and not isinstance(raw_node["paper_count"], bool),
            f"nodes[{i}].paper_count must be int",
        )
        nodes.append(
            Node(
                id=raw_node["id"],
                decade=raw_node["decade"],
                community_id=raw_node["community_id"],
                paper_count=raw_node["paper_count"],
            )
        )

    links: list[Link] = []
    for i, raw_link in enumerate(obj["links"]):
        _require(isinstance(raw_link, dict), f"links[{i}] must be a dict")
        for key in ("source", "target", "value"):
            _require(key in raw_link, f"links[{i}] missing key '{key}'")
        _require(isinstance(raw_link["source"], str), f"links[{i}].source must be str")
        _require(isinstance(raw_link["target"], str), f"links[{i}].target must be str")
        _require(
            isinstance(raw_link["value"], int) and not isinstance(raw_link["value"], bool),
            f"links[{i}].value must be int",
        )
        links.append(
            Link(
                source=raw_link["source"],
                target=raw_link["target"],
                value=raw_link["value"],
            )
        )

    return SankeyData(nodes=tuple(nodes), links=tuple(links))


# ---------------------------------------------------------------------------
# run() — orchestration (no IO write)
# ---------------------------------------------------------------------------


def run(config: Config) -> SankeyData:
    """Load rows, aggregate, and return the in-memory ``SankeyData``.

    Picks the synthetic generator or the DB loader based on ``config``. Does
    NOT write to disk — callers handle serialization so ``--dry-run`` is a
    single-branch concern.
    """
    if config.synthetic:
        logger.info("loading synthetic dataset (deterministic, no DB)")
        rows: Iterable[PaperRow] = load_synthetic()
    else:
        logger.info(
            "loading from DB dsn=%s resolution=%s",
            redact_dsn(config.dsn),
            config.resolution,
        )
        if is_production_dsn(config.dsn):
            # Read-only, so no --allow-prod gate — but log the notice for
            # operator awareness.
            logger.info("dsn appears to point at production (read-only query, proceeding)")
        rows = load_from_db(config.dsn, config.resolution)

    sd = aggregate(rows, top_flows=config.top_flows)
    logger.info(
        "aggregated: %d nodes, %d links (top_flows=%d)",
        len(sd.nodes),
        len(sd.links),
        config.top_flows,
    )
    return sd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a temporal-community Sankey dataset.",
    )
    parser.add_argument(
        "--resolution",
        choices=RESOLUTIONS,
        default=DEFAULT_RESOLUTION,
        help=f"Semantic-community resolution (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--top-flows",
        type=int,
        default=DEFAULT_TOP_FLOWS,
        help=f"Keep the top N flows by value (default: {DEFAULT_TOP_FLOWS}).",
    )
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
        "--dry-run",
        action="store_true",
        help="Do all work except writing the output file.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a deterministic synthetic dataset instead of querying the DB.",
    )
    return parser.parse_args(argv)


def _resolve_output_path(raw: str) -> Path:
    """Resolve the output path. Relative paths are anchored at repo root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        resolution=args.resolution,
        top_flows=int(args.top_flows),
        output=_resolve_output_path(args.output),
        dsn=args.dsn,
        dry_run=bool(args.dry_run),
        synthetic=bool(args.synthetic),
    )


def _summary_log(sd: SankeyData, config: Config) -> None:
    total_papers = sum(n.paper_count for n in sd.nodes)
    decades = sorted({n.decade for n in sd.nodes})
    communities = sorted({n.community_id for n in sd.nodes})
    logger.info(
        "summary: nodes=%d links=%d papers=%d decades=%s communities=%d "
        "dry_run=%s synthetic=%s output=%s",
        len(sd.nodes),
        len(sd.links),
        total_papers,
        decades,
        len(communities),
        config.dry_run,
        config.synthetic,
        config.output,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = _parse_args(argv)
    config = _config_from_args(args)

    if config.top_flows < 0:
        logger.error("--top-flows must be >= 0, got %d", config.top_flows)
        return 2

    sd = run(config)
    _summary_log(sd, config)

    if config.dry_run:
        logger.info("--dry-run set, skipping write of %s", config.output)
        return 0

    serialize_to_json(sd, config.output)
    logger.info("wrote %s", config.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
