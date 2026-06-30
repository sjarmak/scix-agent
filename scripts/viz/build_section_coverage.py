#!/usr/bin/env python3
"""Aggregate full-text section coverage for the V11 "Sections" viz page.

Read-only tool. Answers two questions about ``papers_fulltext``:

1. **Which canonical section roles are populated, and at what rate?**
   Every section heading is classified with
   :func:`scix.section_role.classify_section_role` into one of five roles
   (``background``, ``method``, ``result``, ``conclusion``, ``other``). For a
   random sample of full-text papers we record, per paper, the *set* of roles
   present and the section count, then report the fraction of papers carrying
   each role plus the mean/median sections-per-paper.

2. **How does full-text coverage break down by decade?**
   An exact ``papers_fulltext ⋈ papers`` join counts full-text papers per
   decade, alongside the total papers per decade, so the page can show both
   the absolute volume and the coverage fraction (only ~47% of the corpus has
   full text — the rest is metadata-only).

The role-presence pass samples (headings only — never the section bodies) so
the scan stays in the seconds-to-minutes range instead of detoasting 14.9M
full bodies. The decade pass is exact but touches only ``bibcode``/``year``.

Usage
-----

Smoke-test mode (no DB)::

    python scripts/viz/build_section_coverage.py --synthetic \\
        --output /tmp/section_coverage.json

Production (heavy — run under scix-batch)::

    scix-batch python scripts/viz/build_section_coverage.py \\
        --sample-size 40000 --output data/viz/section_coverage.json

Output JSON schema::

    {
      "corpus_papers": 32440901,
      "fulltext_papers": 14941487,
      "fulltext_fraction": 0.461,
      "sample_size": 40000,
      "mean_sections_per_paper": 35.8,
      "median_sections_per_paper": 21,
      "year_min": 1950,
      "year_max": 2024,
      "roles": [
        {"role": "background", "label": "Background / Intro",
         "present_pct": 0.799, "papers_with": 31967},
        ...                                   # 5 roles, display order
      ],
      "decades": [
        {"decade": 1990, "fulltext": 1234567, "total": 2345678,
         "fulltext_pct": 0.526},
        ...                                   # ascending by decade
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, redact_dsn  # noqa: E402
from scix.section_role import (  # noqa: E402
    ROLE_BACKGROUND,
    ROLE_CONCLUSION,
    ROLE_METHOD,
    ROLE_OTHER,
    ROLE_RESULT,
    classify_section_role,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_section_coverage")

# Display order + human labels for the five canonical roles. The page renders
# the bars in this order; "other" sits last because it is the catch-all.
ROLE_DISPLAY: tuple[tuple[str, str], ...] = (
    (ROLE_BACKGROUND, "Background / Intro"),
    (ROLE_METHOD, "Methods / Data"),
    (ROLE_RESULT, "Results"),
    (ROLE_CONCLUSION, "Discussion / Conclusion"),
    (ROLE_OTHER, "Other (abstract, refs, …)"),
)

# Corpus spans 1800-2026 but the early years are sparse and the trailing two
# are incomplete in our snapshot; default to a window where every decade bin
# holds a meaningful population.
DEFAULT_YEAR_MIN = 1950
DEFAULT_YEAR_MAX = 2024
DEFAULT_SAMPLE_SIZE = 40_000


@dataclass(frozen=True)
class PaperSections:
    """Per-paper roll-up: which roles appear, and how many sections total."""

    roles_present: frozenset[str]
    section_count: int


@dataclass(frozen=True)
class DecadeCoverage:
    """Full-text vs total paper counts for one decade bucket."""

    decade: int
    fulltext: int
    total: int


@dataclass(frozen=True)
class Config:
    dsn: str
    output: Path
    sample_size: int
    year_min: int
    year_max: int
    synthetic: bool


def summarize_paper(headings: Sequence[Optional[str]]) -> PaperSections:
    """Classify a paper's section headings into its set of present roles.

    ``headings`` is the ``$[*].heading`` projection of one paper's ``sections``
    array (entries may be ``None`` when a section carries no heading). The
    section count is the array length, independent of whether headings classify.
    """
    roles = {classify_section_role(h or "") for h in headings}
    return PaperSections(
        roles_present=frozenset(roles),
        section_count=len(headings),
    )


def load_samples_from_db(dsn: str, sample_size: int) -> list[PaperSections]:
    """Sample full-text papers and project just their section headings.

    Uses ``TABLESAMPLE SYSTEM`` (block sampling) so the planner reads a small
    random subset of pages rather than scanning all 14.9M rows. We pull a few
    percent of the table and cap with ``LIMIT`` to land near ``sample_size``.
    """
    import psycopg  # lazy so synthetic / tests don't need a live DB

    # SYSTEM sampling is page-granular; over-request the fraction so the
    # post-LIMIT count reliably reaches sample_size, then cap with LIMIT.
    # ~14.9M rows -> a 1% block sample is ~150k candidate rows.
    fraction_pct = min(100.0, max(0.5, (sample_size / 14_900_000) * 100.0 * 4.0))
    sql = (
        "SELECT jsonb_path_query_array(sections, '$[*].heading')\n"
        f"FROM papers_fulltext TABLESAMPLE SYSTEM ({fraction_pct})\n"
        "WHERE sections IS NOT NULL\n"
        "  AND jsonb_array_length(sections) > 0\n"
        "LIMIT %s"
    )
    out: list[PaperSections] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '20min'")
            cur.execute(sql, (sample_size,))
            for (headings_json,) in cur:
                headings = headings_json if isinstance(headings_json, list) else []
                out.append(summarize_paper(headings))
    return out


def load_decades_from_db(dsn: str, year_min: int, year_max: int) -> list[DecadeCoverage]:
    """Exact full-text-vs-total paper counts per decade.

    Two indexed aggregations: the full-text join (``papers_fulltext ⋈ papers``)
    and the corpus-wide ``papers`` count. Neither touches the ``sections``
    JSONB, so both stay cheap relative to the sampling pass.
    """
    import psycopg

    fulltext_sql = (
        "SELECT (p.year / 10) * 10 AS decade, COUNT(*) AS n\n"
        "FROM papers_fulltext f\n"
        "JOIN papers p USING (bibcode)\n"
        "WHERE p.year BETWEEN %s AND %s\n"
        "GROUP BY decade"
    )
    total_sql = (
        "SELECT (year / 10) * 10 AS decade, COUNT(*) AS n\n"
        "FROM papers\n"
        "WHERE year BETWEEN %s AND %s\n"
        "GROUP BY decade"
    )
    fulltext: dict[int, int] = {}
    total: dict[int, int] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '20min'")
            cur.execute(fulltext_sql, (year_min, year_max))
            for decade, n in cur.fetchall():
                fulltext[int(decade)] = int(n)
            cur.execute(total_sql, (year_min, year_max))
            for decade, n in cur.fetchall():
                total[int(decade)] = int(n)
    return _merge_decades(fulltext, total)


def _merge_decades(fulltext: dict[int, int], total: dict[int, int]) -> list[DecadeCoverage]:
    decades = sorted(set(fulltext) | set(total))
    return [
        DecadeCoverage(
            decade=d,
            fulltext=fulltext.get(d, 0),
            total=total.get(d, 0),
        )
        for d in decades
    ]


def load_corpus_counts_from_db(dsn: str) -> tuple[int, int]:
    """Return ``(corpus_papers, fulltext_papers)`` as exact counts."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '20min'")
            cur.execute("SELECT COUNT(*) FROM papers")
            corpus = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM papers_fulltext")
            fulltext = int(cur.fetchone()[0])
    return corpus, fulltext


def synthesize(
    sample_size: int, year_min: int, year_max: int
) -> tuple[list[PaperSections], list[DecadeCoverage], int, int]:
    """Deterministic offline data so the page renders without a DB."""
    import random

    rng = random.Random(42)
    # Per-role probability a sampled paper carries that role. Background/method/
    # result/other are near-universal in real full text; conclusion is rarer.
    role_prob = {
        ROLE_BACKGROUND: 0.82,
        ROLE_METHOD: 0.74,
        ROLE_RESULT: 0.69,
        ROLE_CONCLUSION: 0.55,
        ROLE_OTHER: 0.95,
    }
    samples: list[PaperSections] = []
    for _ in range(sample_size):
        present = {r for r, p in role_prob.items() if rng.random() < p}
        present.add(ROLE_OTHER)  # abstract/refs always classify as other
        samples.append(
            PaperSections(
                roles_present=frozenset(present),
                section_count=rng.randint(3, 28),
            )
        )

    decades: list[DecadeCoverage] = []
    decade_start = (year_min // 10) * 10
    for decade in range(decade_start, year_max + 1, 10):
        total = 200_000 + rng.randint(0, 3_000_000)
        fulltext = int(total * rng.uniform(0.25, 0.7))
        decades.append(DecadeCoverage(decade=decade, fulltext=fulltext, total=total))

    corpus = sum(d.total for d in decades)
    fulltext_total = sum(d.fulltext for d in decades)
    return samples, decades, corpus, fulltext_total


def build_payload(
    samples: Sequence[PaperSections],
    decades: Sequence[DecadeCoverage],
    corpus_papers: int,
    fulltext_papers: int,
    year_min: int,
    year_max: int,
) -> dict:
    """Reshape sampled papers + decade counts into the page's JSON payload."""
    n = len(samples)
    role_counts = {role: 0 for role, _ in ROLE_DISPLAY}
    section_counts: list[int] = []
    for paper in samples:
        section_counts.append(paper.section_count)
        for role in paper.roles_present:
            if role in role_counts:
                role_counts[role] += 1

    roles = [
        {
            "role": role,
            "label": label,
            "present_pct": (role_counts[role] / n) if n else 0.0,
            "papers_with": role_counts[role],
        }
        for role, label in ROLE_DISPLAY
    ]

    decade_rows = [
        {
            "decade": d.decade,
            "fulltext": d.fulltext,
            "total": d.total,
            "fulltext_pct": (d.fulltext / d.total) if d.total else 0.0,
        }
        for d in decades
    ]

    return {
        "corpus_papers": corpus_papers,
        "fulltext_papers": fulltext_papers,
        "fulltext_fraction": (fulltext_papers / corpus_papers) if corpus_papers else 0.0,
        "sample_size": n,
        "mean_sections_per_paper": (statistics.fmean(section_counts) if section_counts else 0.0),
        "median_sections_per_paper": (
            int(statistics.median(section_counts)) if section_counts else 0
        ),
        "year_min": year_min,
        "year_max": year_max,
        "roles": roles,
        "decades": decade_rows,
    }


def serialize(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Papers to sample for role presence (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=DEFAULT_YEAR_MIN,
        help=f"Lowest year for the decade breakdown (default: {DEFAULT_YEAR_MIN}).",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=DEFAULT_YEAR_MAX,
        help=f"Highest year for the decade breakdown (default: {DEFAULT_YEAR_MAX}).",
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: scix.db.DEFAULT_DSN).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/viz/section_coverage.json).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate deterministic synthetic data instead of querying the DB.",
    )
    return parser.parse_args(argv)


def _resolve_output(raw: Optional[str]) -> Path:
    if raw is not None:
        p = Path(raw)
        return p if p.is_absolute() else _REPO_ROOT / p
    return _REPO_ROOT / "data" / "viz" / "section_coverage.json"


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        dsn=args.dsn,
        output=_resolve_output(args.output),
        sample_size=int(args.sample_size),
        year_min=int(args.year_min),
        year_max=int(args.year_max),
        synthetic=bool(args.synthetic),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)
    if config.year_min > config.year_max:
        logger.error("year_min %d > year_max %d", config.year_min, config.year_max)
        return 2

    if config.synthetic:
        logger.info("synthetic mode: sample_size=%d", config.sample_size)
        samples, decades, corpus, fulltext = synthesize(
            config.sample_size, config.year_min, config.year_max
        )
    else:
        logger.info(
            "loading from DB dsn=%s sample_size=%d years=%d-%d",
            redact_dsn(config.dsn),
            config.sample_size,
            config.year_min,
            config.year_max,
        )
        corpus, fulltext = load_corpus_counts_from_db(config.dsn)
        logger.info("corpus=%d fulltext=%d", corpus, fulltext)
        samples = load_samples_from_db(config.dsn, config.sample_size)
        logger.info("sampled %d full-text papers", len(samples))
        decades = load_decades_from_db(config.dsn, config.year_min, config.year_max)
        logger.info("aggregated %d decade buckets", len(decades))

    payload = build_payload(
        samples,
        decades,
        corpus_papers=corpus,
        fulltext_papers=fulltext,
        year_min=config.year_min,
        year_max=config.year_max,
    )
    serialize(payload, config.output)
    logger.info(
        "wrote %s (%d roles, %d decades, sample=%d)",
        config.output,
        len(payload["roles"]),
        len(payload["decades"]),
        payload["sample_size"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
