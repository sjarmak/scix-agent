#!/usr/bin/env python3
"""Audit the tier-1 keyword-match links produced by ``link_tier1.py``.

Draws a stratified sample across (source, arxiv_class) buckets, writes the
sample to ``build-artifacts/tier1_audit.md`` with placeholder labels, and
prints a Wilson 95% CI stub for the precision number. The real LLM-judge
integration lands in u11; until then, every row is labeled ``"unlabeled"``.

Usage::

    python scripts/audit_tier1.py --db-url "dbname=scix"
    python scripts/audit_tier1.py --sample-size 200 --output build-artifacts/tier1_audit.md
"""

from __future__ import annotations

import argparse
import logging
import math
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_OUTPUT = pathlib.Path("build-artifacts/tier1_audit.md")
LABEL_PLACEHOLDER = "unlabeled"
RNG_SEED = 42

# z for 95% two-sided normal-approx: 1.959963984540054 (commonly 1.96)
_Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Wilson 95% CI (tested in tests/test_tier1.py)
# ---------------------------------------------------------------------------


def wilson_95_ci(successes: int, total: int) -> tuple[float, float]:
    """Return the Wilson 95% confidence interval for a binomial proportion.

    >>> lo, hi = wilson_95_ci(95, 100)
    >>> round(lo, 3), round(hi, 3)
    (0.887, 0.978)

    Degenerate cases:
    - ``total == 0`` returns ``(0.0, 1.0)`` (no information).
    - ``successes == 0`` or ``successes == total`` produce one-sided intervals
      via the standard Wilson formula (no clamping beyond [0, 1]).
    """
    if total <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be in [0, {total}], got {successes}")

    n = float(total)
    p = successes / n
    z = _Z_95
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom

    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Data access + stratification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier1Row:
    bibcode: str
    entity_id: int
    canonical_name: str
    source: str
    arxiv_class: str  # empty string when missing (markdown-friendly)


_FETCH_SQL = """
SELECT
    de.bibcode                              AS bibcode,
    de.entity_id                            AS entity_id,
    e.canonical_name                        AS canonical_name,
    COALESCE(e.source, '')                  AS source,
    COALESCE(
        CASE WHEN array_length(p.arxiv_class, 1) > 0
             THEN p.arxiv_class[1]
             ELSE NULL END,
        ''
    )                                       AS arxiv_class
FROM document_entities de
JOIN entities e ON e.id = de.entity_id
JOIN papers   p ON p.bibcode = de.bibcode
WHERE de.tier = 1
  AND de.link_type = 'keyword_match'
"""


def fetch_tier1_rows(conn: psycopg.Connection) -> list[Tier1Row]:
    """Pull all tier-1 keyword-match rows joined with paper/entity metadata."""
    with conn.cursor() as cur:
        cur.execute(_FETCH_SQL)
        return [
            Tier1Row(
                bibcode=r[0],
                entity_id=r[1],
                canonical_name=r[2],
                source=r[3],
                arxiv_class=r[4],
            )
            for r in cur.fetchall()
        ]


def _bucket_key(row: Tier1Row, use_arxiv_class: bool) -> tuple[str, ...]:
    if use_arxiv_class:
        return (row.source, row.arxiv_class)
    return (row.source,)


def stratified_sample(
    rows: Sequence[Tier1Row],
    sample_size: int,
    seed: int = RNG_SEED,
) -> list[Tier1Row]:
    """Return a stratified sample of up to ``sample_size`` rows.

    Stratifies across ``(source, arxiv_class)``. If only a single distinct
    ``source`` is present, degrades to stratification on ``source`` alone.
    If the total population is at or below ``sample_size``, returns all rows
    (shuffled deterministically).
    """
    if not rows:
        return []
    if sample_size <= 0:
        return []

    distinct_sources = {r.source for r in rows}
    use_arxiv_class = len(distinct_sources) > 1 or any(r.arxiv_class for r in rows)

    rng = random.Random(seed)

    if len(rows) <= sample_size:
        out = list(rows)
        rng.shuffle(out)
        return out

    # Group by bucket
    buckets: dict[tuple[str, ...], list[Tier1Row]] = {}
    for r in rows:
        buckets.setdefault(_bucket_key(r, use_arxiv_class), []).append(r)

    total = len(rows)
    # Proportional allocation with ceil so small buckets always get ≥1.
    allocations: dict[tuple[str, ...], int] = {}
    for key, bucket in buckets.items():
        quota = math.ceil(len(bucket) / total * sample_size)
        allocations[key] = min(quota, len(bucket))

    sampled: list[Tier1Row] = []
    for key, bucket in buckets.items():
        take = allocations[key]
        rng.shuffle(bucket)
        sampled.extend(bucket[:take])

    rng.shuffle(sampled)
    return sampled[:sample_size]


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


_MD_HEADER = (
    "| bibcode | entity_id | canonical_name | source | arxiv_class | label_placeholder |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
)


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    sample: Iterable[Tier1Row],
    total_population: int,
    successes: int = 0,
    example_ci: tuple[int, int] = (95, 100),
) -> str:
    """Render the audit markdown.

    ``successes`` is a placeholder count used for the (stub) precision CI over
    the sample; real labels arrive in u11.  ``example_ci`` also renders a
    worked-example Wilson CI to show the function is wired up.
    """
    lines: list[str] = []
    lines.append("# Tier-1 keyword-match audit\n")
    lines.append("")
    lines.append(f"- Total tier-1 links in corpus: **{total_population}**")
    sample_list = list(sample)
    lines.append(f"- Sample drawn: **{len(sample_list)}**")
    lines.append(f"- Label placeholder: `{LABEL_PLACEHOLDER}` (real LLM-judge audit in u11)")
    lines.append("")

    lo, hi = wilson_95_ci(successes, len(sample_list))
    lines.append("## Precision (placeholder)\n")
    lines.append(
        f"- Wilson 95% CI over sample "
        f"({successes}/{len(sample_list)}): **[{lo:.3f}, {hi:.3f}]**"
    )
    ex_lo, ex_hi = wilson_95_ci(example_ci[0], example_ci[1])
    lines.append(
        f"- Worked example `wilson_95_ci({example_ci[0]}, {example_ci[1]})` → "
        f"**[{ex_lo:.3f}, {ex_hi:.3f}]**"
    )
    lines.append("")

    lines.append("## Sample\n")
    lines.append(_MD_HEADER.rstrip())
    for row in sample_list:
        lines.append(
            "| {bib} | {eid} | {name} | {src} | {ax} | {lbl} |".format(
                bib=_md_escape(row.bibcode),
                eid=row.entity_id,
                name=_md_escape(row.canonical_name),
                src=_md_escape(row.source),
                ax=_md_escape(row.arxiv_class),
                lbl=LABEL_PLACEHOLDER,
            )
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(
    conn: psycopg.Connection,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_path: pathlib.Path = DEFAULT_OUTPUT,
) -> pathlib.Path:
    """Fetch tier-1 rows, stratify-sample, and write the audit markdown."""
    rows = fetch_tier1_rows(conn)
    sample = stratified_sample(rows, sample_size)
    md = render_markdown(sample, total_population=len(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    logger.info(
        "Wrote tier-1 audit (%d sampled from %d) to %s", len(sample), len(rows), output_path
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db_url)
    try:
        path = run_audit(
            conn,
            sample_size=args.sample_size,
            output_path=args.output,
        )
    finally:
        conn.close()

    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
