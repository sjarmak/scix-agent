#!/usr/bin/env python3
"""Full-text coverage-bias report with KL-divergence per facet (PRD M1).

Extends ``scripts/coverage_bias_analysis.py`` (which already computes counts
and percentages per facet). This script ADDS:

  * Per-facet KL-divergence of the full-text distribution (P) against the
    corpus prior (Q), so agents can quantify how skewed each facet is.
  * Two new facets: bibstem (top-N journals via the unnested array column)
    and community_semantic_medium (joined via paper_metrics; gracefully
    skipped if the column does not exist).
  * Machine-readable JSON output at ``results/full_text_coverage_bias.json``.
  * An "Agent guidance: safe vs unsafe queries on the full-text cohort"
    section appended (or replaced in place) inside
    ``docs/full_text_coverage_analysis.md`` between dedicated HTML markers
    so re-runs do not duplicate the section.

The script is read-only — it issues SELECT statements only and never writes
to the database. It still hits prod (32M papers, several LATERAL unnest
GROUP BYs) so the production re-run path should be wrapped with
``scix-batch`` to keep the working-set footprint inside a transient
systemd scope and avoid colliding with the gascity supervisor's oomd
budget:

    scix-batch python scripts/report_full_text_coverage_bias.py \\
        --json-out results/full_text_coverage_bias.json

For local development or schema-shape checks, ``--dry-run`` synthesises a
deterministic toy corpus (no DB connection required) and writes an
example JSON document conforming to the production schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make sibling scripts/ importable so we can reuse coverage_bias_analysis.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))

from coverage_bias_analysis import (  # noqa: E402  (sys.path tweak above)
    DistributionRow,
    get_arxiv_distribution,
    get_citation_distribution,
    get_corpus_summary,
    get_year_distribution,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KL-divergence helper (pure-Python, numerically stable)
# ---------------------------------------------------------------------------


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """Return D_KL(P || Q) in nats.

    Definition: ``D_KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))``.

    Both inputs are normalised to sum to 1 (ignoring length-mismatch is a
    bug — caller must align the supports first). To keep ``log()`` safe
    for empty buckets, we add ``eps`` to every entry before normalising.
    Returns ``float('inf')`` only if ``Q`` contains a true zero in a
    position where ``P`` has mass AND ``eps == 0``.

    The implementation is intentionally pure-Python so the helper is
    importable in environments without scipy (the test suite uses it
    directly). Equivalent to ``scipy.stats.entropy(p, q)`` for the
    smoothed inputs.
    """
    if len(p) != len(q):
        raise ValueError(
            f"kl_divergence: length mismatch — len(P)={len(p)}, len(Q)={len(q)}"
        )
    if len(p) == 0:
        return 0.0

    # Smooth, then normalise so each is a probability distribution.
    p_smoothed = [max(float(x), 0.0) + eps for x in p]
    q_smoothed = [max(float(x), 0.0) + eps for x in q]
    p_total = sum(p_smoothed)
    q_total = sum(q_smoothed)
    if p_total <= 0.0 or q_total <= 0.0:
        return 0.0

    p_norm = [x / p_total for x in p_smoothed]
    q_norm = [x / q_total for x in q_smoothed]

    total = 0.0
    for pi, qi in zip(p_norm, q_norm):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        total += pi * math.log(pi / qi)
    return total


def rows_to_distributions(
    rows: Sequence[DistributionRow],
) -> tuple[list[float], list[float]]:
    """Project a list of DistributionRows onto aligned (P, Q) probability vectors.

    P[i] = with_body_i / sum(with_body); Q[i] = total_i / sum(total).
    Returns the unnormalised counts as floats — caller passes them to
    ``kl_divergence`` which handles smoothing + normalisation.
    """
    p = [float(r.with_body) for r in rows]
    q = [float(r.total) for r in rows]
    return p, q


# ---------------------------------------------------------------------------
# New facet collectors (bibstem + community_semantic_medium)
# ---------------------------------------------------------------------------


def get_bibstem_distribution(conn: Any, limit: int = 20) -> list[DistributionRow]:
    """Top-``limit`` bibstems by paper count, full-text vs abstract-only.

    ``papers.bibstem`` is a TEXT[] (per ``\\d papers``) so we LATERAL-unnest
    the same way ``coverage_bias_analysis.get_arxiv_distribution`` does.
    A paper with multiple bibstem entries is counted once per entry.
    """
    query = f"""
        SELECT
            stem,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers, LATERAL unnest(bibstem) AS stem
        WHERE bibstem IS NOT NULL
        GROUP BY stem
        ORDER BY total DESC
        LIMIT {int(limit)}
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def _semantic_medium_column_exists(conn: Any) -> bool:
    """True iff ``paper_metrics.community_semantic_medium`` exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name   = 'paper_metrics'
               AND column_name  = 'community_semantic_medium'
            """
        )
        return cur.fetchone() is not None


def get_community_semantic_distribution(
    conn: Any, limit: int = 20
) -> list[DistributionRow] | None:
    """Top-``limit`` semantic-medium communities by paper count.

    Returns ``None`` when ``paper_metrics.community_semantic_medium`` is
    absent (e.g. the semantic-communities migration has not been applied)
    so the caller can skip the facet gracefully.
    """
    if not _semantic_medium_column_exists(conn):
        logger.info(
            "paper_metrics.community_semantic_medium not present — "
            "skipping semantic facet"
        )
        return None

    query = f"""
        SELECT
            pm.community_semantic_medium AS cluster_id,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE p.body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE p.body IS NULL) AS without_body
        FROM papers p
        JOIN paper_metrics pm ON pm.bibcode = p.bibcode
        WHERE pm.community_semantic_medium IS NOT NULL
        GROUP BY pm.community_semantic_medium
        ORDER BY total DESC
        LIMIT {int(limit)}
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# JSON payload assembly
# ---------------------------------------------------------------------------


def build_facet_payload(rows: Sequence[DistributionRow]) -> dict[str, Any]:
    """Convert a facet's DistributionRows into JSON-shaped payload."""
    p, q = rows_to_distributions(rows)
    kl = kl_divergence(p, q)

    p_total = sum(p) or 1.0
    q_total = sum(q) or 1.0

    facet_rows: list[dict[str, Any]] = []
    for row in rows:
        p_i = float(row.with_body) / p_total
        q_i = float(row.total) / q_total
        ratio = (p_i / q_i) if q_i > 0 else None
        facet_rows.append(
            {
                "label": row.label,
                "total": int(row.total),
                "with_body": int(row.with_body),
                "without_body": int(row.without_body),
                "pct_with_body": float(row.pct_with_body),
                "p_fulltext": round(p_i, 6),
                "q_corpus": round(q_i, 6),
                "ratio_p_over_q": round(ratio, 4) if ratio is not None else None,
            }
        )

    return {
        "kl_divergence_vs_corpus_prior": round(float(kl), 6),
        "row_count": len(rows),
        "rows": facet_rows,
    }


def synthetic_facets() -> dict[str, list[DistributionRow]]:
    """Deterministic toy distributions used by ``--dry-run``.

    Numbers are illustrative only — they roughly mirror the real shape of
    the corpus (modern arxiv-class papers nearly all have body text;
    historic years and conference abstracts do not). The schema of the
    JSON payload is identical to the production run, so the dry-run
    output is suitable for AC2 verification.
    """
    return {
        "arxiv_class": [
            DistributionRow("cs.LG", 200_000, 198_000, 2_000, 99.0),
            DistributionRow("hep-ph", 185_000, 182_000, 3_000, 98.4),
            DistributionRow("astro-ph.SR", 64_000, 63_500, 500, 99.2),
            DistributionRow("math.CO", 70_000, 68_500, 1_500, 97.9),
        ],
        "year": [
            DistributionRow("1900", 8_000, 2_200, 5_800, 27.5),
            DistributionRow("1950", 29_000, 3_900, 25_100, 13.4),
            DistributionRow("2000", 442_000, 119_000, 323_000, 26.9),
            DistributionRow("2020", 1_042_000, 730_000, 312_000, 70.1),
            DistributionRow("2025", 430_000, 332_000, 98_000, 77.2),
        ],
        "citation_bucket": [
            DistributionRow("0", 15_000_000, 5_700_000, 9_300_000, 38.0),
            DistributionRow("1-5", 8_100_000, 4_000_000, 4_100_000, 49.4),
            DistributionRow("6-20", 5_600_000, 3_000_000, 2_600_000, 53.6),
            DistributionRow("21-100", 3_200_000, 1_900_000, 1_300_000, 59.4),
            DistributionRow("101-500", 460_000, 290_000, 170_000, 63.0),
            DistributionRow("500+", 35_000, 22_000, 13_000, 62.9),
        ],
        "bibstem": [
            DistributionRow("ApJ", 138_000, 137_500, 500, 99.6),
            DistributionRow("PhRvB", 218_000, 218_000, 0, 100.0),
            DistributionRow("Natur", 411_000, 77_000, 334_000, 18.7),
            DistributionRow("AGUFM", 410_000, 30, 409_970, 0.0),
            DistributionRow("APS..MAR", 210_000, 0, 210_000, 0.0),
        ],
        "community_semantic_medium": [
            DistributionRow("12", 180_000, 175_000, 5_000, 97.2),
            DistributionRow("47", 95_000, 30_000, 65_000, 31.6),
            DistributionRow("103", 60_000, 200, 59_800, 0.3),
        ],
    }


def build_payload(
    corpus_total: int,
    fulltext_total: int,
    facet_rows_by_name: dict[str, Sequence[DistributionRow] | None],
    dsn_redacted: str,
) -> dict[str, Any]:
    """Assemble the full JSON payload from facet row collections."""
    facets: dict[str, Any] = {}
    for name, rows in facet_rows_by_name.items():
        if rows is None:
            continue
        facets[name] = build_facet_payload(rows)

    fulltext_pct = (
        round(100.0 * fulltext_total / corpus_total, 4) if corpus_total > 0 else 0.0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dsn_redacted": dsn_redacted,
        "corpus_total": int(corpus_total),
        "fulltext_total": int(fulltext_total),
        "fulltext_pct": fulltext_pct,
        "kl_divergence_basis": (
            "P=full-text distribution (with_body counts); "
            "Q=corpus prior (total counts); "
            "Laplace smoothing eps=1e-12; units=nats"
        ),
        "facets": facets,
    }


# ---------------------------------------------------------------------------
# Agent-guidance section (markdown, in-place upsert)
# ---------------------------------------------------------------------------


_GUIDANCE_START = "<!-- agent-guidance:start -->"
_GUIDANCE_END = "<!-- agent-guidance:end -->"
_GUIDANCE_HEADING = "## Agent guidance: safe vs unsafe queries on the full-text cohort"


def _kl_summary_line(facet_name: str, payload: dict[str, Any]) -> str:
    """One-line summary used in the docs section header."""
    facet = payload["facets"].get(facet_name)
    if facet is None:
        return f"- {facet_name}: not measured"
    return (
        f"- {facet_name}: KL = {facet['kl_divergence_vs_corpus_prior']:.4f} nats "
        f"({facet['row_count']} rows)"
    )


def _top_skewed(
    payload: dict[str, Any], facet_name: str, *, low_pct: bool, n: int = 1
) -> list[dict[str, Any]]:
    """Pull the most-biased rows from a facet (highest or lowest pct_with_body)."""
    facet = payload["facets"].get(facet_name)
    if facet is None:
        return []
    rows = list(facet["rows"])
    rows.sort(key=lambda r: r["pct_with_body"], reverse=not low_pct)
    return rows[:n]


def _format_safe_examples(payload: dict[str, Any]) -> list[str]:
    """3+ safe-query examples grounded in the highest-coverage facet rows."""
    candidates: list[str] = []

    high_arxiv = _top_skewed(payload, "arxiv_class", low_pct=False, n=1)
    if high_arxiv:
        row = high_arxiv[0]
        candidates.append(
            f"1. arxiv_class={row['label']!r} queries — "
            f"{row['pct_with_body']:.1f}% full-text coverage; restricting to "
            f"the body-bearing cohort barely changes the population (ratio "
            f"P/Q = {row['ratio_p_over_q']})."
        )

    high_year = _top_skewed(payload, "year", low_pct=False, n=1)
    if high_year:
        row = high_year[0]
        candidates.append(
            f"2. year={row['label']} queries — "
            f"{row['pct_with_body']:.1f}% full-text coverage; safe to filter "
            f"on body IS NOT NULL without losing representativeness."
        )

    high_cite = _top_skewed(payload, "citation_bucket", low_pct=False, n=1)
    if high_cite:
        row = high_cite[0]
        candidates.append(
            f"3. citation_count bucket {row['label']!r} — "
            f"{row['pct_with_body']:.1f}% full-text coverage; high-impact "
            f"papers are over-represented in the full-text cohort but the "
            f"absolute coverage is high enough to be safe."
        )

    while len(candidates) < 3:
        candidates.append(
            f"{len(candidates) + 1}. Modern, English-language journal articles in "
            f"the full corpus (~46% body coverage overall) — safe baseline."
        )
    return candidates


def _format_unsafe_examples(payload: dict[str, Any]) -> list[str]:
    """3+ unsafe-query examples grounded in the lowest-coverage facet rows."""
    candidates: list[str] = []

    low_year = _top_skewed(payload, "year", low_pct=True, n=1)
    if low_year:
        row = low_year[0]
        candidates.append(
            f"1. year={row['label']} (and earlier) — "
            f"{row['pct_with_body']:.1f}% full-text coverage; the body-bearing "
            f"subset is a non-random sample of the historical literature, so "
            f"restricting to it under-counts pre-modern work."
        )

    low_bib = _top_skewed(payload, "bibstem", low_pct=True, n=1)
    if low_bib:
        row = low_bib[0]
        candidates.append(
            f"2. bibstem={row['label']!r} (e.g. conference-abstract series) — "
            f"only {row['pct_with_body']:.1f}% have body text; full-text "
            f"filtering would silently drop the entire venue."
        )

    low_cite = _top_skewed(payload, "citation_bucket", low_pct=True, n=1)
    if low_cite:
        row = low_cite[0]
        candidates.append(
            f"3. citation_count bucket {row['label']!r} (uncited / lightly "
            f"cited papers) — only {row['pct_with_body']:.1f}% have body text; "
            f"long-tail discovery queries should NOT restrict to the full-text "
            f"cohort."
        )

    while len(candidates) < 3:
        candidates.append(
            f"{len(candidates) + 1}. Pre-1950 historical literature — full-text "
            f"coverage is below 15% across most years; filtering loses the tail."
        )
    return candidates


def render_agent_guidance(payload: dict[str, Any], json_out: Path | None) -> str:
    """Return the markdown for the agent-guidance section."""
    lines: list[str] = []
    lines.append(_GUIDANCE_START)
    lines.append("")
    lines.append(_GUIDANCE_HEADING)
    lines.append("")
    lines.append(
        f"_Generated {payload['generated_at']} by "
        f"`scripts/report_full_text_coverage_bias.py` against "
        f"`{payload['dsn_redacted']}`._"
    )
    lines.append("")
    lines.append(
        f"Corpus total: **{payload['corpus_total']:,}** papers; "
        f"full-text cohort: **{payload['fulltext_total']:,}** "
        f"({payload['fulltext_pct']:.2f}%)."
    )
    if json_out is not None:
        lines.append("")
        lines.append(
            f"Full machine-readable distribution (per-row P, Q, KL contribution) "
            f"is at `{json_out}`."
        )
    lines.append("")
    lines.append("### KL-divergence per facet (P=full-text, Q=corpus prior)")
    lines.append("")
    for facet_name in (
        "arxiv_class",
        "year",
        "citation_bucket",
        "bibstem",
        "community_semantic_medium",
    ):
        lines.append(_kl_summary_line(facet_name, payload))
    lines.append("")
    lines.append("### Safe queries to restrict to the full-text cohort")
    lines.append("")
    lines.extend(_format_safe_examples(payload))
    lines.append("")
    lines.append("### Unsafe queries (filtering to full-text would bias the result)")
    lines.append("")
    lines.extend(_format_unsafe_examples(payload))
    lines.append("")
    lines.append(
        "### Operational rule of thumb"
    )
    lines.append("")
    lines.append(
        "Any MCP tool path that consumes body-text (NER, negative-results, "
        "claim extraction) MUST attach a `coverage_note` referencing this "
        "report so downstream agents do not over-generalise from the "
        "biased subset."
    )
    lines.append("")
    lines.append(_GUIDANCE_END)
    return "\n".join(lines) + "\n"


def upsert_agent_guidance_section(docs_path: Path, section: str) -> None:
    """Replace (or append) the agent-guidance section in the docs file."""
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Expected docs file at {docs_path} — refusing to create it from "
            f"scratch (this script extends, never recreates, the existing "
            f"counts-based report)."
        )

    text = docs_path.read_text(encoding="utf-8")
    if _GUIDANCE_START in text and _GUIDANCE_END in text:
        before, _, rest = text.partition(_GUIDANCE_START)
        _, _, after = rest.partition(_GUIDANCE_END)
        new_text = before.rstrip() + "\n\n" + section + after.lstrip()
    else:
        if not text.endswith("\n"):
            text += "\n"
        new_text = text + "\n" + section

    docs_path.write_text(new_text, encoding="utf-8")
    logger.info("Upserted agent-guidance section in %s", docs_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def collect_from_db(dsn: str | None) -> tuple[dict[str, Any], dict[str, Sequence[DistributionRow] | None]]:
    """Connect, query, and return (corpus_meta, facet_rows_by_name).

    corpus_meta carries ``corpus_total``, ``fulltext_total``, ``dsn_redacted``.
    """
    # Local import so --dry-run does not need psycopg / DB available.
    from scix.db import DEFAULT_DSN, get_connection, redact_dsn

    effective_dsn = dsn or DEFAULT_DSN
    redacted = redact_dsn(effective_dsn)
    logger.info("Connecting to %s", redacted)

    conn = get_connection(dsn)
    try:
        summary = get_corpus_summary(conn)
        corpus_total = summary.total_papers
        fulltext_total = summary.total_with_body

        facets: dict[str, Sequence[DistributionRow] | None] = {}
        logger.info("Collecting arxiv_class facet ...")
        facets["arxiv_class"] = get_arxiv_distribution(conn)
        logger.info("Collecting year facet ...")
        facets["year"] = get_year_distribution(conn)
        logger.info("Collecting citation_bucket facet ...")
        facets["citation_bucket"] = get_citation_distribution(conn)
        logger.info("Collecting bibstem facet ...")
        facets["bibstem"] = get_bibstem_distribution(conn)
        logger.info("Collecting community_semantic_medium facet ...")
        facets["community_semantic_medium"] = get_community_semantic_distribution(conn)
    finally:
        conn.close()

    return (
        {
            "corpus_total": corpus_total,
            "fulltext_total": fulltext_total,
            "dsn_redacted": redacted,
        },
        facets,
    )


def collect_synthetic() -> tuple[dict[str, Any], dict[str, Sequence[DistributionRow] | None]]:
    """Build a deterministic toy payload for ``--dry-run``."""
    facets: dict[str, Sequence[DistributionRow] | None] = dict(synthetic_facets())
    # Derive corpus totals from the citation_bucket facet, which is the
    # only one whose row totals sum to the whole corpus.
    citation_rows = facets["citation_bucket"] or []
    corpus_total = sum(r.total for r in citation_rows)
    fulltext_total = sum(r.with_body for r in citation_rows)
    return (
        {
            "corpus_total": corpus_total,
            "fulltext_total": fulltext_total,
            "dsn_redacted": "synthetic://dry-run",
        },
        facets,
    )


def run_report(
    dsn: str | None = None,
    json_out: Path | None = None,
    docs_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Top-level entry point. Returns the JSON payload (also writes it if json_out)."""
    if dry_run:
        meta, facets = collect_synthetic()
    else:
        meta, facets = collect_from_db(dsn)

    payload = build_payload(
        corpus_total=meta["corpus_total"],
        fulltext_total=meta["fulltext_total"],
        facet_rows_by_name=facets,
        dsn_redacted=meta["dsn_redacted"],
    )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Wrote JSON to %s", json_out)

    if docs_path is not None:
        section = render_agent_guidance(payload, json_out)
        upsert_agent_guidance_section(docs_path, section)

    return payload


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute KL-divergence per-facet for the full-text vs corpus "
            "prior comparison; emit JSON and refresh the agent-guidance "
            "section of docs/full_text_coverage_analysis.md."
        )
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (defaults to SCIX_DSN env var or 'dbname=scix').",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_project_root() / "results" / "full_text_coverage_bias.json",
        help="Where to write the machine-readable JSON payload.",
    )
    parser.add_argument(
        "--docs-path",
        default=str(_project_root() / "docs" / "full_text_coverage_analysis.md"),
        help=(
            "Markdown file whose 'Agent guidance' section will be refreshed. "
            "Pass an empty string ('') or 'none' to skip the docs upsert."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use synthetic distributions instead of querying the database. "
            "Useful for local schema-shape verification."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_docs_path = args.docs_path or ""
    docs_path: Path | None
    if raw_docs_path.strip().lower() in ("", "none"):
        docs_path = None
    else:
        docs_path = Path(raw_docs_path)

    run_report(
        dsn=args.dsn,
        json_out=args.json_out,
        docs_path=docs_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
