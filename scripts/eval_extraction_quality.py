#!/usr/bin/env python3
"""Evaluate extraction quality by resolving mentions against the entity graph.

Samples papers from the extractions table, runs EntityResolver on all
extracted mentions, and computes resolution rate, match_method distribution,
and lists unmatched mentions. Writes a markdown report to the specified
output path.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.db import get_connection
from scix.entity_resolver import EntityCandidate, EntityResolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Results of extraction quality evaluation."""

    papers_sampled: int
    total_mentions: int
    resolved_mentions: int
    unmatched_mentions: tuple[str, ...]
    match_method_counts: dict[str, int]

    @property
    def resolution_rate(self) -> float:
        """Fraction of mentions that resolved to at least one entity (precision proxy)."""
        if self.total_mentions == 0:
            return 0.0
        return self.resolved_mentions / self.total_mentions

    @property
    def recall_proxy(self) -> float:
        """Fraction of mentions that got any match (recall proxy).

        Without ground truth, this equals resolution_rate.
        """
        return self.resolution_rate


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def sample_papers(conn: psycopg.Connection, n: int = 50) -> list[str]:
    """Sample n random bibcodes from the extractions table.

    Args:
        conn: Database connection.
        n: Number of papers to sample.

    Returns:
        List of distinct bibcodes.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT bibcode FROM extractions ORDER BY RANDOM() LIMIT %s",
            (n,),
        )
        bibcodes = [row[0] for row in cur.fetchall()]
    logger.info("Sampled %d papers from extractions table", len(bibcodes))
    return bibcodes


def get_mentions(
    conn: psycopg.Connection,
    bibcodes: list[str],
) -> dict[str, list[str]]:
    """Get all mention strings from extraction payloads for given bibcodes.

    Args:
        conn: Database connection.
        bibcodes: List of bibcodes to query.

    Returns:
        Dict mapping bibcode to flattened list of mention strings.
    """
    if not bibcodes:
        return {}

    mentions_by_bibcode: dict[str, list[str]] = {b: [] for b in bibcodes}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, extraction_type, payload
            FROM extractions
            WHERE bibcode = ANY(%s)
            """,
            (bibcodes,),
        )
        for row in cur.fetchall():
            bibcode = row[0]
            payload = row[2]
            # payload is {"entities": ["mention1", "mention2", ...]}
            if isinstance(payload, str):
                payload = json.loads(payload)
            entities = payload.get("entities", [])
            if isinstance(entities, list):
                mentions_by_bibcode[bibcode].extend(entities)

    total = sum(len(v) for v in mentions_by_bibcode.values())
    logger.info(
        "Extracted %d mentions from %d papers",
        total,
        len(bibcodes),
    )
    return mentions_by_bibcode


def evaluate_mentions(
    resolver: EntityResolver,
    mentions_by_bibcode: dict[str, list[str]],
    *,
    fuzzy: bool = False,
) -> EvalResult:
    """Resolve all mentions and compute evaluation metrics.

    Args:
        resolver: EntityResolver instance.
        mentions_by_bibcode: Dict mapping bibcode to mention strings.
        fuzzy: Enable fuzzy matching in the resolver.

    Returns:
        EvalResult with computed metrics.
    """
    total = 0
    resolved = 0
    unmatched: list[str] = []
    method_counts: dict[str, int] = {}

    for bibcode, mentions in mentions_by_bibcode.items():
        for mention in mentions:
            total += 1
            candidates = resolver.resolve(mention, fuzzy=fuzzy)

            if candidates:
                resolved += 1
                # Count the match_method of the top candidate (highest confidence)
                top_method = candidates[0].match_method
                method_counts[top_method] = method_counts.get(top_method, 0) + 1
            else:
                unmatched.append(mention)

    logger.info(
        "Evaluated %d mentions: %d resolved, %d unmatched",
        total,
        resolved,
        len(unmatched),
    )

    return EvalResult(
        papers_sampled=len(mentions_by_bibcode),
        total_mentions=total,
        resolved_mentions=resolved,
        unmatched_mentions=tuple(unmatched),
        match_method_counts=dict(sorted(method_counts.items())),
    )


def format_report(result: EvalResult) -> str:
    """Format evaluation results as a markdown report.

    Args:
        result: EvalResult to format.

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    lines.append("# Extraction Quality Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Papers sampled**: {result.papers_sampled}")
    lines.append(f"- **Total mentions**: {result.total_mentions}")
    lines.append(f"- **Resolved mentions**: {result.resolved_mentions}")
    lines.append(f"- **Resolution rate (precision proxy)**: {result.resolution_rate:.1%}")
    lines.append(f"- **Recall proxy**: {result.recall_proxy:.1%}")
    lines.append(f"- **Unmatched mentions**: {len(result.unmatched_mentions)}")
    lines.append("")

    # Match method distribution
    lines.append("## Match Method Distribution")
    lines.append("")
    if result.match_method_counts:
        lines.append("| Method | Count | Percentage |")
        lines.append("|--------|------:|----------:|")
        for method, count in result.match_method_counts.items():
            pct = count / result.resolved_mentions * 100 if result.resolved_mentions > 0 else 0.0
            lines.append(f"| {method} | {count} | {pct:.1f}% |")
    else:
        lines.append("No resolved mentions.")
    lines.append("")

    # Unmatched examples
    lines.append("## Unmatched Mention Examples")
    lines.append("")
    if result.unmatched_mentions:
        examples = result.unmatched_mentions[:20]
        for mention in examples:
            lines.append(f"- `{mention}`")
        if len(result.unmatched_mentions) > 20:
            remaining = len(result.unmatched_mentions) - 20
            lines.append(f"- ... and {remaining} more")
    else:
        lines.append("All mentions resolved successfully.")
    lines.append("")

    return "\n".join(lines)


def write_report(report: str, output_path: str) -> Path:
    """Write report string to a file.

    Args:
        report: Markdown report content.
        output_path: Path to write to.

    Returns:
        Path object for the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run extraction quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate extraction quality against the entity graph",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of papers to sample (default: 50)",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy matching in EntityResolver",
    )
    parser.add_argument(
        "--output",
        default=".claude/prd-build-artifacts/extraction-quality-eval.md",
        help="Output path for the evaluation report",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.dsn)
    try:
        bibcodes = sample_papers(conn, n=args.sample_size)
        if not bibcodes:
            logger.warning("No papers found in extractions table")
            report = format_report(
                EvalResult(
                    papers_sampled=0,
                    total_mentions=0,
                    resolved_mentions=0,
                    unmatched_mentions=(),
                    match_method_counts={},
                )
            )
        else:
            mentions = get_mentions(conn, bibcodes)
            resolver = EntityResolver(conn)
            result = evaluate_mentions(resolver, mentions, fuzzy=args.fuzzy)
            report = format_report(result)

        output_path = write_report(report, args.output)
        print(f"Evaluation report written to {output_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
