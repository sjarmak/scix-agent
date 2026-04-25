#!/usr/bin/env python3
"""Orchestrate correction-event ingestion across the four PRD-A3 sources.

Pulls events from:

* Retraction Watch (CC0 CSV)            -> ``scix.sources.retraction_watch``
* OpenAlex ``is_retracted`` filter      -> ``scix.sources.openalex_corrections``
* Crossref ``update-to`` relations      -> ``scix.sources.crossref_update_to``
* Top-15 astronomy Errata RSS feeds     -> ``scix.sources.journal_errata_rss``

Per paper: fetch existing ``correction_events`` JSONB, dedup on the tuple
``(type, source, doi, date)``, append the new events, and re-write the JSONB
in a single transaction. ``papers.retracted_at`` is updated to the earliest
retraction date observed.

USAGE::

    python scripts/ingest_corrections.py --dsn "$SCIX_TEST_DSN" \\
        --sources retraction_watch crossref \\
        --dry-run

SAFETY: Refuses to target a production DSN without ``--yes-production``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running both as a CLI and as an importable module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.db import (
    DEFAULT_DSN,
    IngestLog,
    get_connection,
    is_production_dsn,
    redact_dsn,
)
from scix.sources import (
    crossref_update_to,
    journal_errata_rss,
    openalex_corrections,
    retraction_watch,
)

logger = logging.getLogger(__name__)

ALL_SOURCES = ("retraction_watch", "openalex", "crossref", "journal_rss")


class ProductionGuardError(RuntimeError):
    """Raised when the orchestrator would target a production DSN unsafely."""


@dataclass(frozen=True)
class IngestStats:
    """Result of a single orchestrator run."""

    events_collected: int
    papers_updated: int
    new_events_inserted: int
    retractions_marked: int
    dry_run: bool


def _event_key(event: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(event.get("type", "")),
        str(event.get("source", "")),
        str(event.get("doi", "")),
        str(event.get("date", "")),
    )


def merge_events_for_paper(
    existing: list[dict[str, Any]],
    new_events: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Pure merge: dedup + append + compute earliest retraction date.

    Returns ``(merged_events, num_new_events_added, earliest_retraction_iso)``.
    The earliest-retraction date considers BOTH existing and incoming events,
    so the orchestrator can keep ``retracted_at`` in sync without a separate
    pass.
    """
    seen: set[tuple[str, str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    for ev in existing:
        if not isinstance(ev, dict):
            continue
        key = _event_key(ev)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ev)

    added = 0
    for ev in new_events:
        if not isinstance(ev, dict):
            continue
        key = _event_key(ev)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ev)
        added += 1

    earliest_retraction: str | None = None
    for ev in merged:
        if ev.get("type") != "retraction":
            continue
        date_str = ev.get("date")
        if not isinstance(date_str, str):
            continue
        if earliest_retraction is None or date_str < earliest_retraction:
            earliest_retraction = date_str
    return merged, added, earliest_retraction


def _group_by_doi(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        doi = ev.get("doi")
        if not isinstance(doi, str) or not doi:
            continue
        grouped.setdefault(doi.lower(), []).append(ev)
    return grouped


def _check_production_guard(dsn: str | None, *, yes_production: bool) -> None:
    effective = dsn or DEFAULT_DSN
    if is_production_dsn(effective) and not yes_production:
        raise ProductionGuardError(
            f"Refusing to run correction-event ingest against production DSN "
            f"({redact_dsn(effective)}). Pass --yes-production to override."
        )


def collect_events(
    sources: Iterable[str],
    *,
    crossref_dois: Iterable[str] = (),
    fetchers: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run the configured source modules in sequence and yield their events.

    ``fetchers`` is an optional mapping ``{source_name: callable}`` for tests
    to inject stubbed HTTP responses.
    """
    fetchers = fetchers or {}
    for src in sources:
        if src == "retraction_watch":
            yield from retraction_watch.harvest(fetcher=fetchers.get("retraction_watch"))
        elif src == "openalex":
            yield from openalex_corrections.harvest(fetcher=fetchers.get("openalex"))
        elif src == "crossref":
            yield from crossref_update_to.harvest(
                crossref_dois, fetcher=fetchers.get("crossref")
            )
        elif src == "journal_rss":
            yield from journal_errata_rss.harvest(fetcher=fetchers.get("journal_rss"))
        else:
            logger.warning("Unknown source %r; skipping", src)


def apply_events(
    conn: psycopg.Connection,
    events: Iterable[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> IngestStats:
    """Apply correction events to the ``papers`` table.

    Joins by DOI: ``papers.doi`` is ``text[]`` so we use ``= ANY(doi)``.
    Each affected paper gets a single UPDATE in its own short transaction.
    """
    grouped = _group_by_doi(events)
    events_collected = sum(len(v) for v in grouped.values())
    papers_updated = 0
    new_events_inserted = 0
    retractions_marked = 0

    for doi, doi_events in grouped.items():
        # Find papers (could be 0, 1, or many) keyed by this DOI in the array.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, correction_events, retracted_at "
                "FROM papers WHERE %s = ANY(doi)",
                (doi,),
            )
            rows = cur.fetchall()

        for bibcode, existing_json, prior_retracted_at in rows:
            if isinstance(existing_json, str):
                try:
                    existing = json.loads(existing_json)
                except json.JSONDecodeError:
                    existing = []
            elif isinstance(existing_json, list):
                existing = existing_json
            else:
                existing = []

            merged, added, earliest_retraction = merge_events_for_paper(
                existing, doi_events
            )
            if added == 0:
                continue
            new_events_inserted += added
            papers_updated += 1
            new_retracted_at: Any
            if earliest_retraction is not None:
                new_retracted_at = earliest_retraction
                if prior_retracted_at is None:
                    retractions_marked += 1
            else:
                new_retracted_at = prior_retracted_at  # leave unchanged

            if dry_run:
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE papers SET correction_events = %s::jsonb, "
                    "retracted_at = %s WHERE bibcode = %s",
                    (json.dumps(merged), new_retracted_at, bibcode),
                )
            conn.commit()

    return IngestStats(
        events_collected=events_collected,
        papers_updated=papers_updated,
        new_events_inserted=new_events_inserted,
        retractions_marked=retractions_marked,
        dry_run=dry_run,
    )


def run(
    *,
    dsn: str | None,
    sources: Iterable[str],
    yes_production: bool = False,
    dry_run: bool = False,
    crossref_dois: Iterable[str] = (),
    fetchers: dict[str, Any] | None = None,
) -> IngestStats:
    """Top-level entry point used by both the CLI and integration tests."""
    _check_production_guard(dsn, yes_production=yes_production)

    sources_list = list(sources)
    conn = get_connection(dsn)
    try:
        ingest_log = IngestLog(conn)
        for src in sources_list:
            ingest_log.start(f"corrections:{src}")

        events = list(
            collect_events(
                sources_list,
                crossref_dois=crossref_dois,
                fetchers=fetchers,
            )
        )
        stats = apply_events(conn, events, dry_run=dry_run)

        for src in sources_list:
            ingest_log.update_counts(
                f"corrections:{src}",
                stats.events_collected,
                0,
                stats.new_events_inserted,
            )
            ingest_log.finish(f"corrections:{src}")
        return stats
    finally:
        conn.close()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", default=None, help="Postgres DSN; falls back to SCIX_DSN")
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(ALL_SOURCES),
        choices=ALL_SOURCES,
        help="Which sources to harvest (default: all four).",
    )
    p.add_argument(
        "--yes-production",
        action="store_true",
        help="Required when --dsn points at the production database.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and merge events but skip the UPDATE statements.",
    )
    p.add_argument(
        "--crossref-doi-file",
        default=None,
        help="Path to a newline-delimited list of DOIs to query Crossref for.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    crossref_dois: list[str] = []
    if args.crossref_doi_file:
        with open(args.crossref_doi_file) as fh:
            crossref_dois = [line.strip() for line in fh if line.strip()]

    stats = run(
        dsn=args.dsn,
        sources=args.sources,
        yes_production=args.yes_production,
        dry_run=args.dry_run,
        crossref_dois=crossref_dois,
    )
    logger.info(
        "Correction ingest complete: events_collected=%d papers_updated=%d "
        "new_events=%d retractions_marked=%d dry_run=%s",
        stats.events_collected,
        stats.papers_updated,
        stats.new_events_inserted,
        stats.retractions_marked,
        stats.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
