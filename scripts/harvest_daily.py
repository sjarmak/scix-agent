#!/usr/bin/env python3
"""Incremental ADS harvest: fetch records in a rolling entdate window that the
papers table does not have yet.

ADS stamps arXiv records with a backdated entry_date (the arXiv date) but can
index them days later, and its numFound flaps between replicas mid-query. A
watermark on entdate therefore skipped late records for good, and trusting the
last page's numFound truncated busy days (bead scix_experiments-d4c1). So each
run instead:

  1. lists every bibcode with entdate in the last --window-days days (sorted,
     bibcode-only, repeated passes unioned until the listing covers the largest
     numFound seen);
  2. diffs that list against the papers table;
  3. fetches full records for the missing bibcodes only.

Output goes to data/daily_harvest/ads_daily_<today>.jsonl.gz, written to a
.tmp path and renamed on success so a crash never leaves a partial file at the
path daily_sync.sh ingests. Nothing missing means no file. To recover from an
outage longer than the window, rerun once with a larger --window-days.

Usage:
    python scripts/harvest_daily.py [--window-days 14] [--output-dir data/daily_harvest]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import psycopg
import requests

logger = logging.getLogger(__name__)

# ─── ADS API ─────────────────────────────────────────────────────────────────

API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

FIELDS = ",".join(
    [
        "abstract",
        "ack",
        "aff",
        "alternate_bibcode",
        "alternate_title",
        "arxiv_class",
        "author",
        "bibcode",
        "bibgroup",
        "bibstem",
        "body",
        "citation",
        "citation_count",
        "copyright",
        "database",
        "data",
        "doi",
        "doctype",
        "editor",
        "entry_date",
        "first_author",
        "grant",
        "id",
        "identifier",
        "indexstamp",
        "issue",
        "keyword",
        "lang",
        "orcid_pub",
        "orcid_user",
        "page",
        "property",
        "pub",
        "pub_raw",
        "pubdate",
        "read_count",
        "reference",
        "reference_count",
        "series",
        "title",
        "volume",
        "year",
    ]
)

LIST_ROWS = 2000  # ADS API max; bibcode-only pages are small
FETCH_BATCH = 100  # bibcodes per full-record query (keeps the GET URL short)
MAX_LIST_PASSES = 3
COUNT_PROBES = 3  # count-only requests that seed the expected total
MAX_FETCH = 20_000  # per-run cap; a larger backlog drains over later runs
TIMEOUT = 60
THROTTLE = 1.0  # seconds between requests
MAX_RETRIES = 10
DB_CHUNK = 10_000

Fetch = Callable[[dict], "tuple[list[dict], int]"]
Existing = Callable[[list[str]], "set[str]"]


class HarvestIncomplete(Exception):
    """The window listing never covered ADS's reported record count."""


def _get_headers() -> dict[str, str]:
    api_key = os.environ.get("ADS_API_KEY")
    if not api_key:
        logger.error("ADS_API_KEY environment variable is not set")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def ads_fetcher(headers: dict[str, str]) -> Fetch:
    """Return fetch(params) -> (docs, numFound) against the ADS search API."""

    def fetch(params: dict) -> tuple[list[dict], int]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)
                if resp.status_code == 200:
                    body = resp.json().get("response", {})
                    sleep(THROTTLE)
                    return body.get("docs", []), body.get("numFound", 0)
                if resp.status_code == 400:
                    logger.error("HTTP 400 (bad request, not retrying): %s", resp.text[:500])
                    sys.exit(1)
                logger.warning(
                    "HTTP %d (attempt %d/%d): %s",
                    resp.status_code,
                    attempt + 1,
                    MAX_RETRIES,
                    resp.text[:500],
                )
            except requests.exceptions.RequestException as e:
                logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            sleep(min(60, 2 ** min(attempt, 6)))
        logger.error("Max retries (%d) exceeded for params %s", MAX_RETRIES, params)
        sys.exit(1)

    return fetch


# ─── Listing / diff / fetch ──────────────────────────────────────────────────


def window_query(today: date, window_days: int) -> str:
    return f"entdate:[{today - timedelta(days=window_days)} TO {today}]"


def list_window_bibcodes(
    fetch: Fetch, query: str, rows: int = LIST_ROWS, max_passes: int = MAX_LIST_PASSES
) -> set[str]:
    """Every bibcode matching query, robust to flapping numFound and short pages.

    Pages are sorted by bibcode so offsets are stable. A pass ends at an empty
    page or once the offset passes the largest numFound seen so far (never the
    latest one, which is what truncated 2026-09-23). Passes are unioned until
    the union covers that largest numFound. The expected total is seeded from
    several count-only probes first, so a pass served entirely by one lagging
    replica cannot agree with itself and look complete.
    """
    found: set[str] = set()
    expected = max(fetch({"q": query, "start": 0, "rows": 0})[1] for _ in range(COUNT_PROBES))
    for n in range(1, max_passes + 1):
        start = 0
        while True:
            docs, num_found = fetch(
                {"q": query, "start": start, "rows": rows, "fl": "bibcode", "sort": "bibcode asc"}
            )
            expected = max(expected, num_found)
            found.update(d["bibcode"] for d in docs)
            start += rows
            if not docs or start >= expected:
                break
        logger.info("Listing pass %d: %d / %d bibcodes", n, len(found), expected)
        if len(found) >= expected:
            return found
    raise HarvestIncomplete(
        f"listing covered {len(found)}/{expected} bibcodes after {max_passes} passes"
    )


def fetch_records(
    fetch: Fetch, bibcodes: Iterable[str], batch: int = FETCH_BATCH
) -> Iterator[dict]:
    """Full records for bibcodes. Logs any bibcode ADS no longer returns; it is
    still missing from the DB, so the next run's diff retries it."""
    ordered = sorted(bibcodes)
    for i in range(0, len(ordered), batch):
        chunk = ordered[i : i + batch]
        # Quoted: bibcodes carry Solr-significant characters (A&A, dots).
        # rows is doubled so an extra hit (e.g. an alternate_bibcode match)
        # cannot push a requested record off the page; docs are matched by
        # bibcode, not position.
        terms = " OR ".join(f'"{b}"' for b in chunk)
        docs, _ = fetch(
            {"q": f"bibcode:({terms})", "start": 0, "rows": 2 * len(chunk), "fl": FIELDS}
        )
        wanted = set(chunk)
        hits = [d for d in docs if d.get("bibcode") in wanted]
        missing = wanted - {d["bibcode"] for d in hits}
        if missing:
            logger.warning(
                "ADS returned no record for %d bibcode(s): %s", len(missing), sorted(missing)
            )
        yield from hits


def db_existing(dsn: str) -> Existing:
    """Return existing(bibcodes) -> the subset already in papers (read-only)."""

    def existing(bibcodes: list[str]) -> set[str]:
        present: set[str] = set()
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            for i in range(0, len(bibcodes), DB_CHUNK):
                cur.execute(
                    "SELECT bibcode FROM papers WHERE bibcode = ANY(%s)",
                    (bibcodes[i : i + DB_CHUNK],),
                )
                present.update(row[0] for row in cur.fetchall())
        return present

    return existing


# ─── Main ─────────────────────────────────────────────────────────────────────


def harvest(
    output_dir: Path,
    window_days: int,
    fetch: Fetch,
    existing: Existing,
    today: date,
    fetch_batch: int = FETCH_BATCH,
    max_fetch: int = MAX_FETCH,
) -> Path | None:
    """Harvest window records missing from the DB. Returns the output file or None.

    At most max_fetch records per run, arXiv eprints first, so a bulk ADS
    re-stamp (675k dataset records landed in one window in 2026-09) cannot turn
    a daily run into a multi-hour ingest. The rest stay missing and are picked
    up by later runs while they remain inside the window."""
    query = window_query(today, window_days)
    logger.info("ADS window: %s", query)

    listed = list_window_bibcodes(fetch, query)
    missing = listed - existing(sorted(listed))
    logger.info("%d bibcodes in window, %d missing from papers", len(listed), len(missing))
    if not missing:
        return None
    if len(missing) > max_fetch:
        ordered = sorted(missing, key=lambda b: (b[4:9] != "arXiv", b))
        missing = set(ordered[:max_fetch])
        logger.warning(
            "Backlog of %d exceeds --max-fetch %d; fetching %d (arXiv first), deferring %d",
            len(ordered),
            max_fetch,
            max_fetch,
            len(ordered) - max_fetch,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ads_daily_{today}.jsonl.gz"
    tmp_file = output_file.with_name(output_file.name + ".tmp")
    total_written = 0
    with gzip.open(tmp_file, "wt", encoding="utf-8") as f:
        for doc in fetch_records(fetch, missing, fetch_batch):
            f.write(json.dumps(doc) + "\n")
            total_written += 1
            if total_written % 1000 == 0:
                logger.info("Progress: %d / %d records", total_written, len(missing))
    tmp_file.rename(output_file)
    logger.info("Wrote %d records to %s", total_written, output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental ADS daily harvest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/daily_harvest"),
        help="Output directory (default: data/daily_harvest)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Rolling entdate window to reconcile against the DB (default: 14)",
    )
    parser.add_argument(
        "--max-fetch",
        type=int,
        default=MAX_FETCH,
        help=f"Most records to fetch per run, arXiv first (default: {MAX_FETCH})",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help="PostgreSQL DSN (default: $SCIX_DSN or dbname=scix)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        harvest(
            args.output_dir,
            args.window_days,
            fetch=ads_fetcher(_get_headers()),
            existing=db_existing(args.dsn),
            today=datetime.now(timezone.utc).date(),
            max_fetch=args.max_fetch,
        )
    except HarvestIncomplete as e:
        logger.error("Harvest incomplete, nothing written: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
