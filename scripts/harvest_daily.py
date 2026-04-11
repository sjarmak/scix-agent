#!/usr/bin/env python3
"""Incremental ADS harvest: fetch records added since last watermark.

Writes compressed JSONL to data/daily_harvest/ and updates the watermark
file on success. Designed to run daily via cron (daily_sync.sh).

Usage:
    python scripts/harvest_daily.py [--lookback-days 2] [--output-dir data/daily_harvest]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

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

ROWS_PER_PAGE = 100
TIMEOUT = 60
THROTTLE = 1.0  # seconds between pages


def _get_headers() -> dict[str, str]:
    api_key = os.environ.get("ADS_API_KEY")
    if not api_key:
        logger.error("ADS_API_KEY environment variable is not set")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


MAX_RETRIES = 10


def fetch_page(headers: dict[str, str], query: str, start: int) -> tuple[list[dict], int]:
    """Fetch one page of results. Returns (docs, total_num_found)."""
    params = {"q": query, "start": start, "rows": ROWS_PER_PAGE, "fl": FIELDS}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)
            if resp.status_code == 200:
                body = resp.json().get("response", {})
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
    logger.error("Max retries (%d) exceeded for query start=%d", MAX_RETRIES, start)
    sys.exit(1)


# ─── Watermark ────────────────────────────────────────────────────────────────


def read_watermark(watermark_path: Path, lookback_days: int) -> str:
    """Read ISO date from watermark file, or default to lookback_days ago."""
    if watermark_path.exists():
        text = watermark_path.read_text().strip()
        if text:
            logger.info("Watermark: %s (from %s)", text, watermark_path)
            return text
    default = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    logger.info("No watermark found, defaulting to %s (%d days ago)", default, lookback_days)
    return default


def write_watermark(watermark_path: Path, date_str: str) -> None:
    watermark_path.parent.mkdir(parents=True, exist_ok=True)
    watermark_path.write_text(date_str + "\n")
    logger.info("Watermark updated to %s", date_str)


# ─── Main ─────────────────────────────────────────────────────────────────────


def harvest(output_dir: Path, lookback_days: int) -> Path | None:
    """Harvest new ADS records since watermark. Returns output file path or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    watermark_path = output_dir / "last_run.txt"
    since = read_watermark(watermark_path, lookback_days)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    query = f"entdate:[{since} TO {today}]"
    logger.info("ADS query: %s", query)

    headers = _get_headers()
    output_file = output_dir / f"ads_daily_{today}.jsonl.gz"

    start = 0
    total_written = 0

    # Peek at total count first
    _, num_found = fetch_page(headers, query, 0)
    logger.info("ADS reports %d records matching query", num_found)

    if num_found == 0:
        logger.info("No new records — nothing to harvest")
        write_watermark(watermark_path, today)
        return None

    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        while start < num_found:
            docs, num_found = fetch_page(headers, query, start)
            if not docs:
                break
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
                total_written += 1
            start += ROWS_PER_PAGE
            logger.info("Progress: %d / %d records", min(start, num_found), num_found)
            sleep(THROTTLE)

    logger.info("Wrote %d records to %s", total_written, output_file)
    write_watermark(watermark_path, today)
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
        "--lookback-days",
        type=int,
        default=2,
        help="Days to look back if no watermark exists (default: 2)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    harvest(args.output_dir, args.lookback_days)


if __name__ == "__main__":
    main()
