#!/usr/bin/env python3
"""Re-harvest ADS metadata with full field coverage.

Replaces the per-range harvest scripts with a single configurable script.
Key improvements over harvest_2021_2023.py / harvest_2024_2026.py:
  - All retrievable API fields including body, caption, facility, nedid, simbid
  - rows=2000 (API max) instead of 100 — 20x fewer API calls
  - Respects rate limit headers (X-RateLimit-Remaining)
  - Writes to a separate output dir to avoid clobbering existing data
  - Resume support: counts existing lines and continues from where it left off
  - Validates record count against API numFound

Usage:
    # Re-harvest years missing body (2010-2020):
    python scripts/harvest_full.py --start-year 2010 --end-year 2020

    # Re-harvest everything with full fields:
    python scripts/harvest_full.py --start-year 1800 --end-year 2026

    # Dry run — just show how many records per year:
    python scripts/harvest_full.py --start-year 2010 --end-year 2020 --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# Complete set of retrievable ADS API fields.
# body is documented as non-retrievable but our key returns it.
FIELDS = ",".join(
    [
        # Core metadata
        "abstract",
        "ack",
        "aff",
        "aff_id",
        "alternate_bibcode",
        "alternate_title",
        "arxiv_class",
        "author",
        "author_count",
        "author_norm",
        "bibcode",
        "bibgroup",
        "bibstem",
        # Full text
        "body",
        "caption",
        # Citations & references
        "citation",
        "citation_count",
        "citation_count_norm",
        "cite_read_boost",
        "classic_factor",
        # Identifiers
        "comment",
        "copyright",
        "data",
        "database",
        "date",
        "doi",
        "doctype",
        "eid",
        # Dates
        "entry_date",
        "entdate",
        # Sources & facilities
        "esources",
        "facility",
        # Authors
        "first_author",
        "first_author_norm",
        # Grants
        "grant",
        "grant_agencies",
        "grant_id",
        # IDs
        "id",
        "identifier",
        "indexstamp",
        # ISBN/ISSN
        "isbn",
        "issn",
        "issue",
        # Keywords
        "keyword",
        "keyword_norm",
        "keyword_schema",
        # Other
        "lang",
        "links_data",
        # Cross-references
        "nedid",
        "nedtype",
        # ORCID
        "orcid_pub",
        "orcid_user",
        "orcid_other",
        # Pages
        "page",
        "page_count",
        "page_range",
        # Properties
        "property",
        "pub",
        "pub_raw",
        "pubdate",
        "pubnote",
        # Counts
        "read_count",
        "reference",
        "reference_count",
        # Series
        "series",
        # Cross-references
        "simbid",
        # Title & volume
        "title",
        "vizier",
        "volume",
        "year",
    ]
)

# API max is 2000 rows per request
ROWS = 2000
TIMEOUT = 120
DEFAULT_THROTTLE = 0.5  # seconds between requests when rate limit is healthy
RATE_LIMIT_BUFFER = 50  # slow down when fewer than this many requests remain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """Read API key from environment."""
    key = os.environ.get("ADS_API_KEY", "")
    if not key:
        logger.error("ADS_API_KEY environment variable is not set")
        sys.exit(1)
    return key


def count_lines_gz(path: str) -> int:
    """Count lines in a gzip file. Returns 0 if file doesn't exist."""
    if not os.path.exists(path):
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)


def adaptive_throttle(response: requests.Response, default: float) -> float:
    """Determine sleep time based on rate limit headers."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        remaining = int(remaining)
        if remaining < 10:
            logger.warning("Rate limit nearly exhausted: %d remaining", remaining)
            # Sleep until reset
            reset = response.headers.get("X-RateLimit-Reset")
            if reset:
                sleep_time = max(0, int(reset) - int(time.time())) + 5
                logger.info("Sleeping %d seconds until rate limit reset", sleep_time)
                return sleep_time
            return 300  # 5 min fallback
        if remaining < RATE_LIMIT_BUFFER:
            return default * 3  # slow down
    return default


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_batch(
    api_key: str,
    year: int,
    start: int,
    rows: int = ROWS,
) -> tuple[list[dict], int, requests.Response | None]:
    """Fetch a batch of records from ADS API.

    Returns (records, total_found, response).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "q": f"year:{year}",
        "start": start,
        "rows": rows,
        "fl": FIELDS,
        "sort": "bibcode asc",  # deterministic ordering for resume
    }

    for attempt in range(10):
        try:
            resp = requests.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json().get("response", {})
                return data.get("docs", []), data.get("numFound", 0), resp
            elif resp.status_code == 429:
                # Rate limited
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Rate limited (429). Sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            else:
                logger.warning(
                    "HTTP %d on attempt %d: %s",
                    resp.status_code,
                    attempt + 1,
                    resp.text[:200],
                )
        except requests.exceptions.RequestException as e:
            logger.warning("Request failed attempt %d: %s", attempt + 1, e)

        backoff = min(120, 2 ** min(attempt + 1, 7))
        logger.info("Retrying in %ds...", backoff)
        time.sleep(backoff)

    logger.error("Failed after 10 attempts for year=%d start=%d", year, start)
    return [], 0, None


# ---------------------------------------------------------------------------
# Year harvester
# ---------------------------------------------------------------------------


def get_year_count(api_key: str, year: int) -> int:
    """Get total record count for a year (no data fetched)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"year:{year}", "rows": 0, "fl": "bibcode"}
    resp = requests.get(API_URL, headers=headers, params=params, timeout=30)
    if resp.status_code == 200:
        return resp.json().get("response", {}).get("numFound", 0)
    return -1


def harvest_year(
    api_key: str,
    year: int,
    output_dir: str,
    throttle: float = DEFAULT_THROTTLE,
) -> int:
    """Harvest all records for a single year. Returns total records written."""
    output_path = os.path.join(output_dir, f"ads_metadata_{year}_full.jsonl.gz")
    existing = count_lines_gz(output_path)

    if existing > 0:
        logger.info("Year %d: resuming from record %d", year, existing)

    start = existing
    total_written = existing

    with gzip.open(output_path, "at", encoding="utf-8") as f:
        while True:
            records, num_found, resp = fetch_batch(api_key, year, start)

            if not records:
                if start < num_found:
                    logger.warning(
                        "Year %d: got 0 records at start=%d but numFound=%d",
                        year,
                        start,
                        num_found,
                    )
                break

            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total_written += len(records)
            start += len(records)

            logger.info(
                "Year %d: %d / %d (%.1f%%)",
                year,
                total_written,
                num_found,
                100.0 * total_written / num_found if num_found else 0,
            )

            if start >= num_found:
                break

            # Adaptive throttle
            if resp is not None:
                sleep_time = adaptive_throttle(resp, throttle)
            else:
                sleep_time = throttle
            time.sleep(sleep_time)

    logger.info("Year %d: complete — %d records", year, total_written)
    return total_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest ADS metadata with full field coverage")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        default="ads_metadata_by_year_picard",
        help="Output directory for JSONL files (default: ads_metadata_by_year_picard)",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=DEFAULT_THROTTLE,
        help=f"Base seconds between requests (default: {DEFAULT_THROTTLE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show record counts per year, don't fetch data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-harvest from scratch, backing up existing files with .old suffix",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    os.makedirs(args.output_dir, exist_ok=True)

    years = list(range(args.start_year, args.end_year + 1))

    if args.dry_run:
        total = 0
        for year in years:
            count = get_year_count(api_key, year)
            existing_path = os.path.join(args.output_dir, f"ads_metadata_{year}_full.jsonl.gz")
            existing = count_lines_gz(existing_path) if not args.force else 0
            remaining = max(0, count - existing)
            api_calls = (remaining + ROWS - 1) // ROWS
            label = "(force)" if args.force and existing == 0 else ""
            print(
                f"  {year}: {count:>10,} total, {existing:>10,} existing, "
                f"{remaining:>10,} to fetch ({api_calls:,} API calls) {label}"
            )
            total += remaining
            time.sleep(0.3)  # gentle on rate limit even for dry run
        total_calls = (total + ROWS - 1) // ROWS
        days = total_calls / 5000
        print(
            f"\n  Total: {total:,} records, ~{total_calls:,} API calls, "
            f"~{days:.1f} days at 5K/day rate limit"
        )
        return

    if args.force:
        for year in years:
            existing_path = os.path.join(args.output_dir, f"ads_metadata_{year}_full.jsonl.gz")
            if os.path.exists(existing_path):
                backup = existing_path + ".old"
                logger.info("Backing up %s -> %s", existing_path, backup)
                os.rename(existing_path, backup)

    grand_total = 0
    for year in years:
        try:
            written = harvest_year(api_key, year, args.output_dir, args.throttle)
            grand_total += written
        except KeyboardInterrupt:
            logger.info("Interrupted at year %d. Resume with same command.", year)
            sys.exit(0)
        except Exception:
            logger.exception("Error harvesting year %d — skipping", year)
            continue

    logger.info("Harvest complete: %d total records across %d years", grand_total, len(years))


if __name__ == "__main__":
    main()
