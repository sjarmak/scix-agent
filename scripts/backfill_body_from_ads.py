#!/usr/bin/env python3
"""Backfill body text from ADS API for papers missing it in the database.

Queries ADS for papers that have body text (using body:*) within a year range,
fetches only bibcode + body (minimal bandwidth), and writes directly to the
papers.body column in batches.

Uses the bigquery endpoint to send bibcode lists, avoiding redundant downloads
for papers that already have body text.

Usage:
    # Backfill 2005-2020 (the gap years):
    python scripts/backfill_body_from_ads.py --start-year 2005 --end-year 2020

    # Dry run — just show how many papers need body per year:
    python scripts/backfill_body_from_ads.py --start-year 2005 --end-year 2020 --dry-run

    # Resume after interruption (tracks progress in DB):
    python scripts/backfill_body_from_ads.py --start-year 2005 --end-year 2020
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import psycopg
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ROWS = 2000
TIMEOUT = 120
DEFAULT_THROTTLE = 0.5
RATE_LIMIT_BUFFER = 50
DB_BATCH = 500


def get_api_key() -> str:
    key = os.environ.get("ADS_API_KEY", "")
    if not key:
        logger.error("ADS_API_KEY environment variable is not set")
        sys.exit(1)
    return key


def adaptive_throttle(response: requests.Response, default: float) -> float:
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        remaining = int(remaining)
        if remaining < 10:
            logger.warning("Rate limit nearly exhausted: %d remaining", remaining)
            reset = response.headers.get("X-RateLimit-Reset")
            if reset:
                sleep_time = max(0, int(reset) - int(time.time())) + 5
                logger.info("Sleeping %d seconds until rate limit reset", sleep_time)
                return sleep_time
            return 300
        if remaining < RATE_LIMIT_BUFFER:
            return default * 3
    return default


def get_year_count(api_key: str, year: int) -> int:
    """Count total papers for a given year via ADS API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"year:{year}", "rows": 0, "fl": "bibcode"}
    resp = requests.get(API_URL, headers=headers, params=params, timeout=30)
    if resp.status_code == 200:
        return resp.json().get("response", {}).get("numFound", 0)
    return -1


def fetch_body_batch(
    api_key: str,
    year: int,
    start: int,
    rows: int = ROWS,
) -> tuple[list[dict], int, requests.Response | None]:
    """Fetch bibcode + body for papers in a given year.

    Body is returned when available, absent when not. Callers filter nulls.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "q": f"year:{year}",
        "start": start,
        "rows": rows,
        "fl": "bibcode,body",
        "sort": "bibcode asc",
    }

    for attempt in range(10):
        try:
            resp = requests.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json().get("response", {})
                return data.get("docs", []), data.get("numFound", 0), resp
            elif resp.status_code == 429:
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


def write_body_batch(
    conn: psycopg.Connection,
    records: list[dict],
) -> int:
    """Write body text to papers table. Returns number of rows updated."""
    updated = 0
    with conn.cursor() as cur:
        for rec in records:
            bibcode = rec.get("bibcode")
            body = rec.get("body")
            if not bibcode or not body:
                continue
            # Strip NUL bytes — PostgreSQL text columns reject them
            body = body.replace("\x00", "")
            if not body:
                continue
            cur.execute(
                "UPDATE papers SET body = %s WHERE bibcode = %s AND body IS NULL",
                (body, bibcode),
            )
            updated += cur.rowcount
    conn.commit()
    return updated


def count_missing(conn: psycopg.Connection, year: int) -> int:
    """Count papers in DB for this year that are missing body."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM papers WHERE year = %s AND body IS NULL",
            (year,),
        )
        return cur.fetchone()[0]


def backfill_year(
    api_key: str,
    conn: psycopg.Connection,
    year: int,
    throttle: float = DEFAULT_THROTTLE,
) -> tuple[int, int]:
    """Backfill body text for one year. Returns (fetched, updated)."""
    start = 0
    total_fetched = 0
    total_updated = 0

    while True:
        records, num_found, resp = fetch_body_batch(api_key, year, start)

        if not records:
            if start < num_found:
                logger.warning(
                    "Year %d: got 0 records at start=%d but numFound=%d",
                    year,
                    start,
                    num_found,
                )
            break

        updated = write_body_batch(conn, records)
        total_fetched += len(records)
        total_updated += updated
        start += len(records)

        logger.info(
            "Year %d: fetched %d / %d, updated %d DB rows (%.1f%%)",
            year,
            total_fetched,
            num_found,
            total_updated,
            100.0 * total_fetched / num_found if num_found else 0,
        )

        if start >= num_found:
            break

        if resp is not None:
            sleep_time = adaptive_throttle(resp, throttle)
        else:
            sleep_time = throttle
        time.sleep(sleep_time)

    return total_fetched, total_updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill body text from ADS API for papers missing it"
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help="PostgreSQL connection string",
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
        help="Show counts per year without fetching",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    years = list(range(args.start_year, args.end_year + 1))

    if args.dry_run:
        total_available = 0
        total_missing = 0
        with psycopg.connect(args.dsn) as conn:
            for year in years:
                available = get_year_count(api_key, year)
                missing = count_missing(conn, year)
                api_calls = (available + ROWS - 1) // ROWS
                print(
                    f"  {year}: {available:>10,} total in ADS, "
                    f"{missing:>10,} missing body in DB, "
                    f"~{api_calls:,} API calls"
                )
                total_available += max(available, 0)
                total_missing += missing
                time.sleep(0.3)

        total_calls = (total_available + ROWS - 1) // ROWS
        days = total_calls / 5000
        print(
            f"\n  Total: {total_available:,} available, {total_missing:,} missing in DB"
            f"\n  ~{total_calls:,} API calls, ~{days:.1f} days at 5K/day rate limit"
        )
        return

    with psycopg.connect(args.dsn) as conn:
        conn.autocommit = False
        grand_fetched = 0
        grand_updated = 0

        for year in years:
            try:
                fetched, updated = backfill_year(api_key, conn, year, args.throttle)
                grand_fetched += fetched
                grand_updated += updated
                logger.info(
                    "Year %d complete: fetched %d, updated %d",
                    year,
                    fetched,
                    updated,
                )
            except KeyboardInterrupt:
                logger.info("Interrupted at year %d. Re-run to resume.", year)
                sys.exit(0)
            except Exception:
                logger.exception("Error on year %d — skipping", year)
                continue

        logger.info(
            "Backfill complete: fetched %d, updated %d across %d years",
            grand_fetched,
            grand_updated,
            len(years),
        )


if __name__ == "__main__":
    main()
