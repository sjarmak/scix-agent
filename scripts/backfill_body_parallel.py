#!/usr/bin/env python3
"""Parallel body backfill from ADS API with controlled concurrency.

Runs N years concurrently (default: 3) using threads, with a shared
rate limiter to avoid overwhelming the ADS API.

Usage:
    python scripts/backfill_body_parallel.py --start-year 2012 --end-year 2020 --workers 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

import psycopg
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# Shared rate limiter: minimum seconds between API calls across all threads
_rate_lock = threading.Lock()
_last_call = 0.0
MIN_INTERVAL = 1.0  # seconds between any two API calls


def rate_limited_get(session: requests.Session, **kwargs) -> requests.Response:
    """Make a rate-limited GET request."""
    global _last_call
    with _rate_lock:
        now = time.monotonic()
        wait = MIN_INTERVAL - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
    return session.get(**kwargs)


def fetch_body_batch(
    session: requests.Session,
    year: int,
    start: int,
    rows: int,
    timeout: int,
) -> tuple[list[dict], int, requests.Response | None]:
    """Fetch bibcode + body for papers in a given year."""
    params = {
        "q": f"year:{year}",
        "start": start,
        "rows": rows,
        "fl": "bibcode,body",
        "sort": "bibcode asc",
    }

    for attempt in range(10):
        try:
            resp = rate_limited_get(session, url=API_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json().get("response", {})
                return data.get("docs", []), data.get("numFound", 0), resp
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Year %d: rate limited, sleeping %ds", year, retry_after)
                time.sleep(retry_after)
                continue
            else:
                logger.warning(
                    "Year %d: HTTP %d attempt %d",
                    year,
                    resp.status_code,
                    attempt + 1,
                )
        except requests.exceptions.RequestException as e:
            logger.warning("Year %d: request error attempt %d: %s", year, attempt + 1, e)

        backoff = min(60, 2 ** min(attempt + 1, 6))
        time.sleep(backoff)

    logger.error("Year %d: failed after 10 attempts at start=%d", year, start)
    return [], 0, None


def backfill_year(
    api_key: str,
    dsn: str,
    year: int,
    start_offset: int = 0,
    rows: int = 2000,
    timeout: int = 120,
) -> tuple[int, int]:
    """Backfill body for one year. Returns (fetched, updated)."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"

    conn = psycopg.connect(dsn)
    conn.autocommit = False

    start = start_offset
    total_fetched = 0
    total_updated = 0

    try:
        while True:
            records, num_found, resp = fetch_body_batch(session, year, start, rows, timeout)

            if not records:
                break

            updated = 0
            with conn.cursor() as cur:
                for rec in records:
                    bibcode = rec.get("bibcode")
                    body = rec.get("body")
                    if not bibcode or not body:
                        continue
                    body = body.replace("\x00", "")
                    if not body:
                        continue
                    cur.execute(
                        "UPDATE papers SET body = %s WHERE bibcode = %s AND body IS NULL",
                        (body, bibcode),
                    )
                    updated += cur.rowcount
            conn.commit()

            total_fetched += len(records)
            total_updated += updated
            start += len(records)

            if total_fetched % 20000 == 0 or updated > 0:
                logger.info(
                    "Year %d: %d/%d fetched, %d updated (%.0f%%)",
                    year,
                    total_fetched,
                    num_found,
                    total_updated,
                    100.0 * total_fetched / num_found if num_found else 0,
                )

            if start >= num_found:
                break
    finally:
        conn.close()
        session.close()

    logger.info("Year %d DONE: fetched %d, updated %d", year, total_fetched, total_updated)
    return total_fetched, total_updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel body backfill from ADS API")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3, help="Max concurrent years")
    parser.add_argument("--rows", type=int, default=2000, help="Rows per API request")
    parser.add_argument("--timeout", type=int, default=120, help="API timeout seconds")
    parser.add_argument(
        "--start-offset",
        type=str,
        default="",
        help="Comma-separated year:offset pairs, e.g. '2005:248000,2006:268000'",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("ADS_API_KEY", "")
    if not api_key:
        logger.error("ADS_API_KEY not set")
        sys.exit(1)

    years = list(range(args.start_year, args.end_year + 1))

    offsets: dict[int, int] = {}
    if args.start_offset:
        for pair in args.start_offset.split(","):
            y, o = pair.split(":")
            offsets[int(y)] = int(o)

    logger.info(
        "Backfilling %d years (%d-%d) with %d workers, rows=%d, timeout=%ds",
        len(years),
        years[0],
        years[-1],
        args.workers,
        args.rows,
        args.timeout,
    )
    if offsets:
        logger.info("Start offsets: %s", offsets)

    results: dict[int, tuple[int, int]] = {}
    semaphore = threading.Semaphore(args.workers)

    def worker(year: int) -> None:
        with semaphore:
            try:
                results[year] = backfill_year(
                    api_key,
                    args.dsn,
                    year,
                    start_offset=offsets.get(year, 0),
                    rows=args.rows,
                    timeout=args.timeout,
                )
            except Exception:
                logger.exception("Year %d failed", year)
                results[year] = (0, 0)

    threads = []
    for year in years:
        t = threading.Thread(target=worker, args=(year,), name=f"y{year}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    grand_fetched = sum(f for f, _ in results.values())
    grand_updated = sum(u for _, u in results.values())
    logger.info(
        "All done: %d fetched, %d updated across %d years",
        grand_fetched,
        grand_updated,
        len(years),
    )
    for year in sorted(results):
        f, u = results[year]
        logger.info("  %d: fetched=%d updated=%d", year, f, u)


if __name__ == "__main__":
    main()
