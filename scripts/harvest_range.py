#!/usr/bin/env python3
"""One-off ADS backfill harvest for an entdate range (bead zyuc).

Adapted from harvest_daily.py (field list) and harvest_full.py (rows=2000,
sort=bibcode, 429/backoff handling). Differences from harvest_daily.py:

  - Takes --start-date / --end-date (both inclusive, day granularity) and
    iterates day by day. ADS ``entdate`` is a datetime, so a day D is queried
    as ``entdate:[D TO D+1]`` (the upper bound only adds the midnight instant
    of D+1; records stamped exactly at midnight can appear in two adjacent
    day chunks — ingest's ON CONFLICT dedupes them).
  - Field list EXCLUDES ``body``: the backfill window holds ~10.8M ADS
    records (~3.4M missing locally); with bodies both the download and the
    papers-table growth blow the disk budget on this host. Body enrichment
    for the delta is follow-up work.
  - Filters out bibcodes already present in the papers table before writing
    (read-only SELECTs). This keeps the file to genuinely-missing records —
    required because ingest's merge overwrites every column, so re-ingesting
    an existing paper from a body-less harvest would null its body.
  - Never touches the daily-sync watermark (data/daily_harvest/last_run.txt).
  - Resume support via a sidecar progress file: <output>.progress.json.

Usage:
    python scripts/harvest_range.py --start-date 2025-05-01 --end-date 2025-05-31

Exit codes: 0 = complete, 2 = stopped gracefully mid-range (progress saved).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import psycopg
import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# harvest_daily.py's field list minus "body" (see module docstring).
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

ROWS_PER_PAGE = 2000  # ADS API max
TIMEOUT = 120
MAX_RETRIES = 10
RATE_LIMIT_BUFFER = 50


class HarvestStopped(Exception):
    """Raised when the harvest must stop before the range is complete."""


def _get_headers() -> dict[str, str]:
    api_key = os.environ.get("ADS_API_KEY")
    if not api_key:
        logger.error("ADS_API_KEY environment variable is not set")
        sys.exit(1)
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def adaptive_throttle(resp: requests.Response, default: float) -> float:
    """Sleep time based on rate-limit headers (harvest_full.py pattern)."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        remaining_n = int(remaining)
        if remaining_n < 10:
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset:
                return max(0, int(reset) - int(time.time())) + 5
            return 300
        if remaining_n < RATE_LIMIT_BUFFER:
            return default * 3
    return default


def fetch_page(
    headers: dict[str, str], query: str, start: int, throttle: float
) -> tuple[list[dict], int, float]:
    """Fetch one page. Returns (docs, num_found, sleep_time).

    Raises HarvestStopped after MAX_RETRIES consecutive failures so the
    caller can persist progress and exit gracefully.
    """
    params = {
        "q": query,
        "start": start,
        "rows": ROWS_PER_PAGE,
        "fl": FIELDS,
        "sort": "bibcode asc",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)
            if resp.status_code == 200:
                body = resp.json().get("response", {})
                return (
                    body.get("docs", []),
                    body.get("numFound", 0),
                    adaptive_throttle(resp, throttle),
                )
            if resp.status_code == 400:
                logger.error("HTTP 400 (bad request, not retrying): %s", resp.text[:500])
                raise HarvestStopped("HTTP 400 from ADS")
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Rate limited (429); sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            logger.warning(
                "HTTP %d (attempt %d/%d): %s",
                resp.status_code,
                attempt + 1,
                MAX_RETRIES,
                resp.text[:300],
            )
        except requests.exceptions.RequestException as e:
            logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
        time.sleep(min(120, 2 ** min(attempt + 1, 7)))
    raise HarvestStopped(f"Max retries ({MAX_RETRIES}) exceeded for {query} start={start}")


def existing_bibcodes(conn: psycopg.Connection, bibcodes: list[str]) -> set[str]:
    """Return the subset of bibcodes already present in papers (read-only)."""
    if not bibcodes:
        return set()
    with conn.cursor() as cur:
        cur.execute("SELECT bibcode FROM papers WHERE bibcode = ANY(%s)", (bibcodes,))
        return {row[0] for row in cur.fetchall()}


def load_progress(progress_path: Path) -> dict | None:
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    return None


def _recoverable_chunks(path: Path):
    """Yield decompressed byte chunks from a damaged multi-member gzip file.

    Decodes the maximal prefix of each member; on hitting corrupt bytes (a
    member truncated by a mid-write kill), scans forward for the next gzip
    magic and resumes with a fresh decompressor, so intact members appended
    after a truncated one are still recovered. Do NOT use zcat for this: on
    a decode error it exits without flushing already-decoded stdout, silently
    dropping durable records (observed 2026-07-14).
    """
    import zlib

    magic = b"\x1f\x8b\x08"
    data = path.read_bytes()
    n = len(data)
    pos = 0
    while pos < n:
        start = data.find(magic, pos)
        if start < 0:
            return
        member = data[start:]
        d = zlib.decompressobj(wbits=47)
        try:
            out = d.decompress(member)
            if out:
                yield out
            if d.eof:
                nxt = n - len(d.unused_data)  # clean member; next starts here
                pos = nxt if nxt > start else start + 3  # defensive: no stall
                continue
            return  # trailer-less final member: durable prefix fully decoded
        except zlib.error:
            # Truncated member with bytes after the seam. zlib discards the
            # raising call's output, so bisect the longest error-free prefix
            # (error position is fixed -> monotone -> bisection is exact).
            # Junk decoded past the true seam cannot survive the caller's
            # JSON-line filter; complete flushed lines end in a newline, so
            # junk never glues onto a durable record.
            lo, hi, best = 0, len(member), b""
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                d2 = zlib.decompressobj(wbits=47)
                try:
                    best_mid = d2.decompress(member[:mid])
                except zlib.error:
                    hi = mid
                else:
                    lo, best = mid, best_mid
            if best:
                yield best
            pos = start + 3  # hunt for the next member past this one's start


def salvage_partial_output(output_file: Path, expected_written: int) -> int:
    """Rewrite a possibly-truncated gzip output as a clean single-member file.

    A run killed mid-write leaves the current gzip member without a trailer;
    appending another member after it breaks sequential decompression at the
    seam (observed 2026-07-14: ``zlib.error: Error -3 ... invalid block
    type`` at the kill boundary). Before any resume-append, decode every
    recoverable line (each page is flushed before its progress save, so all
    sidecar-counted pages are durable under SIGKILL), drop unparseable
    partial lines, and atomically replace the file with a clean copy.

    Exits with code 3 if fewer than ``expected_written`` records could be
    recovered — that means real corruption beyond the kill-window and the
    range must be re-harvested fresh rather than silently resumed short.
    """
    tmp = output_file.with_suffix(".salvage.tmp.gz")
    kept = 0
    tail = b""
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for chunk in _recoverable_chunks(output_file):
            tail += chunk
            *lines, tail = tail.split(b"\n")
            for raw in lines:
                if not raw:
                    continue
                try:
                    if not json.loads(raw).get("bibcode"):
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue  # partial line at a kill boundary
                out.write(raw.decode("utf-8") + "\n")
                kept += 1
    if kept < expected_written:
        tmp.unlink(missing_ok=True)
        logger.error(
            "Salvage recovered %d records but progress sidecar says %d were "
            "durably written — refusing to resume short. Delete %s and its "
            ".progress.json sidecar and re-harvest the range.",
            kept,
            expected_written,
            output_file,
        )
        sys.exit(3)
    tmp.replace(output_file)
    logger.info(
        "Salvaged %d records from partial output %s (sidecar expected >= %d)",
        kept,
        output_file,
        expected_written,
    )
    return kept


def save_progress(progress_path: Path, day: str, start: int, written: int, fetched: int) -> None:
    progress_path.write_text(
        json.dumps({"day": day, "start": start, "written": written, "fetched": fetched}) + "\n"
    )


def harvest_range(
    start_date: date,
    end_date: date,
    output_dir: Path,
    dsn: str | None,
    throttle: float,
    filter_existing: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ads_range_{start_date}_{end_date}.jsonl.gz"
    progress_path = Path(str(output_file) + ".progress.json")

    headers = _get_headers()
    conn = psycopg.connect(dsn or os.environ.get("SCIX_DSN", "dbname=scix")) if filter_existing else None

    progress = load_progress(progress_path)
    resume_day: date | None = None
    resume_start = 0
    total_written = 0
    total_fetched = 0
    if progress:
        resume_day = date.fromisoformat(progress["day"])
        resume_start = int(progress["start"])
        total_written = int(progress["written"])
        total_fetched = int(progress["fetched"])
        logger.info("Resuming at day=%s start=%d (written so far: %d)", resume_day, resume_start, total_written)
        if output_file.exists():
            # The previous run may have been killed mid-write, leaving a
            # trailer-less gzip member; appending after it would corrupt the
            # file at the seam. Rewrite the recoverable content as a clean
            # member first (atomic replace), then append.
            salvage_partial_output(output_file, total_written)

    seen_this_run: set[str] = set()
    t0 = time.monotonic()

    try:
        with gzip.open(output_file, "at", encoding="utf-8") as f:
            day = resume_day or start_date
            while day <= end_date:
                day_start = resume_start if (resume_day and day == resume_day) else 0
                query = f"entdate:[{day} TO {day + timedelta(days=1)}]"
                start = day_start
                docs, num_found, sleep_t = fetch_page(headers, query, start, throttle)
                if num_found:
                    logger.info("%s: %d records at ADS", day, num_found)
                while docs:
                    page_bibcodes = [d.get("bibcode") for d in docs if d.get("bibcode")]
                    skip = existing_bibcodes(conn, page_bibcodes) if conn else set()
                    written_page = 0
                    for d in docs:
                        bib = d.get("bibcode")
                        if not bib or bib in skip or bib in seen_this_run:
                            continue
                        f.write(json.dumps(d) + "\n")
                        seen_this_run.add(bib)
                        written_page += 1
                    f.flush()
                    total_fetched += len(docs)
                    total_written += written_page
                    start += ROWS_PER_PAGE
                    save_progress(progress_path, str(day), start, total_written, total_fetched)
                    if start >= num_found:
                        break
                    time.sleep(sleep_t)
                    docs, num_found, sleep_t = fetch_page(headers, query, start, throttle)
                if num_found:
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "%s done: cumulative fetched=%d written=%d (%.0f rec/s)",
                        day,
                        total_fetched,
                        total_written,
                        total_fetched / elapsed if elapsed > 0 else 0,
                    )
                day += timedelta(days=1)
                save_progress(progress_path, str(day), 0, total_written, total_fetched)
    except HarvestStopped as e:
        logger.error("Harvest stopped early: %s — progress saved to %s", e, progress_path)
        if conn:
            conn.close()
        sys.exit(2)
    finally:
        if conn:
            conn.close()

    progress_path.unlink(missing_ok=True)
    logger.info(
        "Range %s..%s complete: fetched=%d, written=%d (new), skipped=%d (existing/dup) → %s",
        start_date,
        end_date,
        total_fetched,
        total_written,
        total_fetched - total_written,
        output_file,
    )
    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(description="One-off ADS entdate-range backfill harvest")
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat, help="inclusive")
    parser.add_argument("--output-dir", type=Path, default=Path("data/daily_harvest"))
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN for the existing-bibcode filter")
    parser.add_argument(
        "--no-filter-existing",
        action="store_true",
        help="Write all fetched records without checking the papers table",
    )
    parser.add_argument("--throttle", type=float, default=0.3, help="seconds between pages")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.end_date < args.start_date:
        parser.error("--end-date must be >= --start-date")

    harvest_range(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        dsn=args.dsn,
        throttle=args.throttle,
        filter_existing=not args.no_filter_existing,
    )


if __name__ == "__main__":
    main()
