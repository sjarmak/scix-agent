"""ADS data sync: fetch records from NASA ADS API into JSONL files."""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# All fields to request from ADS (same as the original harvest scripts).
ADS_FIELDS = ",".join(
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
        # "body" omitted: full text is not stored in our schema and causes
        # ADS gateway timeouts (504) on large batches for high-volume years.
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


@dataclass(frozen=True)
class SyncState:
    """Resume state for a single file/query."""

    filename: str
    records_fetched: int


def _load_state(state_path: Path) -> dict[str, SyncState]:
    """Load sync state from JSON file."""
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        data = json.load(f)
    return {
        k: SyncState(filename=v["filename"], records_fetched=v["records_fetched"])
        for k, v in data.items()
    }


def _save_state(state_path: Path, states: dict[str, SyncState]) -> None:
    """Save sync state to JSON file."""
    data = {
        k: {"filename": v.filename, "records_fetched": v.records_fetched} for k, v in states.items()
    }
    with open(state_path, "w") as f:
        json.dump(data, f, indent=2)


class SyncClient:
    """Fetch records from the ADS API with pagination, retry, and resume."""

    def __init__(
        self,
        api_key: str | None = None,
        batch_size: int = 2000,
        throttle: float = 1.0,
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key or os.environ["ADS_API_KEY"]
        self._batch_size = batch_size
        self._throttle = throttle
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def count(self, query: str) -> int:
        """Get the total record count for a query."""
        resp = self._request(query, start=0, rows=0)
        return resp.get("response", {}).get("numFound", 0)

    def fetch_to_file(
        self,
        query: str,
        output_path: Path,
        label: str,
        resume_offset: int = 0,
    ) -> int:
        """Fetch all records matching query, append to gzipped JSONL file.

        Returns total records written (including resume offset).
        """
        start = resume_offset
        total_written = resume_offset
        t_start = time.monotonic()

        # Get total count for progress reporting
        total_expected = self.count(query)
        logger.info(
            "%s: %d records expected, resuming from %d",
            label,
            total_expected,
            start,
        )

        mode = "at" if resume_offset > 0 else "wt"
        with gzip.open(output_path, mode, encoding="utf-8") as f:
            while True:
                docs = self._fetch_batch(query, start)
                if not docs:
                    break

                for doc in docs:
                    f.write(json.dumps(doc) + "\n")
                    total_written += 1

                start += len(docs)

                elapsed = time.monotonic() - t_start
                rate = (total_written - resume_offset) / elapsed if elapsed > 0 else 0
                pct = total_written / total_expected * 100 if total_expected > 0 else 0
                logger.info(
                    "%s: %d/%d (%.1f%%) %.0f rec/s",
                    label,
                    total_written,
                    total_expected,
                    pct,
                    rate,
                )

                time.sleep(self._throttle)

        logger.info("%s: complete — %d records", label, total_written)
        return total_written

    def _fetch_batch(self, query: str, start: int) -> list[dict[str, Any]]:
        """Fetch a single batch with retry."""
        resp = self._request(query, start=start, rows=self._batch_size)
        return resp.get("response", {}).get("docs", [])

    def _request(
        self, query: str, start: int = 0, rows: int = 0, max_retries: int = 20
    ) -> dict[str, Any]:
        """Make an API request with exponential backoff retry.

        Raises RuntimeError after max_retries consecutive failures.
        Rate-limit (429) retries do not count toward the limit.
        """
        params = {"q": query, "start": start, "rows": rows, "fl": ADS_FIELDS}
        attempt = 0
        while attempt < max_retries:
            try:
                resp = requests.get(
                    ADS_API_URL,
                    headers=self._headers,
                    params=params,
                    timeout=self._timeout,
                )
                # Adaptive throttle from rate limit headers
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining is not None and int(remaining) < 10:
                    reset = resp.headers.get("X-RateLimit-Reset")
                    if reset:
                        wait = max(1, int(reset) - int(time.time()))
                        logger.warning(
                            "Rate limit low (%s remaining), waiting %ds", remaining, wait
                        )
                        time.sleep(wait)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning("Rate limited (429), waiting %ds", wait)
                    time.sleep(wait)
                    continue  # 429 does not count as a failure attempt
                else:
                    logger.warning(
                        "HTTP %d (attempt %d/%d): %s",
                        resp.status_code,
                        attempt + 1,
                        max_retries,
                        resp.text[:200],
                    )
            except requests.exceptions.RequestException as e:
                logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)

            attempt += 1
            time.sleep(min(60, 2 ** min(attempt, 6)))

        raise RuntimeError(
            f"ADS API request failed after {max_retries} retries: q={params['q']}, start={start}"
        )


def harvest_years(
    client: SyncClient,
    years: range,
    data_dir: Path,
) -> list[Path]:
    """Fetch all records for each year, with resume support.

    Returns list of output file paths.
    """
    state_path = data_dir / ".sync_state.json"
    states = _load_state(state_path)
    output_files: list[Path] = []

    for year in years:
        key = f"year_{year}"
        filename = f"ads_metadata_{year}_full.jsonl.gz"
        output_path = data_dir / filename
        resume_offset = states.get(key, SyncState(filename, 0)).records_fetched

        # Check if already complete (state marks it)
        if key in states and states[key].records_fetched > 0:
            expected = client.count(f"year:{year}")
            if states[key].records_fetched >= expected:
                logger.info(
                    "Year %d: already complete (%d records), skipping",
                    year,
                    states[key].records_fetched,
                )
                output_files.append(output_path)
                continue

        try:
            total = client.fetch_to_file(
                query=f"year:{year}",
                output_path=output_path,
                label=f"Year {year}",
                resume_offset=resume_offset,
            )
        except RuntimeError as e:
            logger.error("Year %d: FAILED — %s, moving to next year", year, e)
            continue

        states[key] = SyncState(filename=filename, records_fetched=total)
        _save_state(state_path, states)
        output_files.append(output_path)

    return output_files


def incremental_sync(
    client: SyncClient,
    since: str,
    data_dir: Path,
) -> Path:
    """Fetch records entered/updated since a date using entdate range query.

    Args:
        since: Date string like '2026-03-01'

    Returns path to the output file.
    """
    query = f"entdate:[{since} TO *]"
    filename = f"ads_incremental_{since}.jsonl.gz"
    output_path = data_dir / filename

    client.fetch_to_file(
        query=query,
        output_path=output_path,
        label=f"Incremental since {since}",
    )
    return output_path


def fill_gaps(
    client: SyncClient,
    years: range,
    data_dir: Path,
    dsn: str | None = None,
) -> list[tuple[int, int, int]]:
    """Compare DB record counts vs ADS API counts per year.

    Returns list of (year, db_count, api_count) for years with gaps.
    """
    import psycopg

    from scix.db import get_connection

    conn = get_connection(dsn)
    gaps: list[tuple[int, int, int]] = []

    try:
        for year in years:
            api_count = client.count(f"year:{year}")
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM papers WHERE year = %s", (year,))
                db_count = cur.fetchone()[0]

            if db_count < api_count:
                gap = api_count - db_count
                logger.info(
                    "Year %d: DB has %d, API has %d (gap: %d)", year, db_count, api_count, gap
                )
                gaps.append((year, db_count, api_count))
            else:
                logger.info("Year %d: OK (%d/%d)", year, db_count, api_count)

            time.sleep(1)  # Respect rate limits between API count queries
    finally:
        conn.close()

    return gaps
