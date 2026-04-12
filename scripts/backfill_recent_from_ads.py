#!/usr/bin/env python3
"""Re-fetch recent ADS records that were missing body text or references at
initial harvest time.

When the daily harvest runs shortly after ADS indexes a new arxiv submission,
ADS often hasn't finished extracting full text OR reference lists yet, so
body is NULL and citation_edges has no outgoing edges. This script finds
papers harvested in the last N days that are missing either body or
references in the database, re-fetches them from ADS (which may now have
either populated), and writes a JSONL file compatible with the standard
ingest pipeline.

Only records that actually gained body OR references are written — records
where the re-fetch yields nothing new are skipped to avoid pointless rewrites.

Distinct from backfill_body.py, which promotes body from raw JSONB already
stored in the database (one-time migration, no ADS API calls).

Usage:
    python scripts/backfill_recent_from_ads.py [--days 7] [--output-dir data/daily_harvest]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import psycopg
import requests

# Reuse the canonical field list from the daily harvest script so backfilled
# records match the shape the ingest pipeline expects.
sys.path.insert(0, str(Path(__file__).parent))
from harvest_daily import FIELDS  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

BIGQUERY_URL = "https://api.adsabs.harvard.edu/v1/search/bigquery"
BATCH_SIZE = 500
TIMEOUT = 120
THROTTLE = 1.0
MAX_RETRIES = 10


@dataclass(frozen=True)
class Candidate:
    """A paper that is missing body, references, or both in the DB."""

    bibcode: str
    has_body: bool
    has_edges: bool


def _get_headers() -> dict[str, str]:
    api_key = os.environ.get("ADS_API_KEY")
    if not api_key:
        logger.error("ADS_API_KEY environment variable is not set")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "big-query/csv",
    }


def find_candidates(dsn: str, days: int) -> list[Candidate]:
    """Return candidates harvested within the last N days that are missing
    body text, outgoing citation edges, or both."""
    sql = """
        SELECT p.bibcode,
               (p.body IS NOT NULL AND p.body <> '') AS has_body,
               EXISTS (
                   SELECT 1 FROM citation_edges ce
                   WHERE ce.source_bibcode = p.bibcode
               ) AS has_edges
        FROM papers p
        WHERE p.entry_date >= (CURRENT_DATE - %s::int)::text
          AND (
              p.body IS NULL
              OR p.body = ''
              OR NOT EXISTS (
                  SELECT 1 FROM citation_edges ce
                  WHERE ce.source_bibcode = p.bibcode
              )
          )
        ORDER BY p.bibcode
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (days,))
        return [
            Candidate(bibcode=row[0], has_body=row[1], has_edges=row[2]) for row in cur.fetchall()
        ]


def fetch_batch(headers: dict[str, str], bibcodes: list[str]) -> list[dict]:
    """Fetch full records for a list of bibcodes via ADS bigquery endpoint."""
    params = {"q": "*:*", "fl": FIELDS, "rows": len(bibcodes)}
    data = "bibcode\n" + "\n".join(bibcodes)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                BIGQUERY_URL, headers=headers, params=params, data=data, timeout=TIMEOUT
            )
            if resp.status_code == 200:
                return resp.json().get("response", {}).get("docs", [])
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
    logger.error("Max retries (%d) exceeded", MAX_RETRIES)
    sys.exit(1)


def backfill(output_dir: Path, days: int, dsn: str) -> Path | None:
    """Re-fetch recent papers that lack body or references from ADS."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching DB for recent papers missing body or references (last %d days)...", days)
    candidates = find_candidates(dsn, days)
    if not candidates:
        logger.info("Nothing to backfill — done")
        return None

    missing_body = sum(1 for c in candidates if not c.has_body)
    missing_edges = sum(1 for c in candidates if not c.has_edges)
    logger.info(
        "Found %d candidates (missing body: %d, missing edges: %d)",
        len(candidates),
        missing_body,
        missing_edges,
    )

    state: dict[str, Candidate] = {c.bibcode: c for c in candidates}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_file = output_dir / f"ads_backfill_{today}.jsonl.gz"
    headers = _get_headers()

    inspected = 0
    gained_body = 0
    gained_edges = 0
    written = 0
    bibcodes = [c.bibcode for c in candidates]

    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for start in range(0, len(bibcodes), BATCH_SIZE):
            batch = bibcodes[start : start + BATCH_SIZE]
            docs = fetch_batch(headers, batch)
            for doc in docs:
                inspected += 1
                bibcode = doc.get("bibcode")
                cand = state.get(bibcode) if bibcode else None
                if cand is None:
                    continue
                new_body = bool(doc.get("body"))
                new_refs = bool(doc.get("reference"))
                body_gain = new_body and not cand.has_body
                edge_gain = new_refs and not cand.has_edges
                if body_gain or edge_gain:
                    f.write(json.dumps(doc) + "\n")
                    written += 1
                    if body_gain:
                        gained_body += 1
                    if edge_gain:
                        gained_edges += 1
            logger.info(
                "Progress: %d / %d bibcodes (gained body: %d, gained edges: %d)",
                min(start + BATCH_SIZE, len(bibcodes)),
                len(bibcodes),
                gained_body,
                gained_edges,
            )
            sleep(THROTTLE)

    logger.info(
        "Inspected %d records: %d gained body, %d gained edges, %d written to file",
        inspected,
        gained_body,
        gained_edges,
        written,
    )

    if written == 0:
        logger.info("No records gained body or edges — removing empty output file")
        output_file.unlink(missing_ok=True)
        return None

    logger.info("Wrote %d enriched records to %s", written, output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-fetch missing body/references from ADS")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/daily_harvest"),
        help="Output directory (default: data/daily_harvest)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look back N days for candidates (default: 7)",
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

    backfill(args.output_dir, args.days, args.dsn)


if __name__ == "__main__":
    main()
