#!/usr/bin/env python3
"""Harvest AAS Facility Keywords into entity_dictionary.

Downloads the AAS Facility Keywords HTML table from
https://journals.aas.org/facility-keywords/, parses each facility entry
with its wavelength regime flags, location, and facility type metadata,
and bulk-loads them via scix.dictionary.bulk_load() with
entity_type='instrument', source='aas'.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load

logger = logging.getLogger(__name__)

AAS_FACILITIES_URL = "https://journals.aas.org/facility-keywords/"

# Column name normalization: map header substrings to metadata keys
_WAVELENGTH_REGIMES = (
    "gamma_ray",
    "x_ray",
    "ultraviolet",
    "optical",
    "infrared",
    "millimeter",
    "radio",
    "neutrinos_particles_gw",
)

_FACILITY_FLAGS = (
    "solar",
    "archive_database",
    "computational_center",
)

# Map from lowercase header substring to normalized key
_HEADER_KEY_MAP: dict[str, str] = {
    "gamma": "gamma_ray",
    "x-ray": "x_ray",
    "x ray": "x_ray",
    "ultraviolet": "ultraviolet",
    "optical": "optical",
    "infrared": "infrared",
    "millimeter": "millimeter",
    "radio": "radio",
    "neutrino": "neutrinos_particles_gw",
    "gravitational": "neutrinos_particles_gw",
    "solar": "solar",
    "archive": "archive_database",
    "database": "archive_database",
    "computational": "computational_center",
}


class _FacilityTableParser(HTMLParser):
    """Parse AAS facility keywords HTML table into rows of cell values."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table: bool = False
        self.in_thead: bool = False
        self.in_tbody: bool = False
        self.in_row: bool = False
        self.in_cell: bool = False
        self.current_cell: str = ""
        self.current_row: list[str] = []
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._table_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self.in_table = True
        if not self.in_table:
            return
        if tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                self.in_table = False
        if not self.in_table and self._table_depth == 0:
            return
        if tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag in ("td", "th"):
            if self.in_cell:
                self.current_row.append(self.current_cell.strip())
                self.in_cell = False
        elif tag == "tr":
            if self.in_row:
                if self.in_thead or (not self.headers and self.current_row):
                    # First row with content becomes headers
                    if self.current_row and any(self.current_row):
                        self.headers = self.current_row
                elif self.current_row:
                    self.rows.append(self.current_row)
                self.in_row = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell += data


def _classify_header(header: str) -> str | None:
    """Map a table header string to a normalized metadata key.

    Returns None if the header is not a recognized flag column.
    """
    lower = header.lower()
    for substring, key in _HEADER_KEY_MAP.items():
        if substring in lower:
            return key
    return None


def download_aas_facilities(url: str = AAS_FACILITIES_URL) -> str:
    """Download the AAS Facility Keywords HTML page.

    Uses urllib.request with retry and exponential backoff.

    Args:
        url: URL of the AAS facility keywords page.

    Returns:
        Raw HTML string.

    Raises:
        urllib.error.URLError: If the download fails after retries.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "scix-experiments/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()

            html = data.decode("utf-8", errors="replace")
            logger.info("Downloaded AAS facility keywords: %d bytes", len(data))
            return html

        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Download attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to download AAS facility keywords after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise

    # Unreachable, but satisfies type checker
    raise RuntimeError("download_aas_facilities: unexpected exit from retry loop")


def parse_aas_facilities(html: str) -> list[dict[str, Any]]:
    """Parse AAS facility keywords HTML into entity_dictionary records.

    Each returned dict has keys compatible with dictionary.bulk_load():
    canonical_name, entity_type, source, aliases, metadata.

    Args:
        html: Raw HTML string from the AAS facility keywords page.

    Returns:
        List of entity dictionary entry dicts.
    """
    parser = _FacilityTableParser()
    parser.feed(html)

    if not parser.headers:
        logger.warning("No table headers found in AAS facility keywords HTML")
        return []

    # Build column index mapping
    # First column: full name, second column: keyword, third: location
    # Remaining columns: flag columns
    flag_col_map: dict[int, str] = {}
    for idx, header in enumerate(parser.headers):
        key = _classify_header(header)
        if key is not None:
            flag_col_map[idx] = key

    logger.info(
        "Found %d headers, %d flag columns, %d data rows",
        len(parser.headers),
        len(flag_col_map),
        len(parser.rows),
    )

    entries: list[dict[str, Any]] = []
    skipped = 0

    for row in parser.rows:
        if len(row) < 2:
            skipped += 1
            continue

        full_name = row[0].strip() if len(row) > 0 else ""
        keyword = row[1].strip() if len(row) > 1 else ""
        location = row[2].strip() if len(row) > 2 else ""

        if not full_name and not keyword:
            skipped += 1
            continue

        # Use full name as canonical; fall back to keyword if empty
        canonical_name = full_name if full_name else keyword

        # Build aliases from keyword
        aliases: list[str] = []
        if keyword and keyword != canonical_name:
            aliases.append(keyword)

        # Collect active wavelength regimes and facility flags
        wavelength_regimes: list[str] = []
        facility_flags: list[str] = []

        for col_idx, key in flag_col_map.items():
            if col_idx < len(row) and row[col_idx].strip():
                if key in _WAVELENGTH_REGIMES:
                    wavelength_regimes.append(key)
                elif key in _FACILITY_FLAGS:
                    facility_flags.append(key)

        metadata: dict[str, Any] = {
            "wavelength_regimes": wavelength_regimes,
        }
        if location:
            metadata["location"] = location
        if facility_flags:
            metadata["facility_flags"] = facility_flags

        entries.append(
            {
                "canonical_name": canonical_name,
                "entity_type": "instrument",
                "source": "aas",
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d rows missing facility name and keyword", skipped)

    logger.info("Parsed %d AAS facility entries into dictionary records", len(entries))
    return entries


def run_harvest(dsn: str | None = None, url: str = AAS_FACILITIES_URL) -> int:
    """Run the full AAS facility keywords harvest pipeline.

    Downloads the page, parses entries, and loads them into entity_dictionary.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        url: URL of the AAS facility keywords page.

    Returns:
        Number of entries loaded.
    """
    t0 = time.monotonic()

    html = download_aas_facilities(url)
    entries = parse_aas_facilities(html)

    conn = get_connection(dsn)
    try:
        count = bulk_load(conn, entries)
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "AAS facility harvest complete: %d entries loaded in %.1fs",
        count,
        elapsed,
    )
    return count


def main() -> None:
    """Parse arguments and run the AAS facility keywords harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest AAS Facility Keywords into entity_dictionary",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
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

    count = run_harvest(dsn=args.dsn)
    print(f"Loaded {count} AAS facility entries into entity_dictionary")


if __name__ == "__main__":
    main()
