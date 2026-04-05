#!/usr/bin/env python3
"""Harvest Papers With Code methods into entity_dictionary.

Downloads the PWC methods JSON from their production media endpoint,
parses ML method entries into canonical_name + aliases + description,
and loads into entity_dictionary with entity_type='method', source='pwc'.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load

logger = logging.getLogger(__name__)

PWC_METHODS_URL = "https://production-media.paperswithcode.com/about/methods.json.gz"

_DEFAULT_DEST = Path("data/methods.json.gz")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_methods(dest: Path | None = None) -> Path:
    """Download the PWC methods JSON gzip from the canonical location.

    Skips download if the destination file already exists and is non-empty.

    Args:
        dest: Path to save the downloaded file. Defaults to data/methods.json.gz.

    Returns:
        Path to the downloaded (or existing) file.

    Raises:
        urllib.error.URLError: If the download fails after retries.
    """
    if dest is None:
        dest = _DEFAULT_DEST

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("PWC methods file already exists at %s, skipping download", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PWC methods from %s", PWC_METHODS_URL)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                PWC_METHODS_URL,
                headers={"User-Agent": "scix-experiments/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()

            dest.write_bytes(data)
            logger.info("Downloaded PWC methods: %d bytes -> %s", len(data), dest)
            return dest

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
                    "Failed to download PWC methods after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise

    raise RuntimeError("download_methods: unexpected exit from retry loop")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_methods(data_path: Path) -> list[dict[str, Any]]:
    """Parse the PWC methods JSON into entity_dictionary entries.

    Each method object in the JSON array is expected to have at minimum
    a ``name`` field. Optional fields: ``full_name``, ``description``,
    ``paper``, ``introduced_year``, ``source_url``, ``main_collection``.

    Args:
        data_path: Path to the gzip-compressed JSON file.

    Returns:
        List of dicts suitable for ``dictionary.bulk_load()``.
    """
    logger.info("Parsing PWC methods from %s", data_path)

    with gzip.open(data_path, "rt", encoding="utf-8") as fh:
        raw_methods: list[dict[str, Any]] = json.load(fh)

    entries: list[dict[str, Any]] = []
    skipped = 0

    for method in raw_methods:
        name = (method.get("name") or "").strip()
        full_name = (method.get("full_name") or "").strip()

        if not name and not full_name:
            skipped += 1
            continue

        canonical_name = full_name if full_name else name
        aliases: list[str] = []
        if name and name != canonical_name:
            aliases.append(name)

        # Build metadata from available fields
        metadata: dict[str, Any] = {}
        description = (method.get("description") or "").strip()
        if description:
            metadata["description"] = description

        introduced_year = method.get("introduced_year")
        if introduced_year is not None:
            metadata["introduced_year"] = introduced_year

        source_url = (method.get("source_url") or "").strip()
        if source_url:
            metadata["source_url"] = source_url

        collection = method.get("main_collection")
        if collection:
            # collection may be a dict or string depending on schema version
            if isinstance(collection, dict):
                coll_name = (collection.get("name") or "").strip()
                if coll_name:
                    metadata["collection"] = coll_name
            elif isinstance(collection, str) and collection.strip():
                metadata["collection"] = collection.strip()

        paper = method.get("paper")
        if paper and isinstance(paper, dict):
            paper_meta: dict[str, Any] = {}
            for key in ("title", "url", "arxiv_id"):
                val = paper.get(key)
                if val and isinstance(val, str) and val.strip():
                    paper_meta[key] = val.strip()
            if paper_meta:
                metadata["paper"] = paper_meta

        entries.append(
            {
                "canonical_name": canonical_name,
                "entity_type": "method",
                "source": "pwc",
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d methods with no name", skipped)

    logger.info("Parsed %d method entries from PWC data", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_methods(entries: list[dict[str, Any]], dsn: str | None = None) -> int:
    """Load parsed PWC method entries into entity_dictionary.

    Args:
        entries: List of entry dicts from ``parse_methods()``.
        dsn: Database connection string. Uses SCIX_DSN or default if None.

    Returns:
        Number of rows upserted.
    """
    conn = get_connection(dsn)
    try:
        count = bulk_load(conn, entries)
        logger.info("Loaded %d PWC method entries into entity_dictionary", count)
        return count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(data_path: Path | None = None, dsn: str | None = None) -> int:
    """Run the full PWC methods harvesting pipeline.

    Steps:
        1. Download methods.json.gz (if not provided)
        2. Parse method entries
        3. Load into entity_dictionary

    Args:
        data_path: Path to a local methods.json.gz. Downloads if None.
        dsn: Database connection string. Uses SCIX_DSN or default if None.

    Returns:
        Number of methods loaded.
    """
    t0 = time.monotonic()

    if data_path is None:
        data_path = download_methods()
    elif not data_path.exists():
        raise FileNotFoundError(f"Methods file not found: {data_path}")

    entries = parse_methods(data_path)

    if not entries:
        logger.warning("No method entries parsed — nothing to load")
        return 0

    count = load_methods(entries, dsn=dsn)

    elapsed = time.monotonic() - t0
    logger.info(
        "PWC methods pipeline complete: %d methods loaded in %.1fs",
        count,
        elapsed,
    )
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the PWC methods harvesting pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest Papers With Code methods into entity_dictionary",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Path to methods.json.gz (downloads if not provided)",
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

    count = run_pipeline(data_path=args.data_file, dsn=args.dsn)
    logger.info("Done. %d methods loaded.", count)


if __name__ == "__main__":
    main()
