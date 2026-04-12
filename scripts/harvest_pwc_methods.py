#!/usr/bin/env python3
"""Harvest Papers With Code methods into entity_dictionary.

Downloads the PWC methods parquet from the HuggingFace pwc-archive dataset,
parses ML method entries into canonical_name + aliases + description,
and loads into entity_dictionary with entity_type='method', source='pwc'.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.harvest_utils import HarvestRunLog
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

PWC_METHODS_URL = (
    "https://huggingface.co/datasets/pwc-archive/methods"
    "/resolve/main/data/train-00000-of-00001.parquet"
)

_DEFAULT_DEST = Path("data/methods.parquet")

# ---------------------------------------------------------------------------
# Module-level client (lazy init)
# ---------------------------------------------------------------------------

_client: ResilientClient | None = None


def _get_client() -> ResilientClient:
    """Return a shared ResilientClient instance."""
    global _client
    if _client is None:
        _client = ResilientClient(
            user_agent="scix-experiments/1.0",
            max_retries=3,
            backoff_base=2.0,
            rate_limit=10.0,
            timeout=120.0,
        )
    return _client


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_methods(dest: Path | None = None) -> Path:
    """Download the PWC methods parquet from HuggingFace.

    Skips download if the destination file already exists and is non-empty.
    Uses ResilientClient with built-in retry and exponential backoff.

    Args:
        dest: Path to save the downloaded file. Defaults to data/methods.parquet.

    Returns:
        Path to the downloaded (or existing) file.
    """
    if dest is None:
        dest = _DEFAULT_DEST

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("PWC methods file already exists at %s, skipping download", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PWC methods from %s", PWC_METHODS_URL)

    client = _get_client()
    response = client.get(PWC_METHODS_URL)
    if hasattr(response, "content"):
        data = response.content
    else:
        data = response.text.encode("utf-8")

    dest.write_bytes(data)
    logger.info("Downloaded PWC methods: %d bytes -> %s", len(data), dest)
    return dest


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_methods(data_path: Path) -> list[dict[str, Any]]:
    """Parse the PWC methods parquet into entity_dictionary entries.

    Each row is expected to have at minimum a ``name`` column. Optional
    columns: ``full_name``, ``description``, ``paper``, ``introduced_year``,
    ``source_url``, ``collections``.

    Args:
        data_path: Path to the parquet file.

    Returns:
        List of dicts suitable for ``dictionary.bulk_load()``.
    """
    logger.info("Parsing PWC methods from %s", data_path)

    df = pd.read_parquet(data_path)

    entries: list[dict[str, Any]] = []
    skipped = 0

    def _str(val: Any) -> str:
        """Coerce a value to stripped string, treating NaN/None as empty."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val).strip()

    for row in df.itertuples(index=False):
        name = _str(getattr(row, "name", None))
        full_name = _str(getattr(row, "full_name", None))

        if not name and not full_name:
            skipped += 1
            continue

        canonical_name = full_name if full_name else name
        aliases: list[str] = []
        if name and name != canonical_name:
            aliases.append(name)

        metadata: dict[str, Any] = {}
        description = _str(getattr(row, "description", None))
        if description:
            metadata["description"] = description

        introduced_year = getattr(row, "introduced_year", None)
        if introduced_year is not None and not (
            isinstance(introduced_year, float) and pd.isna(introduced_year)
        ):
            metadata["introduced_year"] = int(introduced_year)

        source_url = _str(getattr(row, "source_url", None))
        if source_url:
            metadata["source_url"] = source_url

        # Collections field (array of dicts with area/collection keys)
        collections = getattr(row, "collections", None)
        if collections is not None and hasattr(collections, "__len__") and len(collections) > 0:
            try:
                first = collections[0]
                if isinstance(first, dict):
                    coll_name = (first.get("collection") or "").strip()
                    if coll_name:
                        metadata["collection"] = coll_name
            except (IndexError, TypeError):
                pass

        paper = getattr(row, "paper", None)
        if paper and isinstance(paper, dict):
            paper_meta: dict[str, Any] = {}
            for key in ("title", "url"):
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
    Logs harvest run to harvest_runs.

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

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, "pwc")
    try:
        run_log.start()
        count = bulk_load(conn, entries)
        run_log.complete(
            records_fetched=len(entries),
            records_upserted=count,
        )
        logger.info("Loaded %d PWC method entries into entity_dictionary", count)
    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to update harvest_run status to 'failed'")
        raise
    finally:
        conn.close()

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
        help="Path to methods.parquet (downloads if not provided)",
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
