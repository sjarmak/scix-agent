#!/usr/bin/env python3
"""Harvest AstroMLab concept vocabulary into entity_dictionary.

Downloads the AstroMLab 5 concept vocabulary (~9,999 concepts with categories)
from GitHub, parses concepts into entries grouped by category, and loads into
entity_dictionary with appropriate entity_type based on category mapping,
source='astromlab'.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.harvest_utils import HarvestRunLog
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

# GitHub raw URL for AstroMLab concept vocabulary.
# The AstroMLab project publishes concept lists as part of their benchmarks.
ASTROMLAB_CONCEPTS_URL = (
    "https://raw.githubusercontent.com/AstroMLab/AstroMLab-Benchmark-Dataset/"
    "main/astro_concepts.csv"
)

_DEFAULT_DEST = Path("data/astromlab_concepts.csv")

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
# Category -> entity_type mapping
# ---------------------------------------------------------------------------

# Substrings (lowercased) that map to specific entity types.
# Order matters: first match wins.
_CATEGORY_INSTRUMENT_KEYWORDS = (
    "instrumental",
    "instrumentation",
    "instrument",
    "telescope",
    "detector",
)

_CATEGORY_DATASET_KEYWORDS = (
    "survey",
    "catalog",
    "catalogue",
    "database",
    "data release",
    "archive",
)


def map_category_to_entity_type(category: str) -> str:
    """Map an AstroMLab category string to an entity_type.

    Mapping rules:
    - Categories containing instrument-related keywords -> 'instrument'
    - Categories containing data/survey-related keywords -> 'dataset'
    - Everything else -> 'method' (default, covers techniques and approaches)

    Args:
        category: The category string from AstroMLab data.

    Returns:
        Entity type string: 'instrument', 'dataset', or 'method'.
    """
    lower = category.lower().strip()

    for keyword in _CATEGORY_INSTRUMENT_KEYWORDS:
        if keyword in lower:
            return "instrument"

    for keyword in _CATEGORY_DATASET_KEYWORDS:
        if keyword in lower:
            return "dataset"

    return "method"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_concepts(
    dest: Path | None = None,
    url: str = ASTROMLAB_CONCEPTS_URL,
) -> Path:
    """Download the AstroMLab concept vocabulary from GitHub.

    Skips download if the destination file already exists and is non-empty.
    Uses ResilientClient with built-in retry and exponential backoff.

    Args:
        dest: Path to save the downloaded file. Defaults to data/astromlab_concepts.csv.
        url: URL to download from.

    Returns:
        Path to the downloaded (or existing) file.
    """
    if dest is None:
        dest = _DEFAULT_DEST

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("AstroMLab concepts file already exists at %s, skipping download", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading AstroMLab concepts from %s", url)

    client = _get_client()
    response = client.get(url)
    # response may be requests.Response or CachedResponse
    if hasattr(response, "content"):
        data = response.content
    else:
        data = response.text.encode("utf-8")

    dest.write_bytes(data)
    logger.info("Downloaded AstroMLab concepts: %d bytes -> %s", len(data), dest)
    return dest


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _parse_csv(text: str) -> list[dict[str, Any]]:
    """Parse CSV-formatted concept data into entry dicts.

    Expected CSV columns (flexible header matching):
    - concept / concept_name / name: the concept name
    - category / topic / field: the category/topic

    Args:
        text: CSV text content.

    Returns:
        List of entity dictionary entry dicts.
    """
    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None:
        logger.warning("No CSV headers found")
        return []

    # Flexible column name matching
    lower_fields = {f.lower().strip(): f for f in reader.fieldnames}

    concept_col = None
    for candidate in ("concept", "concept_name", "name"):
        if candidate in lower_fields:
            concept_col = lower_fields[candidate]
            break

    category_col = None
    for candidate in ("category", "topic", "field", "subject"):
        if candidate in lower_fields:
            category_col = lower_fields[candidate]
            break

    if concept_col is None:
        # Fall back: use first column as concept name
        concept_col = reader.fieldnames[0]
        logger.warning("No recognized concept column; using first column: %s", concept_col)

    entries: list[dict[str, Any]] = []
    skipped = 0

    for row in reader:
        concept_name = (row.get(concept_col) or "").strip()
        if not concept_name:
            skipped += 1
            continue

        category = (row.get(category_col) or "").strip() if category_col else ""
        entity_type = map_category_to_entity_type(category) if category else "method"

        metadata: dict[str, Any] = {}
        if category:
            metadata["category"] = category

        # Include any additional columns as metadata
        description_col = None
        for candidate in ("description", "definition", "desc"):
            if candidate in lower_fields:
                description_col = lower_fields[candidate]
                break

        if description_col:
            desc = (row.get(description_col) or "").strip()
            if desc:
                metadata["description"] = desc

        entries.append(
            {
                "canonical_name": concept_name,
                "entity_type": entity_type,
                "source": "astromlab",
                "aliases": [],
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d rows with no concept name", skipped)

    return entries


def _parse_json(text: str) -> list[dict[str, Any]]:
    """Parse JSON-formatted concept data into entry dicts.

    Expected JSON: array of objects with keys like
    concept/concept_name/name and category/topic/field.

    Args:
        text: JSON text content.

    Returns:
        List of entity dictionary entry dicts.
    """
    raw: list[dict[str, Any]] = json.loads(text)

    entries: list[dict[str, Any]] = []
    skipped = 0

    for item in raw:
        # Flexible key matching
        concept_name = ""
        for key in ("concept", "concept_name", "name"):
            val = item.get(key)
            if val and isinstance(val, str) and val.strip():
                concept_name = val.strip()
                break

        if not concept_name:
            skipped += 1
            continue

        category = ""
        for key in ("category", "topic", "field", "subject"):
            val = item.get(key)
            if val and isinstance(val, str) and val.strip():
                category = val.strip()
                break

        entity_type = map_category_to_entity_type(category) if category else "method"

        metadata: dict[str, Any] = {}
        if category:
            metadata["category"] = category

        description = ""
        for key in ("description", "definition", "desc"):
            val = item.get(key)
            if val and isinstance(val, str) and val.strip():
                description = val.strip()
                break

        if description:
            metadata["description"] = description

        entries.append(
            {
                "canonical_name": concept_name,
                "entity_type": entity_type,
                "source": "astromlab",
                "aliases": [],
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d items with no concept name", skipped)

    return entries


def parse_concepts(data_path: Path) -> list[dict[str, Any]]:
    """Parse AstroMLab concept data into entity_dictionary entries.

    Auto-detects CSV vs JSON format based on file extension and content.

    Args:
        data_path: Path to the concept data file (CSV or JSON).

    Returns:
        List of dicts suitable for ``dictionary.bulk_load()``.
    """
    logger.info("Parsing AstroMLab concepts from %s", data_path)
    text = data_path.read_text(encoding="utf-8")

    if not text.strip():
        logger.warning("Empty concept data file: %s", data_path)
        return []

    # Detect format: try JSON first if extension suggests it, otherwise CSV
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        entries = _parse_json(text)
    elif suffix == ".csv":
        entries = _parse_csv(text)
    else:
        # Auto-detect: try JSON, fall back to CSV
        stripped = text.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            entries = _parse_json(text)
        else:
            entries = _parse_csv(text)

    logger.info("Parsed %d AstroMLab concept entries", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_concepts(entries: list[dict[str, Any]], dsn: str | None = None) -> int:
    """Load parsed AstroMLab concept entries into entity_dictionary.

    Args:
        entries: List of entry dicts from ``parse_concepts()``.
        dsn: Database connection string. Uses SCIX_DSN or default if None.

    Returns:
        Number of rows upserted.
    """
    conn = get_connection(dsn)
    try:
        count = bulk_load(conn, entries)
        logger.info("Loaded %d AstroMLab concept entries into entity_dictionary", count)
        return count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_path: Path | None = None,
    dsn: str | None = None,
    url: str = ASTROMLAB_CONCEPTS_URL,
) -> int:
    """Run the full AstroMLab concept harvesting pipeline.

    Steps:
        1. Download concepts file (if not provided)
        2. Parse concept entries
        3. Load into entity_dictionary
    Logs harvest run to harvest_runs.

    Args:
        data_path: Path to a local concepts file. Downloads if None.
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        url: URL to download from if data_path is not provided.

    Returns:
        Number of concepts loaded.
    """
    t0 = time.monotonic()

    if data_path is None:
        data_path = download_concepts(url=url)
    elif not data_path.exists():
        raise FileNotFoundError(f"Concepts file not found: {data_path}")

    entries = parse_concepts(data_path)

    if not entries:
        logger.warning("No concept entries parsed — nothing to load")
        return 0

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, "astromlab")
    try:
        run_log.start()
        count = bulk_load(conn, entries)
        run_log.complete(
            records_fetched=len(entries),
            records_upserted=count,
        )
        logger.info("Loaded %d AstroMLab concept entries into entity_dictionary", count)
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
        "AstroMLab concepts pipeline complete: %d concepts loaded in %.1fs",
        count,
        elapsed,
    )
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the AstroMLab concept harvesting pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest AstroMLab concept vocabulary into entity_dictionary",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Path to concepts file (downloads from GitHub if not provided)",
    )
    parser.add_argument(
        "--url",
        default=ASTROMLAB_CONCEPTS_URL,
        help="URL to download concept data from",
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

    count = run_pipeline(data_path=args.data_file, dsn=args.dsn, url=args.url)
    print(f"Loaded {count} AstroMLab concept entries into entity_dictionary")


if __name__ == "__main__":
    main()
