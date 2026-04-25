"""Retraction Watch CC0 CSV harvester.

Reads the public Retraction Watch dataset (CC0 license, no API key required)
and emits :class:`CorrectionEvent` dicts of the shape::

    {"type": "retraction",
     "source": "retraction_watch",
     "doi": "<original-paper-doi>",
     "date": "YYYY-MM-DD"}

The Retraction Watch DB is published as a static CSV. The schema we depend on
(both column names and order) is documented at https://retractionwatch.com/
and mirrored on Crossref Labs. We extract:

* ``OriginalPaperDOI`` — the DOI of the paper that was retracted (or had the
  correction applied).
* ``RetractionDate`` — the date the retraction was published.

Rows missing either field are skipped. This module is pure parsing — the
orchestrator script (``scripts/ingest_corrections.py``) handles persistence.

LICENSING: Retraction Watch's database is CC0. No paid SDK is used.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Callable, Iterator
from typing import Any

import requests

logger = logging.getLogger(__name__)

# The canonical CC0 download URL. Retraction Watch publishes the dataset via
# Crossref's "Labs" mirror; this URL is also reachable directly from the
# Retraction Watch site. Override with ``--url`` on the orchestrator CLI when
# the upstream changes.
RW_DEFAULT_URL = "https://api.labs.crossref.org/data/retractionwatch.csv"

SOURCE_TAG = "retraction_watch"
EVENT_TYPE = "retraction"

# Expected columns in the public CSV (only the two we need are required).
_COL_DOI = "OriginalPaperDOI"
_COL_DATE = "RetractionDate"

# Date formats observed in Retraction Watch CSV exports. We try these in order.
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d")


def _normalize_date(raw: str) -> str | None:
    """Normalize a free-form Retraction Watch date string to ISO ``YYYY-MM-DD``.

    Returns ``None`` for empty or unparseable values.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    from datetime import datetime

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    logger.debug("Could not parse retraction date: %r", raw)
    return None


def _normalize_doi(raw: str) -> str | None:
    """Strip URL prefix and whitespace; lower-case for stable dedup keys."""
    if not raw:
        return None
    doi = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    doi = doi.strip()
    return doi.lower() or None


def parse_retraction_watch_csv(text: str) -> Iterator[dict[str, Any]]:
    """Parse a Retraction Watch CSV payload into correction events.

    Skips rows missing ``OriginalPaperDOI`` or ``RetractionDate``. Tolerates
    minor schema drift by reading via ``csv.DictReader`` keyed on header names.

    Yields events of type ``retraction`` from source ``retraction_watch``.
    """
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        doi = _normalize_doi(row.get(_COL_DOI, ""))
        date = _normalize_date(row.get(_COL_DATE, ""))
        if not doi or not date:
            continue
        yield {
            "type": EVENT_TYPE,
            "source": SOURCE_TAG,
            "doi": doi,
            "date": date,
        }


def harvest(
    url: str | None = None,
    *,
    fetcher: Callable[[str], str] | None = None,
    timeout: float = 60.0,
) -> Iterator[dict[str, Any]]:
    """Download the Retraction Watch CSV and yield correction events.

    Args:
        url: Override URL for the CSV. Defaults to :data:`RW_DEFAULT_URL`.
        fetcher: Optional injected callable ``(url) -> csv text`` for tests.
            When ``None``, uses :func:`requests.get` with ``timeout``.
        timeout: HTTP timeout when ``fetcher`` is None.
    """
    target = url or RW_DEFAULT_URL
    if fetcher is None:
        response = requests.get(target, timeout=timeout)
        response.raise_for_status()
        text = response.text
    else:
        text = fetcher(target)
    yield from parse_retraction_watch_csv(text)


__all__ = [
    "RW_DEFAULT_URL",
    "SOURCE_TAG",
    "EVENT_TYPE",
    "parse_retraction_watch_csv",
    "harvest",
]
