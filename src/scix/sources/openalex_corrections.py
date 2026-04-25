"""OpenAlex retracted-works harvester.

Pulls works flagged ``is_retracted=true`` from the public OpenAlex API and
emits correction events of the shape::

    {"type": "retraction",
     "source": "openalex",
     "doi": "<doi>",
     "date": "YYYY-MM-DD"}

OpenAlex is free and key-free; we identify ourselves via a polite-pool email
in the ``mailto`` query parameter when available. No paid SDK is involved.

The pagination uses cursor-based traversal (``cursor=*``), which the OpenAlex
docs recommend for large result sets.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"
SOURCE_TAG = "openalex"
EVENT_TYPE = "retraction"

# OpenAlex date fields, in priority order: prefer the explicit retraction
# `updated_date` (when the is_retracted flag was applied) and fall back to
# `publication_date`.
_DATE_FIELDS = ("retracted_date", "updated_date", "publication_date")

PolitePoolEmail = os.environ.get("SCIX_OPENALEX_MAILTO")


def _strip_doi_prefix(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :]
    return s.strip().lower() or None


def _pick_date(work: dict[str, Any]) -> str | None:
    """Pick the most informative date from an OpenAlex work record."""
    for field in _DATE_FIELDS:
        val = work.get(field)
        if isinstance(val, str) and val:
            # OpenAlex returns dates as ISO strings; trim to YYYY-MM-DD.
            return val[:10]
    return None


def parse_openalex_works(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Parse one OpenAlex works-page response into correction events.

    Yields a retraction event for every work whose ``is_retracted`` is true and
    that exposes a DOI.
    """
    results = payload.get("results")
    if not isinstance(results, list):
        return
    for work in results:
        if not isinstance(work, dict):
            continue
        if not work.get("is_retracted"):
            continue
        doi = _strip_doi_prefix(work.get("doi"))
        if not doi:
            continue
        date = _pick_date(work)
        if not date:
            continue
        yield {
            "type": EVENT_TYPE,
            "source": SOURCE_TAG,
            "doi": doi,
            "date": date,
        }


def _next_cursor(payload: dict[str, Any]) -> str | None:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    cursor = meta.get("next_cursor")
    if isinstance(cursor, str) and cursor:
        return cursor
    return None


def _build_url(base: str, *, cursor: str = "*", per_page: int = 200) -> str:
    params: dict[str, str] = {
        "filter": "is_retracted:true",
        "per_page": str(per_page),
        "cursor": cursor,
    }
    if PolitePoolEmail:
        params["mailto"] = PolitePoolEmail
    return f"{base}?{urlencode(params)}"


def harvest(
    *,
    base: str = OPENALEX_BASE,
    fetcher: Callable[[str], dict[str, Any]] | None = None,
    max_pages: int = 50,
    per_page: int = 200,
    timeout: float = 60.0,
) -> Iterator[dict[str, Any]]:
    """Walk the OpenAlex retracted-works cursor and yield correction events.

    Args:
        base: API base URL. Override for staging endpoints.
        fetcher: Optional injected ``(url) -> json dict`` for tests.
        max_pages: Safety cap on cursor pages to walk.
        per_page: Records per cursor page (max 200 in OpenAlex).
        timeout: HTTP timeout when ``fetcher`` is None.
    """
    cursor: str | None = "*"
    pages = 0
    while cursor is not None and pages < max_pages:
        url = _build_url(base, cursor=cursor, per_page=per_page)
        if fetcher is None:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        else:
            payload = fetcher(url)
        if not isinstance(payload, dict):
            return
        yield from parse_openalex_works(payload)
        cursor = _next_cursor(payload)
        pages += 1


__all__ = [
    "OPENALEX_BASE",
    "SOURCE_TAG",
    "EVENT_TYPE",
    "parse_openalex_works",
    "harvest",
]
