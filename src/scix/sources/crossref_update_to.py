"""Crossref ``update-to`` relation harvester.

For a known DOI, Crossref records an ``update-to`` relation in the work's
``message.relation`` block whenever the DOI itself is a correction, erratum,
retraction, or expression-of-concern of *another* DOI. To find correction
events for a target paper, we query Crossref for the paper's DOI and inspect
``message.relation.update-to`` — these are the corrections that have been
issued FOR the paper.

We emit events of the shape::

    {"type": "<mapped>",
     "source": "crossref",
     "doi": "<correcting-doi-or-target-doi>",
     "date": "YYYY-MM-DD"}

The Crossref API is free and key-free. No paid SDK is involved.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org/works"
SOURCE_TAG = "crossref"

# Mapping from Crossref relation/update labels to our event-type vocabulary.
# Crossref uses both "update_type" inside `update-to` and the relation-key
# ("is-corrected-by", "is-retracted-by", ...). We accept either.
_TYPE_MAP: dict[str, str] = {
    "correction": "correction",
    "erratum": "erratum",
    "retraction": "retraction",
    "expression_of_concern": "expression_of_concern",
    "expression-of-concern": "expression_of_concern",
    "removal": "retraction",
    "withdrawal": "retraction",
    "is-corrected-by": "correction",
    "is-retracted-by": "retraction",
    "is-erratum-for": "erratum",
}


def _map_crossref_type(label: str | None) -> str:
    """Map a Crossref label to one of our five canonical event types.

    Falls back to ``correction`` for unknown labels (the most generic of the
    five). We log an info message so the orchestrator audit trail records the
    drift.
    """
    if not label:
        return "correction"
    key = label.strip().lower().replace(" ", "_")
    mapped = _TYPE_MAP.get(key) or _TYPE_MAP.get(label.strip().lower())
    if mapped is None:
        logger.info("Unknown Crossref update label %r; defaulting to correction", label)
        return "correction"
    return mapped


def _normalize_doi(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :]
    return s.strip().lower() or None


def _extract_date(rel: dict[str, Any]) -> str | None:
    """Pull a ``YYYY-MM-DD`` date from a Crossref relation entry.

    Crossref encodes dates as ``{"date-parts": [[YYYY, MM, DD]]}``.
    """
    raw = rel.get("date") or rel.get("updated")
    if isinstance(raw, str) and raw:
        return raw[:10]
    if isinstance(raw, dict):
        parts_outer = raw.get("date-parts")
        if isinstance(parts_outer, list) and parts_outer:
            parts = parts_outer[0]
            if isinstance(parts, list) and parts:
                year = parts[0]
                month = parts[1] if len(parts) > 1 else 1
                day = parts[2] if len(parts) > 2 else 1
                try:
                    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                except (TypeError, ValueError):
                    return None
    return None


def parse_crossref_work(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Parse one Crossref ``/works/{doi}`` response into correction events.

    Looks at ``message.relation`` and emits one event per ``update-to`` entry
    (and its synonyms ``is-corrected-by`` / ``is-retracted-by`` /
    ``is-erratum-for`` if present).

    The DOI on the emitted event is the *target* paper (the one being
    corrected) — i.e. the DOI of the work this payload belongs to — so the
    orchestrator can join straight against ``papers.doi``.
    """
    message = payload.get("message")
    if not isinstance(message, dict):
        return
    target_doi = _normalize_doi(message.get("DOI"))
    relation = message.get("relation")
    if not isinstance(relation, dict):
        return

    relevant_keys = (
        "update-to",
        "is-corrected-by",
        "is-retracted-by",
        "is-erratum-for",
        "has-correction",
        "has-erratum",
        "has-retraction",
    )
    for key in relevant_keys:
        entries = relation.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            update_type = entry.get("update_type") or entry.get("update-type") or key
            mapped = _map_crossref_type(update_type)
            doi = _normalize_doi(entry.get("id")) or target_doi
            if not doi:
                continue
            date = _extract_date(entry) or _extract_date(message)
            if not date:
                continue
            yield {
                "type": mapped,
                "source": SOURCE_TAG,
                "doi": target_doi or doi,
                "date": date,
            }


def harvest(
    dois: Iterable[str],
    *,
    base: str = CROSSREF_BASE,
    fetcher: Callable[[str], dict[str, Any]] | None = None,
    timeout: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Query Crossref for each DOI and yield correction events.

    Args:
        dois: Iterable of DOIs (with or without ``doi.org/`` prefix).
        base: Crossref ``/works`` endpoint. Override for staging.
        fetcher: Optional injected ``(url) -> json dict`` for tests.
        timeout: HTTP timeout when ``fetcher`` is None.
    """
    seen: set[str] = set()
    for raw in dois:
        doi = _normalize_doi(raw)
        if not doi or doi in seen:
            continue
        seen.add(doi)
        url = f"{base}/{quote(doi, safe='/:')}"
        try:
            if fetcher is None:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
            else:
                payload = fetcher(url)
        except requests.RequestException as exc:
            logger.warning("Crossref request failed for %s: %s", doi, exc)
            continue
        if not isinstance(payload, dict):
            continue
        yield from parse_crossref_work(payload)


__all__ = [
    "CROSSREF_BASE",
    "SOURCE_TAG",
    "_map_crossref_type",
    "parse_crossref_work",
    "harvest",
]
