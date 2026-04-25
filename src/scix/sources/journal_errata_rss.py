"""Journal Errata/Correction RSS harvester (top-15 astronomy venues).

Implements the journal-RSS leg of PRD ``docs/prd/scix_deep_search_v1.md``
amendment A3. Walks publisher-supplied RSS feeds for the top-15 astronomy
journals and emits correction events of the shape::

    {"type": "<erratum|correction|retraction|expression_of_concern>",
     "source": "journal_rss",
     "doi": "<doi-of-corrected-paper-or-the-erratum-itself>",
     "date": "YYYY-MM-DD"}

Note on ZFC compliance (CLAUDE.md ``patterns.md``): the title-classification
step matches publisher-side rigid conventions ("Erratum: ...", "Correction
to: ...", "Retraction Notice: ..."). This is mechanical structural matching
on tokens publishers themselves emit, not semantic classification — the same
role keyword matching plays in DOI extraction. Any drift in publisher
conventions surfaces as a single event-type mis-classification, not as a
correctness failure for the overall pipeline (the event still lands as
``correction``, the safe default).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from xml.etree import ElementTree as ET  # noqa: N817 — stdlib alias

import requests

logger = logging.getLogger(__name__)

SOURCE_TAG = "journal_rss"

# Top-15 astronomy venues from PRD A3. The exact upstream URLs are subject to
# publisher reorganisation; treat them as defaults the orchestrator can
# override via CLI. Tests do not hit these URLs live.
JOURNAL_FEEDS: dict[str, str] = {
    "ApJ": "https://iopscience.iop.org/journal/0004-637X/rss/recent",
    "ApJL": "https://iopscience.iop.org/journal/2041-8205/rss/recent",
    "ApJS": "https://iopscience.iop.org/journal/0067-0049/rss/recent",
    "A&A": "https://www.aanda.org/component/issues/?task=rss",
    "AJ": "https://iopscience.iop.org/journal/1538-3881/rss/recent",
    "MNRAS": "https://academic.oup.com/mnras/issue-feed",
    "Nature Astronomy": "https://www.nature.com/natastron.rss",
    "Science": "https://www.science.org/rss/news_current.xml",
    "Icarus": "https://rss.sciencedirect.com/publication/science/00191035",
    "Solar Physics": "https://link.springer.com/search.rss?facet-journal-id=11207",
    "JGR Space Physics": (
        "https://agupubs.onlinelibrary.wiley.com/feed/21699402/most-recent"
    ),
    "Space Sci. Rev.": "https://link.springer.com/search.rss?facet-journal-id=11214",
    "PASP": "https://iopscience.iop.org/journal/1538-3873/rss/recent",
    "Living Reviews": "https://link.springer.com/search.rss?facet-journal-id=41114",
    "ARA&A": "https://www.annualreviews.org/rss/content/journals/astro/latest",
}

# Publisher-side rigid title prefixes for correction posts.
_TYPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bretraction\b", re.I), "retraction"),
    (re.compile(r"\bwithdrawal\b", re.I), "retraction"),
    (re.compile(r"\bexpression\s+of\s+concern\b", re.I), "expression_of_concern"),
    (re.compile(r"\berratum\b", re.I), "erratum"),
    (re.compile(r"\bcorrigendum\b", re.I), "erratum"),
    (re.compile(r"\bcorrection\b", re.I), "correction"),
    (re.compile(r"\brecalibration\b", re.I), "recalibration_supersession"),
    (re.compile(r"\bsupersed", re.I), "recalibration_supersession"),
)

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)
_PUBDATE_FORMATS = (
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
)


def _classify_title(title: str | None) -> str | None:
    """Return one of our event types when the title looks like a correction.

    Returns ``None`` for non-correction items (which should be filtered out).
    """
    if not title:
        return None
    for pat, event_type in _TYPE_PATTERNS:
        if pat.search(title):
            return event_type
    return None


def _extract_doi(item: ET.Element) -> str | None:
    """Pull a DOI from an RSS item.

    Looks at, in order: ``<dc:identifier>``, ``<prism:doi>``, ``<guid>``,
    ``<link>``, the title-suffix fallback. The first match wins.
    """
    candidates: list[str] = []
    for child in item.iter():
        # Strip namespace prefix when present.
        tag = child.tag.split("}", 1)[-1].lower()
        if tag in ("identifier", "doi", "guid", "link") and child.text:
            candidates.append(child.text)
    title = item.findtext("title") or ""
    candidates.append(title)
    for cand in candidates:
        match = _DOI_RE.search(cand or "")
        if match:
            doi = match.group(0).rstrip(".,);]")
            return doi.lower()
    return None


def _normalize_pubdate(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    from datetime import datetime

    for fmt in _PUBDATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    # Last-ditch: accept a bare YYYY-MM-DD prefix.
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return None


def parse_rss_feed(
    xml_text: str,
    *,
    source_label: str = SOURCE_TAG,
) -> Iterator[dict[str, Any]]:
    """Parse a publisher RSS feed XML payload into correction events.

    Skips items whose title doesn't match any correction-type pattern. Skips
    items missing a DOI or publication date.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("RSS parse failed: %s", exc)
        return

    # Both RSS 2.0 (<channel><item>...</item></channel>) and Atom-ish layouts
    # are common; iterate items by descendant search to cover both.
    for item in root.iter():
        local = item.tag.split("}", 1)[-1].lower()
        if local not in ("item", "entry"):
            continue
        title = item.findtext("title") or item.findtext(
            "{http://www.w3.org/2005/Atom}title"
        )
        event_type = _classify_title(title)
        if event_type is None:
            continue
        doi = _extract_doi(item)
        if not doi:
            continue
        pubdate = item.findtext("pubDate") or item.findtext(
            "{http://www.w3.org/2005/Atom}published"
        ) or item.findtext("{http://purl.org/dc/elements/1.1/}date")
        date = _normalize_pubdate(pubdate)
        if not date:
            continue
        yield {
            "type": event_type,
            "source": source_label,
            "doi": doi,
            "date": date,
        }


def harvest(
    feeds: Mapping[str, str] | None = None,
    *,
    fetcher: Callable[[str], str] | None = None,
    timeout: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Walk all configured journal RSS feeds and yield correction events.

    Args:
        feeds: Mapping of journal-label -> feed URL. Defaults to
            :data:`JOURNAL_FEEDS`.
        fetcher: Optional injected ``(url) -> xml text`` for tests.
        timeout: HTTP timeout when ``fetcher`` is None.
    """
    targets = feeds if feeds is not None else JOURNAL_FEEDS
    for label, url in targets.items():
        try:
            if fetcher is None:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                xml_text = response.text
            else:
                xml_text = fetcher(url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s feed (%s): %s", label, url, exc)
            continue
        yield from parse_rss_feed(xml_text)


__all__ = [
    "SOURCE_TAG",
    "JOURNAL_FEEDS",
    "parse_rss_feed",
    "harvest",
]
