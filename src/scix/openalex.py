"""OpenAlex API integration: DOI-based linking and topic retrieval.

Uses urllib.request (stdlib) to avoid adding external HTTP dependencies.
Respects the OpenAlex polite pool by including a mailto parameter.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

OPENALEX_API_BASE = "https://api.openalex.org"

# Polite pool: max 10 req/s — sleep 0.1s between requests
_MIN_REQUEST_INTERVAL = 0.1


def fetch_openalex_by_doi(doi: str, mailto: str) -> dict[str, Any] | None:
    """Fetch an OpenAlex work record by DOI.

    Args:
        doi: A DOI string (e.g. "10.1234/example").
        mailto: Email address for OpenAlex polite pool.

    Returns:
        The work object dict, or None if not found / error.
    """
    # Strip common prefixes if present
    clean_doi = doi.strip()
    if clean_doi.startswith("https://doi.org/"):
        clean_doi = clean_doi[len("https://doi.org/") :]
    elif clean_doi.startswith("http://doi.org/"):
        clean_doi = clean_doi[len("http://doi.org/") :]

    url = f"{OPENALEX_API_BASE}/works/doi:{clean_doi}?mailto={urllib.request.quote(mailto)}"

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"SciX-Experiments/1.0 (mailto:{mailto})",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.debug("OpenAlex: no work found for DOI %s", doi)
            return None
        logger.warning("OpenAlex HTTP error %d for DOI %s: %s", exc.code, doi, exc.reason)
        return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("OpenAlex request failed for DOI %s: %s", doi, exc)
        return None


def _extract_topics(work: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract topic labels from an OpenAlex work object."""
    topics = work.get("topics", [])
    return [
        {
            "id": t.get("id", ""),
            "display_name": t.get("display_name", ""),
            "score": t.get("score"),
            "subfield": (t.get("subfield") or {}).get("display_name"),
            "field": (t.get("field") or {}).get("display_name"),
            "domain": (t.get("domain") or {}).get("display_name"),
        }
        for t in topics
    ]


def link_papers_batch(
    conn: psycopg.Connection,
    dois_and_bibcodes: list[tuple[str, str]],
    mailto: str,
) -> int:
    """Fetch OpenAlex IDs and topics for a batch of DOI/bibcode pairs.

    Updates the papers table with openalex_id and openalex_topics.
    Sleeps between requests to respect the 10 req/s polite pool limit.

    Args:
        conn: Database connection.
        dois_and_bibcodes: List of (doi, bibcode) tuples.
        mailto: Email for OpenAlex polite pool.

    Returns:
        Number of papers successfully linked.
    """
    linked = 0

    for doi, bibcode in dois_and_bibcodes:
        work = fetch_openalex_by_doi(doi, mailto)
        if work is None:
            continue

        openalex_id = work.get("id", "")
        topics = _extract_topics(work)
        topics_json = json.dumps(topics)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE papers
                SET openalex_id = %s,
                    openalex_topics = %s::jsonb
                WHERE bibcode = %s
                """,
                (openalex_id, topics_json, bibcode),
            )
        conn.commit()
        linked += 1
        logger.debug("Linked bibcode %s -> %s (%d topics)", bibcode, openalex_id, len(topics))

        # Rate limiting: 10 req/s max
        time.sleep(_MIN_REQUEST_INTERVAL)

    return linked
