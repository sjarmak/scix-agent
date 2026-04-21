"""JSON API routes for the SciX viz frontend.

The viz HTML pages are pure static bundles served from ``web/viz/``. A few of
them — the UMAP browser in particular — need to resolve a ``bibcode`` to a
human-readable title on hover. Exposing the MCP surface to the browser would
be overkill (and a security hazard), so we add one tiny read-only lookup
endpoint here and include it on the viz FastAPI app.

The DB call is expressed as a FastAPI dependency (`get_fetcher`) so tests can
override it via ``app.dependency_overrides`` without needing a live database.
"""

from __future__ import annotations

from typing import Callable, Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Path

from scix.db import get_connection

router = APIRouter(prefix="/viz/api", tags=["viz"])

# Bibcode regex kept pragmatic — real ADS bibcodes are 19 chars with a narrow
# alphabet, but relaxing to [A-Za-z0-9.:&'+%-]{4,32} keeps us correct for every
# bibcode observed in the corpus while still excluding path-traversal tricks.
_BIBCODE_PATTERN = r"^[A-Za-z0-9.:&'+%\-]{4,32}$"


PaperDict = dict[str, object]
FetcherCallable = Callable[[str], Optional[PaperDict]]


def _fetch_paper(bibcode: str) -> Optional[PaperDict]:
    """Single-paper lookup against the live ``scix`` database.

    Returns a dict shaped ``{bibcode, title, abstract, community_id}`` or
    ``None`` when the bibcode is unknown. Network/DB errors propagate — do
    not swallow them (see coding-style guidance).
    """
    sql = (
        "SELECT p.bibcode, p.title, p.abstract, pm.community_id_coarse "
        "FROM papers p "
        "LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode "
        "WHERE p.bibcode = %s LIMIT 1"
    )
    conn: psycopg.Connection = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (bibcode,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "bibcode": row[0],
        "title": row[1],
        "abstract": row[2],
        "community_id": row[3],
    }


def get_fetcher() -> FetcherCallable:
    """FastAPI dependency returning the concrete paper-fetcher callable.

    Exposing this as a dependency is what lets the test suite swap in a stub
    via ``app.dependency_overrides[get_fetcher] = lambda: <stub>``.
    """
    return _fetch_paper


@router.get("/paper/{bibcode}")
def read_paper(
    bibcode: str = Path(..., pattern=_BIBCODE_PATTERN),
    fetcher: FetcherCallable = Depends(get_fetcher),
) -> PaperDict:
    """Return ``{bibcode,title,abstract,community_id}`` or 404."""
    record = fetcher(bibcode)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {bibcode}")
    return record
