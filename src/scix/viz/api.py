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

import json
import logging
import re
import time
from pathlib import Path as _FsPath
from typing import Any, Callable, Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from scix import mcp_server
from scix.db import get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/viz/api", tags=["viz"])

# Bibcode regex kept pragmatic — real ADS bibcodes are 19 chars with a narrow
# alphabet, but relaxing to [A-Za-z0-9.:&'+%-]{4,32} keeps us correct for every
# bibcode observed in the corpus while still excluding path-traversal tricks.
_BIBCODE_PATTERN = r"^[A-Za-z0-9.:&'+%\-]{4,32}$"


PaperDict = dict[str, object]
FetcherCallable = Callable[[str], Optional[PaperDict]]
EgoFetcherCallable = Callable[[str, int, int, int], Optional[PaperDict]]


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


# ---------------------------------------------------------------------------
# Resolution config for the nav-level toggle (coarse / medium / fine).
#
# The shared frontend helpers in ``web/viz/shared.js`` persist the selected
# resolution in localStorage and read the candidate data-file URLs from this
# endpoint at startup. Exposing it as an API — rather than baking the config
# into the HTML — means ops can drop new ``umap.<res>.json`` and
# ``community_labels.<res>.json`` files under ``web/viz/`` or ``data/viz/``
# without touching the HTML bundle.
# ---------------------------------------------------------------------------


_WEB_VIZ_DIR: _FsPath = _FsPath(__file__).resolve().parents[3] / "web" / "viz"
_DATA_VIZ_DIR: _FsPath = _FsPath(__file__).resolve().parents[3] / "data" / "viz"

_RESOLUTION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "coarse",
        "label": "Coarse (~20 domains)",
        "description": "20 broad domain-level communities.",
        "umap_filenames": ("umap.coarse.json", "umap.json"),
        "labels_filenames": ("community_labels.coarse.json", "community_labels.json"),
        "stream_filenames": ("stream.coarse.json",),
    },
    {
        "id": "medium",
        "label": "Medium (~200 topics)",
        "description": "~200 fine-grained topic communities.",
        "umap_filenames": ("umap.medium.json",),
        "labels_filenames": (
            "community_labels.medium.json",
            "community_labels_medium.json",
        ),
        "stream_filenames": ("stream.medium.json",),
    },
    {
        "id": "fine",
        "label": "Fine (~2000 subtopics)",
        "description": "Fine-grained subtopic communities (experimental).",
        "umap_filenames": ("umap.fine.json",),
        "labels_filenames": (
            "community_labels.fine.json",
            "community_labels_fine.json",
        ),
        "stream_filenames": ("stream.fine.json",),
    },
)
_DEFAULT_RESOLUTION = "coarse"


def _locate_first(filenames: tuple[str, ...]) -> Optional[dict[str, str]]:
    """Return the first filename that exists on disk, as ``{url, source}``.

    Checks ``web/viz/`` first (served at ``/viz/``) and falls back to
    ``data/viz/`` (served at ``/data/viz/``). Returns ``None`` if nothing
    matches — callers use that signal to mark a resolution as unavailable.
    """
    for name in filenames:
        web_path = _WEB_VIZ_DIR / name
        if web_path.exists():
            return {"url": f"/viz/{name}", "source": "web"}
        data_path = _DATA_VIZ_DIR / name
        if data_path.exists():
            return {"url": f"/data/viz/{name}", "source": "data"}
    return None


@router.get("/resolution")
def read_resolution_config() -> dict[str, Any]:
    """Return the resolution catalog the frontend should render.

    Shape::

        {
            "default": "coarse",
            "current": "coarse",
            "resolutions": [
                {
                    "id": "coarse",
                    "label": "Coarse (~20 domains)",
                    "description": "...",
                    "available": true,
                    "umap_url": "/viz/umap.json",
                    "labels_url": "/viz/community_labels.json",
                    "umap_candidates": ["umap.coarse.json", "umap.json"],
                    "labels_candidates": ["community_labels.coarse.json",
                                           "community_labels.json"]
                },
                ...
            ]
        }

    The frontend uses ``current`` as its startup default when
    ``localStorage`` has no stored choice, and uses ``available`` + the
    resolved URLs to decide whether to offer each option.
    """
    resolutions: list[dict[str, Any]] = []
    for spec in _RESOLUTION_SPECS:
        umap = _locate_first(spec["umap_filenames"])
        labels = _locate_first(spec["labels_filenames"])
        stream = _locate_first(spec["stream_filenames"])
        resolutions.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "description": spec["description"],
                "available": bool(umap and labels),
                "umap_url": umap["url"] if umap else None,
                "labels_url": labels["url"] if labels else None,
                "stream_url": stream["url"] if stream else None,
                "umap_candidates": list(spec["umap_filenames"]),
                "labels_candidates": list(spec["labels_filenames"]),
                "stream_candidates": list(spec["stream_filenames"]),
            }
        )
    return {
        "default": _DEFAULT_RESOLUTION,
        "current": _DEFAULT_RESOLUTION,
        "resolutions": resolutions,
    }


# ---------------------------------------------------------------------------
# Live demo: natural-language query -> real MCP tool dispatch -> trace events.
# ---------------------------------------------------------------------------


class DemoSearchRequest(BaseModel):
    """Request payload for POST /viz/api/demo/search."""

    query: str = Field(..., min_length=2, max_length=400)
    top_n: int = Field(default=5, ge=1, le=15)


@router.post("/demo/search")
def demo_search(payload: DemoSearchRequest) -> dict:
    """Run a real MCP search and let the instrumentation hook narrate the trace.

    Dispatches through :func:`scix.mcp_server.call_tool`, which already wraps
    every tool call with ``_log_query`` + ``_emit_trace_event``. Each call
    here therefore yields exactly one TraceEvent on the SSE bus — sourced
    from the live tool, not synthesized by this endpoint. The frontend
    animates them on the UMAP overlay via ``/viz/api/trace/stream``.

    Not authenticated — intended for local demo driving over localhost.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")

    # Real search dispatch — the MCP hook publishes one TraceEvent with the
    # full result set (bibcodes drive the line-flash on UMAP).
    search_args = {"query": query, "mode": "keyword", "limit": payload.top_n}
    search_json = mcp_server.call_tool("search", search_args)
    search_payload = json.loads(search_json)

    if "error" in search_payload:
        raise HTTPException(status_code=502, detail=search_payload["error"])

    papers = [p for p in (search_payload.get("papers") or []) if isinstance(p, dict)]
    bibcodes = tuple(str(p.get("bibcode")) for p in papers if p.get("bibcode"))
    timing = search_payload.get("timing_ms") or {}
    if isinstance(timing, dict):
        # Sum across whatever stages a hybrid/keyword/semantic search reports.
        search_ms = float(sum(v for v in timing.values() if isinstance(v, (int, float))))
    else:
        search_ms = float(timing or 0.0)

    # Drill into the top-3 with real get_paper calls — each produces its own
    # trace event via the same instrumentation hook.
    drill: list[str] = []
    for bib in bibcodes[:3]:
        try:
            mcp_server.call_tool("get_paper", {"bibcode": bib})
        except Exception:
            logger.debug("demo drill get_paper failed for %s", bib, exc_info=True)
            continue
        drill.append(bib)

    return {
        "query": query,
        "latency_ms": round(search_ms, 1),
        "total": search_payload.get("total", len(papers)),
        "bibcodes": list(bibcodes),
        "drilled": drill,
        "papers": [
            {
                "bibcode": p.get("bibcode"),
                "title": p.get("title"),
                "score": p.get("rrf_score") or p.get("score"),
            }
            for p in papers
        ],
    }


# ---------------------------------------------------------------------------
# Composite multi-step demo endpoints — each runs a real MCP tool sequence
# and relies entirely on the instrumentation hook inside
# :func:`scix.mcp_server.call_tool` to publish TraceEvents. No hand-coded
# ``publish_trace`` calls anywhere in these handlers.
# ---------------------------------------------------------------------------


# Mechanical title tokenizer (ZFC-safe: keyword frequency, no semantic
# classification). Minimum three ASCII letters — keeps acronyms like "JWST",
# drops digits / single letters / short function words that leak through.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z]+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "these",
        "those",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "using",
        "based",
        "study",
        "analysis",
        "paper",
        "abstract",
        "title",
        "observation",
        "observations",
        "new",
        "first",
        "their",
        "between",
        "its",
        "how",
        "what",
        "why",
    }
)

# Keyword → entity_type map used by /viz/api/demo/disambig. Pure lookup —
# mechanical routing, no semantic classification.
_ENTITY_TYPE_HINTS: dict[str, str] = {
    "telescope": "instruments",
    "telescopes": "instruments",
    "instrument": "instruments",
    "instruments": "instruments",
    "jwst": "instruments",
    "hst": "instruments",
    "chandra": "instruments",
    "spitzer": "instruments",
    "dataset": "datasets",
    "datasets": "datasets",
    "catalog": "datasets",
    "catalogs": "datasets",
    "survey": "datasets",
    "method": "methods",
    "methods": "methods",
    "algorithm": "methods",
    "algorithms": "methods",
    "pipeline": "methods",
    "model": "methods",
    "material": "materials",
    "materials": "materials",
    "alloy": "materials",
    "compound": "materials",
}


def _call_mcp(name: str, arguments: dict) -> tuple[dict, float]:
    """Invoke an MCP tool and return ``(parsed_result, latency_ms)``.

    Every invocation produces one TraceEvent via the ``_emit_trace_event``
    hook in the ``finally`` block of :func:`scix.mcp_server.call_tool`.
    Exceptions from the inner handler are captured and surfaced as
    ``{"error": ...}`` so a composite sequence can keep narrating even when
    one step fails.
    """
    t0 = time.monotonic()
    try:
        raw = mcp_server.call_tool(name, arguments)
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000.0
        logger.debug("demo composite: %s raised %s", name, exc, exc_info=True)
        return {"error": str(exc)}, latency
    latency = (time.monotonic() - t0) * 1000.0
    try:
        return json.loads(raw), latency
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid JSON from tool", "raw": raw[:400]}, latency


def _bibcodes_of(result: dict, *, limit: int = 20) -> list[str]:
    """Extract bibcodes from any ``{papers:[{bibcode}...]}`` or ``{bibcode}`` shape."""
    out: list[str] = []
    if not isinstance(result, dict):
        return out
    papers = result.get("papers")
    if isinstance(papers, list):
        for paper in papers:
            if isinstance(paper, dict):
                bib = paper.get("bibcode")
                if isinstance(bib, str):
                    out.append(bib)
                    if len(out) >= limit:
                        return out
    if not out:
        bib = result.get("bibcode")
        if isinstance(bib, str):
            out.append(bib)
    return out


def _titles_of(result: dict, *, limit: int = 5) -> list[str]:
    """Pull paper titles out of a SearchResult-shaped payload."""
    out: list[str] = []
    if not isinstance(result, dict):
        return out
    papers = result.get("papers")
    if isinstance(papers, list):
        for paper in papers:
            if isinstance(paper, dict):
                title = paper.get("title")
                if isinstance(title, str) and title.strip():
                    out.append(title.strip())
                    if len(out) >= limit:
                        break
    return out


def _concept_terms(titles: list[str], *, fallback_query: str) -> list[str]:
    """Pick up to two content words suitable for :func:`concept_search`.

    Ranks title tokens by frequency, drops stopwords and words shorter than
    three characters, returns the top two. Falls back to tokens from the
    user's query when titles yield nothing usable.
    """
    counts: dict[str, int] = {}
    for title in titles:
        for tok in _WORD_RE.findall(title.lower()):
            if tok in _STOPWORDS or len(tok) < 3:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        for tok in _WORD_RE.findall(fallback_query.lower()):
            if tok in _STOPWORDS or len(tok) < 3:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:2]]


def _guess_entity_type(query: str) -> str:
    """Map free-text to ``{instruments,datasets,methods,materials}``; default instruments."""
    lowered = query.lower()
    for token, bucket in _ENTITY_TYPE_HINTS.items():
        if token in lowered:
            return bucket
    return "instruments"


def _record_step(
    steps: list[dict],
    *,
    tool: str,
    args: dict,
    bibcodes: list[str],
    latency_ms: float,
    error: str | None = None,
) -> None:
    """Append one step to the response-side step log."""
    step: dict[str, Any] = {
        "tool": tool,
        "args": {k: v for k, v in args.items() if k != "query_embedding"},
        "bibcodes": bibcodes,
        "latency_ms": round(latency_ms, 1),
    }
    if error is not None:
        step["error"] = error
    steps.append(step)


def _error_of(res: dict) -> str | None:
    return res.get("error") if isinstance(res, dict) else None


@router.post("/demo/survey")
def demo_survey(payload: DemoSearchRequest) -> dict:
    """Literature-survey scenario.

    Pipeline (each step fires one TraceEvent via the MCP instrumentation
    hook): ``search`` → ``citation_similarity(co_citation)×3`` →
    ``concept_search×≤2`` → ``get_paper``. Designed to emit 8+ real trace
    events inside the 15-second budget.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")

    t_start = time.monotonic()
    steps: list[dict] = []

    # 1 — seed search.
    search_args = {"query": query, "mode": "keyword", "limit": payload.top_n}
    search_res, ms = _call_mcp("search", search_args)
    seed_bibs = _bibcodes_of(search_res)
    _record_step(
        steps,
        tool="search",
        args={"query": query[:120], "mode": "keyword", "limit": payload.top_n},
        bibcodes=seed_bibs,
        latency_ms=ms,
        error=_error_of(search_res),
    )

    # 2 — co-citation on the top 3 seeds.
    for bib in seed_bibs[:3]:
        cc_args = {"bibcode": bib, "method": "co_citation", "limit": 5}
        cc_res, ms = _call_mcp("citation_similarity", cc_args)
        _record_step(
            steps,
            tool="citation_similarity",
            args=cc_args,
            bibcodes=_bibcodes_of(cc_res),
            latency_ms=ms,
            error=_error_of(cc_res),
        )

    # 3 — concept_search on up to two title-derived terms.
    for term in _concept_terms(_titles_of(search_res), fallback_query=query):
        cs_args = {"query": term, "limit": 5}
        cs_res, ms = _call_mcp("concept_search", cs_args)
        _record_step(
            steps,
            tool="concept_search",
            args=cs_args,
            bibcodes=_bibcodes_of(cs_res),
            latency_ms=ms,
            error=_error_of(cs_res),
        )

    # 4 — get_paper on the top seed.
    if seed_bibs:
        gp_args = {"bibcode": seed_bibs[0]}
        gp_res, ms = _call_mcp("get_paper", gp_args)
        _record_step(
            steps,
            tool="get_paper",
            args=gp_args,
            bibcodes=_bibcodes_of(gp_res),
            latency_ms=ms,
            error=_error_of(gp_res),
        )

    # 5 — graph_context on the top seed — floor the event count at 8 even
    # when concept_search only yields one term (or none). Community limit=5
    # keeps this sub-second on hub papers.
    if seed_bibs:
        gc_args = {
            "bibcode": seed_bibs[0],
            "include_community": True,
            "limit": 5,
        }
        gc_res, ms = _call_mcp("graph_context", gc_args)
        _record_step(
            steps,
            tool="graph_context",
            args=gc_args,
            bibcodes=_bibcodes_of(gc_res) or [seed_bibs[0]],
            latency_ms=ms,
            error=_error_of(gc_res),
        )

    return {
        "query": query,
        "scenario": "survey",
        "total_latency_ms": round((time.monotonic() - t_start) * 1000.0, 1),
        "bibcodes": seed_bibs,
        "steps": steps,
    }


@router.post("/demo/methods")
def demo_methods(payload: DemoSearchRequest) -> dict:
    """Related-methods scenario.

    Pipeline: ``search`` → ``citation_similarity(coupling)`` →
    ``graph_context×3`` → ``concept_search`` → ``get_paper`` (top seed) →
    ``get_paper`` (top coupled). Designed for 8+ real trace events.

    ``citation_chain`` was explicitly excluded from this pipeline — on the
    production 299M-edge graph it can exceed the 15-second endpoint budget
    in depth-3 traversals. Two ``get_paper`` narration steps keep the event
    count at 8+ without adding latency risk.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")

    t_start = time.monotonic()
    steps: list[dict] = []

    # 1 — seed search.
    search_args = {"query": query, "mode": "keyword", "limit": payload.top_n}
    search_res, ms = _call_mcp("search", search_args)
    seed_bibs = _bibcodes_of(search_res)
    _record_step(
        steps,
        tool="search",
        args={"query": query[:120], "mode": "keyword", "limit": payload.top_n},
        bibcodes=seed_bibs,
        latency_ms=ms,
        error=_error_of(search_res),
    )

    # 2 — bibliographic coupling on the top seed.
    coupled_bibs: list[str] = []
    if seed_bibs:
        cp_args = {"bibcode": seed_bibs[0], "method": "coupling", "limit": 10}
        cp_res, ms = _call_mcp("citation_similarity", cp_args)
        coupled_bibs = _bibcodes_of(cp_res)
        _record_step(
            steps,
            tool="citation_similarity",
            args=cp_args,
            bibcodes=coupled_bibs,
            latency_ms=ms,
            error=_error_of(cp_res),
        )

    # 3 — graph_context on up to 3 coupled papers. The first pull includes
    # community siblings (limit=5 to stay inside the endpoint budget); the
    # next two omit community and return metrics-only, which stays sub-
    # second on highly-cited hubs.
    for idx, bib in enumerate(coupled_bibs[:3]):
        include_community = idx == 0
        gc_args = {
            "bibcode": bib,
            "include_community": include_community,
            "limit": 5,
        }
        gc_res, ms = _call_mcp("graph_context", gc_args)
        _record_step(
            steps,
            tool="graph_context",
            args=gc_args,
            bibcodes=_bibcodes_of(gc_res) or [bib],
            latency_ms=ms,
            error=_error_of(gc_res),
        )

    # 4 — concept_search on up to two title-derived terms.
    for term in _concept_terms(_titles_of(search_res), fallback_query=query):
        cs_args = {"query": term, "limit": 5}
        cs_res, ms = _call_mcp("concept_search", cs_args)
        _record_step(
            steps,
            tool="concept_search",
            args=cs_args,
            bibcodes=_bibcodes_of(cs_res),
            latency_ms=ms,
            error=_error_of(cs_res),
        )

    # 5 — get_paper on the top seed (the paper the agent would read first).
    if seed_bibs:
        gp_args: dict = {"bibcode": seed_bibs[0]}
        gp_res, ms = _call_mcp("get_paper", gp_args)
        _record_step(
            steps,
            tool="get_paper",
            args=gp_args,
            bibcodes=_bibcodes_of(gp_res),
            latency_ms=ms,
            error=_error_of(gp_res),
        )

    # 6 — get_paper on the top coupled paper (the methodological sibling).
    if coupled_bibs and (not seed_bibs or coupled_bibs[0] != seed_bibs[0]):
        gp2_args: dict = {"bibcode": coupled_bibs[0]}
        gp2_res, ms = _call_mcp("get_paper", gp2_args)
        _record_step(
            steps,
            tool="get_paper",
            args=gp2_args,
            bibcodes=_bibcodes_of(gp2_res),
            latency_ms=ms,
            error=_error_of(gp2_res),
        )

    return {
        "query": query,
        "scenario": "methods",
        "total_latency_ms": round((time.monotonic() - t_start) * 1000.0, 1),
        "bibcodes": seed_bibs,
        "coupled_bibcodes": coupled_bibs,
        "steps": steps,
    }


@router.post("/demo/disambig")
def demo_disambig(payload: DemoSearchRequest) -> dict:
    """Entity-disambiguation scenario.

    Pipeline: ``entity(resolve)`` → ``entity(search)`` → ``entity_context``
    → ``search(filtered by entity_ids)`` → optional unfiltered ``search``
    fallback (fires only when the entity filter zeroes out, common for
    newer instruments with sparse entity coverage) → ``graph_context×3``
    → ``temporal_evolution`` → ``get_paper``. Designed for 8+ real trace
    events.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")

    t_start = time.monotonic()
    steps: list[dict] = []
    entity_type = _guess_entity_type(query)

    # 1 — resolve.
    resolve_args: dict = {"action": "resolve", "query": query, "fuzzy": True}
    resolved_res, ms = _call_mcp("entity", resolve_args)
    resolved_candidates = (
        resolved_res.get("candidates") if isinstance(resolved_res, dict) else None
    )
    if not isinstance(resolved_candidates, list):
        resolved_candidates = []
    resolved_ids = [
        int(c["entity_id"])
        for c in resolved_candidates
        if isinstance(c, dict) and isinstance(c.get("entity_id"), int)
    ]
    _record_step(
        steps,
        tool="entity",
        args={"action": "resolve", "query": query[:120], "fuzzy": True},
        bibcodes=[],
        latency_ms=ms,
        error=_error_of(resolved_res),
    )

    # 2 — typed entity search.
    type_args: dict = {
        "action": "search",
        "query": query,
        "entity_type": entity_type,
        "limit": 10,
    }
    typed_res, ms = _call_mcp("entity", type_args)
    typed_candidates = (
        typed_res.get("candidates") if isinstance(typed_res, dict) else None
    )
    if isinstance(typed_candidates, list):
        for c in typed_candidates:
            if isinstance(c, dict) and isinstance(c.get("entity_id"), int):
                eid = int(c["entity_id"])
                if eid not in resolved_ids:
                    resolved_ids.append(eid)
    _record_step(
        steps,
        tool="entity",
        args={"action": "search", "entity_type": entity_type, "query": query[:120]},
        bibcodes=[],
        latency_ms=ms,
        error=_error_of(typed_res),
    )

    # 3 — entity_context on the top candidate.
    if resolved_ids:
        ctx_args: dict = {"entity_id": resolved_ids[0]}
        ctx_res, ms = _call_mcp("entity_context", ctx_args)
        _record_step(
            steps,
            tool="entity_context",
            args=ctx_args,
            bibcodes=_bibcodes_of(ctx_res),
            latency_ms=ms,
            error=_error_of(ctx_res),
        )

    # 4 — filtered (or unfiltered) search.
    if resolved_ids:
        filt_args: dict = {
            "query": query,
            "mode": "keyword",
            "limit": payload.top_n,
            "filters": {"entity_ids": resolved_ids[:5]},
        }
    else:
        filt_args = {
            "query": query,
            "mode": "keyword",
            "limit": payload.top_n,
        }
    filt_res, ms = _call_mcp("search", filt_args)
    filtered_bibs = _bibcodes_of(filt_res)
    _record_step(
        steps,
        tool="search",
        args={
            "query": query[:120],
            "mode": "keyword",
            "entity_ids": resolved_ids[:5] if resolved_ids else [],
        },
        bibcodes=filtered_bibs,
        latency_ms=ms,
        error=_error_of(filt_res),
    )

    # 4b — fallback unfiltered search when the entity filter zeroed out
    # (sparse entity coverage on newer instruments like JWST/NIRCam). Keeps
    # the graph_context loop and the event count healthy.
    graph_seeds: list[str] = filtered_bibs
    if not graph_seeds and resolved_ids:
        fb_args: dict = {
            "query": query,
            "mode": "keyword",
            "limit": payload.top_n,
        }
        fb_res, ms = _call_mcp("search", fb_args)
        graph_seeds = _bibcodes_of(fb_res)
        _record_step(
            steps,
            tool="search",
            args={"query": query[:120], "mode": "keyword", "fallback": True},
            bibcodes=graph_seeds,
            latency_ms=ms,
            error=_error_of(fb_res),
        )

    # 5 — graph_context on the top 3 bibcodes (filtered or fallback). First
    # call pulls community siblings (limit=5); the rest are metrics-only so
    # the loop stays well under the 15 s endpoint budget on hub papers.
    for idx, bib in enumerate(graph_seeds[:3]):
        include_community = idx == 0
        gc_args: dict = {
            "bibcode": bib,
            "include_community": include_community,
            "limit": 5,
        }
        gc_res, ms = _call_mcp("graph_context", gc_args)
        _record_step(
            steps,
            tool="graph_context",
            args=gc_args,
            bibcodes=_bibcodes_of(gc_res) or [bib],
            latency_ms=ms,
            error=_error_of(gc_res),
        )

    # 6 — temporal_evolution on the query.
    te_args: dict = {"bibcode_or_query": query}
    te_res, ms = _call_mcp("temporal_evolution", te_args)
    _record_step(
        steps,
        tool="temporal_evolution",
        args={"query": query[:120]},
        bibcodes=_bibcodes_of(te_res),
        latency_ms=ms,
        error=_error_of(te_res),
    )

    # 7 — get_paper on the top bibcode (if any) — final narration drill.
    if graph_seeds:
        gp_args: dict = {"bibcode": graph_seeds[0]}
        gp_res, ms = _call_mcp("get_paper", gp_args)
        _record_step(
            steps,
            tool="get_paper",
            args=gp_args,
            bibcodes=_bibcodes_of(gp_res),
            latency_ms=ms,
            error=_error_of(gp_res),
        )

    return {
        "query": query,
        "scenario": "disambig",
        "entity_type": entity_type,
        "resolved_entity_ids": resolved_ids,
        "total_latency_ms": round((time.monotonic() - t_start) * 1000.0, 1),
        "bibcodes": graph_seeds,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Ego-network viewer: /viz/api/ego/{bibcode}
#
# Returns the 1-hop citation neighborhood of a paper plus a weighted sample
# of its 2-hop neighborhood, shaped for a force-directed graph view. Direct
# edges ("ref" = center cites neighbor, "cite" = neighbor cites center) come
# straight from ``citation_edges``. Second-hop nodes are ranked by the number
# of distinct 1-hop neighbors that link to them — that is the "weight".
#
# Edge set is intentionally flat: each edge carries {source, target, kind} so
# the client can color/thicken without re-deriving topology. No weight >1 is
# exposed on individual edges because ``citation_edges`` is a boolean relation
# (PK on (source, target)); edge thickness on the frontend is driven by the
# node weight of the 2-hop endpoint.
# ---------------------------------------------------------------------------


# Caps are exposed as per-request Query params (clamped) so the frontend can
# ask for a smaller view on mobile without a server redeploy.
MAX_DIRECT_REFS_DEFAULT = 100
MAX_DIRECT_CITES_DEFAULT = 100
MAX_SECOND_HOP_DEFAULT = 200

# Per-neighbor cap on edges pulled during the 2-hop expansion. Highly-cited
# papers (review articles, seminal works) can have 100k+ inbound edges; without
# this cap, the second-hop query scans tens of millions of rows. 60 was chosen
# empirically as the largest value that keeps a cold-cache fetch under 1s for
# typical papers (100 refs + ~20 cites) on the 299M-edge production graph.
_PER_NEIGHBOR_EDGE_CAP = 60


def _fetch_ego_network(
    center_bibcode: str,
    max_refs: int,
    max_cites: int,
    max_second_hop: int,
) -> Optional[PaperDict]:
    """Build a 1-hop + weighted 2-hop citation neighborhood around a bibcode.

    Returns ``None`` if the center bibcode is not present in ``papers``. All
    DB errors propagate — do not swallow them (see coding-style guidance).

    The structure returned:

    ``{
        "center": {bibcode, title, community_id},
        "direct_refs":  [{bibcode, title, community_id}, ...],   # center cites
        "direct_cites": [{bibcode, title, community_id}, ...],   # cites center
        "second_hop_sample": [
            {bibcode, title, community_id, weight: int}, ...
        ],
        "edges": [{source, target, kind: 'ref'|'cite'|'hop'}, ...]
    }``
    """
    conn: psycopg.Connection = get_connection()
    try:
        with conn.cursor() as cur:
            # --- 1. Center paper -------------------------------------------
            cur.execute(
                "SELECT p.bibcode, p.title, pm.community_id_coarse "
                "FROM papers p "
                "LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode "
                "WHERE p.bibcode = %s LIMIT 1",
                (center_bibcode,),
            )
            center_row = cur.fetchone()
            if center_row is None:
                return None
            center = {
                "bibcode": center_row[0],
                "title": center_row[1],
                "community_id": center_row[2],
            }

            # --- 2. Direct refs + cites (one indexed lookup each) ----------
            cur.execute(
                "SELECT target_bibcode FROM citation_edges "
                "WHERE source_bibcode = %s LIMIT %s",
                (center_bibcode, max_refs),
            )
            ref_bibs = [r[0] for r in cur.fetchall()]

            cur.execute(
                "SELECT source_bibcode FROM citation_edges "
                "WHERE target_bibcode = %s LIMIT %s",
                (center_bibcode, max_cites),
            )
            cite_bibs = [r[0] for r in cur.fetchall()]

            # --- 3. Second hop, weighted by 1-hop-path count ---------------
            neighbor_bibs = list(dict.fromkeys(ref_bibs + cite_bibs))
            second_hop_rows: list[tuple[str, int]] = []
            # Map 2-hop bib -> list of 1-hop neighbors that reached it, for
            # edge emission below.
            second_hop_edges: dict[str, list[str]] = {}

            if neighbor_bibs and max_second_hop > 0:
                # LATERAL per-neighbor LIMIT avoids blowups on highly-cited
                # hubs: each neighbor contributes at most
                # ``_PER_NEIGHBOR_EDGE_CAP`` rows per direction.
                cur.execute(
                    """
                    WITH n AS (
                        SELECT unnest(%s::text[]) AS bib
                    ),
                    hop_raw AS (
                        SELECT n.bib AS via, h.bib AS hop
                        FROM n,
                        LATERAL (
                            (SELECT target_bibcode AS bib
                               FROM citation_edges
                              WHERE source_bibcode = n.bib
                              LIMIT %s)
                            UNION ALL
                            (SELECT source_bibcode AS bib
                               FROM citation_edges
                              WHERE target_bibcode = n.bib
                              LIMIT %s)
                        ) h
                    )
                    SELECT hop,
                           count(DISTINCT via) AS weight,
                           array_agg(DISTINCT via) AS via_list
                    FROM hop_raw
                    WHERE hop <> %s
                      AND hop <> ALL(%s::text[])
                    GROUP BY hop
                    ORDER BY weight DESC, hop ASC
                    LIMIT %s
                    """,
                    (
                        neighbor_bibs,
                        _PER_NEIGHBOR_EDGE_CAP,
                        _PER_NEIGHBOR_EDGE_CAP,
                        center_bibcode,
                        neighbor_bibs,
                        max_second_hop,
                    ),
                )
                for hop, weight, via_list in cur.fetchall():
                    second_hop_rows.append((hop, int(weight)))
                    second_hop_edges[hop] = list(via_list or [])

            # --- 4. Hydrate titles + community for every node in one shot -
            all_bibs: list[str] = list(
                dict.fromkeys(
                    [center_bibcode]
                    + ref_bibs
                    + cite_bibs
                    + [hop for hop, _ in second_hop_rows]
                )
            )
            cur.execute(
                "SELECT p.bibcode, p.title, pm.community_id_coarse "
                "FROM papers p "
                "LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode "
                "WHERE p.bibcode = ANY(%s::text[])",
                (all_bibs,),
            )
            meta: dict[str, dict[str, object]] = {}
            for row in cur.fetchall():
                meta[row[0]] = {"title": row[1], "community_id": row[2]}
    finally:
        conn.close()

    def _node(bib: str) -> dict[str, object]:
        m = meta.get(bib, {})
        return {
            "bibcode": bib,
            "title": m.get("title"),
            "community_id": m.get("community_id"),
        }

    direct_refs = [_node(b) for b in ref_bibs]
    direct_cites = [_node(b) for b in cite_bibs]
    second_hop_sample = [{**_node(b), "weight": w} for b, w in second_hop_rows]

    # Edges: center->ref for each ref, cite->center for each cite, and for
    # each 2-hop node, one edge per 1-hop neighbor that reached it.
    edges: list[dict[str, object]] = []
    for b in ref_bibs:
        edges.append({"source": center_bibcode, "target": b, "kind": "ref"})
    for b in cite_bibs:
        edges.append({"source": b, "target": center_bibcode, "kind": "cite"})
    for hop, _ in second_hop_rows:
        for via in second_hop_edges.get(hop, []):
            edges.append({"source": via, "target": hop, "kind": "hop"})

    # Fill in community_id on the center from the hydrated metadata (prefer
    # the just-fetched value, which may be richer than the quick lookup above).
    if center_bibcode in meta:
        center["community_id"] = meta[center_bibcode].get(
            "community_id", center["community_id"]
        )
        if meta[center_bibcode].get("title"):
            center["title"] = meta[center_bibcode]["title"]

    return {
        "center": center,
        "direct_refs": direct_refs,
        "direct_cites": direct_cites,
        "second_hop_sample": second_hop_sample,
        "edges": edges,
        "counts": {
            "direct_refs": len(direct_refs),
            "direct_cites": len(direct_cites),
            "second_hop": len(second_hop_sample),
            "edges": len(edges),
        },
    }


def get_ego_fetcher() -> EgoFetcherCallable:
    """FastAPI dependency returning the concrete ego-network fetcher.

    Like ``get_fetcher`` above, this lets tests swap in a stub via
    ``app.dependency_overrides[get_ego_fetcher] = lambda: <stub>``.
    """
    return _fetch_ego_network


@router.get("/ego/{bibcode}")
def read_ego_network(
    bibcode: str = Path(..., pattern=_BIBCODE_PATTERN),
    max_refs: int = Query(MAX_DIRECT_REFS_DEFAULT, ge=1, le=500),
    max_cites: int = Query(MAX_DIRECT_CITES_DEFAULT, ge=1, le=500),
    max_second_hop: int = Query(MAX_SECOND_HOP_DEFAULT, ge=0, le=1000),
    fetcher: EgoFetcherCallable = Depends(get_ego_fetcher),
) -> PaperDict:
    """Return the citation ego network for ``bibcode`` or 404 if unknown."""
    t0 = time.time()
    record = fetcher(bibcode, max_refs, max_cites, max_second_hop)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {bibcode}")
    record["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
    return record
