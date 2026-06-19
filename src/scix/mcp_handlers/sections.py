"""Section-retrieval MCP handler and its dense+BM25 RRF fusion helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

import psycopg

from scix import mcp_server as _srv
from scix.mcp_errors import ErrorCode
from scix.mcp_server import (
    _NOMIC_QUERY_PREFIX,
    _RRF_K_DEFAULT,
    _SNIPPET_MAX_CHARS,
    _coerce_year,
    _rrf_fuse,
    _truncate_snippet,
)
from scix.synthesize import MAX_WORKING_SET_BIBCODES

logger = logging.getLogger("scix.mcp_server")


def _encode_section_query(query: str, dimensions: int = 1024) -> list[float]:
    """Encode a query string with the local nomic-embed-text-v1.5 model.

    Reuses :func:`scix.embeddings.section_pipeline._load_model` (lazy) and
    :func:`scix.embeddings.section_pipeline.encode_batch` so this module
    inherits the same model loader the indexing pipeline uses. The query
    is prefixed with ``"search_query: "`` per the nomic model card.

    Returns a 1024-dim Python list[float]. Raises ImportError if
    sentence_transformers is not installed (caller is expected to wrap and
    return a structured MCP error).
    """
    from scix.embeddings.section_pipeline import (  # local import — lazy
        DEFAULT_MODEL,
        _load_model,
        encode_batch,
    )

    model = _load_model(DEFAULT_MODEL)
    prefixed = _NOMIC_QUERY_PREFIX + (query or "")
    vectors = encode_batch(model, [prefixed], dimensions=dimensions)
    if not vectors:
        raise RuntimeError("section query encoder returned no vectors")
    return vectors[0]

def _section_filter_clauses(
    filters: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Build SQL fragments + parameter list for the section_retrieval filters.

    Returns a tuple of ``(extra_sql, params)`` where ``extra_sql`` is a
    string of zero or more ``AND <clause>`` fragments referring to columns
    on the ``papers`` row aliased as ``p``. Params are bound positionally.

    Filter contract (matches _SECTION_FILTERS_SCHEMA):
        - year_min     -> p.year >= %s
        - year_max     -> p.year <= %s
        - bibcode_prefix -> p.bibcode LIKE %s   (caller-supplied trailing % logic)

    Unknown keys are silently ignored. ``discipline`` is intentionally not
    accepted because ``papers`` has no ``discipline`` column — see
    scix_experiments-9zyw and scix_experiments-dbl.10.
    """
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    year_min = _coerce_year(filters.get("year_min"), "year_min")
    if year_min is not None:
        clauses.append("AND p.year >= %s")
        params.append(year_min)
    year_max = _coerce_year(filters.get("year_max"), "year_max")
    if year_max is not None:
        clauses.append("AND p.year <= %s")
        params.append(year_max)
    bibcode_prefix = filters.get("bibcode_prefix")
    if bibcode_prefix is not None:
        clauses.append("AND p.bibcode LIKE %s")
        params.append(f"{bibcode_prefix}%")
    return (" " + " ".join(clauses)) if clauses else "", params

def _section_dense_retrieve(
    conn: psycopg.Connection,
    query_vector: Sequence[float],
    filter_sql: str,
    filter_params: list[Any],
    fanout: int,
) -> list[tuple[str, int, float]]:
    """Run the dense leg of section retrieval inside an explicit transaction.

    Sets ``hnsw.iterative_scan = 'relaxed'`` and ``hnsw.ef_search = 100``
    via ``SET LOCAL`` so they roll back on transaction end and don't leak
    to other pool consumers.

    Returns a list of ``(bibcode, section_index, distance)`` tuples ordered
    by distance ascending (best first).
    """
    if fanout <= 0:
        return []
    vector_literal = "[" + ",".join(repr(float(v)) for v in query_vector) + "]"
    sql = f"""
        SELECT se.bibcode, se.section_index,
               (se.embedding <=> %s::halfvec) AS distance
        FROM section_embeddings se
        JOIN papers p ON p.bibcode = se.bibcode
        WHERE TRUE
        {filter_sql}
        ORDER BY se.embedding <=> %s::halfvec
        LIMIT %s
    """
    params = [vector_literal, *filter_params, vector_literal, fanout]
    rows: list[tuple[str, int, float]] = []
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed'")
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            cur.execute(sql, params)
            for row in cur.fetchall():
                rows.append((row[0], int(row[1]), float(row[2])))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
    return rows

# Candidate-pool cap for the section BM25 leg (scix_experiments-ynt8). Mirrors
# the lexical_search cap (search._LEXICAL_POOL_DEFAULT, bead 3t37): without it a
# common single-token query matches a large slice of papers_fulltext (14.4M
# rows) and forces ts_rank over the whole match set — the same
# ORDER-BY-ts_rank cost that times out lexical_search on broad terms. The
# section leg is heavier per candidate (jsonb unnest + per-section to_tsvector),
# so the cap matters even at the smaller corpus size. The cap LIMITs candidates
# in bitmap-heap (TID) order *before* ranking, so it is a blunt recall
# instrument — acceptable because the leg is RRF-fused with the dense leg. The
# default borrows the lexical knee (30000) pending section-specific tuning;
# operators retune via SCIX_SECTIONS_POOL without a restart. A separate knob
# (not SCIX_LEXICAL_POOL) keeps the two lanes independently tunable.
_SECTIONS_POOL_DEFAULT: int = 30000

# Token values of SCIX_SECTIONS_POOL that disable the cap (rank the full match
# set). Mirrors search._LEXICAL_POOL_UNBOUNDED; for eval harnesses only, not the
# live server.
_SECTIONS_POOL_UNBOUNDED: frozenset[str] = frozenset({"inf", "all", "none"})

def _resolve_sections_pool() -> int | None:
    """Resolve the section BM25 candidate-pool cap from ``SCIX_SECTIONS_POOL``.

    Returns the row cap, or ``None`` for an unbounded pool (passed to SQL as
    ``LIMIT NULL``, which Postgres treats as no limit). Read on every call so
    operators can tune the running container without a restart. Misconfigured
    values log a warning and fall back to :data:`_SECTIONS_POOL_DEFAULT`.
    """
    raw = os.environ.get("SCIX_SECTIONS_POOL")
    if raw is None:
        return _SECTIONS_POOL_DEFAULT
    token = raw.strip().lower()
    if token in _SECTIONS_POOL_UNBOUNDED:
        return None
    try:
        value = int(token)
    except ValueError:
        logger.warning(
            "SCIX_SECTIONS_POOL=%r is not an integer or one of %s; falling back to %d",
            raw,
            sorted(_SECTIONS_POOL_UNBOUNDED),
            _SECTIONS_POOL_DEFAULT,
        )
        return _SECTIONS_POOL_DEFAULT
    if value <= 0:
        logger.warning(
            "SCIX_SECTIONS_POOL=%d must be positive (use INF for unbounded); falling back to %d",
            value,
            _SECTIONS_POOL_DEFAULT,
        )
        return _SECTIONS_POOL_DEFAULT
    return value

def _section_bm25_retrieve(
    conn: psycopg.Connection,
    query: str,
    filter_sql: str,
    filter_params: list[Any],
    fanout: int,
) -> list[tuple[str, int, float]]:
    """Run the BM25 leg over papers_fulltext.sections_tsv.

    The tsvector index ranks the *paper*; we then unnest each matching
    paper's sections and emit one row per section whose body text matches
    the query terms via plainto_tsquery, scored by ts_rank on that section
    text. Returns ``(bibcode, section_index, ts_rank)`` tuples sorted by
    rank descending (higher = better).
    """
    if fanout <= 0:
        return []
    pool_size = _resolve_sections_pool()
    sql = f"""
        WITH matching_candidates AS (
            SELECT pf.bibcode, pf.sections, pf.sections_tsv
            FROM papers_fulltext pf
            JOIN papers p ON p.bibcode = pf.bibcode
            WHERE pf.sections_tsv @@ plainto_tsquery('english', %s)
            {filter_sql}
            LIMIT %s
        ),
        matching_papers AS (
            SELECT bibcode, sections,
                   ts_rank(sections_tsv, plainto_tsquery('english', %s)) AS paper_rank
            FROM matching_candidates
            ORDER BY paper_rank DESC
            LIMIT %s
        ),
        per_section AS (
            SELECT mp.bibcode,
                   (sec.ord - 1)::int AS section_index,
                   ts_rank(
                       to_tsvector('english',
                           coalesce(sec.value->>'heading', '') || ' ' ||
                           coalesce(sec.value->>'text', '')
                       ),
                       plainto_tsquery('english', %s)
                   ) AS section_rank
            FROM matching_papers mp,
                 jsonb_array_elements(mp.sections) WITH ORDINALITY AS sec(value, ord)
            WHERE to_tsvector('english',
                      coalesce(sec.value->>'heading', '') || ' ' ||
                      coalesce(sec.value->>'text', '')
                  ) @@ plainto_tsquery('english', %s)
        )
        SELECT bibcode, section_index, section_rank
        FROM per_section
        ORDER BY section_rank DESC, bibcode, section_index
        LIMIT %s
    """
    params = [
        query,  # candidate match: sections_tsv @@ plainto_tsquery
        *filter_params,
        pool_size,  # candidate-pool cap (LIMIT NULL = unbounded)
        query,  # paper_rank ts_rank
        fanout,  # paper LIMIT
        query,  # section_rank ts_rank
        query,  # section match
        fanout,  # final LIMIT
    ]
    rows: list[tuple[str, int, float]] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            rows.append((row[0], int(row[1]), float(row[2])))
    return rows

def _hydrate_section_payload(
    conn: psycopg.Connection,
    keys: Sequence[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Fetch heading + text for each (bibcode, section_index) key.

    Reads ``papers_fulltext.sections`` JSONB once per bibcode and indexes
    into the requested section. Returns a dict keyed by (bibcode, idx)
    whose values carry ``section_heading`` and ``snippet`` (truncated).
    """
    if not keys:
        return {}
    bibcodes = sorted({k[0] for k in keys})
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    sections_by_bibcode: dict[str, list[Any]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, sections FROM papers_fulltext WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        for bibcode, sections in cur.fetchall():
            if isinstance(sections, (str, bytes)):
                try:
                    sections = json.loads(sections)
                except (TypeError, ValueError, json.JSONDecodeError):
                    sections = []
            if isinstance(sections, list):
                sections_by_bibcode[bibcode] = sections
    for bibcode, idx in keys:
        sections = sections_by_bibcode.get(bibcode) or []
        section: dict[str, Any] | None = None
        if 0 <= idx < len(sections) and isinstance(sections[idx], dict):
            section = sections[idx]
        heading = (section.get("heading") if section else None) or ""
        text = (section.get("text") if section else None) or ""
        payloads[(bibcode, idx)] = {
            "section_heading": heading,
            "snippet": _truncate_snippet(text, _SNIPPET_MAX_CHARS),
        }
    return payloads

def _hydrate_canonical_urls(
    conn: psycopg.Connection,
    bibcodes: Sequence[str],
) -> dict[str, str | None]:
    """Map each bibcode to a canonical_url.

    Uses the first identifier in ``papers.identifier`` matching the arXiv
    pattern (mirrors :func:`scix.search._lookup_arxiv_id`) and feeds it to
    :func:`scix.sources.ar5iv._build_canonical_url`. bibcodes without an
    arXiv identifier map to None.
    """
    if not bibcodes:
        return {}
    from scix.sources.ar5iv import _ARXIV_ID_RE, _build_canonical_url

    out: dict[str, str | None] = {bibcode: None for bibcode in bibcodes}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, identifier FROM papers WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
        for bibcode, identifiers in cur.fetchall():
            if not identifiers:
                continue
            for ident in identifiers:
                if isinstance(ident, str) and _ARXIV_ID_RE.match(ident):
                    out[bibcode] = _build_canonical_url(ident)
                    break
    return out

def _handle_section_retrieval(conn: psycopg.Connection, args: dict[str, Any]) -> str:
    """Dispatch handler for ``section_retrieval``.

    Encodes the query with the local nomic model, runs dense HNSW + BM25
    retrieval in parallel (sequentially in code, structurally independent),
    fuses ranks via Reciprocal Rank Fusion (k=60), hydrates section text +
    canonical_url, and returns the top ``k`` items.
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return json.dumps(
            {
                "error": "query must be a non-empty string",
                "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            }
        )

    try:
        k = int(args.get("k", 10))
    except (TypeError, ValueError):
        return json.dumps(
            {"error": "k must be an integer", "error_code": ErrorCode.INVALID_PARAM_TYPE}
        )
    if k <= 0:
        return json.dumps(
            {"error": "k must be positive", "error_code": ErrorCode.INVALID_PARAM_VALUE}
        )
    # Cap fanout to keep blast radius bounded; matches the convention used
    # elsewhere in this module (find_gaps caps at MAX_WORKING_SET_BIBCODES).
    k = min(k, MAX_WORKING_SET_BIBCODES)

    try:
        filter_sql, filter_params = _section_filter_clauses(args.get("filters"))
    except ValueError as exc:
        return json.dumps({"error": str(exc), "error_code": ErrorCode.INVALID_FILTERS})

    # Encode the query with the local nomic model.
    try:
        query_vector = _srv._encode_section_query(query)
    except ImportError:
        return json.dumps(
            {
                "error": "embedding_dependency_missing",
                "error_code": ErrorCode.DEPENDENCY_MISSING,
                "hint": (
                    "section_retrieval requires sentence-transformers. "
                    "Install with: pip install -e .[search]"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("section_retrieval encode failed")
        return json.dumps({"error": f"encode_failed: {exc}", "error_code": ErrorCode.ENCODE_FAILED})

    fanout = max(50, k * 10)

    # Dense leg — explicit txn so SET LOCAL settings apply.
    try:
        dense_rows = _section_dense_retrieve(conn, query_vector, filter_sql, filter_params, fanout)
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("section_retrieval dense leg failed")
        return json.dumps(
            {
                "error": f"dense_retrieve_failed: {exc}",
                "error_code": ErrorCode.DENSE_RETRIEVE_FAILED,
            }
        )

    # BM25 leg.
    try:
        bm25_rows = _section_bm25_retrieve(conn, query, filter_sql, filter_params, fanout)
    except Exception as exc:  # noqa: BLE001 — boundary
        logger.exception("section_retrieval bm25 leg failed")
        return json.dumps(
            {
                "error": f"bm25_retrieve_failed: {exc}",
                "error_code": ErrorCode.BM25_RETRIEVE_FAILED,
            }
        )

    dense_keys: list[tuple[str, int]] = [(b, i) for (b, i, _d) in dense_rows]
    bm25_keys: list[tuple[str, int]] = [(b, i) for (b, i, _r) in bm25_rows]

    fused = _rrf_fuse([dense_keys, bm25_keys], k_rrf=_RRF_K_DEFAULT)
    top_keys = [key for (key, _score) in fused[:k]]

    payloads = _hydrate_section_payload(conn, top_keys)
    bibcodes = sorted({k_[0] for k_ in top_keys})
    canonical_urls = _hydrate_canonical_urls(conn, bibcodes)

    score_by_key = {key: score for (key, score) in fused}
    results: list[dict[str, Any]] = []
    for key in top_keys:
        bibcode, idx = key
        payload = payloads.get(key) or {}
        results.append(
            {
                "bibcode": bibcode,
                "section_heading": payload.get("section_heading", ""),
                "snippet": payload.get("snippet", ""),
                "score": float(score_by_key.get(key, 0.0)),
                "canonical_url": canonical_urls.get(bibcode),
            }
        )
    return json.dumps(
        {"results": results, "total": len(results)},
        indent=2,
        default=str,
    )
