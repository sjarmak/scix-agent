"""Core search module with timing instrumentation.

Every public function returns a SearchResult that includes timing metadata
(milliseconds) for benchmarking and observability. This is a hard contract:
callers can always access result.timing_ms to understand latency breakdown.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from scix.db import IterativeScanMode, configure_iterative_scan
from scix.sources.ar5iv import _ARXIV_ID_RE, LATEX_DERIVED_SOURCES, _build_canonical_url
from scix.sources.licensing import enforce_snippet_budget
from scix.stubs import PaperStub

logger = logging.getLogger(__name__)

# Stub columns used in all search queries that return PaperStubs
STUB_COLUMNS = "p.bibcode, p.title, p.first_author, p.year, p.citation_count, p.abstract"

# Default RRF constant (controls how much rank position matters vs raw score)
RRF_K = 60

# Halfvec cutover gate. Migrations 053/054 add paper_embeddings.embedding_hv
# and idx_embed_hnsw_indus_hv, but were not applied to prod scix as of
# 2026-04-22 (see bead scix_experiments-d0a). Default False so vector_search
# stays on the legacy vector(768) column + idx_embed_hnsw_indus for INDUS.
# Set SCIX_USE_HALFVEC=1 only after migration + scripts/backfill_halfvec.py
# completes; see docs/runbooks/halfvec_migration.md.
_HALFVEC_ENABLED = os.environ.get("SCIX_USE_HALFVEC", "0") == "1"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Result from any search/query operation. Always includes timing metadata."""

    papers: list[dict[str, Any]]
    total: int
    timing_ms: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchFilters:
    """Optional filters applied to search queries.

    Scalar column filters (year, doctype, etc.) emit WHERE fragments via
    ``to_where_clause``. Entity-graph filters (``entity_types`` /
    ``entity_ids``) emit a correlated EXISTS fragment against
    ``document_entities_canonical`` via ``to_entity_filter_clause`` — kept
    separate so callers can control where the JOIN is anchored in complex
    query shapes (e.g. CTE-based filter-first vector search).

    Empty sequences are normalized to ``None`` so accidental ``filters={}``
    payloads from agents do not silently drop every result. Lists are
    coerced to tuples so the frozen dataclass remains hashable/immutable.
    """

    year_min: int | None = None
    year_max: int | None = None
    arxiv_class: str | None = None
    doctype: str | None = None
    first_author: str | None = None
    entity_types: tuple[str, ...] | None = None
    entity_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Normalize sequences to tuples; empty -> None.
        object.__setattr__(
            self, "entity_types", _normalize_seq(self.entity_types, "entity_types", str)
        )
        object.__setattr__(self, "entity_ids", _normalize_seq(self.entity_ids, "entity_ids", int))

    def to_where_clause(self, table_alias: str = "p") -> tuple[str, list[Any]]:
        """Generate a WHERE clause fragment and parameter list.

        Returns ("AND ... AND ...", [param1, param2, ...]) or ("", []) if no filters.
        Entity-graph filters are emitted separately via ``to_entity_filter_clause``.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if self.year_min is not None:
            clauses.append(f"{table_alias}.year >= %s")
            params.append(self.year_min)
        if self.year_max is not None:
            clauses.append(f"{table_alias}.year <= %s")
            params.append(self.year_max)
        if self.arxiv_class is not None:
            clauses.append(f"{table_alias}.arxiv_class @> ARRAY[%s]")
            params.append(self.arxiv_class)
        if self.doctype is not None:
            clauses.append(f"{table_alias}.doctype = %s")
            params.append(self.doctype)
        if self.first_author is not None:
            clauses.append(f"{table_alias}.first_author ILIKE %s")
            params.append(f"%{self.first_author}%")

        if not clauses:
            return "", []
        return " AND " + " AND ".join(clauses), params

    def to_entity_filter_clause(self, table_alias: str = "p") -> tuple[str, list[Any]]:
        """Generate an EXISTS clause restricting *table_alias*.bibcode to papers
        linked to the configured entities.

        Uses the ``document_entities_canonical`` materialized view which has
        ``(bibcode)`` and ``(entity_id, fused_confidence DESC)`` indexes; the
        ``entities`` table has a btree on ``entity_type``. The per-paper
        semi-join is cheap because the outer search has already narrowed
        candidates (lexical/vector LIMIT). Returns ("", []) when neither
        filter is set.
        """
        if self.entity_types is None and self.entity_ids is None:
            return "", []

        inner_conds: list[str] = [f"dec.bibcode = {table_alias}.bibcode"]
        params: list[Any] = []

        if self.entity_ids is not None:
            inner_conds.append("dec.entity_id = ANY(%s)")
            params.append(list(self.entity_ids))

        # Only JOIN entities when an entity_type filter is set — keeps the
        # plan simple for the id-only path.
        if self.entity_types is not None:
            join = " JOIN entities e ON e.id = dec.entity_id"
            inner_conds.append("e.entity_type = ANY(%s)")
            params.append(list(self.entity_types))
        else:
            join = ""

        where = " AND ".join(inner_conds)
        clause = (
            f" AND EXISTS ("
            f"SELECT 1 FROM document_entities_canonical dec{join} "
            f"WHERE {where})"
        )
        return clause, params


def _normalize_seq(value: Any, name: str, element_type: type) -> tuple[Any, ...] | None:
    """Coerce a sequence to a tuple; None/empty => None; validate element types.

    Rejects bare strings/bytes (which are iterable but rarely the intent when
    the field expects a sequence of items), and explicitly rejects booleans
    when ``element_type`` is ``int`` so ``True`` is not silently coerced to 1.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(
            f"{name} must be a sequence of {element_type.__name__}, " f"got {type(value).__name__}"
        )
    seq = tuple(value)
    if not seq:
        return None
    for item in seq:
        if element_type is int and isinstance(item, bool):
            raise TypeError(f"{name} items must be int, got bool")
        if not isinstance(item, element_type):
            raise TypeError(
                f"{name} items must be {element_type.__name__}, got {type(item).__name__}"
            )
    return seq


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------


def _elapsed_ms(t0: float) -> float:
    """Milliseconds elapsed since t0 (from time.perf_counter)."""
    return round((time.perf_counter() - t0) * 1000, 2)


# ---------------------------------------------------------------------------
# Lexical search (tsvector with custom scix_english config)
# ---------------------------------------------------------------------------


def lexical_search(
    conn: psycopg.Connection,
    query_text: str,
    *,
    filters: SearchFilters | None = None,
    limit: int = 20,
    ts_config: str = "scix_english",
) -> SearchResult:
    """Full-text search using PostgreSQL tsvector with ts_rank_cd scoring.

    Uses the custom scix_english text search config by default, which handles
    scientific text (hyphens like X-ray, numeric tokens) better than built-in english.
    Falls back to 'english' if scix_english config does not exist.
    """
    t0 = time.perf_counter()

    effective = filters or SearchFilters()
    filter_clause, filter_params = effective.to_where_clause("p")
    entity_clause, entity_params = effective.to_entity_filter_clause("p")

    # plainto_tsquery is more robust than websearch_to_tsquery for programmatic use:
    # it doesn't fail on unmatched quotes or special chars in user input.
    query = f"""
        SELECT {STUB_COLUMNS},
               ts_rank_cd(p.tsv, plainto_tsquery('{ts_config}', %s), 32) AS rank
        FROM papers p
        WHERE p.tsv @@ plainto_tsquery('{ts_config}', %s)
        {filter_clause}
        {entity_clause}
        ORDER BY rank DESC
        LIMIT %s
    """
    params: list[Any] = [query_text, query_text] + filter_params + entity_params + [limit]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    lexical_ms = _elapsed_ms(t0)

    papers = []
    for row in rows:
        stub = PaperStub.from_row(row).to_dict()
        stub["score"] = float(row["rank"])
        papers.append(stub)

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"lexical_ms": lexical_ms},
    )


# ---------------------------------------------------------------------------
# Body lexical search (GIN expression index on papers.body tsvector)
# ---------------------------------------------------------------------------

# Matches the partial-index predicate in migration 039 — bodies above the
# tsvector limit are not indexed and the expression cannot be evaluated.
_BODY_TSVECTOR_MAX_BYTES = 1_048_575


def lexical_search_body(
    conn: psycopg.Connection,
    query_text: str,
    *,
    filters: SearchFilters | None = None,
    limit: int = 20,
) -> SearchResult:
    """Body-text full-text search using the GIN expression index on papers.body.

    Uses the built-in 'english' config (matching the index expression in
    migration 039). Rows with NULL or oversized bodies are skipped to match
    the partial-index predicate, ensuring the planner uses the GIN index.

    Ranking uses the title+abstract tsvector (papers.tsv) rather than
    recomputing to_tsvector on the full body text — the body expression
    index handles the match filter, but ts_rank_cd on 65KB bodies is
    prohibitively slow (~400s per query). Since body results are fused
    via RRF with other signals, approximate ordering is sufficient.
    """
    t0 = time.perf_counter()

    effective = filters or SearchFilters()
    filter_clause, filter_params = effective.to_where_clause("p")
    entity_clause, entity_params = effective.to_entity_filter_clause("p")

    query = f"""
        SELECT {STUB_COLUMNS},
               ts_rank_cd(p.tsv, plainto_tsquery('english', %s), 32) AS rank
        FROM papers p
        WHERE p.body IS NOT NULL
          AND length(p.body) <= {_BODY_TSVECTOR_MAX_BYTES}
          AND to_tsvector('english', p.body) @@ plainto_tsquery('english', %s)
        {filter_clause}
        {entity_clause}
        ORDER BY rank DESC
        LIMIT %s
    """
    params: list[Any] = [query_text, query_text] + filter_params + entity_params + [limit]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    body_lexical_ms = _elapsed_ms(t0)

    papers = []
    for row in rows:
        stub = PaperStub.from_row(row).to_dict()
        stub["score"] = float(row["rank"])
        papers.append(stub)

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"body_lexical_ms": body_lexical_ms},
    )


# ---------------------------------------------------------------------------
# Vector search (pgvector cosine similarity with tunable ef_search)
# ---------------------------------------------------------------------------


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    model_name: str = "indus",
    filters: SearchFilters | None = None,
    limit: int = 20,
    ef_search: int = 100,
    iterative_scan: IterativeScanMode | None = None,
) -> SearchResult:
    """Approximate nearest neighbor search using pgvector HNSW.

    Args:
        query_embedding: 768-dim float vector.
        model_name: Filter to a specific embedding model (default: indus).
        ef_search: HNSW ef_search parameter (higher = more accurate, slower).
            When iterative_scan is enabled, ef_search is less critical because
            pgvector automatically expands the search to satisfy the LIMIT.
        iterative_scan: pgvector 0.8.0+ iterative scan mode. When set to
            "relaxed_order" or "strict_order", pgvector automatically expands
            the index scan until LIMIT results pass the WHERE filters.
            When None (default), iterative scan is auto-enabled with
            "relaxed_order" if filters are present and pgvector >= 0.8.0.
    """
    t0 = time.perf_counter()

    ndim = len(query_embedding)
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    # INDUS halfvec path is gated on SCIX_USE_HALFVEC=1 because migrations
    # 053/054 (add embedding_hv shadow column + HNSW index) have not been
    # applied to prod scix yet — see bead scix_experiments-d0a. Default off
    # falls back to the original vector(768) column and idx_embed_hnsw_indus.
    # Flip the env var after the migration + backfill runbook completes.
    use_halfvec = _HALFVEC_ENABLED and model_name == "indus"
    vec_col = "pe.embedding_hv" if use_halfvec else "pe.embedding"
    vec_cast = f"halfvec({ndim})" if use_halfvec else f"vector({ndim})"
    effective = filters or SearchFilters()
    filter_clause, filter_params = effective.to_where_clause("p")
    entity_clause, entity_params = effective.to_entity_filter_clause("p")

    with conn.cursor() as cur:
        # Tune HNSW probe depth for this transaction
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")

    # Auto-enable iterative scan for filtered queries on pgvector >= 0.8.0
    has_filters = bool(filter_clause) or bool(entity_clause)
    scan_mode = iterative_scan
    if scan_mode is None and has_filters:
        scan_mode = "relaxed_order"

    iterative_applied = False
    if scan_mode is not None:
        iterative_applied = configure_iterative_scan(conn, mode=scan_mode)

    # Match the per-model partial HNSW expression: halfvec(768) for INDUS,
    # vector(N) for pilots. The LHS cast `({vec_col})::{vec_cast}` is required
    # for the planner to match the indexed expression `((embedding)::vector(768))`
    # — without it, the planner falls back to a Seq Scan over 32M rows + Sort
    # (verified 2026-04-26: omitting the cast produces cost=11.5M / ~44s wall-clock;
    # adding it produces cost=4k / sub-100ms HNSW lookup).
    cast_vec = f"({vec_col})::{vec_cast}"
    query = f"""
        SELECT {STUB_COLUMNS},
               1 - ({cast_vec} <=> %s::{vec_cast}) AS similarity
        FROM paper_embeddings pe
        JOIN papers p ON p.bibcode = pe.bibcode
        WHERE pe.model_name = %s
        {filter_clause}
        {entity_clause}
        ORDER BY {cast_vec} <=> %s::{vec_cast}
        LIMIT %s
    """
    params: list[Any] = [vec_str, model_name] + filter_params + entity_params + [vec_str, limit]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    vector_ms = _elapsed_ms(t0)

    papers = []
    for row in rows:
        stub = PaperStub.from_row(row).to_dict()
        stub["score"] = float(row["similarity"])
        papers.append(stub)

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"vector_ms": vector_ms},
        metadata={"iterative_scan": iterative_applied},
    )


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------


def rrf_fuse(
    results_lists: list[list[dict[str, Any]]],
    *,
    k: int = RRF_K,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    RRF score = sum(1 / (k + rank_i)) for each list containing the paper.
    Papers appearing in multiple lists get boosted.
    """
    scores: dict[str, float] = {}
    paper_data: dict[str, dict[str, Any]] = {}

    for results in results_lists:
        for rank, paper in enumerate(results, start=1):
            bib = paper["bibcode"]
            scores[bib] = scores.get(bib, 0.0) + 1.0 / (k + rank)
            if bib not in paper_data:
                paper_data[bib] = paper

    fused = []
    for bib, score in sorted(scores.items(), key=lambda x: -x[1]):
        item = {**paper_data[bib], "rrf_score": round(score, 6)}
        fused.append(item)
        if len(fused) >= top_n:
            break

    return fused


# ---------------------------------------------------------------------------
# Cardinality estimation and filter-first vector search
# ---------------------------------------------------------------------------

# Selectivity threshold: filters matching < 1% of the corpus use
# filter-first CTE + brute-force cosine instead of HNSW iterative scan.
SELECTIVITY_THRESHOLD = 0.01


def _estimate_filter_selectivity(
    conn: psycopg.Connection,
    filters: SearchFilters,
) -> float:
    """Estimate the fraction of the corpus matching *filters*.

    Uses pg_class.reltuples for the total corpus size (no seq scan) and a
    fast COUNT on the filtered subset.  Returns a ratio in [0.0, 1.0].
    A return value of 1.0 means "no filters" or "cannot estimate".
    """
    filter_clause, filter_params = filters.to_where_clause("p")
    entity_clause, entity_params = filters.to_entity_filter_clause("p")
    if not filter_clause and not entity_clause:
        return 1.0

    with conn.cursor() as cur:
        # Total corpus estimate from planner stats (instant, no scan)
        cur.execute("SELECT GREATEST(reltuples, 1) FROM pg_class WHERE relname = 'papers'")
        row = cur.fetchone()
        if row is None:
            return 1.0
        total: float = float(row[0])

        # Cap the probe: if we find more than (threshold * total + 1) rows,
        # selectivity is already above threshold — no need to count further.
        # This bounds worst-case scan to ~1% of the corpus (~320K rows on
        # 32M) instead of a full sequential scan.
        cap = max(1, int(SELECTIVITY_THRESHOLD * total) + 1)
        count_sql = (
            f"SELECT count(*) FROM ("
            f"SELECT 1 FROM papers p WHERE TRUE {filter_clause} {entity_clause} LIMIT {cap}"
            f") sub"
        )
        cur.execute(count_sql, filter_params + entity_params)
        matched: int = cur.fetchone()[0]

        # If we hit the cap, we know selectivity ≥ threshold — return a
        # value just above threshold so the caller routes to HNSW.
        if matched >= cap:
            return SELECTIVITY_THRESHOLD + 0.001

    return matched / total if total > 0 else 1.0


def _filter_first_vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    model_name: str = "indus",
    filters: SearchFilters | None = None,
    limit: int = 20,
) -> SearchResult:
    """Filter-first CTE: get matching bibcodes, then brute-force cosine.

    Used when filter selectivity is very low (<1% of corpus) so that HNSW
    iterative scan would waste I/O expanding the index.  Instead we
    materialise the small filtered set and compute exact cosine distance.
    """
    t0 = time.perf_counter()

    ndim = len(query_embedding)
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    # See gate explanation on _vector_search_hnsw — same story here.
    use_halfvec = _HALFVEC_ENABLED and model_name == "indus"
    vec_col = "pe.embedding_hv" if use_halfvec else "pe.embedding"
    vec_cast = f"halfvec({ndim})" if use_halfvec else f"vector({ndim})"
    effective = filters or SearchFilters()
    filter_clause, filter_params = effective.to_where_clause("p")
    entity_clause, entity_params = effective.to_entity_filter_clause("p")

    query = f"""
        WITH filtered AS MATERIALIZED (
            SELECT p.bibcode
            FROM papers p
            WHERE TRUE {filter_clause}
            {entity_clause}
        )
        SELECT {STUB_COLUMNS},
               1 - ({vec_col} <=> %s::{vec_cast}) AS similarity
        FROM paper_embeddings pe
        JOIN filtered f ON f.bibcode = pe.bibcode
        JOIN papers p   ON p.bibcode = pe.bibcode
        WHERE pe.model_name = %s
        ORDER BY {vec_col} <=> %s::{vec_cast}
        LIMIT %s
    """
    params: list[Any] = filter_params + entity_params + [vec_str, model_name, vec_str, limit]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    vector_ms = _elapsed_ms(t0)

    papers = []
    for row in rows:
        stub = PaperStub.from_row(row).to_dict()
        stub["score"] = float(row["similarity"])
        papers.append(stub)

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"vector_ms": vector_ms},
        metadata={"filter_first": True},
    )


def _model_has_embeddings(conn: psycopg.Connection, model_name: str) -> bool:
    """Fast EXISTS check for whether a model has any rows in paper_embeddings."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM paper_embeddings WHERE model_name = %s LIMIT 1)",
            (model_name,),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Entity-aware query intent (alias expansion + ontology parsing)
# ---------------------------------------------------------------------------

# Cap on how many alias expansions become extra lexical RRF lanes per query —
# bounds DB load when a query mentions many entities (e.g. "JWST MIRI NIRSpec
# NIRCam imaging" would otherwise fan out to 4 extra SELECTs).
_MAX_ALIAS_LEXICAL_LANES = 3

# Cap on entity_ids lifted from a single properties_filter. The entities table
# has 1.58M rows; we never want to materialise every match into the IN list.
_MAX_PROPERTIES_FILTER_IDS = 200


def _resolve_entity_ids_for_properties(
    conn: psycopg.Connection,
    properties_filters: tuple[dict[str, str], ...],
    entity_types: tuple[str, ...] | None,
) -> tuple[int, ...]:
    """Look up entity_ids whose ``properties`` JSONB contains any of the supplied filters.

    Each filter is applied as ``entities.properties @> %s::jsonb`` and the
    union is returned. ``entity_types`` further restricts the lookup when
    supplied. Capped at :data:`_MAX_PROPERTIES_FILTER_IDS` per call.
    """
    if not properties_filters:
        return ()

    clauses: list[str] = []
    params: list[Any] = []
    for payload in properties_filters:
        clauses.append("properties @> %s::jsonb")
        params.append(json.dumps(payload))

    type_clause = ""
    if entity_types:
        type_clause = " AND entity_type = ANY(%s)"
        params.append(list(entity_types))

    sql = f"SELECT id FROM entities WHERE ({' OR '.join(clauses)}){type_clause} LIMIT %s"
    params.append(_MAX_PROPERTIES_FILTER_IDS)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return tuple(int(row[0]) for row in cur.fetchall())


def _merge_filters(
    base: SearchFilters | None,
    *,
    extra_entity_types: tuple[str, ...] = (),
    extra_entity_ids: tuple[int, ...] = (),
) -> SearchFilters:
    """Return a new :class:`SearchFilters` with extra entity scopes UNION'd in.

    Empty extras leave the field unchanged. Existing filter scalar fields
    (year, doctype, etc.) carry through verbatim.
    """
    base = base or SearchFilters()

    if extra_entity_types:
        merged_types = tuple(dict.fromkeys((*(base.entity_types or ()), *extra_entity_types)))
    else:
        merged_types = base.entity_types

    if extra_entity_ids:
        merged_ids = tuple(dict.fromkeys((*(base.entity_ids or ()), *extra_entity_ids)))
    else:
        merged_ids = base.entity_ids

    return SearchFilters(
        year_min=base.year_min,
        year_max=base.year_max,
        arxiv_class=base.arxiv_class,
        doctype=base.doctype,
        first_author=base.first_author,
        entity_types=merged_types,
        entity_ids=merged_ids,
    )


# ---------------------------------------------------------------------------
# Hybrid search (vector + lexical + RRF + optional cross-encoder rerank)
# ---------------------------------------------------------------------------


def hybrid_search(
    conn: psycopg.Connection,
    query_text: str,
    query_embedding: list[float] | None = None,
    *,
    model_name: str = "indus",
    openai_embedding: list[float] | None = None,
    filters: SearchFilters | None = None,
    vector_limit: int = 60,
    lexical_limit: int = 60,
    rrf_k: int = RRF_K,
    top_n: int = 20,
    ef_search: int = 100,
    reranker: Any | None = None,
    include_body: bool = True,
    enable_alias_expansion: bool = False,
    enable_ontology_parser: bool = False,
    alias_automaton: Any | None = None,
) -> SearchResult:
    """Hybrid search combining vector and lexical via RRF, with optional reranking.

    If query_embedding is None, falls back to lexical-only mode (BM25-only).
    If openai_embedding is provided *and* text-embedding-3-large has rows in
    paper_embeddings, runs a second vector search and fuses via RRF.
    If reranker is provided, re-ranks the top RRF results.

    Cardinality-aware routing: when filters match <1% of the corpus, uses a
    filter-first CTE with brute-force cosine instead of HNSW iterative scan.

    Args:
        model_name: Embedding model for primary vector search (default: indus).
        openai_embedding: Optional pre-computed OpenAI text-embedding-3-large vector.
        reranker: Optional callable(query_text, papers) -> list[dict] with 'rerank_score'.
        include_body: When True (default), runs body BM25 as a 4th RRF signal
            against the GIN expression index on papers.body. Set to False to
            skip when body coverage is irrelevant or for benchmarking.
        enable_alias_expansion: When True, runs :func:`scix.alias_expansion.expand_query`
            on ``query_text`` and adds one extra lexical RRF lane per matched
            entity (canonical name as the lane query), capped at
            :data:`_MAX_ALIAS_LEXICAL_LANES`. Default False.
        enable_ontology_parser: When True, runs :func:`scix.ontology_query_parser.parse_query`
            on ``query_text`` and lifts ``entity_types`` plus
            ``properties_filters`` (resolved to entity_ids) into ``filters``.
            Default False.
        alias_automaton: Optional pre-built :class:`AliasAutomaton`. When None
            and ``enable_alias_expansion`` is True, a global cached automaton
            is built lazily from the ``entities``/``entity_aliases`` tables.
    """
    timing: dict[str, float] = {}
    metadata: dict[str, Any] = {}

    # Ontology parsing — lift to filters BEFORE lexical/vector calls so all
    # downstream stages see the augmented scope.
    if enable_ontology_parser:
        from scix.ontology_query_parser import parse_query

        t_parse = time.perf_counter()
        parsed = parse_query(query_text)
        property_ids: tuple[int, ...] = ()
        if parsed.properties_filters:
            property_ids = _resolve_entity_ids_for_properties(
                conn,
                parsed.properties_filters,
                parsed.entity_types or None,
            )
        if parsed.entity_types or property_ids:
            filters = _merge_filters(
                filters,
                extra_entity_types=parsed.entity_types,
                extra_entity_ids=property_ids,
            )
        timing["ontology_parse_ms"] = _elapsed_ms(t_parse)
        metadata["ontology_clauses"] = len(parsed.clauses)
        metadata["ontology_entity_ids"] = len(property_ids)

    # Alias expansion — collect extra lexical lanes (no filter mutation, since
    # an entity-id filter would over-restrict recall vs. simple OR'd lexical
    # rerank). Each extra lane is a lexical_search on the matched entity's
    # canonical name; RRF fusion handles deduplication.
    alias_lane_queries: list[str] = []
    if enable_alias_expansion:
        from scix.alias_expansion import expand_query

        t_alias = time.perf_counter()
        expansion = expand_query(conn, query_text, automaton=alias_automaton)
        seen: set[str] = set()
        for match in expansion.matches:
            canonical = match.canonical_name.strip()
            key = canonical.lower()
            if not canonical or key in seen:
                continue
            seen.add(key)
            alias_lane_queries.append(canonical)
            if len(alias_lane_queries) >= _MAX_ALIAS_LEXICAL_LANES:
                break
        timing["alias_expansion_ms"] = _elapsed_ms(t_alias)
        metadata["alias_matches"] = len(expansion.matches)
        metadata["alias_lanes"] = len(alias_lane_queries)

    # Lexical search (title + abstract tsvector)
    lex_result = lexical_search(conn, query_text, filters=filters, limit=lexical_limit)
    timing["lexical_ms"] = lex_result.timing_ms["lexical_ms"]

    results_lists: list[list[dict[str, Any]]] = [lex_result.papers]

    # Extra alias lexical lanes — one per matched entity canonical name.
    if alias_lane_queries:
        t_alias_lex = time.perf_counter()
        for lane_query in alias_lane_queries:
            try:
                lane = lexical_search(conn, lane_query, filters=filters, limit=lexical_limit)
            except Exception:
                logger.warning(
                    "Alias lexical lane failed for %r; continuing", lane_query, exc_info=True
                )
                continue
            if lane.papers:
                results_lists.append(lane.papers)
        timing["alias_lexical_ms"] = _elapsed_ms(t_alias_lex)

    # Body BM25 (full-text via GIN expression index on papers.body)
    timing["body_lexical_ms"] = 0.0
    if include_body:
        try:
            body_result = lexical_search_body(
                conn, query_text, filters=filters, limit=lexical_limit
            )
            timing["body_lexical_ms"] = body_result.timing_ms["body_lexical_ms"]
            if body_result.papers:
                results_lists.append(body_result.papers)
        except Exception:
            logger.warning("Body BM25 search failed; continuing without it", exc_info=True)

    # Cardinality-aware routing: estimate filter selectivity once for reuse
    use_filter_first = False
    if query_embedding is not None and filters is not None:
        selectivity = _estimate_filter_selectivity(conn, filters)
        if selectivity < SELECTIVITY_THRESHOLD:
            use_filter_first = True
            logger.debug(
                "Filter selectivity %.4f < %.2f — using filter-first CTE",
                selectivity,
                SELECTIVITY_THRESHOLD,
            )

    # Primary vector search (if embeddings available)
    if query_embedding is not None:
        if use_filter_first:
            vec_result = _filter_first_vector_search(
                conn,
                query_embedding,
                model_name=model_name,
                filters=filters,
                limit=vector_limit,
            )
        else:
            vec_result = vector_search(
                conn,
                query_embedding,
                model_name=model_name,
                filters=filters,
                limit=vector_limit,
                ef_search=ef_search,
            )
        timing["vector_ms"] = vec_result.timing_ms["vector_ms"]
        results_lists.append(vec_result.papers)
    else:
        timing["vector_ms"] = 0.0

    # OpenAI vector search — skip entirely when model has 0 rows
    timing["openai_vector_ms"] = 0.0
    if openai_embedding is not None and _model_has_embeddings(conn, "text-embedding-3-large"):
        try:
            openai_vec_result = vector_search(
                conn,
                openai_embedding,
                model_name="text-embedding-3-large",
                filters=filters,
                limit=vector_limit,
                ef_search=ef_search,
            )
            timing["openai_vector_ms"] = openai_vec_result.timing_ms["vector_ms"]
            results_lists.append(openai_vec_result.papers)
        except Exception:
            logger.warning(
                "OpenAI vector search failed; falling back to primary+lexical only",
                exc_info=True,
            )
    elif openai_embedding is not None:
        logger.debug("Skipping OpenAI vector search: text-embedding-3-large has 0 rows")

    # RRF fusion
    t_fuse = time.perf_counter()
    fused = rrf_fuse(results_lists, k=rrf_k, top_n=top_n)
    timing["fusion_ms"] = _elapsed_ms(t_fuse)

    # Optional cross-encoder reranking
    if reranker is not None and fused:
        t_rerank = time.perf_counter()
        fused = reranker(query_text, fused)
        timing["rerank_ms"] = _elapsed_ms(t_rerank)
    else:
        timing["rerank_ms"] = 0.0

    timing["total_ms"] = round(sum(timing.values()), 2)

    return SearchResult(
        papers=fused,
        total=len(fused),
        timing_ms=timing,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Cross-encoder reranker (lazy-loaded, batched inference)
# ---------------------------------------------------------------------------

# HuggingFace commit SHA pin for BAAI/bge-reranker-large.
# Resolved against https://huggingface.co/BAAI/bge-reranker-large on 2026-04-25.
# Pinning to a specific revision makes inference reproducible and prevents
# silent upstream weight changes from altering rerank scores in production.
BGE_RERANKER_LARGE_SHA: str = "55611d7bca2a7133960a6d3b71e083071bbfc312"

# Local snapshot directory for bge-reranker-large weights. Tracked under
# models/ which is excluded from version control by .gitignore. The
# CrossEncoderReranker prefers this path when present so production runs
# never have to hit the network for the ~1.3 GB checkpoint.
BGE_RERANKER_LARGE_LOCAL_DIR: str = "models/bge-reranker-large"

# Public registry of supported cross-encoder model names. Anything not
# listed here is treated as an opaque sentence-transformers identifier and
# loaded directly (no local-cache resolution, no SHA pin).
_SUPPORTED_RERANKER_MODELS: tuple[str, ...] = (
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-large",
)


class CrossEncoderReranker:
    """Re-rank papers using a cross-encoder model.

    Lazy-loads the model on first use to avoid import overhead.
    Batches all candidates in a single forward pass.

    Two model paths are first-class:

    * ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (default, ~80 MB) — fast,
      general-purpose MS-MARCO baseline. Loaded directly by name.
    * ``BAAI/bge-reranker-large`` (~1.3 GB) — stronger reranker. Loaded
      from a local snapshot under :data:`BGE_RERANKER_LARGE_LOCAL_DIR`
      when that directory exists; otherwise falls back to the HF Hub
      identifier pinned at :data:`BGE_RERANKER_LARGE_SHA`.

    Any other ``model_name`` is passed through to ``CrossEncoder`` as-is.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _resolve_model_source(self) -> tuple[str, str | None]:
        """Return ``(model_path_or_name, revision)`` for ``CrossEncoder``.

        For ``BAAI/bge-reranker-large`` we prefer a local snapshot under
        :data:`BGE_RERANKER_LARGE_LOCAL_DIR` when present (no network),
        and otherwise fall back to the HF Hub identifier pinned at
        :data:`BGE_RERANKER_LARGE_SHA`. All other names are returned
        verbatim with no revision pin.
        """
        if self._model_name == "BAAI/bge-reranker-large":
            if os.path.isdir(BGE_RERANKER_LARGE_LOCAL_DIR):
                return BGE_RERANKER_LARGE_LOCAL_DIR, None
            return self._model_name, BGE_RERANKER_LARGE_SHA
        return self._model_name, None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            source, revision = self._resolve_model_source()
            if revision is not None:
                self._model = CrossEncoder(source, revision=revision)
            else:
                self._model = CrossEncoder(source)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )

    def __call__(
        self,
        query: str,
        papers: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Re-rank papers by cross-encoder score. Returns sorted list."""
        # Short-circuit empty input — no need to load weights or score.
        if not papers:
            return []

        self._load()

        pairs: list[tuple[str, str]] = []
        for p in papers:
            doc_text = (p.get("title") or "") + ". " + (p.get("abstract_snippet") or "")
            pairs.append((query, doc_text))

        scores = self._model.predict(pairs)

        scored = []
        for paper, score in zip(papers, scores):
            scored.append({**paper, "rerank_score": float(score)})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_n is not None:
            scored = scored[:top_n]

        return scored


# ---------------------------------------------------------------------------
# Paper detail and graph queries (all return SearchResult with timing)
# ---------------------------------------------------------------------------


def get_paper(conn: psycopg.Connection, bibcode: str) -> SearchResult:
    """Fetch full metadata for a single paper by bibcode."""
    t0 = time.perf_counter()

    sql = """
        SELECT p.*,
               (SELECT count(*) FROM citation_edges WHERE source_bibcode = p.bibcode) AS ref_count,
               (SELECT count(*) FROM citation_edges WHERE target_bibcode = p.bibcode) AS cite_count
        FROM papers p
        WHERE p.bibcode = %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (bibcode,))
        row = cur.fetchone()

    query_ms = _elapsed_ms(t0)

    if row is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": query_ms},
            metadata={"error": f"Paper not found: {bibcode}"},
        )

    # Convert non-JSON-serializable types
    result = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        else:
            result[k] = v

    return SearchResult(
        papers=[result],
        total=1,
        timing_ms={"query_ms": query_ms},
    )


_EDGE_COLS = frozenset({"source_bibcode", "target_bibcode"})
_COUNT_ALIASES = frozenset({"overlap_count", "shared_refs"})


def _citation_edge_query(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    join_col: str,
    where_col: str,
    limit: int,
) -> SearchResult:
    """Shared implementation for get_citations and get_references."""
    assert join_col in _EDGE_COLS, f"invalid join_col: {join_col}"
    assert where_col in _EDGE_COLS, f"invalid where_col: {where_col}"

    t0 = time.perf_counter()

    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM citation_edges ce
        JOIN papers p ON p.bibcode = ce.{join_col}
        WHERE ce.{where_col} = %s
        ORDER BY p.citation_count DESC NULLS LAST
        LIMIT %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (bibcode, limit))
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)
    papers = [PaperStub.from_row(row).to_dict() for row in rows]

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"query_ms": query_ms},
    )


def get_citations(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    limit: int = 20,
) -> SearchResult:
    """Get forward citations (papers that cite this paper). Returns stubs."""
    return _citation_edge_query(
        conn, bibcode, join_col="source_bibcode", where_col="target_bibcode", limit=limit
    )


def get_references(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    limit: int = 20,
) -> SearchResult:
    """Get backward references (papers this paper cites). Returns stubs."""
    return _citation_edge_query(
        conn, bibcode, join_col="target_bibcode", where_col="source_bibcode", limit=limit
    )


def _author_name_variants(author_name: str) -> list[str]:
    """Generate plausible ADS name variants from user input.

    ADS stores names as "Surname, Given" with varying abbreviation levels.
    Given "Jarmak, Stephanie G." this produces:
        Jarmak, Stephanie G.
        Jarmak, Stephanie
        Jarmak, S. G.
        Jarmak, S.
    Given "Stephanie Jarmak" (western order) it normalizes first.
    """
    # Normalize to "Surname, Given" form
    if "," in author_name:
        surname, given = author_name.split(",", 1)
        surname = surname.strip()
        given = given.strip()
    else:
        parts = author_name.strip().split()
        if len(parts) >= 2:
            surname = parts[-1]
            given = " ".join(parts[:-1])
        else:
            # Single name — return as-is
            return [author_name.strip()]

    variants: list[str] = []
    given_parts = given.split()

    if given_parts:
        # Full name: "Jarmak, Stephanie G."
        variants.append(f"{surname}, {given}")
        # First name only: "Jarmak, Stephanie"
        if len(given_parts) > 1:
            variants.append(f"{surname}, {given_parts[0]}")
        # All initials: "Jarmak, S. G."
        initials_full = " ".join(f"{p[0]}." for p in given_parts if p)
        variants.append(f"{surname}, {initials_full}")
        # First initial only: "Jarmak, S."
        if len(given_parts) > 1:
            variants.append(f"{surname}, {given_parts[0][0]}.")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def get_author_papers(
    conn: psycopg.Connection,
    author_name: str,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 50,
) -> SearchResult:
    """Get papers by an author.

    Generates plausible name variants from the input, then queries
    the GIN index on authors via array containment (@>).
    Each variant hits a fast bitmap index scan — no sequential scan needed.
    """
    t0 = time.perf_counter()

    year_clause = ""
    year_params: list[Any] = []

    if year_min is not None:
        year_clause += " AND p.year >= %s"
        year_params.append(year_min)
    if year_max is not None:
        year_clause += " AND p.year <= %s"
        year_params.append(year_max)

    variants = _author_name_variants(author_name)

    # Use GIN index (authors @>) for each variant — fast BitmapOr
    or_clauses = " OR ".join(["p.authors @> ARRAY[%s]"] * len(variants))
    params: list[Any] = [*variants, *year_params, limit]

    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM papers p
        WHERE ({or_clauses})
        {year_clause}
        ORDER BY p.year DESC, p.citation_count DESC NULLS LAST
        LIMIT %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)

    papers = [PaperStub.from_row(row).to_dict() for row in rows]

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"query_ms": query_ms},
    )


# ---------------------------------------------------------------------------
# Graph analysis queries (co-citation, bibliographic coupling, citation chain)
# ---------------------------------------------------------------------------


def _overlap_query(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    join_col: str,
    where_col: str,
    result_col: str,
    count_alias: str,
    min_overlap: int,
    limit: int,
) -> SearchResult:
    """Shared implementation for co-citation analysis and bibliographic coupling.

    Both queries find papers that share citation edges with the given bibcode,
    differing only in which edge direction is joined and which is filtered.
    """
    assert join_col in _EDGE_COLS, f"invalid join_col: {join_col}"
    assert where_col in _EDGE_COLS, f"invalid where_col: {where_col}"
    assert result_col in _EDGE_COLS, f"invalid result_col: {result_col}"
    assert count_alias in _COUNT_ALIASES, f"invalid count_alias: {count_alias}"

    t0 = time.perf_counter()

    sql = f"""
        SELECT {STUB_COLUMNS}, COUNT(*) AS {count_alias}
        FROM citation_edges ce1
        JOIN citation_edges ce2 ON ce1.{join_col} = ce2.{join_col}
        JOIN papers p ON p.bibcode = ce2.{result_col}
        WHERE ce1.{where_col} = %s
          AND ce2.{result_col} != %s
        GROUP BY {STUB_COLUMNS}
        HAVING COUNT(*) >= %s
        ORDER BY {count_alias} DESC
        LIMIT %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (bibcode, bibcode, min_overlap, limit))
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)

    papers = []
    for row in rows:
        stub = PaperStub.from_row(row).to_dict()
        stub[count_alias] = row[count_alias]
        papers.append(stub)

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"query_ms": query_ms},
    )


def co_citation_analysis(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    min_overlap: int = 2,
    limit: int = 20,
) -> SearchResult:
    """Find papers frequently co-cited with the given paper.

    Two papers are co-cited when a third paper cites both. This finds papers
    that share the most citing papers with the given bibcode.
    """
    return _overlap_query(
        conn,
        bibcode,
        join_col="source_bibcode",
        where_col="target_bibcode",
        result_col="target_bibcode",
        count_alias="overlap_count",
        min_overlap=min_overlap,
        limit=limit,
    )


def bibliographic_coupling(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    min_overlap: int = 2,
    limit: int = 20,
) -> SearchResult:
    """Find papers that share references with the given paper.

    Two papers are bibliographically coupled when they both cite the same paper.
    This finds papers that share the most references with the given bibcode.
    """
    return _overlap_query(
        conn,
        bibcode,
        join_col="target_bibcode",
        where_col="source_bibcode",
        result_col="source_bibcode",
        count_alias="shared_refs",
        min_overlap=min_overlap,
        limit=limit,
    )


def citation_chain(
    conn: psycopg.Connection,
    source_bibcode: str,
    target_bibcode: str,
    *,
    max_depth: int = 5,
) -> SearchResult:
    """Find the shortest citation path between two papers via iterative BFS.

    Walks forward along citation edges (source cites target). Returns the
    ordered path of papers from source to target, or empty if no path exists
    within max_depth hops.

    Uses Python-driven BFS rather than recursive CTEs to control fan-out
    and enable early termination.
    """
    t0 = time.perf_counter()

    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    if source_bibcode == target_bibcode:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={"path_length": 0, "path_bibcodes": [source_bibcode]},
        )

    # BFS with memory cap to prevent runaway expansion on high-degree graphs
    max_visited = 100_000
    visited: set[str] = {source_bibcode}
    parent: dict[str, str] = {}
    frontier = [source_bibcode]

    for _depth in range(max_depth):
        if not frontier:
            break

        # Exclusion list is best-effort: nodes discovered within this BFS level
        # are filtered by the Python guard below, not by the DB query.
        exclusion_list = list(visited)
        row_cap = 2 * max_visited
        sql = """
            SELECT source_bibcode, target_bibcode
            FROM citation_edges
            WHERE source_bibcode = ANY(%s)
              AND target_bibcode != ALL(%s)
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (frontier, exclusion_list, row_cap))
            edges = cur.fetchall()

        next_frontier: list[str] = []
        for src, tgt in edges:
            if tgt not in visited:
                visited.add(tgt)
                parent[tgt] = src
                next_frontier.append(tgt)

                if tgt == target_bibcode:
                    # Reconstruct path
                    path = [tgt]
                    node = tgt
                    while node in parent:
                        node = parent[node]
                        path.append(node)
                    path.reverse()

                    # Fetch paper stubs for the path
                    path_sql = f"""
                        SELECT {STUB_COLUMNS}
                        FROM papers p
                        WHERE p.bibcode = ANY(%s)
                    """
                    with conn.cursor(row_factory=dict_row) as stub_cur:
                        stub_cur.execute(path_sql, (path,))
                        stub_rows = stub_cur.fetchall()

                    # Order stubs to match path order
                    stub_by_bib = {r["bibcode"]: r for r in stub_rows}
                    papers = []
                    for bib in path:
                        if bib in stub_by_bib:
                            papers.append(PaperStub.from_row(stub_by_bib[bib]).to_dict())

                    query_ms = _elapsed_ms(t0)
                    return SearchResult(
                        papers=papers,
                        total=len(papers),
                        timing_ms={"query_ms": query_ms},
                        metadata={"path_length": len(path) - 1, "path_bibcodes": path},
                    )

        frontier = next_frontier

        if len(visited) > max_visited:
            query_ms = _elapsed_ms(t0)
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": query_ms},
                metadata={"path_length": -1, "path_bibcodes": [], "truncated": True},
            )

    query_ms = _elapsed_ms(t0)
    return SearchResult(
        papers=[],
        total=0,
        timing_ms={"query_ms": query_ms},
        metadata={"path_length": -1, "path_bibcodes": []},
    )


ANCHORS_PER_BUCKET = 5
MAX_BUCKET_YEARS = 30
_NO_COMMUNITY_SENTINEL = -1


def _fetch_query_mode_buckets(
    conn: psycopg.Connection,
    query: str,
    yearly_counts: list[dict[str, int]],
    ts_config: str,
    anchors_per_bucket: int = ANCHORS_PER_BUCKET,
) -> list[dict[str, Any]]:
    """For each year, fetch top-K papers matching the query ranked by PageRank.

    Returns per-year bucket dicts with shape:
        {"year": int, "count": int,
         "anchors": list[PaperStub.to_dict() + pagerank],
         "communities": list[{"community_id", "label", "anchor_count"}]}

    `anchor_count` counts community members within this bucket's anchor list
    (capped at ``anchors_per_bucket``), not the community's total paper count.

    Runs a single CTE with ``ROW_NUMBER() OVER (PARTITION BY year)`` rather
    than N per-year queries, so cost is O(1) round-trips regardless of year
    span. Bucket ordering matches the input ``yearly_counts`` ordering.

    The communities list is populated from the semantic partition
    (``community_semantic_medium``) — citation Leiden Phase B has never
    completed, so ``community_id_medium`` holds only the ``-1`` sentinel
    or NULL. Communities with no matching row in the ``communities``
    table are omitted rather than emitted with a null label.
    """
    if not yearly_counts:
        return []

    years = [yc["year"] for yc in yearly_counts]
    count_by_year = {yc["year"]: yc["count"] for yc in yearly_counts}

    anchor_sql = f"""
        WITH ranked AS (
            SELECT {STUB_COLUMNS},
                   pm.pagerank,
                   pm.community_semantic_medium,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.year
                       ORDER BY pm.pagerank DESC NULLS LAST,
                                p.citation_count DESC NULLS LAST
                   ) AS rn
            FROM papers p
            LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode
            WHERE p.tsv @@ plainto_tsquery(%s, %s)
              AND p.year = ANY(%s)
        )
        SELECT * FROM ranked WHERE rn <= %s
    """

    anchors_by_year: dict[int, list[dict[str, Any]]] = {y: [] for y in years}
    community_counts_by_year: dict[int, Counter[int]] = {y: Counter() for y in years}
    all_community_ids: set[int] = set()

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(anchor_sql, [ts_config, query, years, anchors_per_bucket])
        for row in cur.fetchall():
            year = row["year"]
            stub = PaperStub.from_row(row).to_dict()
            stub["pagerank"] = row["pagerank"]
            anchors_by_year[year].append(stub)

            cid = row["community_semantic_medium"]
            if cid is not None and cid != _NO_COMMUNITY_SENTINEL:
                community_counts_by_year[year][cid] += 1
                all_community_ids.add(cid)

        label_map: dict[int, str] = {}
        if all_community_ids:
            cur.execute(
                "SELECT community_id, label FROM communities "
                "WHERE signal = 'semantic' AND resolution = 'medium' "
                "AND community_id = ANY(%s) AND label IS NOT NULL",
                (list(all_community_ids),),
            )
            label_map = {r["community_id"]: r["label"] for r in cur.fetchall()}

    buckets: list[dict[str, Any]] = []
    for year in years:
        communities = [
            {
                "community_id": cid,
                "label": label_map[cid],
                "anchor_count": cnt,
            }
            for cid, cnt in community_counts_by_year[year].most_common()
            if cid in label_map
        ]
        buckets.append(
            {
                "year": year,
                "count": count_by_year[year],
                "anchors": anchors_by_year[year],
                "communities": communities,
            }
        )

    return buckets


def temporal_evolution(
    conn: psycopg.Connection,
    bibcode_or_query: str,
    *,
    year_start: int | None = None,
    year_end: int | None = None,
    ts_config: str = "scix_english",
    anchors_per_bucket: int = ANCHORS_PER_BUCKET,
) -> SearchResult:
    """Show temporal trends: citations received by a paper, or publication volume for a query.

    Mode detection:
    - If bibcode_or_query matches an existing paper bibcode, shows citations per year.
    - Otherwise, treats it as a search query and shows publications per year
      plus per-year "buckets" with anchor papers (ranked by PageRank) and
      dominant communities — so a single call yields a usable temporal
      narrative instead of raw counts.

    Query-mode bucket generation is capped at the ``MAX_BUCKET_YEARS`` most
    recent years to bound work regardless of the matched year span.
    """
    t0 = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM papers WHERE bibcode = %s", (bibcode_or_query,))
        is_bibcode = cur.fetchone() is not None

    year_clause = ""
    year_params: list[Any] = []
    if year_start is not None:
        year_clause += " AND p.year >= %s"
        year_params.append(year_start)
    if year_end is not None:
        year_clause += " AND p.year <= %s"
        year_params.append(year_end)

    if is_bibcode:
        sql = f"""
            SELECT p.year, COUNT(*) AS count
            FROM citation_edges ce
            JOIN papers p ON p.bibcode = ce.source_bibcode
            WHERE ce.target_bibcode = %s
            {year_clause}
            GROUP BY p.year
            ORDER BY p.year
        """
        params: list[Any] = [bibcode_or_query] + year_params
        mode = "citations"
    else:
        sql = f"""
            SELECT p.year, COUNT(*) AS count
            FROM papers p
            WHERE p.tsv @@ plainto_tsquery(%s, %s)
            {year_clause}
            GROUP BY p.year
            ORDER BY p.year
        """
        params = [ts_config, bibcode_or_query] + year_params
        mode = "publications"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)

    yearly_counts = [
        {"year": row["year"], "count": row["count"]} for row in rows if row["year"] is not None
    ]

    metadata: dict[str, Any] = {
        "mode": mode,
        "bibcode_found": is_bibcode,
        "yearly_counts": yearly_counts,
    }

    timing: dict[str, float] = {"query_ms": query_ms}
    if not is_bibcode and yearly_counts:
        t_buckets = time.perf_counter()
        bucket_input = yearly_counts[-MAX_BUCKET_YEARS:]
        metadata["buckets"] = _fetch_query_mode_buckets(
            conn,
            bibcode_or_query,
            bucket_input,
            ts_config,
            anchors_per_bucket=anchors_per_bucket,
        )
        timing["buckets_ms"] = _elapsed_ms(t_buckets)

    return SearchResult(
        papers=[],
        total=len(yearly_counts),
        timing_ms=timing,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Facet counts
# ---------------------------------------------------------------------------


def facet_counts(
    conn: psycopg.Connection,
    facet_field: str,
    *,
    filters: SearchFilters | None = None,
    limit: int = 50,
) -> SearchResult:
    """Get distribution counts for a facet field.

    Supported fields: year, doctype, arxiv_class, database, bibgroup, property.
    """
    t0 = time.perf_counter()

    # Whitelist fields to prevent SQL injection
    allowed_simple = {"year", "doctype"}
    allowed_array = {"arxiv_class", "database", "bibgroup", "property"}

    effective = filters or SearchFilters()
    filter_clause, filter_params = effective.to_where_clause("p")
    entity_clause, entity_params = effective.to_entity_filter_clause("p")

    if facet_field in allowed_simple:
        sql = f"""
            SELECT p.{facet_field}::text AS val, count(*) AS cnt
            FROM papers p
            WHERE p.{facet_field} IS NOT NULL
            {filter_clause}
            {entity_clause}
            GROUP BY p.{facet_field}
            ORDER BY cnt DESC
            LIMIT %s
        """
        params: list[Any] = filter_params + entity_params + [limit]
    elif facet_field in allowed_array:
        sql = f"""
            SELECT elem AS val, count(*) AS cnt
            FROM papers p, unnest(p.{facet_field}) AS elem
            WHERE TRUE
            {filter_clause}
            {entity_clause}
            GROUP BY elem
            ORDER BY cnt DESC
            LIMIT %s
        """
        params = filter_params + entity_params + [limit]
    else:
        raise ValueError(
            f"Unsupported facet field: {facet_field}. "
            f"Allowed: {sorted(allowed_simple | allowed_array)}"
        )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)

    facets = [{"value": row["val"], "count": row["cnt"]} for row in rows]

    return SearchResult(
        papers=[],
        total=len(facets),
        timing_ms={"query_ms": query_ms},
        metadata={"facet_field": facet_field, "facets": facets},
    )


# ---------------------------------------------------------------------------
# Composite lit-review tool
# ---------------------------------------------------------------------------


def lit_review(
    conn: psycopg.Connection,
    query: str,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    top_seeds: int = 20,
    expand_per_seed: int = 20,
    expansion_seeds: int = 5,
    sample_abstracts: int = 5,
    discipline: str | None = None,
    session_state: Any = None,
) -> SearchResult:
    """One-call composite for opening a literature-review session.

    Composes the 4-5-call sequence every research-copilot agent makes:

    1. ``hybrid_search`` for ``top_seeds`` seed papers (year-filtered).
    2. For each of the top ``expansion_seeds`` seeds, expand via
       ``get_references`` + ``get_citations`` (``expand_per_seed`` each
       direction). Year-filter the expansion.
    3. Aggregate seeds + expansion into a working set; populate
       ``_session_state`` so follow-up tool calls can read the set
       without re-listing bibcodes.
    4. Compute community distribution (``community_semantic_medium``
       with labels), top venues, and year distribution over the working
       set in a single SQL round-trip.
    5. Pull full abstracts for ``sample_abstracts`` of the highest-ranked
       seeds so the agent has synthesis material in the same response.
    6. Surface ``citation_contexts`` coverage on the working set so
       downstream calls to ``claim_blame`` / ``find_replications`` know
       whether to expect substantive results.

    Returns a SearchResult whose ``papers`` carry the seed list (with
    full abstracts on the first ``sample_abstracts`` rows) and whose
    ``metadata`` carries the structural characterization. The full
    working set is in ``metadata['working_set_bibcodes']``.

    ``session_state`` is the optional SessionState singleton from
    mcp_server. When provided, every working-set bibcode is added
    via ``add_to_working_set(source_tool='lit_review', ...)`` so
    follow-up tool calls in the same session see the working set
    without re-listing bibcodes. Pure search.py (no MCP) callers can
    omit it.
    """
    t0 = time.perf_counter()

    query = (query or "").strip()
    if not query:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={"error": "query must be a non-empty string"},
        )

    top_seeds = max(1, min(top_seeds, 100))
    expand_per_seed = max(0, min(expand_per_seed, 50))
    expansion_seeds = max(0, min(expansion_seeds, top_seeds))
    sample_abstracts = max(0, min(sample_abstracts, top_seeds))

    # ---- Step 1: seed retrieval --------------------------------------------
    filters = SearchFilters(year_min=year_min, year_max=year_max)
    seed_result = hybrid_search(conn, query, filters=filters, top_n=top_seeds)
    seeds = list(seed_result.papers)
    seed_bibs = [p["bibcode"] for p in seeds if p.get("bibcode")]

    # ---- Step 2: citation expansion ----------------------------------------
    expanded: set[str] = set()
    if expansion_seeds and expand_per_seed:
        for bib in seed_bibs[:expansion_seeds]:
            for fn in (get_references, get_citations):
                try:
                    cit = fn(conn, bib, limit=expand_per_seed)
                except Exception:
                    # Skip missing / error rows; lit_review is best-effort.
                    continue
                for p in cit.papers:
                    b = p.get("bibcode")
                    y = p.get("year")
                    if not b or b in seed_bibs:
                        continue
                    if year_min is not None and (y is None or y < year_min):
                        continue
                    if year_max is not None and (y is None or y > year_max):
                        continue
                    expanded.add(b)

    working_set = list(dict.fromkeys(seed_bibs + sorted(expanded)))

    # ---- Step 3: side-effect — populate session working set ----------------
    if session_state is not None:
        for bib in working_set:
            if not session_state.is_in_working_set(bib):
                session_state.add_to_working_set(
                    bibcode=bib,
                    source_tool="lit_review",
                    source_context=query[:120],
                )

    # ---- Step 4: structural characterization (single round-trip) -----------
    communities: list[dict[str, Any]] = []
    top_venues: list[dict[str, Any]] = []
    year_distribution: list[dict[str, Any]] = []
    if working_set:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pm.community_semantic_medium AS cid,
                       (SELECT label FROM communities c
                        WHERE c.signal = 'semantic'
                          AND c.resolution = 'medium'
                          AND c.community_id = pm.community_semantic_medium) AS label,
                       (SELECT top_keywords FROM communities c
                        WHERE c.signal = 'semantic'
                          AND c.resolution = 'medium'
                          AND c.community_id = pm.community_semantic_medium) AS top_keywords,
                       count(*) AS n
                FROM paper_metrics pm
                WHERE pm.bibcode = ANY(%s)
                  AND pm.community_semantic_medium IS NOT NULL
                GROUP BY pm.community_semantic_medium
                ORDER BY n DESC
                LIMIT 8
                """,
                (working_set,),
            )
            communities = [
                {
                    "community_id": cid,
                    "label": label,
                    "top_keywords": list(kws) if kws else [],
                    "count": n,
                }
                for cid, label, kws, n in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT b AS bibstem, count(*) AS n
                FROM papers, unnest(bibstem) AS b
                WHERE bibcode = ANY(%s)
                GROUP BY b
                ORDER BY n DESC
                LIMIT 8
                """,
                (working_set,),
            )
            top_venues = [{"bibstem": b, "count": n} for b, n in cur.fetchall()]

            cur.execute(
                """
                SELECT year, count(*) AS n
                FROM papers
                WHERE bibcode = ANY(%s) AND year IS NOT NULL
                GROUP BY year
                ORDER BY year
                """,
                (working_set,),
            )
            year_distribution = [{"year": y, "count": n} for y, n in cur.fetchall()]

    # ---- Step 5: full abstracts for the top sample_abstracts seeds ---------
    if sample_abstracts and seed_bibs:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT bibcode, abstract
                FROM papers
                WHERE bibcode = ANY(%s)
                """,
                (seed_bibs[:sample_abstracts],),
            )
            ab_by_bib = {row["bibcode"]: row["abstract"] for row in cur.fetchall()}
        for p in seeds[:sample_abstracts]:
            ab = ab_by_bib.get(p.get("bibcode"))
            if ab:
                p["abstract"] = ab

    # ---- Step 6: citation_contexts coverage --------------------------------
    covered = 0
    coverage_pct = 0.0
    contexts_rows = 0
    if working_set:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM (
                    SELECT DISTINCT b FROM (
                        SELECT source_bibcode AS b FROM citation_contexts WHERE source_bibcode = ANY(%s)
                        UNION
                        SELECT target_bibcode AS b FROM citation_contexts WHERE target_bibcode = ANY(%s)
                    ) t
                ) tt
                """,
                (working_set, working_set),
            )
            covered = cur.fetchone()[0]
            coverage_pct = (covered / len(working_set)) * 100.0
            cur.execute(
                """
                SELECT count(*) FROM citation_contexts
                WHERE source_bibcode = ANY(%s) OR target_bibcode = ANY(%s)
                """,
                (working_set, working_set),
            )
            contexts_rows = cur.fetchone()[0]

    if working_set:
        coverage_note = (
            f"citation_contexts touches {covered}/{len(working_set)} "
            f"working-set papers ({coverage_pct:.1f}%); "
        )
        if coverage_pct < 5.0:
            coverage_note += (
                "claim_blame / find_replications will return mostly empty for this "
                "topic — the substrate covers ~0.27% of edges corpus-wide (bead 79n). "
            )
    else:
        coverage_note = ""
    coverage_note += "See docs/full_text_coverage_analysis.md."

    return SearchResult(
        papers=seeds,
        total=len(seeds),
        timing_ms={
            "query_ms": _elapsed_ms(t0),
            "seed_query_ms": seed_result.timing_ms.get("query_ms", 0.0),
        },
        metadata={
            "query": query,
            "year_min": year_min,
            "year_max": year_max,
            "discipline": discipline,
            "working_set_size": len(working_set),
            "working_set_bibcodes": working_set,
            "communities": communities,
            "top_venues": top_venues,
            "year_distribution": year_distribution,
            "citation_contexts_coverage": {
                "covered_papers": covered if working_set else 0,
                "total_papers": len(working_set),
                "coverage_pct": round(coverage_pct, 2),
                "context_rows": contexts_rows,
                "note": coverage_note,
            },
        },
    )


# ---------------------------------------------------------------------------
# Graph metrics queries
# ---------------------------------------------------------------------------

_VALID_RESOLUTIONS = frozenset({"coarse", "medium", "fine"})

# Community signals supported by explore_community and graph_context.
# - 'citation' reads paper_metrics.community_id_{resolution}
# - 'semantic' reads paper_metrics.community_semantic_{resolution}
# - 'taxonomic' reads paper_metrics.community_taxonomic (single resolution,
#   stored at 'coarse' in the communities table by convention).
_VALID_COMMUNITY_SIGNALS = frozenset({"citation", "semantic", "taxonomic"})

# Per-signal resolution lists mirroring scripts/generate_community_labels.py.
_RESOLUTIONS_BY_SIGNAL: dict[str, tuple[str, ...]] = {
    "citation": ("coarse", "medium", "fine"),
    "semantic": ("coarse", "medium", "fine"),
    "taxonomic": ("coarse",),
}


def _community_column_for_signal(signal: str, resolution: str) -> str:
    """Return the paper_metrics column for a given (signal, resolution).

    Mirrors scripts/generate_community_labels._community_column so that
    labels in `communities` align with ids stored in `paper_metrics`.
    """
    if signal == "citation":
        return f"community_id_{resolution}"
    if signal == "semantic":
        return f"community_semantic_{resolution}"
    if signal == "taxonomic":
        return "community_taxonomic"
    raise ValueError(
        f"invalid community signal: {signal!r}. "
        f"Must be one of {sorted(_VALID_COMMUNITY_SIGNALS)}"
    )


def _taxonomic_community_id(text_label: str) -> int:
    """Map a taxonomic string (e.g. 'astro-ph.GA') to a stable non-negative INT.

    Mirrors scripts/generate_community_labels._taxonomic_id so that a taxonomic
    community_id in `paper_metrics.community_taxonomic` (TEXT) can be joined
    against `communities.community_id` (INT).
    """
    import zlib

    return zlib.adler32(text_label.encode("utf-8")) & 0x7FFFFFFF


def _fetch_communities_for_paper(
    conn: psycopg.Connection, bibcode: str
) -> dict[str, dict[str, Any]]:
    """Return the full per-signal communities block for a paper.

    Shape:
        {
          "citation":  {"coarse": {...}, "medium": {...}, "fine": {...}},
          "semantic":  {"coarse": {...}, "medium": {...}, "fine": {...}},
          "taxonomic": {"coarse": {...}},
        }
    where each inner block is
        {"community_id": int, "label": str|None, "top_keywords": list[str]}
    and resolutions with no community assignment are omitted.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT community_id_coarse, community_id_medium, community_id_fine, "
            "       community_semantic_coarse, community_semantic_medium, "
            "       community_semantic_fine, community_taxonomic "
            "FROM paper_metrics WHERE bibcode = %s",
            (bibcode,),
        )
        pm_row = cur.fetchone()

    result: dict[str, dict[str, Any]] = {}
    if pm_row is None:
        return result

    # Build list of (signal, resolution, community_id) tuples to look up.
    signal_ids: list[tuple[str, str, int]] = []

    for res in ("coarse", "medium", "fine"):
        cid = pm_row.get(f"community_id_{res}")
        if cid is not None:
            signal_ids.append(("citation", res, int(cid)))

    for res in ("coarse", "medium", "fine"):
        cid = pm_row.get(f"community_semantic_{res}")
        if cid is not None:
            signal_ids.append(("semantic", res, int(cid)))

    tax_text = pm_row.get("community_taxonomic")
    if tax_text:
        signal_ids.append(("taxonomic", "coarse", _taxonomic_community_id(tax_text)))

    if not signal_ids:
        return result

    # Fetch labels in a single query keyed by (signal, resolution, community_id).
    placeholders = ",".join(["(%s,%s,%s)"] * len(signal_ids))
    params: list[Any] = []
    for s, r, c in signal_ids:
        params.extend([s, r, c])

    label_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT signal, resolution, community_id, label, top_keywords "
            "FROM communities "
            f"WHERE (signal, resolution, community_id) IN ({placeholders})",
            params,
        )
        for row in cur.fetchall():
            label_by_key[(row["signal"], row["resolution"], row["community_id"])] = {
                "label": row["label"],
                "top_keywords": list(row["top_keywords"] or []),
            }

    for signal, res, cid in signal_ids:
        block = result.setdefault(signal, {})
        entry: dict[str, Any] = {"community_id": cid}
        if signal == "taxonomic":
            # Preserve the original text so callers can display it.
            entry["community_taxonomic"] = tax_text
        label_row = label_by_key.get((signal, res, cid))
        if label_row:
            entry["label"] = label_row["label"]
            entry["top_keywords"] = label_row["top_keywords"]
        block[res] = entry

    return result


def get_paper_metrics(conn: psycopg.Connection, bibcode: str) -> SearchResult:
    """Get precomputed graph metrics for a paper (PageRank, HITS, communities).

    Community labels are surfaced via the per-signal/per-resolution
    ``communities`` block assembled in ``_fetch_communities_for_paper``,
    which is the single source of truth for label-joined community info.
    We previously LEFT JOIN-ed ``communities`` directly against the
    citation columns at the top level, but those columns hold only the
    Phase-A sentinel (-1) or NULL today and would have to also filter by
    ``signal`` to be correct once Phase B lands — duplicating the work
    already done in ``_fetch_communities_for_paper``. The top-level
    metrics row now carries only ``pm.*``.
    """
    t0 = time.perf_counter()

    sql = "SELECT pm.* FROM paper_metrics pm WHERE pm.bibcode = %s"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (bibcode,))
        row = cur.fetchone()

    query_ms = _elapsed_ms(t0)

    if row is None:
        return SearchResult(papers=[], total=0, timing_ms={"query_ms": query_ms})

    # Convert to serializable dict
    metrics = {k: v for k, v in row.items() if k != "updated_at"}
    if row.get("updated_at"):
        metrics["updated_at"] = row["updated_at"].isoformat()

    communities_block = _fetch_communities_for_paper(conn, bibcode)

    return SearchResult(
        papers=[],
        total=1,
        timing_ms={"query_ms": query_ms},
        metadata={"metrics": metrics, "communities": communities_block},
    )


def explore_community(
    conn: psycopg.Connection,
    bibcode: str,
    resolution: str = "coarse",
    limit: int = 20,
    signal: str = "semantic",
) -> SearchResult:
    """Find a paper's community and return sibling papers by PageRank.

    Parameters
    ----------
    signal:
        One of ``citation`` (co-citation-derived Leiden), ``semantic``
        (INDUS-embedding k-means), or ``taxonomic`` (arXiv-class). Selects
        which ``paper_metrics`` column and which ``communities`` rows to
        read. Invalid values raise ``ValueError``.
    resolution:
        One of ``coarse``, ``medium``, ``fine``. For ``signal='taxonomic'``
        only ``coarse`` is populated (the other resolutions are stored as
        aliases of coarse); callers passing ``medium`` or ``fine`` for
        taxonomic will still receive the taxonomic row.
    """
    if signal not in _VALID_COMMUNITY_SIGNALS:
        raise ValueError(
            f"invalid community signal: {signal!r}. "
            f"Must be one of {sorted(_VALID_COMMUNITY_SIGNALS)}"
        )
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"invalid resolution: {resolution!r}. " f"Must be one of {sorted(_VALID_RESOLUTIONS)}"
        )
    # Taxonomic has a single resolution regardless of the requested value.
    lookup_resolution = "coarse" if signal == "taxonomic" else resolution
    t0 = time.perf_counter()

    col = _community_column_for_signal(signal, resolution)

    # Get the paper's community id for this (signal, resolution).
    with conn.cursor() as cur:
        cur.execute(f"SELECT {col} FROM paper_metrics WHERE bibcode = %s", (bibcode,))
        row = cur.fetchone()

    if row is None or row[0] is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={
                "signal": signal,
                "community_id": None,
                "resolution": resolution,
            },
        )

    raw_cid = row[0]
    # paper_metrics.community_taxonomic is TEXT; normalise to the integer
    # used in `communities.community_id`.
    if signal == "taxonomic":
        taxonomic_text = str(raw_cid)
        community_id: int = _taxonomic_community_id(taxonomic_text)
    else:
        taxonomic_text = ""
        community_id = int(raw_cid)

    # Get community metadata filtered by (signal, resolution, community_id).
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT label, paper_count, top_keywords FROM communities "
            "WHERE signal = %s AND resolution = %s AND community_id = %s",
            (signal, lookup_resolution, community_id),
        )
        community_row = cur.fetchone()

    # Get top papers in this community by PageRank. For taxonomic the
    # column is TEXT, so we filter by the original text value; for
    # citation/semantic we filter by the integer id.
    filter_value: object = taxonomic_text if signal == "taxonomic" else community_id
    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM paper_metrics pm
        JOIN papers p ON p.bibcode = pm.bibcode
        WHERE pm.{col} = %s
        ORDER BY pm.pagerank DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (filter_value, limit))
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)
    papers = [PaperStub.from_row(row).to_dict() for row in rows]

    meta: dict[str, Any] = {
        "signal": signal,
        "community_id": community_id,
        "resolution": resolution,
    }
    if signal == "taxonomic":
        meta["community_taxonomic"] = taxonomic_text
    if community_row:
        meta["label"] = community_row["label"]
        meta["paper_count"] = community_row["paper_count"]
        meta["top_keywords"] = list(community_row["top_keywords"] or [])

    return SearchResult(
        papers=papers, total=len(papers), timing_ms={"query_ms": query_ms}, metadata=meta
    )


#: Maximum recursion depth for UAT descendant expansion in concept_search.
#: Bounds the recursive CTE to avoid pathological traversal of deep/cyclic
#: hierarchies. UAT is typically <=6 levels deep, so this covers full expansion.
CONCEPT_DESCENDANT_MAX_DEPTH = 6

#: Vocabularies the router knows about. ``uat`` is virtual — it lives in the
#: legacy ``uat_concepts`` / ``uat_relationships`` tables. Everything else
#: lives in the unified ``concepts`` table (migration 056).
CONCEPT_VOCABULARIES: tuple[str, ...] = (
    "uat",
    # dbl.1 — small open vocabularies
    "openalex",
    "acm_ccs",
    "msc",
    "physh",
    "gcmd",
    # dbl.2 — biomed vocabularies
    "mesh",
    "ncbi_tax",
    "chebi",
    "gene_ontology",
)


def _normalize_vocabulary_arg(
    vocabulary: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Normalize the ``vocabulary`` parameter to a deduplicated tuple.

    ``None`` → all known vocabularies. A single string is wrapped. A list is
    deduplicated while preserving caller-supplied order. Unknown vocabulary
    names are rejected with ``ValueError`` so typos surface immediately
    instead of silently returning empty results.
    """
    if vocabulary is None:
        return CONCEPT_VOCABULARIES
    if isinstance(vocabulary, str):
        names = (vocabulary,)
    else:
        names = tuple(vocabulary)
    if not names:
        return CONCEPT_VOCABULARIES
    seen: dict[str, None] = {}
    for v in names:
        if v not in CONCEPT_VOCABULARIES:
            raise ValueError(f"unknown vocabulary {v!r}; allowed: {sorted(CONCEPT_VOCABULARIES)}")
        seen.setdefault(v, None)
    return tuple(seen)


def _lookup_concepts_unified(
    conn: psycopg.Connection,
    query: str,
    vocabularies: tuple[str, ...],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Find concept candidates across the requested vocabularies.

    Query is matched against ``preferred_label`` (exact / prefix / substring,
    case-insensitive) and ``alternate_labels`` (exact, case-insensitive).
    URI-shaped queries (starting with ``http``) match ``concept_id`` directly
    on UAT and either ``concept_id`` or ``external_uri`` on the unified
    ``concepts`` table.

    Returns a list of hit dicts with ``vocabulary``, ``concept_id``,
    ``preferred_label``, ``alternate_labels``, ``definition``,
    ``external_uri``, and ``score`` ∈ {1.0, 0.7, 0.5, 0.3}, ordered by score
    desc then (vocabulary, concept_id) for deterministic tiebreak.
    """
    is_uri = query.startswith("http")
    q = query.strip()
    if not q:
        return []

    hits: list[dict[str, Any]] = []

    if "uat" in vocabularies:
        if is_uri:
            sql_uat = """
                SELECT concept_id, preferred_label, alternate_labels, definition,
                       1.0::float AS score
                FROM uat_concepts
                WHERE concept_id = %(q)s
            """
        else:
            sql_uat = """
                SELECT concept_id, preferred_label, alternate_labels, definition,
                       CASE
                         WHEN lower(preferred_label) = lower(%(q)s) THEN 1.0
                         WHEN EXISTS (
                           SELECT 1 FROM unnest(alternate_labels) a
                           WHERE lower(a) = lower(%(q)s)
                         ) THEN 0.7
                         WHEN lower(preferred_label) LIKE lower(%(q)s) || '%%' THEN 0.5
                         ELSE 0.3
                       END::float AS score
                FROM uat_concepts
                WHERE lower(preferred_label) = lower(%(q)s)
                   OR lower(preferred_label) LIKE '%%' || lower(%(q)s) || '%%'
                   OR EXISTS (
                       SELECT 1 FROM unnest(alternate_labels) a
                       WHERE lower(a) = lower(%(q)s)
                   )
            """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_uat, {"q": q})
            for row in cur.fetchall():
                hits.append(
                    {
                        "vocabulary": "uat",
                        "concept_id": row["concept_id"],
                        "preferred_label": row["preferred_label"],
                        "alternate_labels": list(row["alternate_labels"] or []),
                        "definition": row["definition"],
                        "external_uri": (
                            row["concept_id"] if row["concept_id"].startswith("http") else None
                        ),
                        "score": float(row["score"]),
                    }
                )

    other_vocabs = tuple(v for v in vocabularies if v != "uat")
    if other_vocabs:
        if is_uri:
            sql_other = """
                SELECT vocabulary, concept_id, preferred_label, alternate_labels,
                       definition, external_uri,
                       1.0::float AS score
                FROM concepts
                WHERE vocabulary = ANY(%(vocabs)s)
                  AND (concept_id = %(q)s OR external_uri = %(q)s)
            """
        else:
            sql_other = """
                SELECT vocabulary, concept_id, preferred_label, alternate_labels,
                       definition, external_uri,
                       CASE
                         WHEN lower(preferred_label) = lower(%(q)s) THEN 1.0
                         WHEN EXISTS (
                           SELECT 1 FROM unnest(alternate_labels) a
                           WHERE lower(a) = lower(%(q)s)
                         ) THEN 0.7
                         WHEN lower(preferred_label) LIKE lower(%(q)s) || '%%' THEN 0.5
                         ELSE 0.3
                       END::float AS score
                FROM concepts
                WHERE vocabulary = ANY(%(vocabs)s)
                  AND (
                    lower(preferred_label) = lower(%(q)s)
                    OR lower(preferred_label) LIKE '%%' || lower(%(q)s) || '%%'
                    OR EXISTS (
                        SELECT 1 FROM unnest(alternate_labels) a
                        WHERE lower(a) = lower(%(q)s)
                    )
                  )
            """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_other, {"q": q, "vocabs": list(other_vocabs)})
            for row in cur.fetchall():
                hits.append(
                    {
                        "vocabulary": row["vocabulary"],
                        "concept_id": row["concept_id"],
                        "preferred_label": row["preferred_label"],
                        "alternate_labels": list(row["alternate_labels"] or []),
                        "definition": row["definition"],
                        "external_uri": row["external_uri"],
                        "score": float(row["score"]),
                    }
                )

    # Tiebreak: at equal score, prefer UAT (sort key 0 vs 1) so that legacy
    # callers querying a label that exists in both UAT and a new vocabulary
    # (e.g. "Galaxies" -> UAT + PhySH at 1.0) still get UAT-driven paper
    # retrieval. UAT is currently the only vocabulary with paper mappings.
    hits.sort(
        key=lambda h: (
            -h["score"],
            0 if h["vocabulary"] == "uat" else 1,
            h["vocabulary"],
            h["concept_id"],
        )
    )
    return hits[:limit]


def _fetch_uat_papers(
    conn: psycopg.Connection,
    concept_id: str,
    *,
    include_descendants: bool,
    limit: int,
) -> list[dict[str, Any]]:
    """Return papers tagged with a UAT concept (optionally with descendants).

    Recursive CTE bounded at ``CONCEPT_DESCENDANT_MAX_DEPTH`` for the
    expansion path; depth cap lives on the recursive step so that a
    descendant at depth N is produced iff N <= max_depth.
    """
    if include_descendants:
        sql = f"""
            WITH RECURSIVE descendants AS (
                SELECT child_id AS concept_id, 1 AS depth
                FROM uat_relationships
                WHERE parent_id = %(root)s
                UNION ALL
                SELECT r.child_id, d.depth + 1
                FROM uat_relationships r
                JOIN descendants d ON r.parent_id = d.concept_id
                WHERE d.depth < %(max_depth)s
            ),
            all_concepts AS (
                SELECT %(root)s AS concept_id
                UNION
                SELECT concept_id FROM descendants WHERE depth <= %(max_depth)s
            )
            SELECT DISTINCT {STUB_COLUMNS}
            FROM all_concepts ac
            JOIN paper_uat_mappings m ON m.concept_id = ac.concept_id
            JOIN papers p ON p.bibcode = m.bibcode
            ORDER BY p.citation_count DESC NULLS LAST
            LIMIT %(limit)s
        """
        params: dict[str, Any] = {
            "root": concept_id,
            "max_depth": CONCEPT_DESCENDANT_MAX_DEPTH,
            "limit": limit,
        }
    else:
        sql = f"""
            SELECT DISTINCT {STUB_COLUMNS}
            FROM paper_uat_mappings m
            JOIN papers p ON p.bibcode = m.bibcode
            WHERE m.concept_id = %(root)s
            ORDER BY p.citation_count DESC NULLS LAST
            LIMIT %(limit)s
        """
        params = {"root": concept_id, "limit": limit}

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [PaperStub.from_row(row).to_dict() for row in rows]


def _concept_search_lexical_fallback(
    conn: psycopg.Connection,
    *,
    query: str,
    vocabs: tuple[str, ...],
    limit: int,
    t0: float,
) -> SearchResult:
    """Fallback path used when the vocabulary router returns zero hits.

    Runs a plain :func:`lexical_search` (BM25 over title+abstract tsvector)
    so a natural-language phrase like ``"granular mechanics in microgravity"``
    still returns useful papers from the corpus. We use lexical search rather
    than full hybrid here because (a) the embedding pipeline isn't part of
    this module's dependency surface (lives in mcp_server), (b) lexical
    requires no external state, and (c) it's the right tool for queries
    that have already failed exact label matching — they tend to be free-form
    multi-word phrases where BM25 is competitive.

    Returns the result with ``metadata['fallback']='lexical_search'`` so
    callers can see the path taken; ``concept_found`` stays False because
    no controlled-vocabulary concept resolved.
    """
    fallback_result = lexical_search(conn, query, limit=limit)
    return SearchResult(
        papers=fallback_result.papers,
        total=fallback_result.total,
        timing_ms={
            "query_ms": _elapsed_ms(t0),
            **fallback_result.timing_ms,
        },
        metadata={
            "concept_found": False,
            "query": query,
            "vocabularies_searched": list(vocabs),
            "concepts": [],
            "fallback": "lexical_search",
        },
    )


def concept_search(
    conn: psycopg.Connection,
    query: str,
    *,
    vocabulary: str | list[str] | tuple[str, ...] | None = None,
    include_descendants: bool = True,
    include_subtopics: bool | None = None,
    limit: int = 20,
    fallback: bool = True,
) -> SearchResult:
    """Search for concepts across one or more controlled vocabularies.

    Searches the listed vocabularies (default: all of
    :data:`CONCEPT_VOCABULARIES`) for concepts whose preferred label, alternate
    labels, or URI match ``query`` and returns ranked candidates tagged with
    their source vocabulary in ``metadata['concepts']``.

    For UAT — the only vocabulary with paper mappings today — the function
    additionally returns papers tagged with the best-matching UAT concept (or
    its descendants when ``include_descendants=True``) in ``papers``. This
    preserves backwards compatibility with callers that read ``papers``
    directly. Other vocabularies have no paper-side mappings yet, so
    ``papers`` is empty when the best hit is non-UAT.

    Free-text fallback (scix_experiments-2ixv): when the vocabulary lookup
    yields zero hits AND ``fallback=True`` (the default), the function falls
    through to :func:`lexical_search` so a natural-language query like
    ``"granular mechanics in microgravity"`` returns plausible papers
    instead of an empty list. The result carries ``metadata['fallback'] =
    'lexical_search'`` so callers can see the path taken. Set
    ``fallback=False`` to preserve strict legacy behavior (empty papers
    when no concept resolves).

    Parameters
    ----------
    vocabulary
        Restrict search to a single vocabulary or a list of vocabularies.
        ``None`` (default) searches all of :data:`CONCEPT_VOCABULARIES`.
    include_descendants
        When the best hit is a UAT concept, expand to descendants via the
        ``uat_relationships`` recursive CTE bounded at
        :data:`CONCEPT_DESCENDANT_MAX_DEPTH`.
    include_subtopics
        Legacy alias for ``include_descendants``. When provided (non-None),
        overrides ``include_descendants`` for backward compatibility.
    limit
        Maximum number of papers AND maximum number of concept hits to
        return. Concept hits are ranked by score; ties broken by
        ``(vocabulary, concept_id)``.
    fallback
        When True (default), free-text queries that resolve to no
        vocabulary concept fall through to :func:`lexical_search` so the
        tool returns useful papers instead of an empty list. Set False
        for strict vocabulary-only behavior.

    Returns
    -------
    SearchResult
        ``papers`` is populated when the best vocabulary hit is a UAT
        concept that has paper mappings (or descendants do), OR when the
        vocabulary lookup misses and the lexical fallback returns hits.
        ``metadata['concepts']`` is the full ranked list of concept
        candidates with vocabulary tags. ``metadata['concept_found']`` is
        True iff at least one concept matched. ``metadata['fallback']`` is
        ``'lexical_search'`` iff the fallback path ran (regardless of
        whether it returned papers). Legacy keys ``concept_id``,
        ``concept_label``, ``include_descendants``, and
        ``include_subtopics`` are populated from the best hit for
        backwards compatibility.
    """
    if include_subtopics is not None:
        include_descendants = include_subtopics

    t0 = time.perf_counter()
    vocabs = _normalize_vocabulary_arg(vocabulary)
    hits = _lookup_concepts_unified(conn, query, vocabs, limit=limit)

    if not hits:
        # Empty / whitespace-only queries short-circuit before fallback so we
        # don't run a tsvector match on the empty string.
        if fallback and query.strip():
            return _concept_search_lexical_fallback(
                conn,
                query=query,
                vocabs=vocabs,
                limit=limit,
                t0=t0,
            )
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={
                "concept_found": False,
                "query": query,
                "vocabularies_searched": list(vocabs),
                "concepts": [],
            },
        )

    best = hits[0]
    papers: list[dict[str, Any]] = []
    if best["vocabulary"] == "uat":
        papers = _fetch_uat_papers(
            conn,
            best["concept_id"],
            include_descendants=include_descendants,
            limit=limit,
        )

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"query_ms": _elapsed_ms(t0)},
        metadata={
            "concept_found": True,
            "query": query,
            "vocabularies_searched": list(vocabs),
            "concepts": hits,
            "concept_id": best["concept_id"],
            "concept_label": best["preferred_label"],
            "concept_vocabulary": best["vocabulary"],
            "include_descendants": include_descendants,
            # Legacy alias kept for callers that read it from metadata.
            "include_subtopics": include_descendants,
        },
    )


# ---------------------------------------------------------------------------
# Agent context queries (materialized views)
# ---------------------------------------------------------------------------


def get_document_context(
    conn: psycopg.Connection,
    bibcode: str,
) -> SearchResult:
    """Get full document context for a paper from the agent_document_context matview.

    Returns paper metadata (bibcode, title, abstract, year, citation_count,
    reference_count) plus all linked entities as a JSONB array. Replaces the
    separate get_paper + entity_search workflow with a single query against
    the precomputed materialized view.

    Returns an empty SearchResult with an error in metadata when the bibcode
    is not found, rather than raising an exception.
    """
    t0 = time.perf_counter()

    sql = """
        SELECT bibcode, title, abstract, year, citation_count, reference_count, linked_entities
        FROM agent_document_context
        WHERE bibcode = %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (bibcode,))
        row = cur.fetchone()

    query_ms = _elapsed_ms(t0)

    if row is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": query_ms},
            metadata={"error": f"Paper not found in document context: {bibcode}"},
        )

    return SearchResult(
        papers=[dict(row)],
        total=1,
        timing_ms={"query_ms": query_ms},
    )


def get_entity_context(
    conn: psycopg.Connection,
    entity_id: int,
) -> SearchResult:
    """Get full entity context for a single entity.

    Returns an entity card: entity_id, canonical_name, entity_type, discipline,
    source, identifiers, aliases, properties (raw JSONB from
    ``public.entities``), relationships (each enriched with the object
    entity's name, type, and properties), and citing_paper_count.

    The ``agent_entity_context`` matview supplies the aggregates
    (identifiers, aliases, citing_paper_count). ``properties`` and the
    relationship enrichment are read directly from the source tables so
    agents get the full structural context without waiting for a matview
    schema change (xz4.5 tracks that). Relationships join through
    ``entity_relationships`` and surface ``(predicate, object_entity,
    properties_of_object)`` for one-hop navigation.

    Returns an empty SearchResult with an error in metadata when the
    entity_id is not found, rather than raising an exception.
    """
    t0 = time.perf_counter()

    matview_sql = """
        SELECT entity_id, canonical_name, entity_type, discipline, source,
               identifiers, aliases, citing_paper_count
        FROM agent_entity_context
        WHERE entity_id = %s
    """

    properties_sql = "SELECT properties FROM entities WHERE id = %s"

    # Surface both out-edges (queried entity is SUBJECT) and in-edges
    # (queried entity is OBJECT). Without the in-edge half, hub
    # entities like JWST appear bare even though edges like NIRSpec
    # part_of JWST exist (scix_experiments-1fi). The `direction` column
    # tells consumers which way the predicate points; in both directions
    # `object_*` always refers to the OTHER end of the edge from the
    # queried entity's perspective.
    relationships_sql = """
        SELECT 'out' AS direction,
               er.predicate,
               er.object_entity_id AS object_id,
               er.confidence,
               er.source AS relationship_source,
               obj.canonical_name AS object_name,
               obj.entity_type AS object_entity_type,
               obj.properties AS object_properties
        FROM entity_relationships er
        LEFT JOIN entities obj ON obj.id = er.object_entity_id
        WHERE er.subject_entity_id = %s
        UNION ALL
        SELECT 'in' AS direction,
               er.predicate,
               er.subject_entity_id AS object_id,
               er.confidence,
               er.source AS relationship_source,
               subj.canonical_name AS object_name,
               subj.entity_type AS object_entity_type,
               subj.properties AS object_properties
        FROM entity_relationships er
        LEFT JOIN entities subj ON subj.id = er.subject_entity_id
        WHERE er.object_entity_id = %s
        ORDER BY direction, predicate, object_name
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(matview_sql, (entity_id,))
        matview_row = cur.fetchone()

        if matview_row is None:
            # Fall back to entities when the matview has not refreshed yet.
            # Without this, newly-inserted entities return "not found"
            # even though they exist in the canonical table.
            cur.execute(
                """
                SELECT id AS entity_id, canonical_name, entity_type,
                       discipline, source, properties
                FROM entities
                WHERE id = %s
                """,
                (entity_id,),
            )
            fallback = cur.fetchone()
            if fallback is None:
                query_ms = _elapsed_ms(t0)
                return SearchResult(
                    papers=[],
                    total=0,
                    timing_ms={"query_ms": query_ms},
                    metadata={"error": f"Entity not found: {entity_id}"},
                )
            result = dict(fallback)
            result.setdefault("identifiers", [])
            result.setdefault("aliases", [])
            result.setdefault("citing_paper_count", 0)
        else:
            result = dict(matview_row)
            # Convert aliases from SQL array to list for JSON serialization
            if result.get("aliases") is not None:
                result["aliases"] = list(result["aliases"])
            cur.execute(properties_sql, (entity_id,))
            props_row = cur.fetchone()
            result["properties"] = (
                props_row["properties"] if props_row and props_row["properties"] is not None else {}
            )

        cur.execute(relationships_sql, (entity_id, entity_id))
        rel_rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)

    result["relationships"] = [
        {
            "direction": rel["direction"],
            "predicate": rel["predicate"],
            "object_id": rel["object_id"],
            "object_name": rel["object_name"],
            "object_entity_type": rel["object_entity_type"],
            "object_properties": rel["object_properties"] or {},
            "confidence": rel["confidence"],
            "source": rel["relationship_source"],
        }
        for rel in rel_rows
    ]

    return SearchResult(
        papers=[result],
        total=1,
        timing_ms={"query_ms": query_ms},
    )


# ---------------------------------------------------------------------------
# Citation context queries
# ---------------------------------------------------------------------------


def get_citation_context(
    conn: psycopg.Connection,
    source_bibcode: str,
    target_bibcode: str,
) -> SearchResult:
    """Get citation context(s) for a source-target bibcode pair.

    Returns the context text, intent label, and character offset for each
    in-text citation mention where source cites target. Returns an empty
    result (not an error) when no context exists for the pair.
    """
    t0 = time.perf_counter()

    sql = """
        SELECT context_text, intent, char_offset, section_name
        FROM citation_contexts
        WHERE source_bibcode = %s AND target_bibcode = %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (source_bibcode, target_bibcode))
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)

    contexts = [
        {
            "context_text": row["context_text"],
            "intent": row["intent"],
            "char_offset": row["char_offset"],
            "section_name": row["section_name"],
        }
        for row in rows
    ]

    return SearchResult(
        papers=contexts,
        total=len(contexts),
        timing_ms={"query_ms": query_ms},
    )


# ---------------------------------------------------------------------------
# ADR-006 guard: detect LaTeX-derived provenance for papers.body
# ---------------------------------------------------------------------------


def _check_body_latex_provenance(
    conn: psycopg.Connection,
    bibcode: str,
) -> str | None:
    """Check if a bibcode has LaTeX-derived fulltext in papers_fulltext.

    If papers_fulltext has a row for this bibcode with a source in
    LATEX_DERIVED_SOURCES (ar5iv, arxiv_local), returns the source tag.
    Otherwise returns None.

    This is a defense-in-depth guard for read_paper_section and
    search_within_paper, which read from papers.body. Currently
    papers.body is ADS-only, but if LaTeX-derived text were ever
    promoted to papers.body, this guard ensures ADR-006 snippet budget
    enforcement is applied.

    If the papers_fulltext table is absent (migration 041 not yet
    applied), the guard short-circuits to None — there is no provenance
    record to inspect, so by definition no body can be flagged as
    LaTeX-derived. This keeps read_paper / search_within_paper usable
    against papers.body alone.
    """
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT source FROM papers_fulltext WHERE bibcode = %s",
                (bibcode,),
            )
            row = cur.fetchone()
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return None

    if row is None:
        return None

    source = row["source"]
    if source in LATEX_DERIVED_SOURCES:
        return source
    return None


# ---------------------------------------------------------------------------
# Paper section reading (body + section_parser)
# ---------------------------------------------------------------------------


def _read_section_from_papers_fulltext(
    conn: psycopg.Connection,
    *,
    bibcode: str,
    section: str,
    role: str | None,
    char_offset: int,
    limit: int,
    title: str,
    classify_fn,
    elapsed_ms: float = 0.0,
) -> SearchResult | None:
    """Read a section from the structured papers_fulltext.sections JSONB.

    Returns a SearchResult if a matching section is found, None if the
    paper has no papers_fulltext entry or no section matches. The caller
    falls back to body-parsing on None.

    Heading matching is case-insensitive substring (e.g. 'methods' matches
    'METHODS', 'Materials and Methods', '2 Methods'). Role matching uses
    ``classify_fn(heading)`` to pick the first heading whose canonical
    role matches.

    Schema reminder: papers_fulltext.sections is jsonb NOT NULL — array of
    {text: str, heading: str, level: int, offset: int}.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT sections FROM papers_fulltext WHERE bibcode = %s",
            (bibcode,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    sections_json = row.get("sections") or []
    if not sections_json:
        return None

    # Normalize each section into (heading, text, level, offset)
    parsed = [
        (
            (s.get("heading") or "").strip(),
            (s.get("text") or ""),
            int(s.get("level") or 0),
            int(s.get("offset") or 0),
        )
        for s in sections_json
        if isinstance(s, dict)
    ]
    if not parsed:
        return None

    section_lower = section.lower().strip()
    matched_heading = None
    matched_text = None

    if role is not None:
        role_lower = role.lower()
        for h, t, _lvl, _off in parsed:
            if classify_fn(h) == role_lower:
                matched_heading, matched_text = h, t
                break
        if matched_heading is None:
            available = [h for h, _t, _l, _o in parsed]
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": elapsed_ms},
                metadata={
                    "error": f"No section with role '{role}' found",
                    "available_sections": available,
                    "has_body": True,
                    "source": "papers_fulltext.sections",
                },
            )
    else:
        # Substring match (case-insensitive). Multiple candidates: prefer
        # the heading that starts with the query (less generic match), then
        # any substring match. e.g. section='methods' prefers 'Methods'
        # over 'Methodology and Approach'.
        starts_with: list[tuple[str, str]] = []
        contains: list[tuple[str, str]] = []
        for h, t, _lvl, _off in parsed:
            hl = h.lower()
            if hl.startswith(section_lower):
                starts_with.append((h, t))
            elif section_lower in hl:
                contains.append((h, t))
        candidates = starts_with or contains
        if candidates:
            matched_heading, matched_text = candidates[0]
        else:
            available = [h for h, _t, _l, _o in parsed]
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": elapsed_ms},
                metadata={
                    "error": f"Section '{section}' not found",
                    "available_sections": available,
                    "has_body": True,
                    "source": "papers_fulltext.sections",
                },
            )

    total_chars = len(matched_text)
    section_text = matched_text[char_offset : char_offset + limit]

    return SearchResult(
        papers=[
            {
                "bibcode": bibcode,
                "title": title,
                "section_name": matched_heading,
                "section_text": section_text,
                "has_body": True,
                "char_offset": char_offset,
                "total_chars": total_chars,
            }
        ],
        total=1,
        timing_ms={"query_ms": elapsed_ms},
        metadata={
            "has_body": True,
            "available_sections": [h for h, _t, _l, _o in parsed],
            "source": "papers_fulltext.sections",
        },
    )


def read_paper_section(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    section: str = "full",
    char_offset: int = 0,
    limit: int = 5000,
    role: str | None = None,
) -> SearchResult:
    """Read a section of a paper's full-text body with pagination.

    Parameters
    ----------
    conn : psycopg.Connection
    bibcode : str
        ADS bibcode.
    section : str
        Section name (e.g. 'introduction', 'methods', 'full') or 'full' for
        the entire body. Defaults to 'full'.
    char_offset : int
        Character offset for pagination within the section text.
    limit : int
        Maximum characters to return (default 5000).
    role : str, optional
        If provided, select the first parsed section whose name maps to
        this canonical role (one of ``background``, ``method``, ``result``,
        ``conclusion``, ``other``). Takes precedence over ``section`` when
        a matching section is found. Falls back to ``section`` selection
        when no parsed section matches the requested role.

    Returns
    -------
    SearchResult
        papers list contains a single dict with section_text, section_name,
        has_body, char_offset, total_chars, and bibcode.
    """
    from scix.section_parser import parse_sections
    from scix.section_role import classify_section_role

    t0 = time.perf_counter()

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT body, abstract, title FROM papers WHERE bibcode = %s",
            (bibcode,),
        )
        row = cur.fetchone()

    query_ms = _elapsed_ms(t0)

    if row is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": query_ms},
            metadata={"error": f"Paper not found: {bibcode}"},
        )

    body = row.get("body")
    abstract = row.get("abstract") or ""
    title = row.get("title") or ""
    has_body = body is not None and len(body) > 0

    # When a specific section or role is requested, try the structured
    # papers_fulltext.sections JSONB first (14.4M papers populated, ~96.2%
    # of full-text rows). The structured sections come from GROBID/parser
    # output and are far more accurate than regex heuristics on flat body
    # text. Fall through to the body-parsing path only if the paper has no
    # papers_fulltext entry or no matching section.
    if section != "full" or role is not None:
        structured = _read_section_from_papers_fulltext(
            conn,
            bibcode=bibcode,
            section=section,
            role=role,
            char_offset=char_offset,
            limit=limit,
            title=title,
            classify_fn=classify_section_role,
            elapsed_ms=_elapsed_ms(t0),
        )
        if structured is not None:
            return structured

    if not has_body:
        # Fallback to abstract
        section_text = abstract[char_offset : char_offset + limit]
        return SearchResult(
            papers=[
                {
                    "bibcode": bibcode,
                    "title": title,
                    "section_name": "abstract",
                    "section_text": section_text,
                    "has_body": False,
                    "char_offset": char_offset,
                    "total_chars": len(abstract),
                }
            ],
            total=1,
            timing_ms={"query_ms": query_ms},
            metadata={"has_body": False},
        )

    # Parse sections from body
    sections = parse_sections(body)
    section_lower = section.lower()

    # Role-based selection takes precedence when supplied — pick the first
    # parsed section whose name maps to the requested role.
    role_matched: tuple[str, int, int, str] | None = None
    if role is not None:
        role_lower = role.lower()
        for s in sections:
            if classify_section_role(s[0]) == role_lower:
                role_matched = s
                break
        if role_matched is None:
            available = [s[0] for s in sections]
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": query_ms},
                metadata={
                    "error": f"No section with role '{role}' found",
                    "available_sections": available,
                    "has_body": True,
                },
            )

    if role_matched is not None:
        section_name = role_matched[0]
        text = role_matched[3]
    elif section_lower == "full":
        text = body
        section_name = "full"
    else:
        # Find the requested section
        matched = [s for s in sections if s[0] == section_lower]
        if matched:
            section_name = matched[0][0]
            text = matched[0][3]
        else:
            # Section not found — list available sections
            available = [s[0] for s in sections]
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": query_ms},
                metadata={
                    "error": f"Section '{section}' not found",
                    "available_sections": available,
                    "has_body": True,
                },
            )

    total_chars = len(text)
    section_text = text[char_offset : char_offset + limit]

    # ADR-006 guard: check if this bibcode's body is LaTeX-derived.
    # Currently papers.body is ADS-only, but if ar5iv text were ever
    # promoted to papers.body, this guard prevents ADR-006 bypass.
    latex_source = _check_body_latex_provenance(conn, bibcode)
    if latex_source is not None:
        arxiv_id = _get_arxiv_id_for_bibcode(conn, bibcode)
        budget_result = apply_snippet_budget_if_needed(
            body_text=section_text,
            source=latex_source,
            bibcode=bibcode,
            arxiv_id=arxiv_id,
        )
        return SearchResult(
            papers=[
                {
                    "bibcode": bibcode,
                    "title": title,
                    "section_name": section_name,
                    "section_text": budget_result["snippet"],
                    "has_body": True,
                    "char_offset": char_offset,
                    "total_chars": total_chars,
                    "canonical_url": budget_result["canonical_url"],
                }
            ],
            total=1,
            timing_ms={"query_ms": query_ms},
            metadata={
                "has_body": True,
                "available_sections": [s[0] for s in sections],
                "adr006_guarded": True,
                "source": latex_source,
            },
        )

    return SearchResult(
        papers=[
            {
                "bibcode": bibcode,
                "title": title,
                "section_name": section_name,
                "section_text": section_text,
                "has_body": True,
                "char_offset": char_offset,
                "total_chars": total_chars,
            }
        ],
        total=1,
        timing_ms={"query_ms": query_ms},
        metadata={
            "has_body": True,
            "available_sections": [s[0] for s in sections],
        },
    )


# ---------------------------------------------------------------------------
# Search within paper (section-level rerank — M5 of prd_full_text_applications_v2)
# ---------------------------------------------------------------------------

# Map env-var values to model_name strings consumed by CrossEncoderReranker.
# Duplicated from mcp_server._RERANK_MODEL_ALIASES to avoid an upward import
# (search.py is below mcp_server.py in the layering). Keep in sync.
_SEARCH_RERANK_MODEL_ALIASES: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-large": "BAAI/bge-reranker-large",
}


def _resolve_section_rerank_model() -> str | None:
    """Return the configured cross-encoder model name, or ``None`` when off.

    Reads ``SCIX_RERANK_DEFAULT_MODEL`` (default ``'off'``). Unknown values
    fall back to ``None`` with a warning, matching mcp_server's behaviour.
    """
    raw = os.environ.get("SCIX_RERANK_DEFAULT_MODEL", "off").strip().lower()
    if raw == "off":
        return None
    if raw in _SEARCH_RERANK_MODEL_ALIASES:
        return _SEARCH_RERANK_MODEL_ALIASES[raw]
    logger.warning(
        "Unknown SCIX_RERANK_DEFAULT_MODEL=%r in search_within_paper; "
        "falling back to 'off'. Allowed values: 'off', 'minilm', 'bge-large'.",
        raw,
    )
    return None


# Cached reranker per-process to amortise the lazy weight load.
_section_rerank_cache: dict[str, Any] = {}


def _get_section_reranker() -> CrossEncoderReranker | None:
    """Return a CrossEncoderReranker honoring SCIX_RERANK_DEFAULT_MODEL.

    Returns None when the env var is 'off' (the default), which is the M4-FAIL
    ship-with-default-off behaviour. Cached so repeated calls within a process
    reuse the same lazy-loaded model.
    """
    model_name = _resolve_section_rerank_model()
    if model_name is None:
        return None
    cached = _section_rerank_cache.get(model_name)
    if cached is not None:
        return cached
    reranker = CrossEncoderReranker(model_name=model_name)
    _section_rerank_cache[model_name] = reranker
    return reranker


def _reset_section_rerank_cache() -> None:
    """Test hook: drop the cached singleton so env changes take effect."""
    _section_rerank_cache.clear()


def _section_snippet(section_text: str, query: str, window: int = 150) -> str:
    """Build a snippet around the first query-token match in a section.

    Mechanical context-window extraction — no semantic judgement. Falls back
    to the first ``2 * window`` characters when no token matches.
    """
    if not section_text:
        return ""
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 2]
    if not tokens:
        return section_text[: 2 * window].strip()
    lower = section_text.lower()
    best_pos: int | None = None
    for tok in tokens:
        pos = lower.find(tok)
        if pos == -1:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
    if best_pos is None:
        return section_text[: 2 * window].strip()
    start = max(0, best_pos - window)
    end = min(len(section_text), best_pos + window)
    snippet = section_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(section_text):
        snippet = snippet + "..."
    return snippet


def _python_section_score(section_text: str, query: str) -> float:
    """Deterministic Python fallback when PostgreSQL ts_rank is unavailable.

    Counts query-token occurrences in the section (case-insensitive, word
    boundaries) and normalises by the section length. This is a transparent
    mechanical proxy — no semantic judgement — used as a last-resort fallback
    when the ts_rank SQL call fails (e.g. inside unit-test mocks).
    """
    if not section_text:
        return 0.0
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 2]
    if not tokens:
        return 0.0
    lower = section_text.lower()
    matches = 0
    for tok in tokens:
        matches += len(re.findall(rf"\b{re.escape(tok)}\b", lower))
    if matches == 0:
        return 0.0
    # Length-normalised count; the +50 guards against tiny-section blow-up.
    return matches / (len(section_text) + 50)


def _score_sections_ts_rank(
    conn: psycopg.Connection,
    section_texts: list[str],
    query: str,
) -> list[float]:
    """Score each section against the query using PostgreSQL ts_rank.

    Issues a single batched query via ``unnest`` so we get one round-trip
    regardless of section count. Returns scores in input order. Falls back to
    a deterministic Python proxy when the SQL call raises (keeps the function
    usable inside unit-test mocks that don't simulate full cursor behaviour).
    """
    if not section_texts:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ts_rank(to_tsvector('english', t), plainto_tsquery('english', %s)) AS score
                FROM unnest(%s::text[]) WITH ORDINALITY AS u(t, ord)
                ORDER BY ord
                """,
                (query, list(section_texts)),
            )
            rows = list(cur.fetchall())
    except Exception:
        logger.debug(
            "ts_rank section scoring failed; falling back to Python proxy",
            exc_info=True,
        )
        return [_python_section_score(t, query) for t in section_texts]

    if len(rows) != len(section_texts):
        # Mock cursors often return an empty/MagicMock fetchall; degrade
        # gracefully rather than emitting zeros that hide real-world bugs.
        return [_python_section_score(t, query) for t in section_texts]

    scores: list[float] = []
    for row in rows:
        # Cursor may be tuple-returning or dict-returning depending on caller.
        if isinstance(row, dict):
            scores.append(float(row.get("score") or 0.0))
        else:
            scores.append(float(row[0] or 0.0))
    return scores


def search_within_paper(
    conn: psycopg.Connection,
    bibcode: str,
    query: str,
    *,
    top_k: int = 20,
    use_rerank: bool = True,
) -> SearchResult:
    """Search within a paper's body text with section-level ranking.

    The body is split into sections via :func:`scix.section_parser.parse_sections`,
    each section is scored against the query via PostgreSQL ``ts_rank``, the
    top-``top_k`` candidates are optionally re-scored by a cross-encoder
    (``CrossEncoderReranker``) and the best three are returned in a new
    ``sections`` field on the result paper. The legacy ``ts_headline`` blob
    is still surfaced as ``headline`` (top-1 section snippet for backward
    compatibility with callers that read only that field).

    Parameters
    ----------
    conn : psycopg.Connection
    bibcode : str
        ADS bibcode.
    query : str
        Search terms to find within the paper body.
    top_k : int, optional
        Number of BM25 candidate sections to keep before reranking. Default 20.
    use_rerank : bool, optional
        If True (default), route candidates through the cross-encoder reranker
        when ``SCIX_RERANK_DEFAULT_MODEL`` resolves to a model. With the
        default env value ``'off'`` no model is constructed and the candidates
        retain their ts_rank ordering — mirrors the ship-with-default-off
        gate established in the prior cross-encoder-reranker build.

    Returns
    -------
    SearchResult
        ``papers`` list contains a single dict with ``bibcode``, ``title``,
        ``headline`` (top-1 section snippet for backward compat), ``has_body``,
        and ``sections`` (up to 3 entries of
        ``{"section_name", "score", "snippet"}``).
    """
    from scix.section_parser import parse_sections

    t0 = time.perf_counter()

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT bibcode, title, body,
                   ts_headline(
                       'english',
                       body,
                       plainto_tsquery('english', %s),
                       'MaxWords=60, MinWords=20, MaxFragments=5'
                   ) AS headline
            FROM papers
            WHERE bibcode = %s AND body IS NOT NULL AND body != ''
            """,
            (query, bibcode),
        )
        row = cur.fetchone()

    query_ms = _elapsed_ms(t0)

    if row is None:
        # Check if paper exists but has no body
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT bibcode, title FROM papers WHERE bibcode = %s",
                (bibcode,),
            )
            paper_row = cur.fetchone()

        if paper_row is None:
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": query_ms},
                metadata={"error": f"Paper not found: {bibcode}"},
            )
        else:
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": query_ms},
                metadata={
                    "error": "Paper has no body text for searching",
                    "has_body": False,
                    "bibcode": bibcode,
                },
            )

    body = row.get("body") or ""
    headline = row.get("headline") or ""

    # Section-level candidate retrieval (M5).
    parsed = parse_sections(body)
    # parse_sections always returns at least one tuple. Drop empty-text entries
    # so we don't waste a rerank slot on a section that'll score zero.
    candidate_sections: list[tuple[str, str]] = [
        (name, text) for (name, _start, _end, text) in parsed if (text or "").strip()
    ]

    if candidate_sections:
        scores = _score_sections_ts_rank(
            conn, [text for _name, text in candidate_sections], query
        )
        scored = [
            (name, text, score)
            for (name, text), score in zip(candidate_sections, scores)
        ]
    else:
        scored = []

    scored.sort(key=lambda x: x[2], reverse=True)
    candidates = scored[: max(1, top_k)]

    # Optional cross-encoder rerank.
    rerank_used = False
    if use_rerank and len(candidates) > 1:
        reranker = _get_section_reranker()
        if reranker is not None:
            paper_dicts = [
                {
                    "section_name": name,
                    "title": name,
                    "abstract_snippet": text,
                    "_section_text": text,
                    "_ts_rank": score,
                }
                for (name, text, score) in candidates
            ]
            reranked = reranker(query, paper_dicts, top_n=3)
            top_sections = [
                (
                    p["section_name"],
                    p["_section_text"],
                    float(p.get("rerank_score", p.get("_ts_rank", 0.0))),
                )
                for p in reranked
            ]
            rerank_used = True
        else:
            top_sections = [(n, t, s) for (n, t, s) in candidates[:3]]
    else:
        top_sections = [(n, t, s) for (n, t, s) in candidates[:3]]

    sections_payload: list[dict[str, Any]] = [
        {
            "section_name": name,
            "score": round(score, 6),
            "snippet": _section_snippet(text, query),
        }
        for (name, text, score) in top_sections
    ]

    # Backward-compat: prefer the legacy ts_headline blob when present;
    # otherwise fall back to the top-1 section snippet so callers that only
    # read ``headline`` still get something.
    top1_snippet = sections_payload[0]["snippet"] if sections_payload else ""
    headline_out = headline or top1_snippet

    # ADR-006 guard: check if this bibcode's body is LaTeX-derived.
    latex_source = _check_body_latex_provenance(conn, bibcode)
    if latex_source is not None:
        arxiv_id = _get_arxiv_id_for_bibcode(conn, bibcode)
        budget_result = apply_snippet_budget_if_needed(
            body_text=headline_out,
            source=latex_source,
            bibcode=bibcode,
            arxiv_id=arxiv_id,
        )
        # Apply snippet budget to each section snippet too so ADR-006 holds
        # uniformly across the new 'sections' field.
        guarded_sections: list[dict[str, Any]] = []
        for sec in sections_payload:
            guarded = apply_snippet_budget_if_needed(
                body_text=sec["snippet"],
                source=latex_source,
                bibcode=bibcode,
                arxiv_id=arxiv_id,
            )
            guarded_sections.append({**sec, "snippet": guarded["snippet"]})
        return SearchResult(
            papers=[
                {
                    "bibcode": row["bibcode"],
                    "title": row["title"],
                    "headline": budget_result["snippet"],
                    "has_body": True,
                    "canonical_url": budget_result["canonical_url"],
                    "sections": guarded_sections,
                }
            ],
            total=1,
            timing_ms={"query_ms": query_ms},
            metadata={
                "has_body": True,
                "adr006_guarded": True,
                "source": latex_source,
                "rerank_used": rerank_used,
                "candidate_count": len(candidates),
            },
        )

    return SearchResult(
        papers=[
            {
                "bibcode": row["bibcode"],
                "title": row["title"],
                "headline": headline_out,
                "has_body": True,
                "sections": sections_payload,
            }
        ],
        total=1,
        timing_ms={"query_ms": query_ms},
        metadata={
            "has_body": True,
            "rerank_used": rerank_used,
            "candidate_count": len(candidates),
        },
    )


# ---------------------------------------------------------------------------
# papers_fulltext read path (ADR-006 snippet budget enforcement)
# ---------------------------------------------------------------------------


def _get_arxiv_id_for_bibcode(
    conn: psycopg.Connection,
    bibcode: str,
) -> str | None:
    """Look up the arXiv identifier from the papers.identifier array.

    Returns the first identifier matching an arXiv ID pattern, or None.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT identifier FROM papers WHERE bibcode = %s",
            (bibcode,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    identifiers = row.get("identifier") or []
    for ident in identifiers:
        if _ARXIV_ID_RE.match(ident):
            return ident
    return None


def apply_snippet_budget_if_needed(
    *,
    body_text: str,
    source: str,
    bibcode: str,
    arxiv_id: str | None,
) -> dict[str, Any]:
    """Conditionally apply snippet budget based on source provenance.

    For LaTeX-derived sources (ar5iv, arxiv_local), enforces the snippet
    budget and attaches a canonical_url per ADR-006. For non-LaTeX sources,
    returns the body text unmodified.

    Parameters
    ----------
    body_text : str
        The body text to potentially truncate.
    source : str
        The source tag from papers_fulltext (e.g. 'ar5iv', 'ads_body').
    bibcode : str
        The paper's bibcode.
    arxiv_id : str or None
        The arXiv ID for building canonical_url. Required for LaTeX-derived
        sources; raises ValueError if None for those sources.

    Returns
    -------
    dict
        Keys: snippet, truncated, canonical_url (if LaTeX-derived),
        original_length, budget.
    """
    if source in LATEX_DERIVED_SOURCES:
        if not arxiv_id:
            raise ValueError(
                f"LaTeX-derived source '{source}' for bibcode '{bibcode}' "
                f"requires an arxiv_id to build canonical_url"
            )
        canonical_url = _build_canonical_url(arxiv_id)
        payload = enforce_snippet_budget(body_text, canonical_url)
        return {
            "snippet": payload.snippet,
            "truncated": payload.truncated,
            "canonical_url": payload.canonical_url,
            "original_length": payload.original_length,
            "budget": payload.budget,
        }

    # Non-LaTeX source: pass through without truncation
    return {
        "snippet": body_text,
        "truncated": False,
        "canonical_url": None,
        "original_length": len(body_text),
    }


def read_fulltext(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    section: str = "full",
    char_offset: int = 0,
    limit: int = 5000,
) -> SearchResult:
    """Read structured fulltext from papers_fulltext with ADR-006 enforcement.

    For LaTeX-derived sources (ar5iv, arxiv_local), the body text is
    passed through :func:`enforce_snippet_budget` and the response includes
    ``canonical_url``. For non-LaTeX sources, the body text is returned
    as-is with standard pagination.

    Parameters
    ----------
    conn : psycopg.Connection
    bibcode : str
        ADS bibcode.
    section : str
        Section name or 'full' for the entire body.
    char_offset : int
        Character offset for pagination (non-LaTeX sources only; LaTeX
        sources are always returned from the start within budget).
    limit : int
        Maximum characters to return (non-LaTeX sources only).

    Returns
    -------
    SearchResult
        papers list contains a single dict with section_text (or snippet
        for LaTeX sources), source, and optional canonical_url.
    """
    t0 = time.perf_counter()

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT bibcode, source, sections, parser_version "
                "FROM papers_fulltext WHERE bibcode = %s",
                (bibcode,),
            )
            row = cur.fetchone()
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={
                "error": "Structured fulltext store is unavailable on this deployment",
                "fallback_hint": "Use read_paper without 'section' argument to read papers.body directly",
            },
        )

    query_ms = _elapsed_ms(t0)

    if row is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": query_ms},
            metadata={"error": f"No fulltext found for: {bibcode}"},
        )

    source = row["source"]
    sections_raw = row["sections"]

    # Parse sections JSON
    if isinstance(sections_raw, str):
        sections_data = json.loads(sections_raw)
    else:
        sections_data = sections_raw  # already parsed by psycopg JSONB

    # Reconstruct body text from sections
    section_lower = section.lower()
    if section_lower == "full":
        body_parts = [s["text"] for s in sections_data if s.get("text")]
        body_text = "\n\n".join(body_parts)
        section_name = "full"
    else:
        matched = [
            s
            for s in sections_data
            if s.get("heading", "").lower().strip().endswith(section_lower)
            or section_lower in s.get("heading", "").lower()
        ]
        if matched:
            section_name = matched[0].get("heading", section_lower)
            body_text = matched[0].get("text", "")
        else:
            available = [s.get("heading", "") for s in sections_data]
            return SearchResult(
                papers=[],
                total=0,
                timing_ms={"query_ms": query_ms},
                metadata={
                    "error": f"Section '{section}' not found in fulltext",
                    "available_sections": available,
                    "source": source,
                },
            )

    if source in LATEX_DERIVED_SOURCES:
        # LaTeX-derived: enforce snippet budget (ADR-006)
        arxiv_id = _get_arxiv_id_for_bibcode(conn, bibcode)
        budget_result = apply_snippet_budget_if_needed(
            body_text=body_text,
            source=source,
            bibcode=bibcode,
            arxiv_id=arxiv_id,
        )
        return SearchResult(
            papers=[
                {
                    "bibcode": bibcode,
                    "source": source,
                    "section_name": section_name,
                    "section_text": budget_result["snippet"],
                    "truncated": budget_result["truncated"],
                    "canonical_url": budget_result["canonical_url"],
                    "original_length": budget_result["original_length"],
                    "parser_version": row["parser_version"],
                }
            ],
            total=1,
            timing_ms={"query_ms": query_ms},
            metadata={"source": source, "latex_derived": True},
        )

    # Non-LaTeX source: standard pagination
    total_chars = len(body_text)
    section_text = body_text[char_offset : char_offset + limit]

    return SearchResult(
        papers=[
            {
                "bibcode": bibcode,
                "source": source,
                "section_name": section_name,
                "section_text": section_text,
                "char_offset": char_offset,
                "total_chars": total_chars,
                "parser_version": row["parser_version"],
            }
        ],
        total=1,
        timing_ms={"query_ms": query_ms},
        metadata={"source": source, "latex_derived": False},
    )


# ---------------------------------------------------------------------------
# Sibling-fallback fulltext lookup (pure, injected-fetcher)
# ---------------------------------------------------------------------------

# LATEX_DERIVED_SOURCES is imported from scix.sources.ar5iv at the top of
# this module — canonical definition per ADR-006.

# Non-LaTeX sibling sources fall through to a "miss-with-hint" response; the
# caller should invoke read_paper(bibcode=<sibling>) directly rather than
# receiving row data under a different bibcode.
_NON_LATEX_FULLTEXT_SOURCES = frozenset({"s2orc", "ads_body", "docling"})


def read_fulltext_with_sibling_fallback(
    bibcode: str,
    fetch_row,  # Callable[[str], dict | None]
    fetch_aliases,  # Callable[[str], Iterable[str]]
    fetch_canonical_url,  # Callable[[str], str | None]
) -> dict:
    """Resolve a fulltext row for ``bibcode``, falling back to siblings.

    This is a **pure** function: it performs no IO and holds no DB
    connections. All side-effects (DB reads, URL construction) are
    delegated to three injected callables.

    Semantics (PRD R3):

    1. Direct hit — ``fetch_row(bibcode)`` returns a row ⇒ return
       ``{"hit": True, "row": row, "sibling": None}``.
    2. Otherwise resolve ``fetch_aliases(bibcode)``, de-duplicate while
       preserving order, and exclude the requested bibcode itself
       (cycle-guard).
    3. For each sibling, ``srow = fetch_row(sibling)``. If ``srow`` is
       present and its ``source`` is a LaTeX-derived source, return the
       sibling's row with ``served_from_sibling_bibcode`` and
       ``canonical_url`` populated.
    4. Else, if any sibling has a hit in a non-LaTeX source
       ({'s2orc','ads_body','docling'}), return a miss-with-hint
       response that tells the caller which sibling bibcode to read.
       The sibling's ``row`` is NOT propagated.
    5. Else, return ``{"hit": False, "miss_with_hint": False}``.

    The function never recurses and never calls ``fetch_row`` on the
    requested bibcode more than once, even if ``fetch_aliases`` returns
    the requested bibcode in its alias list.
    """
    # Rule 1: direct hit
    row = fetch_row(bibcode)
    if row is not None:
        return {"hit": True, "row": row, "sibling": None}

    # Rule 2: resolve aliases, de-duplicate (preserve order), drop self
    raw_aliases = fetch_aliases(bibcode) or ()
    seen: set[str] = {bibcode}
    siblings: list[str] = []
    for alias in raw_aliases:
        if alias in seen:
            continue
        seen.add(alias)
        siblings.append(alias)

    # Rule 3 / 4: walk siblings, preferring LaTeX-derived hits, but
    # recording the first non-LaTeX hint we encounter so we can return
    # it if no LaTeX sibling is found.
    non_latex_hint_sibling: str | None = None

    for sibling in siblings:
        srow = fetch_row(sibling)
        if srow is None:
            continue
        source = srow.get("source") if isinstance(srow, dict) else None
        if source in LATEX_DERIVED_SOURCES:
            return {
                "hit": True,
                "row": srow,
                "sibling": sibling,
                "served_from_sibling_bibcode": sibling,
                "canonical_url": fetch_canonical_url(sibling),
            }
        if non_latex_hint_sibling is None and source in _NON_LATEX_FULLTEXT_SOURCES:
            non_latex_hint_sibling = sibling

    # Rule 4: no LaTeX hit, but a non-LaTeX sibling exists
    if non_latex_hint_sibling is not None:
        return {
            "hit": False,
            "miss_with_hint": True,
            "fulltext_available_under_sibling": non_latex_hint_sibling,
            "hint": f"call read_paper(bibcode={non_latex_hint_sibling})",
        }

    # Rule 5: complete miss
    return {"hit": False, "miss_with_hint": False}
