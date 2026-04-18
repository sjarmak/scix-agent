"""Core search module with timing instrumentation.

Every public function returns a SearchResult that includes timing metadata
(milliseconds) for benchmarking and observability. This is a hard contract:
callers can always access result.timing_ms to understand latency breakdown.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from scix.db import IterativeScanMode, configure_iterative_scan
from scix.sources.ar5iv import LATEX_DERIVED_SOURCES, _ARXIV_ID_RE, _build_canonical_url
from scix.sources.licensing import enforce_snippet_budget
from scix.stubs import PaperStub

logger = logging.getLogger(__name__)

# Stub columns used in all search queries that return PaperStubs
STUB_COLUMNS = "p.bibcode, p.title, p.first_author, p.year, p.citation_count, p.abstract"

# Default RRF constant (controls how much rank position matters vs raw score)
RRF_K = 60


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
    """Optional filters applied to search queries."""

    year_min: int | None = None
    year_max: int | None = None
    arxiv_class: str | None = None
    doctype: str | None = None
    first_author: str | None = None

    def to_where_clause(self, table_alias: str = "p") -> tuple[str, list[Any]]:
        """Generate a WHERE clause fragment and parameter list.

        Returns ("AND ... AND ...", [param1, param2, ...]) or ("", []) if no filters.
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

    filter_clause, filter_params = (filters or SearchFilters()).to_where_clause("p")

    # plainto_tsquery is more robust than websearch_to_tsquery for programmatic use:
    # it doesn't fail on unmatched quotes or special chars in user input.
    query = f"""
        SELECT {STUB_COLUMNS},
               ts_rank_cd(p.tsv, plainto_tsquery('{ts_config}', %s), 32) AS rank
        FROM papers p
        WHERE p.tsv @@ plainto_tsquery('{ts_config}', %s)
        {filter_clause}
        ORDER BY rank DESC
        LIMIT %s
    """
    params: list[Any] = [query_text, query_text] + filter_params + [limit]

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

    filter_clause, filter_params = (filters or SearchFilters()).to_where_clause("p")

    query = f"""
        SELECT {STUB_COLUMNS},
               ts_rank_cd(p.tsv, plainto_tsquery('english', %s), 32) AS rank
        FROM papers p
        WHERE p.body IS NOT NULL
          AND length(p.body) <= {_BODY_TSVECTOR_MAX_BYTES}
          AND to_tsvector('english', p.body) @@ plainto_tsquery('english', %s)
        {filter_clause}
        ORDER BY rank DESC
        LIMIT %s
    """
    params: list[Any] = [query_text, query_text] + filter_params + [limit]

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
    vec_cast = f"vector({ndim})"
    filter_clause, filter_params = (filters or SearchFilters()).to_where_clause("p")

    with conn.cursor() as cur:
        # Tune HNSW probe depth for this transaction
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")

    # Auto-enable iterative scan for filtered queries on pgvector >= 0.8.0
    has_filters = bool(filter_clause)
    scan_mode = iterative_scan
    if scan_mode is None and has_filters:
        scan_mode = "relaxed_order"

    iterative_applied = False
    if scan_mode is not None:
        iterative_applied = configure_iterative_scan(conn, mode=scan_mode)

    # Cast embedding to vector(N) to match per-model partial HNSW expression
    # indexes which are defined on (embedding::vector(N)).
    query = f"""
        SELECT {STUB_COLUMNS},
               1 - (pe.embedding::{vec_cast} <=> %s::{vec_cast}) AS similarity
        FROM paper_embeddings pe
        JOIN papers p ON p.bibcode = pe.bibcode
        WHERE pe.model_name = %s
        {filter_clause}
        ORDER BY pe.embedding::{vec_cast} <=> %s::{vec_cast}
        LIMIT %s
    """
    params: list[Any] = [vec_str, model_name] + filter_params + [vec_str, limit]

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
    if not filter_clause:
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
        count_sql = f"SELECT count(*) FROM (SELECT 1 FROM papers p WHERE TRUE {filter_clause} LIMIT {cap}) sub"
        cur.execute(count_sql, filter_params)
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
    vec_cast = f"vector({ndim})"
    filter_clause, filter_params = (filters or SearchFilters()).to_where_clause("p")

    query = f"""
        WITH filtered AS MATERIALIZED (
            SELECT p.bibcode
            FROM papers p
            WHERE TRUE {filter_clause}
        )
        SELECT {STUB_COLUMNS},
               1 - (pe.embedding::{vec_cast} <=> %s::{vec_cast}) AS similarity
        FROM paper_embeddings pe
        JOIN filtered f ON f.bibcode = pe.bibcode
        JOIN papers p   ON p.bibcode = pe.bibcode
        WHERE pe.model_name = %s
        ORDER BY pe.embedding::{vec_cast} <=> %s::{vec_cast}
        LIMIT %s
    """
    params: list[Any] = filter_params + [vec_str, model_name, vec_str, limit]

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
    """
    timing: dict[str, float] = {}

    # Lexical search (title + abstract tsvector)
    lex_result = lexical_search(conn, query_text, filters=filters, limit=lexical_limit)
    timing["lexical_ms"] = lex_result.timing_ms["lexical_ms"]

    results_lists: list[list[dict[str, Any]]] = [lex_result.papers]

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
    )


# ---------------------------------------------------------------------------
# Cross-encoder reranker (lazy-loaded, batched inference)
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Re-rank papers using a cross-encoder model.

    Lazy-loads the model on first use to avoid import overhead.
    Batches all candidates in a single forward pass.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
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

    The communities list is populated only for papers whose
    ``community_id_medium`` is real (non-NULL, non-sentinel). When the Leiden
    giant-component run has not been persisted (bead scix_experiments-8r3),
    this degrades gracefully to an empty list per bucket. Communities with no
    matching row in the ``communities`` table are omitted rather than emitted
    with a null label.
    """
    if not yearly_counts:
        return []

    years = [yc["year"] for yc in yearly_counts]
    count_by_year = {yc["year"]: yc["count"] for yc in yearly_counts}

    anchor_sql = f"""
        WITH ranked AS (
            SELECT {STUB_COLUMNS},
                   pm.pagerank,
                   pm.community_id_medium,
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

            cid = row["community_id_medium"]
            if cid is not None and cid != _NO_COMMUNITY_SENTINEL:
                community_counts_by_year[year][cid] += 1
                all_community_ids.add(cid)

        label_map: dict[int, str] = {}
        if all_community_ids:
            cur.execute(
                "SELECT community_id, label FROM communities "
                "WHERE resolution = 'medium' AND community_id = ANY(%s) "
                "AND label IS NOT NULL",
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

    filter_clause, filter_params = (filters or SearchFilters()).to_where_clause("p")

    if facet_field in allowed_simple:
        sql = f"""
            SELECT p.{facet_field}::text AS val, count(*) AS cnt
            FROM papers p
            WHERE p.{facet_field} IS NOT NULL
            {filter_clause}
            GROUP BY p.{facet_field}
            ORDER BY cnt DESC
            LIMIT %s
        """
        params: list[Any] = filter_params + [limit]
    elif facet_field in allowed_array:
        sql = f"""
            SELECT elem AS val, count(*) AS cnt
            FROM papers p, unnest(p.{facet_field}) AS elem
            WHERE TRUE
            {filter_clause}
            GROUP BY elem
            ORDER BY cnt DESC
            LIMIT %s
        """
        params = filter_params + [limit]
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
    """Get precomputed graph metrics for a paper (PageRank, HITS, communities)."""
    t0 = time.perf_counter()

    sql = """
        SELECT pm.*,
               c_coarse.label AS community_label_coarse,
               c_medium.label AS community_label_medium,
               c_fine.label AS community_label_fine
        FROM paper_metrics pm
        LEFT JOIN communities c_coarse
            ON c_coarse.community_id = pm.community_id_coarse
            AND c_coarse.resolution = 'coarse'
        LEFT JOIN communities c_medium
            ON c_medium.community_id = pm.community_id_medium
            AND c_medium.resolution = 'medium'
        LEFT JOIN communities c_fine
            ON c_fine.community_id = pm.community_id_fine
            AND c_fine.resolution = 'fine'
        WHERE pm.bibcode = %s
    """
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
            f"invalid resolution: {resolution!r}. "
            f"Must be one of {sorted(_VALID_RESOLUTIONS)}"
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


def concept_search(
    conn: psycopg.Connection,
    query: str,
    *,
    include_descendants: bool = True,
    include_subtopics: bool | None = None,
    limit: int = 20,
) -> SearchResult:
    """Search for papers by UAT concept, optionally including descendants.

    When ``include_descendants=True`` (default), the query concept is expanded
    to its full descendant set via a recursive CTE against ``uat_relationships``
    bounded at depth ``CONCEPT_DESCENDANT_MAX_DEPTH`` (6). Papers tagged with
    the root concept or any descendant are returned.

    When ``include_descendants=False``, only papers tagged with the exact
    root concept are returned (no expansion).

    ``include_subtopics`` is a legacy alias for ``include_descendants``. When
    provided (non-None), it overrides ``include_descendants`` for backward
    compatibility.

    Accepts a concept label (looked up case-insensitively) or concept_id URI.
    """
    # Backward-compat: legacy ``include_subtopics`` wins when explicitly set.
    if include_subtopics is not None:
        include_descendants = include_subtopics

    t0 = time.perf_counter()

    # Resolve concept_id from label or URI
    with conn.cursor() as cur:
        if query.startswith("http"):
            cur.execute("SELECT concept_id FROM uat_concepts WHERE concept_id = %s", (query,))
        else:
            cur.execute(
                "SELECT concept_id FROM uat_concepts WHERE lower(preferred_label) = lower(%s)",
                (query,),
            )
        row = cur.fetchone()

    if row is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={"concept_found": False, "query": query},
        )

    concept_id = row[0]

    if include_descendants:
        # Recursive CTE bounded at CONCEPT_DESCENDANT_MAX_DEPTH (6). The
        # depth-cap guard lives on the recursive step so that a descendant at
        # depth N can still be produced iff N <= max_depth.
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

    query_ms = _elapsed_ms(t0)
    papers = [PaperStub.from_row(row).to_dict() for row in rows]

    # Get concept label
    with conn.cursor() as cur:
        cur.execute("SELECT preferred_label FROM uat_concepts WHERE concept_id = %s", (concept_id,))
        label_row = cur.fetchone()

    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={"query_ms": query_ms},
        metadata={
            "concept_found": True,
            "concept_id": concept_id,
            "concept_label": label_row[0] if label_row else None,
            "include_descendants": include_descendants,
            # Keep legacy key for backward compatibility with consumers that
            # still read ``include_subtopics`` from the metadata envelope.
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
    """Get full entity context from the agent_entity_context matview.

    Returns entity card: entity_id, canonical_name, entity_type, discipline,
    source, identifiers, aliases, relationships, and citing_paper_count.

    Returns an empty SearchResult with an error in metadata when the entity_id
    is not found, rather than raising an exception.
    """
    t0 = time.perf_counter()

    sql = """
        SELECT entity_id, canonical_name, entity_type, discipline, source,
               identifiers, aliases, relationships, citing_paper_count
        FROM agent_entity_context
        WHERE entity_id = %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (entity_id,))
        row = cur.fetchone()

    query_ms = _elapsed_ms(t0)

    if row is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": query_ms},
            metadata={"error": f"Entity not found: {entity_id}"},
        )

    result = dict(row)
    # Convert aliases from SQL array to list for JSON serialization
    if result.get("aliases") is not None:
        result["aliases"] = list(result["aliases"])

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


def read_paper_section(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    section: str = "full",
    char_offset: int = 0,
    limit: int = 5000,
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

    Returns
    -------
    SearchResult
        papers list contains a single dict with section_text, section_name,
        has_body, char_offset, total_chars, and bibcode.
    """
    from scix.section_parser import parse_sections

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

    if section_lower == "full":
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
# Search within paper (ts_headline on body column)
# ---------------------------------------------------------------------------


def search_within_paper(
    conn: psycopg.Connection,
    bibcode: str,
    query: str,
) -> SearchResult:
    """Search within a paper's body text using PostgreSQL ts_headline.

    Parameters
    ----------
    conn : psycopg.Connection
    bibcode : str
        ADS bibcode.
    query : str
        Search terms to find within the paper body.

    Returns
    -------
    SearchResult
        papers list contains a single dict with bibcode, headline (matching
        passages with context), and has_body flag.
    """
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

    headline = row["headline"] or ""

    # ADR-006 guard: check if this bibcode's body is LaTeX-derived.
    latex_source = _check_body_latex_provenance(conn, bibcode)
    if latex_source is not None:
        arxiv_id = _get_arxiv_id_for_bibcode(conn, bibcode)
        budget_result = apply_snippet_budget_if_needed(
            body_text=headline,
            source=latex_source,
            bibcode=bibcode,
            arxiv_id=arxiv_id,
        )
        return SearchResult(
            papers=[
                {
                    "bibcode": row["bibcode"],
                    "title": row["title"],
                    "headline": budget_result["snippet"],
                    "has_body": True,
                    "canonical_url": budget_result["canonical_url"],
                }
            ],
            total=1,
            timing_ms={"query_ms": query_ms},
            metadata={"has_body": True, "adr006_guarded": True, "source": latex_source},
        )

    return SearchResult(
        papers=[
            {
                "bibcode": row["bibcode"],
                "title": row["title"],
                "headline": headline,
                "has_body": True,
            }
        ],
        total=1,
        timing_ms={"query_ms": query_ms},
        metadata={"has_body": True},
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
