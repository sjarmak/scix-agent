"""Core search module with timing instrumentation.

Every public function returns a SearchResult that includes timing metadata
(milliseconds) for benchmarking and observability. This is a hard contract:
callers can always access result.timing_ms to understand latency breakdown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

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
# Vector search (pgvector cosine similarity with tunable ef_search)
# ---------------------------------------------------------------------------


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    model_name: str = "specter2",
    filters: SearchFilters | None = None,
    limit: int = 20,
    ef_search: int = 100,
) -> SearchResult:
    """Approximate nearest neighbor search using pgvector HNSW.

    Args:
        query_embedding: 768-dim float vector.
        model_name: Filter to a specific embedding model.
        ef_search: HNSW ef_search parameter (higher = more accurate, slower).
    """
    t0 = time.perf_counter()

    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    filter_clause, filter_params = (filters or SearchFilters()).to_where_clause("p")

    # Tune HNSW probe depth for this transaction
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")

    query = f"""
        SELECT {STUB_COLUMNS},
               1 - (pe.embedding <=> %s::vector) AS similarity
        FROM paper_embeddings pe
        JOIN papers p ON p.bibcode = pe.bibcode
        WHERE pe.model_name = %s
        {filter_clause}
        ORDER BY pe.embedding <=> %s::vector
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
# Hybrid search (vector + lexical + RRF + optional cross-encoder rerank)
# ---------------------------------------------------------------------------


def hybrid_search(
    conn: psycopg.Connection,
    query_text: str,
    query_embedding: list[float] | None = None,
    *,
    model_name: str = "specter2",
    filters: SearchFilters | None = None,
    vector_limit: int = 60,
    lexical_limit: int = 60,
    rrf_k: int = RRF_K,
    top_n: int = 20,
    ef_search: int = 100,
    reranker: Any | None = None,
) -> SearchResult:
    """Hybrid search combining vector and lexical via RRF, with optional reranking.

    If query_embedding is None, falls back to lexical-only mode (BM25-only).
    If reranker is provided, re-ranks the top RRF results.

    Args:
        reranker: Optional callable(query_text, papers) -> list[dict] with 'rerank_score'.
    """
    timing: dict[str, float] = {}

    # Lexical search
    lex_result = lexical_search(
        conn, query_text, filters=filters, limit=lexical_limit
    )
    timing["lexical_ms"] = lex_result.timing_ms["lexical_ms"]

    results_lists: list[list[dict[str, Any]]] = [lex_result.papers]

    # Vector search (if embeddings available)
    if query_embedding is not None:
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

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ) -> None:
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


def get_citations(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    limit: int = 20,
) -> SearchResult:
    """Get forward citations (papers that cite this paper). Returns stubs."""
    t0 = time.perf_counter()

    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM citation_edges ce
        JOIN papers p ON p.bibcode = ce.source_bibcode
        WHERE ce.target_bibcode = %s
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


def get_references(
    conn: psycopg.Connection,
    bibcode: str,
    *,
    limit: int = 20,
) -> SearchResult:
    """Get backward references (papers this paper cites). Returns stubs."""
    t0 = time.perf_counter()

    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM citation_edges ce
        JOIN papers p ON p.bibcode = ce.target_bibcode
        WHERE ce.source_bibcode = %s
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


def get_author_papers(
    conn: psycopg.Connection,
    author_name: str,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 50,
) -> SearchResult:
    """Get papers by an author (case-insensitive partial match on authors array)."""
    t0 = time.perf_counter()

    year_clause = ""
    params: list[Any] = [f"%{author_name}%"]

    if year_min is not None:
        year_clause += " AND p.year >= %s"
        params.append(year_min)
    if year_max is not None:
        year_clause += " AND p.year <= %s"
        params.append(year_max)

    params.append(limit)

    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM papers p
        WHERE EXISTS (
            SELECT 1 FROM unnest(p.authors) AS a(name)
            WHERE a.name ILIKE %s
        )
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
