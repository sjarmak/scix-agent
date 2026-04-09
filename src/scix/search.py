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

from scix.db import IterativeScanMode, configure_iterative_scan
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
    iterative_scan: IterativeScanMode | None = None,
) -> SearchResult:
    """Approximate nearest neighbor search using pgvector HNSW.

    Args:
        query_embedding: 768-dim float vector.
        model_name: Filter to a specific embedding model.
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
# Hybrid search (vector + lexical + RRF + optional cross-encoder rerank)
# ---------------------------------------------------------------------------


def hybrid_search(
    conn: psycopg.Connection,
    query_text: str,
    query_embedding: list[float] | None = None,
    *,
    model_name: str = "specter2",
    openai_embedding: list[float] | None = None,
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
    If openai_embedding is provided, runs a second vector search with
    text-embedding-3-large and fuses it alongside SPECTER2 results via RRF.
    If the OpenAI vector search fails, falls back to SPECTER2+lexical only.
    If reranker is provided, re-ranks the top RRF results.

    Args:
        openai_embedding: Optional pre-computed OpenAI text-embedding-3-large vector.
        reranker: Optional callable(query_text, papers) -> list[dict] with 'rerank_score'.
    """
    timing: dict[str, float] = {}

    # Lexical search
    lex_result = lexical_search(conn, query_text, filters=filters, limit=lexical_limit)
    timing["lexical_ms"] = lex_result.timing_ms["lexical_ms"]

    results_lists: list[list[dict[str, Any]]] = [lex_result.papers]

    # SPECTER2 vector search (if embeddings available)
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

    # OpenAI vector search (circuit breaker: fall back on failure)
    timing["openai_vector_ms"] = 0.0
    if openai_embedding is not None:
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
                "OpenAI vector search failed; falling back to SPECTER2+lexical only",
                exc_info=True,
            )

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


def temporal_evolution(
    conn: psycopg.Connection,
    bibcode_or_query: str,
    *,
    year_start: int | None = None,
    year_end: int | None = None,
    ts_config: str = "scix_english",
) -> SearchResult:
    """Show temporal trends: citations received by a paper, or publication volume for a query.

    Mode detection:
    - If bibcode_or_query matches an existing paper bibcode, shows citations per year.
    - Otherwise, treats it as a search query and shows matching publications per year.
    """
    t0 = time.perf_counter()

    # Detect mode: try to look up as a bibcode first
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
        # Citations received per year
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
        # Publication volume matching query
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

    yearly_counts = [{"year": row["year"], "count": row["count"]} for row in rows]

    return SearchResult(
        papers=[],
        total=len(yearly_counts),
        timing_ms={"query_ms": query_ms},
        metadata={"mode": mode, "bibcode_found": is_bibcode, "yearly_counts": yearly_counts},
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

    return SearchResult(
        papers=[],
        total=1,
        timing_ms={"query_ms": query_ms},
        metadata={"metrics": metrics},
    )


def explore_community(
    conn: psycopg.Connection,
    bibcode: str,
    resolution: str = "coarse",
    limit: int = 20,
) -> SearchResult:
    """Find a paper's community and return sibling papers by PageRank."""
    assert resolution in _VALID_RESOLUTIONS, f"invalid resolution: {resolution}"
    t0 = time.perf_counter()

    col = f"community_id_{resolution}"

    # Get the paper's community
    with conn.cursor() as cur:
        cur.execute(f"SELECT {col} FROM paper_metrics WHERE bibcode = %s", (bibcode,))
        row = cur.fetchone()

    if row is None or row[0] is None:
        return SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": _elapsed_ms(t0)},
            metadata={"community_id": None, "resolution": resolution},
        )

    community_id = row[0]

    # Get community metadata
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT label, paper_count, top_keywords FROM communities "
            "WHERE community_id = %s AND resolution = %s",
            (community_id, resolution),
        )
        community_row = cur.fetchone()

    # Get top papers in this community by PageRank
    sql = f"""
        SELECT {STUB_COLUMNS}
        FROM paper_metrics pm
        JOIN papers p ON p.bibcode = pm.bibcode
        WHERE pm.{col} = %s
        ORDER BY pm.pagerank DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (community_id, limit))
        rows = cur.fetchall()

    query_ms = _elapsed_ms(t0)
    papers = [PaperStub.from_row(row).to_dict() for row in rows]

    meta: dict[str, Any] = {
        "community_id": community_id,
        "resolution": resolution,
    }
    if community_row:
        meta["label"] = community_row["label"]
        meta["paper_count"] = community_row["paper_count"]
        meta["top_keywords"] = community_row["top_keywords"]

    return SearchResult(
        papers=papers, total=len(papers), timing_ms={"query_ms": query_ms}, metadata=meta
    )


def concept_search(
    conn: psycopg.Connection,
    query: str,
    *,
    include_subtopics: bool = True,
    limit: int = 20,
) -> SearchResult:
    """Search for papers by UAT concept, optionally including subtopics.

    Accepts a concept label (looked up case-insensitively) or concept_id URI.
    """
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

    if include_subtopics:
        sql = f"""
            WITH RECURSIVE descendants AS (
                SELECT child_id AS cid FROM uat_relationships WHERE parent_id = %s
                UNION
                SELECT r.child_id FROM uat_relationships r JOIN descendants d ON r.parent_id = d.cid
            ),
            all_concepts AS (
                SELECT %s AS cid
                UNION ALL
                SELECT cid FROM descendants
            )
            SELECT DISTINCT {STUB_COLUMNS}
            FROM all_concepts ac
            JOIN paper_uat_mappings m ON m.concept_id = ac.cid
            JOIN papers p ON p.bibcode = m.bibcode
            ORDER BY p.citation_count DESC NULLS LAST
            LIMIT %s
        """
        params: list[Any] = [concept_id, concept_id, limit]
    else:
        sql = f"""
            SELECT DISTINCT {STUB_COLUMNS}
            FROM paper_uat_mappings m
            JOIN papers p ON p.bibcode = m.bibcode
            WHERE m.concept_id = %s
            ORDER BY p.citation_count DESC NULLS LAST
            LIMIT %s
        """
        params = [concept_id, limit]

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
            "include_subtopics": include_subtopics,
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
