"""Real-corpus helpers for PRD §M4 three-way eval and §M4.5 lane consistency.

At u12 the three-way eval and the lane-consistency eval shipped with mock
backends only. This module bridges the eval runners to the populated prod
database so the PRD hinge can be evaluated for real:

* :func:`sample_seed_papers` — picks N seed bibcodes that simultaneously
  have an ``indus`` embedding, at least one entity link in
  ``document_entities_canonical``, and a rich citation network.
* :func:`ground_truth_for_seed` — returns the citation neighbors of a
  seed (capped) as the binary-relevance set.
* :class:`RealEvalContext` — holds a psycopg connection plus cached
  per-bibcode entity lookups so the three lanes can share DB reads.
* :func:`static_filter_retrieve` / :func:`jit_rerank_retrieve` — the two
  entity-enrichment lanes (static-core pre-filter and JIT re-ranker) that
  sit on top of :func:`baseline_retrieve`.
* :func:`citation_chain_entities`, :func:`hybrid_enrich_entities`,
  :func:`static_canonical_entities` — the three lanes consumed by the
  §M4.5 consistency gate.

Design notes
------------
* Read-only. All queries against ``dbname=scix`` are SELECTs.
* Every DB access funnels through parameterised queries.
* The module is free of hardcoded credentials; callers supply a DSN or
  rely on ``SCIX_DSN`` via :func:`scix.db.get_connection`.
* Cached entity reads are per-connection — safe to reuse within a single
  script run, not safe to share across threads.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from scix.search import hybrid_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_NEIGHBORS = 10
DEFAULT_MAX_RELEVANT = 500
BASELINE_TOP_N = 50
JIT_OVERLAP_BOOST = 0.25  # bump applied per overlapping entity when re-ranking

# Title token scrubbing — same conventions as eval_retrieval_50q.
_TITLE_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_ENTITY_RE = re.compile(r"&[a-zA-Z]+;")
_ALPHA_TOKEN_RE = re.compile(r"[a-z]{4,}")
_LEXICAL_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "was", "one", "our", "out", "has", "have", "from", "with", "this",
        "that", "they", "been", "were", "which", "their", "will", "each",
        "many", "some", "than", "them", "then", "what", "when", "over",
        "such", "into", "most", "between", "these", "using", "based",
        "also", "about", "more", "new", "first", "two",
    }
)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedPaper:
    """A single seed paper sampled for the real-data eval."""

    bibcode: str
    title: str
    abstract: str | None
    year: int | None
    citation_count: int
    n_neighbors: int
    n_entities: int

    @property
    def lexical_query(self) -> str:
        """Build a short OR-joined lexical query from the title.

        Mirrors ``scripts/eval_retrieval_50q._make_lexical_query`` so
        downstream retrieval behaves identically to the paper's Section
        4.4 baseline.
        """
        raw = self.title or ""
        cleaned = _TITLE_ENTITY_RE.sub(" ", _TITLE_TAG_RE.sub(" ", raw)).lower()
        tokens = _ALPHA_TOKEN_RE.findall(cleaned)
        content: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in _LEXICAL_STOP_WORDS or tok in seen:
                continue
            seen.add(tok)
            content.append(tok)
            if len(content) >= 6:
                break
        return " ".join(content)


@dataclass
class RealEvalContext:
    """Shared state for a real-data eval run.

    Attributes
    ----------
    conn
        Live psycopg connection — the caller retains ownership and is
        responsible for closing.
    entity_cache
        Bibcode -> frozenset[int] map of canonical entity IDs. Populated
        lazily via :meth:`entities_for`.
    embedding_cache
        Bibcode -> list[float] of the ``indus`` embedding for a bibcode.
    """

    conn: psycopg.Connection
    entity_cache: dict[str, frozenset[int]] = field(default_factory=dict)
    embedding_cache: dict[str, list[float]] = field(default_factory=dict)

    def entities_for(self, bibcode: str) -> frozenset[int]:
        """Return entity IDs for ``bibcode`` from the canonical MV."""
        if bibcode in self.entity_cache:
            return self.entity_cache[bibcode]
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT entity_id FROM document_entities_canonical WHERE bibcode = %s",
                [bibcode],
            )
            ids = frozenset(int(row[0]) for row in cur.fetchall())
        self.entity_cache[bibcode] = ids
        return ids

    def embedding_for(self, bibcode: str, model_name: str = "indus") -> list[float] | None:
        """Return the stored embedding for ``bibcode`` as a list of floats."""
        key = f"{bibcode}::{model_name}"
        if key in self.embedding_cache:
            return self.embedding_cache[key]
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT embedding FROM paper_embeddings "
                "WHERE bibcode = %s AND model_name = %s",
                [bibcode, model_name],
            )
            row = cur.fetchone()
        if row is None:
            return None
        emb_raw = row["embedding"]
        if isinstance(emb_raw, str):
            embedding = [float(x) for x in emb_raw.strip("[]").split(",")]
        else:
            embedding = list(emb_raw)
        self.embedding_cache[key] = embedding
        return embedding


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_seed_papers(
    conn: psycopg.Connection,
    n_seeds: int,
    min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
    random_seed: int = 42,
) -> list[SeedPaper]:
    """Sample seed papers for the three-way eval.

    Candidates must (a) have an ``indus`` embedding, (b) have at least
    one entity link in ``document_entities_canonical`` so the static-core
    filter lane has coverage, and (c) have at least ``min_neighbors``
    citation neighbors so the ground-truth set is non-trivial.

    The sample is stratified across citation-count quintiles to avoid the
    naive bias toward only highly-cited papers.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", [random_seed / 2**31])

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH seed_candidates AS (
                SELECT p.bibcode, p.title, p.abstract, p.year, p.citation_count
                FROM papers p
                WHERE p.abstract IS NOT NULL
                  AND length(p.title) > 20
                  AND p.citation_count >= %s
                  AND EXISTS (
                      SELECT 1 FROM paper_embeddings pe
                      WHERE pe.bibcode = p.bibcode AND pe.model_name = 'indus'
                  )
                  AND EXISTS (
                      SELECT 1 FROM document_entities_canonical dec
                      WHERE dec.bibcode = p.bibcode
                  )
                ORDER BY random()
                LIMIT %s
            ),
            neighbor_counts AS (
                SELECT sc.bibcode,
                       (SELECT COUNT(*) FROM citation_edges ce
                        WHERE ce.source_bibcode = sc.bibcode) +
                       (SELECT COUNT(*) FROM citation_edges ce
                        WHERE ce.target_bibcode = sc.bibcode) AS n_neighbors
                FROM seed_candidates sc
            ),
            entity_counts AS (
                SELECT sc.bibcode,
                       (SELECT COUNT(*) FROM document_entities_canonical dec
                        WHERE dec.bibcode = sc.bibcode) AS n_entities
                FROM seed_candidates sc
            )
            SELECT sc.bibcode, sc.title, sc.abstract, sc.year, sc.citation_count,
                   nc.n_neighbors, ec.n_entities,
                   NTILE(5) OVER (ORDER BY sc.citation_count) AS cite_tier
            FROM seed_candidates sc
            JOIN neighbor_counts nc ON nc.bibcode = sc.bibcode
            JOIN entity_counts ec ON ec.bibcode = sc.bibcode
            WHERE nc.n_neighbors >= %s
            """,
            [min_neighbors, n_seeds * 20, min_neighbors],
        )
        rows = cur.fetchall()

    # Stratify across citation-count tiers (approximately even split).
    per_tier = max(1, n_seeds // 5 + 1)
    by_tier: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_tier.setdefault(row["cite_tier"], []).append(row)

    sampled: list[SeedPaper] = []
    for tier in sorted(by_tier.keys()):
        for row in by_tier[tier][:per_tier]:
            sampled.append(
                SeedPaper(
                    bibcode=row["bibcode"],
                    title=row["title"],
                    abstract=row.get("abstract"),
                    year=row.get("year"),
                    citation_count=row.get("citation_count") or 0,
                    n_neighbors=row["n_neighbors"],
                    n_entities=row["n_entities"],
                )
            )
            if len(sampled) >= n_seeds:
                break
        if len(sampled) >= n_seeds:
            break

    logger.info(
        "Sampled %d seed papers (requested %d, min_neighbors=%d)",
        len(sampled),
        n_seeds,
        min_neighbors,
    )
    return sampled


def ground_truth_for_seed(
    conn: psycopg.Connection,
    seed_bibcode: str,
    max_relevant: int = DEFAULT_MAX_RELEVANT,
) -> set[str]:
    """Return the citation-based relevance set for ``seed_bibcode``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            (SELECT target_bibcode FROM citation_edges
             WHERE source_bibcode = %s LIMIT %s)
            UNION
            (SELECT source_bibcode FROM citation_edges
             WHERE target_bibcode = %s LIMIT %s)
            """,
            [seed_bibcode, max_relevant, seed_bibcode, max_relevant],
        )
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Retrieval lanes
# ---------------------------------------------------------------------------


def _hybrid_candidates(
    ctx: RealEvalContext,
    seed: SeedPaper,
    limit: int = BASELINE_TOP_N,
) -> tuple[list[dict[str, Any]], float]:
    """Run production hybrid_search for a seed. Returns (papers, latency_ms)."""
    embedding = ctx.embedding_for(seed.bibcode)
    t0 = time.perf_counter()
    result = hybrid_search(
        ctx.conn,
        seed.lexical_query,
        query_embedding=embedding,
        model_name="indus",
        vector_limit=limit,
        lexical_limit=limit,
        top_n=limit,
        include_body=False,
    )
    latency = (time.perf_counter() - t0) * 1000.0
    papers = [p for p in result.papers if p.get("bibcode") != seed.bibcode]
    return papers, latency


def baseline_retrieve(
    ctx: RealEvalContext,
    seed: SeedPaper,
    limit: int = BASELINE_TOP_N,
) -> tuple[list[str], float]:
    """Lane A — plain hybrid search, no entity enrichment."""
    papers, latency = _hybrid_candidates(ctx, seed, limit=limit)
    return [p["bibcode"] for p in papers], latency


def static_filter_retrieve(
    ctx: RealEvalContext,
    seed: SeedPaper,
    limit: int = BASELINE_TOP_N,
) -> tuple[list[str], float]:
    """Lane B — hybrid + static-core filter.

    Restricts the baseline candidate list to bibcodes that appear in
    ``document_entities_canonical``. This pre-filter models the
    "static-core only" universe the PRD hinge probes — if it fails to
    beat the plain baseline the static lane carries no incremental value.
    """
    papers, latency = _hybrid_candidates(ctx, seed, limit=limit)
    filtered: list[str] = []
    for p in papers:
        bib = p["bibcode"]
        if ctx.entities_for(bib):
            filtered.append(bib)
    return filtered, latency


def jit_rerank_retrieve(
    ctx: RealEvalContext,
    seed: SeedPaper,
    limit: int = BASELINE_TOP_N,
    overlap_boost: float = JIT_OVERLAP_BOOST,
) -> tuple[list[str], float]:
    """Lane C — hybrid + JIT enrichment.

    Retrieves the full baseline candidate list, then re-ranks by adding
    ``overlap_boost`` per entity-ID overlap between the candidate and the
    seed. Entities are resolved at query time by the resolver lane
    (``document_entities_canonical`` read); papers with no entities keep
    their baseline rank. This is the operational analog of calling
    ``resolve_entities(mode='static')`` on the top-K candidates.
    """
    papers, latency = _hybrid_candidates(ctx, seed, limit=limit)
    seed_entities = ctx.entities_for(seed.bibcode)
    if not seed_entities:
        return [p["bibcode"] for p in papers], latency

    scored: list[tuple[float, int, str]] = []
    for idx, p in enumerate(papers):
        bib = p["bibcode"]
        cand_entities = ctx.entities_for(bib)
        overlap = len(seed_entities & cand_entities)
        # Use negative index as primary tiebreaker so stable re-ranking
        # preserves baseline order when overlap is zero.
        score = float(overlap) * overlap_boost - idx * 1e-6
        scored.append((score, idx, bib))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [bib for _, _, bib in scored], latency


# ---------------------------------------------------------------------------
# M4.5 lane readers
# ---------------------------------------------------------------------------


def citation_chain_entities(
    ctx: RealEvalContext,
    bibcode: str,
    max_neighbors: int = 20,
) -> frozenset[int]:
    """Lane A — union of entities across citation neighbors.

    Mirrors what the ``citation_chain`` MCP tool would derive from a
    bibcode's citation cluster.
    """
    with ctx.conn.cursor() as cur:
        cur.execute(
            """
            (SELECT target_bibcode FROM citation_edges
             WHERE source_bibcode = %s LIMIT %s)
            UNION
            (SELECT source_bibcode FROM citation_edges
             WHERE target_bibcode = %s LIMIT %s)
            """,
            [bibcode, max_neighbors, bibcode, max_neighbors],
        )
        neighbors = [row[0] for row in cur.fetchall()]

    ids: set[int] = set()
    for neighbor in neighbors:
        ids.update(ctx.entities_for(neighbor))
    return frozenset(ids)


def hybrid_enrich_entities(
    ctx: RealEvalContext,
    bibcode: str,
) -> frozenset[int]:
    """Lane B — what ``hybrid_search[enrich_entities=True]`` reads.

    In production the enrich-entities path reads the canonical MV by
    bibcode, which is what :meth:`RealEvalContext.entities_for` does.
    """
    return ctx.entities_for(bibcode)


def static_canonical_entities(
    ctx: RealEvalContext,
    bibcode: str,
    min_fused_confidence: float = 0.5,
) -> frozenset[int]:
    """Lane C — canonical read with a confidence floor.

    Same source as Lane B but filtered at ``fused_confidence >=
    min_fused_confidence``; this surfaces any divergence introduced by the
    confidence threshold used in downstream MCP tools.
    """
    with ctx.conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id FROM document_entities_canonical "
            "WHERE bibcode = %s AND fused_confidence >= %s",
            [bibcode, min_fused_confidence],
        )
        return frozenset(int(row[0]) for row in cur.fetchall())


# ---------------------------------------------------------------------------
# M4.5 fixture sampler
# ---------------------------------------------------------------------------


def sample_lane_consistency_bibcodes(
    conn: psycopg.Connection,
    n_samples: int = 100,
    min_entities: int = 2,
    random_seed: int = 42,
    modern_floor_year: int = 2000,
) -> list[str]:
    """Return bibcodes for the §M4.5 consistency fixture.

    Picks modern (``year >= modern_floor_year``) bibcodes that have both
    canonical entities AND at least one citation neighbor, so all three
    consistency lanes can produce a non-empty set. Restricting to modern
    papers avoids the historical tail (pre-1950 books) where citation
    graphs are thin and entity coverage is sparse.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", [random_seed / 2**31])

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH modern_ents AS (
                SELECT dec.bibcode
                FROM document_entities_canonical dec
                JOIN papers p ON p.bibcode = dec.bibcode
                WHERE p.year >= %s
                GROUP BY dec.bibcode
                HAVING COUNT(*) >= %s
                ORDER BY random()
                LIMIT %s
            )
            SELECT me.bibcode
            FROM modern_ents me
            WHERE EXISTS (
                SELECT 1 FROM citation_edges ce
                WHERE ce.source_bibcode = me.bibcode OR ce.target_bibcode = me.bibcode
            )
            LIMIT %s
            """,
            [modern_floor_year, min_entities, n_samples * 5, n_samples],
        )
        return [row["bibcode"] for row in cur.fetchall()]


__all__ = [
    "BASELINE_TOP_N",
    "DEFAULT_MAX_RELEVANT",
    "DEFAULT_MIN_NEIGHBORS",
    "JIT_OVERLAP_BOOST",
    "RealEvalContext",
    "SeedPaper",
    "baseline_retrieve",
    "citation_chain_entities",
    "ground_truth_for_seed",
    "hybrid_enrich_entities",
    "jit_rerank_retrieve",
    "sample_lane_consistency_bibcodes",
    "sample_seed_papers",
    "static_canonical_entities",
    "static_filter_retrieve",
]
