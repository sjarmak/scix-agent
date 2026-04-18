"""Tests for :mod:`scix.eval.real_data` — real-corpus helpers for M4/M4.5."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from scix.eval.real_data import (
    BASELINE_TOP_N,
    JIT_OVERLAP_BOOST,
    RealEvalContext,
    SeedPaper,
    baseline_retrieve,
    citation_chain_entities,
    hybrid_enrich_entities,
    jit_rerank_retrieve,
    static_canonical_entities,
    static_filter_retrieve,
)


# ---------------------------------------------------------------------------
# SeedPaper.lexical_query
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lexical_query_strips_html_and_stopwords():
    seed = SeedPaper(
        bibcode="2024TEST",
        title="The <sub>Efficient</sub> Simulation of Galaxy Cluster Dynamics",
        abstract=None,
        year=2024,
        citation_count=100,
        n_neighbors=10,
        n_entities=3,
    )
    tokens = seed.lexical_query.split()
    assert "efficient" in tokens
    assert "simulation" in tokens
    assert "galaxy" in tokens
    assert "cluster" in tokens
    assert "dynamics" in tokens
    # "the" is a stopword, "sub" is length 3 (below threshold)
    assert "the" not in tokens
    assert "sub" not in tokens


@pytest.mark.unit
def test_lexical_query_caps_at_six_terms():
    seed = SeedPaper(
        bibcode="2024TEST",
        title="alpha beta gamma delta epsilon zeta eta theta iota kappa",
        abstract=None,
        year=2024,
        citation_count=10,
        n_neighbors=10,
        n_entities=1,
    )
    assert len(seed.lexical_query.split()) == 6


# ---------------------------------------------------------------------------
# RealEvalContext caching
# ---------------------------------------------------------------------------


def _mock_conn_for_entities(bibcode_to_ids: dict[str, list[int]]) -> MagicMock:
    """Build a MagicMock psycopg connection that returns seeded entity IDs."""
    conn = MagicMock()

    def cursor_factory(*args, **kwargs):
        cur = MagicMock()
        executed: dict[str, list] = {}

        def execute(sql: str, params=None):
            executed["sql"] = sql
            executed["params"] = list(params) if params else []

        def fetchall():
            sql = executed.get("sql", "")
            params = executed.get("params", [])
            if "document_entities_canonical" in sql and "entity_id" in sql:
                bib = params[0]
                return [(eid,) for eid in bibcode_to_ids.get(bib, [])]
            return []

        cur.execute.side_effect = execute
        cur.fetchall.side_effect = fetchall
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        return cur

    conn.cursor.side_effect = cursor_factory
    return conn


@pytest.mark.unit
def test_context_entity_cache_hits_db_once_per_bibcode():
    conn = _mock_conn_for_entities({"2024A": [1, 2, 3]})
    ctx = RealEvalContext(conn=conn)

    first = ctx.entities_for("2024A")
    second = ctx.entities_for("2024A")

    assert first == frozenset({1, 2, 3})
    assert first is second  # cached object
    # Cursor called only once per lookup, and the second call is memoised.
    assert conn.cursor.call_count == 1


@pytest.mark.unit
def test_context_returns_empty_frozenset_for_unknown_bibcode():
    conn = _mock_conn_for_entities({})
    ctx = RealEvalContext(conn=conn)
    assert ctx.entities_for("2024MISSING") == frozenset()


# ---------------------------------------------------------------------------
# Retrieval lane fixtures — stub hybrid_search
# ---------------------------------------------------------------------------


class _StubHybridSearch:
    """Context manager replacement for hybrid_search + embedding_for."""

    def __init__(self, papers: list[dict]):
        self.papers = papers

    def __call__(self, *args, **kwargs):
        class _Result:
            def __init__(self, papers):
                self.papers = papers
                self.timing_ms = {"lexical_ms": 0.0, "vector_ms": 0.0}

        return _Result(list(self.papers))


@pytest.fixture
def stub_seed() -> SeedPaper:
    return SeedPaper(
        bibcode="2024SEED",
        title="galaxy cluster simulation dynamics",
        abstract=None,
        year=2024,
        citation_count=100,
        n_neighbors=10,
        n_entities=3,
    )


@pytest.mark.unit
def test_baseline_retrieve_excludes_seed(monkeypatch, stub_seed):
    papers = [
        {"bibcode": "2024SEED"},
        {"bibcode": "2024A"},
        {"bibcode": "2024B"},
    ]
    monkeypatch.setattr("scix.eval.real_data.hybrid_search", _StubHybridSearch(papers))

    conn = _mock_conn_for_entities({})
    ctx = RealEvalContext(conn=conn)
    # Prime embedding cache so embedding_for doesn't hit DB.
    ctx.embedding_cache["2024SEED::indus"] = [0.1, 0.2, 0.3]

    bibs, latency = baseline_retrieve(ctx, stub_seed)
    assert bibs == ["2024A", "2024B"]
    assert latency >= 0.0


@pytest.mark.unit
def test_static_filter_removes_bibcodes_without_entities(monkeypatch, stub_seed):
    papers = [
        {"bibcode": "2024A"},  # has entities
        {"bibcode": "2024B"},  # no entities -> filtered
        {"bibcode": "2024C"},  # has entities
    ]
    monkeypatch.setattr("scix.eval.real_data.hybrid_search", _StubHybridSearch(papers))

    conn = _mock_conn_for_entities({"2024A": [1], "2024C": [2]})
    ctx = RealEvalContext(conn=conn)
    ctx.embedding_cache["2024SEED::indus"] = [0.1]

    bibs, _ = static_filter_retrieve(ctx, stub_seed)
    assert bibs == ["2024A", "2024C"]


@pytest.mark.unit
def test_jit_rerank_lifts_overlapping_candidate(monkeypatch, stub_seed):
    papers = [
        {"bibcode": "2024A"},  # baseline rank 0, no overlap -> should fall
        {"bibcode": "2024B"},  # baseline rank 1, 1 overlap -> should rise
        {"bibcode": "2024C"},  # baseline rank 2, 2 overlaps -> should rise most
    ]
    monkeypatch.setattr("scix.eval.real_data.hybrid_search", _StubHybridSearch(papers))

    conn = _mock_conn_for_entities({
        "2024SEED": [10, 20],
        "2024A": [99],
        "2024B": [10],
        "2024C": [10, 20],
    })
    ctx = RealEvalContext(conn=conn)
    ctx.embedding_cache["2024SEED::indus"] = [0.1]

    bibs, _ = jit_rerank_retrieve(ctx, stub_seed)
    # C (2 overlaps) > B (1 overlap) > A (0 overlaps, preserves baseline)
    assert bibs == ["2024C", "2024B", "2024A"]


@pytest.mark.unit
def test_jit_rerank_noop_when_seed_has_no_entities(monkeypatch, stub_seed):
    papers = [
        {"bibcode": "2024A"},
        {"bibcode": "2024B"},
    ]
    monkeypatch.setattr("scix.eval.real_data.hybrid_search", _StubHybridSearch(papers))

    conn = _mock_conn_for_entities({})  # seed has no entities
    ctx = RealEvalContext(conn=conn)
    ctx.embedding_cache["2024SEED::indus"] = [0.1]

    bibs, _ = jit_rerank_retrieve(ctx, stub_seed)
    assert bibs == ["2024A", "2024B"]  # baseline order preserved


# ---------------------------------------------------------------------------
# M4.5 lane readers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hybrid_enrich_equals_context_entities_for():
    conn = _mock_conn_for_entities({"2024X": [7, 8]})
    ctx = RealEvalContext(conn=conn)
    assert hybrid_enrich_entities(ctx, "2024X") == frozenset({7, 8})


@pytest.mark.unit
def test_constants_are_sensible():
    assert BASELINE_TOP_N >= 20
    assert 0.0 < JIT_OVERLAP_BOOST <= 1.0


# ---------------------------------------------------------------------------
# Integration tests — require real DB, opt-in
# ---------------------------------------------------------------------------


REAL_DB_AVAILABLE = os.environ.get("SCIX_DSN") or os.path.exists("/var/run/postgresql")


@pytest.mark.integration
@pytest.mark.skipif(not REAL_DB_AVAILABLE, reason="requires prod-like DB")
def test_sample_seed_papers_returns_requested_count():
    """End-to-end sampling on prod — read-only."""
    from scix.db import get_connection
    from scix.eval.real_data import sample_seed_papers

    with get_connection() as conn:
        seeds = sample_seed_papers(conn, n_seeds=5, min_neighbors=10)
    assert len(seeds) >= 1
    for seed in seeds:
        assert seed.bibcode
        assert seed.n_neighbors >= 10
        assert seed.n_entities >= 1
