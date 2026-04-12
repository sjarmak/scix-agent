"""Tests for src/scix/query_expansion.py (PRD §S3)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scix import query_expansion as qx


def test_expand_deterministic_for_same_input() -> None:
    """AC3: identical inputs always produce identical output."""
    index = qx.build_fixture_index(n=10, seed=7)
    out1 = qx.expand("dark matter", k=5, index=index)
    out2 = qx.expand("dark matter", k=5, index=index)
    assert out1 == out2
    assert len(out1) == 5
    assert all(isinstance(x, int) for x in out1)


def test_expand_different_queries_differ() -> None:
    """Sanity: different queries should generally produce different rankings."""
    index = qx.build_fixture_index(n=50, seed=11)
    a = qx.expand("Hubble Space Telescope", k=10, index=index)
    b = qx.expand("quasar luminosity function", k=10, index=index)
    assert a != b


def test_expand_returns_ids_from_index() -> None:
    """Every returned id must be in the index."""
    index = qx.build_fixture_index(n=10, seed=3)
    out = qx.expand("galaxy cluster", k=5, index=index)
    valid = set(int(x) for x in index.ids.tolist())
    assert set(out).issubset(valid)


def test_expand_respects_k() -> None:
    index = qx.build_fixture_index(n=10, seed=3)
    assert len(qx.expand("x", k=3, index=index)) == 3
    assert len(qx.expand("x", k=10, index=index)) == 10
    assert len(qx.expand("x", k=20, index=index)) == 10  # clamped to size
    assert qx.expand("x", k=0, index=index) == []


def test_expand_empty_index_returns_empty() -> None:
    empty = qx.EntityIndex(
        ids=np.empty((0,), dtype=np.int64),
        vectors=np.empty((0, 8), dtype=np.float32),
    )
    assert qx.expand("anything", k=5, index=empty) == []


def test_expand_latency_under_20ms_on_100_vector_index() -> None:
    """AC3: latency < 20ms on a 100-vector fixture index.

    Warm numpy caches with one call, then time a second call. We give
    ourselves a generous ceiling — numpy on 100x64 should finish in
    single-digit microseconds on anything more modern than a toaster.
    """
    index = qx.build_fixture_index(n=100, seed=0)
    # Warm.
    qx.expand("warmup query", k=5, index=index)

    start = time.perf_counter()
    out = qx.expand("latency test query", k=5, index=index)
    elapsed = time.perf_counter() - start

    assert len(out) == 5
    assert elapsed < 0.020, f"expand() took {elapsed * 1000:.2f} ms, budget 20 ms"


def test_build_index_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        qx.EntityIndex(
            ids=np.array([1, 2, 3], dtype=np.int64),
            vectors=np.zeros((2, 4), dtype=np.float32),
        )


def test_expand_ordering_stable_on_ties() -> None:
    """When two entities have identical scores, ties must break by id."""
    # Two identical vectors -> identical cosine with any query -> tie.
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = qx.build_index([10, 7, 22], vectors)
    # Whichever query we pass, the first two entities tie; expect id 7 before 10.
    out = qx.expand("anything at all", k=3, index=index)
    # The tied pair must come out in ascending-id order (7, then 10).
    pos_7 = out.index(7)
    pos_10 = out.index(10)
    assert pos_7 < pos_10
