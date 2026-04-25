"""Unit tests for :class:`scix.search.CrossEncoderReranker`.

These tests exercise both supported model names
(``cross-encoder/ms-marco-MiniLM-L-12-v2`` and ``BAAI/bge-reranker-large``)
plus the edge cases of the ``__call__`` contract: empty inputs, single
candidates, full reshuffles, and ``top_n`` truncation.

All ``CrossEncoder`` constructions and ``predict`` calls are mocked, so
the tests never touch the network and never download model weights.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix.search import (
    BGE_RERANKER_LARGE_LOCAL_DIR,
    BGE_RERANKER_LARGE_SHA,
    CrossEncoderReranker,
)

# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

MODEL_NAMES: tuple[str, ...] = (
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-large",
)


def _make_papers(n: int) -> list[dict[str, Any]]:
    """Build n synthetic papers with stable bibcodes/titles for ordering checks."""
    return [
        {
            "bibcode": f"paper-{i:04d}",
            "title": f"Title {i}",
            "abstract_snippet": f"Abstract snippet {i}",
        }
        for i in range(n)
    ]


@pytest.fixture
def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a stub ``sentence_transformers`` module exposing ``CrossEncoder``.

    Returns the ``CrossEncoder`` MagicMock so individual tests can assert on
    construction args and configure ``predict`` return values via
    ``stub_cross_encoder.return_value.predict.return_value``.
    """
    fake_module = types.ModuleType("sentence_transformers")
    cross_encoder_cls = MagicMock(name="CrossEncoder")
    # Default predict: zeros — concrete tests override per-case.
    cross_encoder_cls.return_value.predict.return_value = []
    fake_module.CrossEncoder = cross_encoder_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return cross_encoder_cls


# ---------------------------------------------------------------------------
# (a) Construction with each of the two supported model names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_construction_with_supported_models(
    model_name: str,
    stub_cross_encoder: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each supported model name constructs cleanly and lazy-loads on first call.

    Also verifies the ``BAAI/bge-reranker-large`` path passes the pinned
    revision SHA when no local snapshot is present.
    """
    # Force the local-cache check to miss so the HF Hub fallback path runs.
    monkeypatch.setattr("scix.search.os.path.isdir", lambda _: False)

    reranker = CrossEncoderReranker(model_name=model_name)
    assert reranker._model_name == model_name
    assert reranker._model is None  # lazy

    stub_cross_encoder.return_value.predict.return_value = [0.1]
    papers = _make_papers(1)
    reranker("query", papers)

    assert stub_cross_encoder.call_count == 1
    args, kwargs = stub_cross_encoder.call_args
    if model_name == "BAAI/bge-reranker-large":
        # HF Hub identifier + pinned revision.
        assert args[0] == "BAAI/bge-reranker-large"
        assert kwargs.get("revision") == BGE_RERANKER_LARGE_SHA
    else:
        assert args[0] == model_name
        assert "revision" not in kwargs


def test_bge_prefers_local_snapshot_when_present(
    stub_cross_encoder: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``models/bge-reranker-large/`` exists, load from disk (no revision)."""
    monkeypatch.setattr(
        "scix.search.os.path.isdir",
        lambda path: path == BGE_RERANKER_LARGE_LOCAL_DIR,
    )
    stub_cross_encoder.return_value.predict.return_value = [0.5]

    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-large")
    reranker("query", _make_papers(1))

    args, kwargs = stub_cross_encoder.call_args
    assert args[0] == BGE_RERANKER_LARGE_LOCAL_DIR
    assert "revision" not in kwargs


def test_sha_constant_is_sha_like() -> None:
    """The pinned SHA must be a 40-char lowercase hex digest."""
    assert len(BGE_RERANKER_LARGE_SHA) == 40
    assert all(c in "0123456789abcdef" for c in BGE_RERANKER_LARGE_SHA)


# ---------------------------------------------------------------------------
# (b) Empty input list
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list_without_loading_model(
    stub_cross_encoder: MagicMock,
) -> None:
    """Empty papers must short-circuit — no model load, no predict call."""
    reranker = CrossEncoderReranker()
    out = reranker("query", [])

    assert out == []
    # Model should not have been instantiated for an empty batch.
    assert stub_cross_encoder.call_count == 0
    assert reranker._model is None


# ---------------------------------------------------------------------------
# (c) Single candidate
# ---------------------------------------------------------------------------


def test_single_candidate_attaches_score_and_returns_one(
    stub_cross_encoder: MagicMock,
) -> None:
    """A single candidate must come back annotated with rerank_score."""
    stub_cross_encoder.return_value.predict.return_value = [0.42]

    reranker = CrossEncoderReranker()
    papers = _make_papers(1)
    out = reranker("query", papers)

    assert len(out) == 1
    assert out[0]["bibcode"] == "paper-0000"
    assert out[0]["rerank_score"] == pytest.approx(0.42)
    # Original paper dict must not be mutated (immutability invariant).
    assert "rerank_score" not in papers[0]


# ---------------------------------------------------------------------------
# (d) Top-20 reshuffle
# ---------------------------------------------------------------------------


def test_top_20_reshuffle_orders_by_score_desc(
    stub_cross_encoder: MagicMock,
) -> None:
    """20 inputs, scores in ascending order — output must reverse them."""
    n = 20
    # Ascending scores [0.0, 0.05, ..., 0.95] — best is the last input.
    scores = [round(i * 0.05, 4) for i in range(n)]
    stub_cross_encoder.return_value.predict.return_value = scores

    reranker = CrossEncoderReranker()
    papers = _make_papers(n)
    out = reranker("query", papers)

    assert len(out) == n
    # Highest score first, lowest last.
    assert out[0]["bibcode"] == f"paper-{n - 1:04d}"
    assert out[-1]["bibcode"] == "paper-0000"
    # Scores are monotonically non-increasing.
    out_scores = [r["rerank_score"] for r in out]
    assert out_scores == sorted(out_scores, reverse=True)


# ---------------------------------------------------------------------------
# (e) top_n truncation
# ---------------------------------------------------------------------------


def test_top_n_truncates_to_requested_size(
    stub_cross_encoder: MagicMock,
) -> None:
    """top_n=5 over 20 inputs returns the 5 highest-scored papers."""
    n = 20
    scores = [float(i) for i in range(n)]  # strictly increasing
    stub_cross_encoder.return_value.predict.return_value = scores

    reranker = CrossEncoderReranker()
    out = reranker("query", _make_papers(n), top_n=5)

    assert len(out) == 5
    expected = [f"paper-{i:04d}" for i in range(n - 1, n - 6, -1)]
    assert [r["bibcode"] for r in out] == expected
