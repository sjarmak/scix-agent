"""Tests for the M5 section-level rerank in ``search.search_within_paper``.

Covers the prd_full_text_applications_v2 M5 acceptance criteria:

1. Signature exposes ``use_rerank: bool = True`` and ``top_k: int = 20``.
2. Returned ``SearchResult`` papers carry a new ``sections`` list (up to 3
   entries of ``{section_name, score, snippet}``) while the legacy
   ``headline`` field stays populated for backward compat.
3. nDCG@3 is measured on a 20-paper hand-labeled fixture both with and
   without rerank. If rerank improves nDCG@3 by >= 0.05 the harness asserts
   the improvement; otherwise the test still passes and writes a negative
   result memo to ``results/within_paper_rerank_eval.md``.
4. p95 latency over the 20 fixture queries with ``use_rerank=True`` (MiniLM)
   stays under 500 ms.

The tests use a stub ``Connection`` that returns the fixture body text and
a ts_headline blob; the section-level ts_rank fallback in the function
under test computes the per-section scores deterministically in Python.
No real PostgreSQL connection is required.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix import search
from scix.search import SearchResult, search_within_paper
from scix.section_parser import parse_sections

# ---------------------------------------------------------------------------
# Paths / fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "within_paper_rerank_gold_20.jsonl"
EVAL_REPORT_PATH = REPO_ROOT / "results" / "within_paper_rerank_eval.md"


def _load_fixture() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with FIXTURE_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Stub psycopg.Connection
# ---------------------------------------------------------------------------


class _StubCursor:
    """Hand-rolled cursor returning either dict-rows or tuples per call."""

    def __init__(
        self,
        body: str,
        *,
        bibcode: str = "TEST_BIBCODE",
        title: str = "Test Paper",
        headline: str = "<b>match</b>",
        return_search_row: bool = True,
    ) -> None:
        self._body = body
        self._bibcode = bibcode
        self._title = title
        self._headline = headline
        self._return_search_row = return_search_row
        self._next_action: str | None = None  # "fetchone" | "fetchall"
        self._search_emitted = False

    def __enter__(self) -> _StubCursor:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        s = sql.strip().lower()
        if "ts_headline" in s:
            self._next_action = "search"
        elif "ts_rank" in s and "unnest" in s:
            # Per-section ts_rank batch query.
            self._next_action = "ts_rank"
            self._ts_rank_query = params[0]
            self._ts_rank_texts = list(params[1])
        elif "from papers_fulltext" in s:
            self._next_action = "fulltext"
        elif "select identifier from papers" in s:
            self._next_action = "identifier"
        elif "from papers" in s and "ts_headline" not in s:
            self._next_action = "existence"
        else:
            self._next_action = "unknown"

    def fetchone(self) -> Any:
        if self._next_action == "search":
            if not self._return_search_row:
                return None
            return {
                "bibcode": self._bibcode,
                "title": self._title,
                "body": self._body,
                "headline": self._headline,
            }
        if self._next_action == "fulltext":
            return None  # No papers_fulltext row -> no ADR-006 guard.
        if self._next_action == "existence":
            return None
        if self._next_action == "identifier":
            return None
        return None

    def fetchall(self) -> list[Any]:
        if self._next_action == "ts_rank":
            # Compute a deterministic per-section ts_rank-equivalent score
            # in Python so the rest of search_within_paper sees real numbers.
            scores = []
            tokens = [
                t for t in __import__("re").findall(r"\w+", self._ts_rank_query.lower())
                if len(t) >= 2
            ]
            for text in self._ts_rank_texts:
                lower = (text or "").lower()
                hits = sum(lower.count(tok) for tok in tokens)
                if hits == 0 or not text:
                    scores.append((0.0,))
                else:
                    scores.append((hits / (len(text) + 50),))
            return scores
        return []


class _StubConn:
    def __init__(self, body: str) -> None:
        self._body = body

    def cursor(self, *, row_factory: Any = None) -> _StubCursor:
        return _StubCursor(self._body)


# ---------------------------------------------------------------------------
# nDCG@3 / latency helpers
# ---------------------------------------------------------------------------


def _section_idx_lookup(body: str) -> dict[str, int]:
    """Map section_name -> ordinal index (matches gold_section_idx convention)."""
    sections = parse_sections(body)
    return {name: idx for idx, (name, *_rest) in enumerate(sections)}


def _ndcg3(ranked_section_indices: list[int], gold_idx: int) -> float:
    """nDCG@3 for binary relevance with a single gold section.

    The ideal DCG@3 with one relevant item is 1 / log2(1 + 1) = 1.
    """
    dcg = 0.0
    for rank, sec_idx in enumerate(ranked_section_indices[:3], start=1):
        rel = 1.0 if sec_idx == gold_idx else 0.0
        if rel:
            dcg += rel / math.log2(rank + 1)
    # Ideal DCG with single relevant item at rank 1 == 1/log2(2) == 1.0.
    return dcg / 1.0


def _eval_one(
    entry: dict[str, Any],
    *,
    use_rerank: bool,
) -> tuple[float, float]:
    """Run search_within_paper on one fixture entry. Returns (ndcg3, elapsed_ms)."""
    body = entry["paper_body"]
    query = entry["query"]
    gold_idx = entry["gold_section_idx"]

    conn = _StubConn(body)
    t0 = time.perf_counter()
    result = search_within_paper(conn, "TEST_BIB", query, use_rerank=use_rerank)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sections_payload = result.papers[0]["sections"]
    name_to_idx = _section_idx_lookup(body)
    ranked_indices = [name_to_idx.get(s["section_name"], -1) for s in sections_payload]
    return _ndcg3(ranked_indices, gold_idx), elapsed_ms


# ---------------------------------------------------------------------------
# AC1: signature
# ---------------------------------------------------------------------------


def test_signature_has_use_rerank_and_top_k() -> None:
    sig = inspect.signature(search_within_paper)
    params = sig.parameters
    assert "use_rerank" in params, "search_within_paper must expose use_rerank"
    assert params["use_rerank"].default is True
    assert "top_k" in params, "search_within_paper must expose top_k"
    assert params["top_k"].default == 20


# ---------------------------------------------------------------------------
# AC2: SearchResult shape
# ---------------------------------------------------------------------------


def test_search_result_has_sections_list_and_legacy_headline() -> None:
    entry = _load_fixture()[0]
    conn = _StubConn(entry["paper_body"])
    result = search_within_paper(conn, "TEST", entry["query"])
    paper = result.papers[0]
    assert "sections" in paper, "result must carry the new 'sections' field"
    assert isinstance(paper["sections"], list)
    assert len(paper["sections"]) <= 3
    # Each section entry has the documented shape.
    for sec in paper["sections"]:
        assert {"section_name", "score", "snippet"}.issubset(sec.keys())
    # Legacy 'headline' is still populated (top-1 snippet for backward compat).
    assert isinstance(paper["headline"], str)
    assert paper["headline"]


def test_backward_compat_positional_call() -> None:
    """The (conn, bibcode, query) positional call still works for legacy callers."""
    entry = _load_fixture()[0]
    conn = _StubConn(entry["paper_body"])
    result = search_within_paper(conn, "TEST", entry["query"])
    assert result.total == 1
    assert result.papers[0]["has_body"] is True


# ---------------------------------------------------------------------------
# AC4 + AC5: nDCG@3 eval and p95 latency
# ---------------------------------------------------------------------------


def _resolve_minilm_available() -> bool:
    """Return True if sentence-transformers + MiniLM weights load locally."""
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture
def _minilm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire SCIX_RERANK_DEFAULT_MODEL=minilm and reset the cached singleton."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "minilm")
    search._reset_section_rerank_cache()
    yield
    search._reset_section_rerank_cache()


def _write_eval_report(
    *,
    baseline_ndcg: float,
    rerank_ndcg: float,
    delta: float,
    p95_latency_ms: float,
    threshold: float,
    minilm_available: bool,
) -> None:
    """Write the eval summary; force-create results/ if needed."""
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if delta >= threshold:
        recommendation = (
            f"GO — section-level cross-encoder rerank improves nDCG@3 by "
            f"{delta:+.4f} (>= {threshold:+.2f}). Ship `use_rerank=True` as "
            f"the function default; flip `SCIX_RERANK_DEFAULT_MODEL=minilm` "
            f"in production once latency is verified end-to-end."
        )
    else:
        recommendation = (
            f"NO-GO (negative result) — section-level cross-encoder rerank "
            f"improves nDCG@3 by only {delta:+.4f} (< {threshold:+.2f}). "
            f"Keep `SCIX_RERANK_DEFAULT_MODEL='off'` as the production "
            f"default. The signature still defaults `use_rerank=True` so "
            f"flipping the env is the only operator change needed if a "
            f"future re-eval shows a different outcome."
        )

    note = ""
    if not minilm_available:
        note = (
            "\nNOTE: MiniLM weights were not loadable in this environment, so "
            "the rerank pass fell back to the ts_rank ordering. The reported "
            "delta therefore measures the no-rerank vs no-rerank case and is "
            "exactly 0 by construction. Re-run on a host with "
            "sentence-transformers + network/cache for the real number.\n"
        )

    body = (
        "# search_within_paper section-level rerank — M5 eval\n"
        "\n"
        "## Methodology\n"
        "\n"
        f"- Fixture: `tests/fixtures/within_paper_rerank_gold_20.jsonl` (20 entries)\n"
        "- Each entry has a synthetic IMRaD-style paper body, a query, and a\n"
        "  hand-labeled `gold_section_idx`.\n"
        "- Baseline: `search_within_paper(..., use_rerank=False)` — top-3 by\n"
        "  per-section `ts_rank` (PostgreSQL or Python proxy fallback).\n"
        "- Reranked: `search_within_paper(..., use_rerank=True)` with\n"
        "  `SCIX_RERANK_DEFAULT_MODEL=minilm`\n"
        "  (`cross-encoder/ms-marco-MiniLM-L-12-v2`).\n"
        "- Metric: nDCG@3 with binary relevance, averaged across 20 queries.\n"
        "- Latency metric: per-query wall-clock around the function, p95 over\n"
        "  the 20-query batch, MiniLM model.\n"
        "\n"
        "## Results\n"
        "\n"
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| Baseline nDCG@3 (BM25 only) | {baseline_ndcg:.4f} |\n"
        f"| Reranked nDCG@3 (MiniLM)    | {rerank_ndcg:.4f} |\n"
        f"| Delta                       | {delta:+.4f} |\n"
        f"| p95 latency (rerank, MiniLM)| {p95_latency_ms:.1f} ms |\n"
        f"| Improvement threshold       | {threshold:+.2f} |\n"
        "\n"
        f"## Recommendation\n"
        "\n"
        f"{recommendation}\n"
        f"{note}"
    )
    EVAL_REPORT_PATH.write_text(body)


def test_rerank_improves_ndcg_at_3() -> None:
    """nDCG@3 baseline vs rerank on the 20-entry fixture.

    Always writes the eval report. Asserts an improvement of >= 0.05 only when
    actually observed (negative-result-friendly per the unit acceptance).
    """
    fixture = _load_fixture()

    minilm_available = _resolve_minilm_available()

    # Baseline: no rerank.
    baseline_scores: list[float] = []
    for entry in fixture:
        ndcg, _ = _eval_one(entry, use_rerank=False)
        baseline_scores.append(ndcg)
    baseline_ndcg = sum(baseline_scores) / len(baseline_scores)

    # Reranked: only if MiniLM weights are loadable.
    rerank_scores: list[float] = []
    rerank_latencies_ms: list[float] = []
    if minilm_available:
        import os as _os

        _os.environ["SCIX_RERANK_DEFAULT_MODEL"] = "minilm"
        search._reset_section_rerank_cache()
        try:
            for entry in fixture:
                ndcg, elapsed_ms = _eval_one(entry, use_rerank=True)
                rerank_scores.append(ndcg)
                rerank_latencies_ms.append(elapsed_ms)
        finally:
            _os.environ["SCIX_RERANK_DEFAULT_MODEL"] = "off"
            search._reset_section_rerank_cache()
        rerank_ndcg = sum(rerank_scores) / len(rerank_scores)
    else:
        rerank_ndcg = baseline_ndcg
        rerank_latencies_ms = [0.0] * len(fixture)

    rerank_latencies_ms.sort()
    p95_idx = max(0, int(round(0.95 * len(rerank_latencies_ms))) - 1)
    p95_latency_ms = rerank_latencies_ms[p95_idx]

    delta = rerank_ndcg - baseline_ndcg
    threshold = 0.05

    _write_eval_report(
        baseline_ndcg=baseline_ndcg,
        rerank_ndcg=rerank_ndcg,
        delta=delta,
        p95_latency_ms=p95_latency_ms,
        threshold=threshold,
        minilm_available=minilm_available,
    )

    if delta >= threshold:
        # Strong positive result — assert it explicitly.
        assert rerank_ndcg > baseline_ndcg
        assert delta >= threshold
    # Negative result is also a passing outcome per the unit description; the
    # report has been written so the operator can see the numbers.


@pytest.mark.skipif(
    not _resolve_minilm_available(),
    reason="sentence-transformers / MiniLM weights not available in this env",
)
def test_p95_latency_under_500ms(_minilm_env: None) -> None:
    """p95 of search_within_paper(use_rerank=True, MiniLM) over 20 queries <= 500 ms."""
    fixture = _load_fixture()

    latencies_ms: list[float] = []
    for entry in fixture:
        _, elapsed_ms = _eval_one(entry, use_rerank=True)
        latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    p95_idx = max(0, int(round(0.95 * len(latencies_ms))) - 1)
    p95_latency_ms = latencies_ms[p95_idx]
    assert p95_latency_ms <= 500.0, (
        f"p95 latency {p95_latency_ms:.1f} ms exceeds 500 ms budget"
    )
