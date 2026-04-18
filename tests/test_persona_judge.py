"""Unit tests for the single-persona relevance judge.

Covers:
- JudgeScore / JudgeTriple DTOs
- Snippet construction honoring the licensing budget
- JSON response parser (strict + tolerant paths)
- Quadratic-weighted Cohen's kappa
- Spearman rank correlation
- Async dispatcher orchestration (batching, retry, semaphore)

All tests use a StubDispatcher — the real dispatcher shells out to
``claude -p`` and is exercised by the integration test gated behind
``--live``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from scix.eval.persona_judge import (
    DispatcherError,
    JudgeScore,
    JudgeTriple,
    PersonaJudge,
    StubDispatcher,
    build_snippet,
    parse_judge_response,
    quadratic_weighted_kappa,
    spearman_rho,
)


# ---------------------------------------------------------------------------
# Snippet construction
# ---------------------------------------------------------------------------


class TestBuildSnippet:
    def test_includes_title_abstract_and_500_chars_of_body(self) -> None:
        title = "A Deep Learning Approach to Stellar Classification"
        abstract = "We present a novel CNN for star/galaxy separation."
        body_marker = "Q" * 2000  # distinctive char not in title / abstract / labels
        snippet = build_snippet(title=title, abstract=abstract, body=body_marker)
        assert title in snippet
        assert abstract in snippet
        # body must be truncated to exactly 500 chars of the marker
        assert snippet.count("Q") == 500

    def test_handles_missing_body(self) -> None:
        snippet = build_snippet(title="T", abstract="A", body=None)
        assert "T" in snippet
        assert "A" in snippet

    def test_handles_missing_abstract_and_body(self) -> None:
        snippet = build_snippet(title="T", abstract=None, body=None)
        assert "T" in snippet

    def test_rejects_empty_title(self) -> None:
        with pytest.raises(ValueError):
            build_snippet(title="", abstract="A", body=None)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_parses_clean_json(self) -> None:
        raw = '{"score": 2, "reason": "Methodology matches."}'
        score = parse_judge_response(raw)
        assert score.score == 2
        assert score.reason == "Methodology matches."

    def test_parses_json_embedded_in_prose(self) -> None:
        raw = (
            'Here is my assessment:\n\n'
            '{"score": 3, "reason": "Strongly relevant — same method and subfield."}\n\n'
            'End of response.'
        )
        score = parse_judge_response(raw)
        assert score.score == 3

    def test_rejects_out_of_range_score(self) -> None:
        raw = '{"score": 5, "reason": "off-scale"}'
        with pytest.raises(ValueError):
            parse_judge_response(raw)

    def test_rejects_non_integer_score(self) -> None:
        raw = '{"score": "high", "reason": "off-scale"}'
        with pytest.raises(ValueError):
            parse_judge_response(raw)

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_response('{"score": 2}')

    def test_rejects_no_json(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_response("no json here")


# ---------------------------------------------------------------------------
# Metrics: quadratic-weighted kappa
# ---------------------------------------------------------------------------


class TestQuadraticWeightedKappa:
    def test_perfect_agreement_returns_one(self) -> None:
        human = [0, 1, 2, 3, 2, 1, 0]
        judge_scores = [0, 1, 2, 3, 2, 1, 0]
        assert quadratic_weighted_kappa(human, judge_scores) == pytest.approx(1.0, abs=1e-9)

    def test_off_by_one_is_mildly_penalized(self) -> None:
        human = [0, 1, 2, 3] * 5
        judge_scores = [1, 2, 3, 2] * 5  # each off by 1
        k = quadratic_weighted_kappa(human, judge_scores)
        # Off-by-one should yield positive kappa bounded above, not zero
        assert 0.0 < k < 1.0

    def test_max_disagreement_heavily_penalized(self) -> None:
        human = [0, 0, 3, 3]
        judge_scores = [3, 3, 0, 0]
        # Opposite extremes → very negative kappa
        assert quadratic_weighted_kappa(human, judge_scores) < 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([0, 1], [0, 1, 2])

    def test_empty_returns_zero(self) -> None:
        assert quadratic_weighted_kappa([], []) == 0.0

    def test_scale_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([0, 5], [0, 1])


# ---------------------------------------------------------------------------
# Metrics: Spearman rank correlation
# ---------------------------------------------------------------------------


class TestSpearmanRho:
    def test_perfect_positive_rank_returns_one(self) -> None:
        assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_rank_returns_minus_one(self) -> None:
        assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0, abs=1e-9)

    def test_handles_ties(self) -> None:
        # Both sequences have ties; Spearman should still be well-defined
        rho = spearman_rho([1, 2, 2, 3], [1, 2, 2, 3])
        assert rho == pytest.approx(1.0, abs=1e-9)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            spearman_rho([1, 2], [1, 2, 3])

    def test_empty_returns_zero(self) -> None:
        assert spearman_rho([], []) == 0.0


# ---------------------------------------------------------------------------
# StubDispatcher — verifies the protocol shape
# ---------------------------------------------------------------------------


class TestStubDispatcher:
    def test_returns_deterministic_scores_for_fixed_inputs(self) -> None:
        stub = StubDispatcher(fixed_score=2, reason="stub")
        triple = JudgeTriple(query="q", bibcode="2099X", snippet="Title\nAbstract")

        async def _run() -> JudgeScore:
            return await stub.judge(triple)

        result = asyncio.run(_run())
        assert result.score == 2
        assert result.reason == "stub"

    def test_stub_records_calls(self) -> None:
        stub = StubDispatcher(fixed_score=1)
        triples = [JudgeTriple(query="q", bibcode=f"B{i}", snippet="s") for i in range(3)]

        async def _run() -> list[JudgeScore]:
            return [await stub.judge(t) for t in triples]

        results = asyncio.run(_run())
        assert len(results) == 3
        assert len(stub.calls) == 3
        assert {c.bibcode for c in stub.calls} == {"B0", "B1", "B2"}


# ---------------------------------------------------------------------------
# PersonaJudge — orchestration (batching, retry, semaphore)
# ---------------------------------------------------------------------------


@dataclass
class _FlakyDispatcher:
    """Stub that fails the first ``fail_count`` calls for a given bibcode, then succeeds."""

    fail_count: int = 2
    _attempts: dict[str, int] = None

    def __post_init__(self) -> None:
        self._attempts = {}

    async def judge(self, triple: JudgeTriple) -> JudgeScore:
        self._attempts[triple.bibcode] = self._attempts.get(triple.bibcode, 0) + 1
        if self._attempts[triple.bibcode] <= self.fail_count:
            raise DispatcherError(f"transient failure #{self._attempts[triple.bibcode]}")
        return JudgeScore(score=1, reason=f"ok after {self._attempts[triple.bibcode]} attempts")


class TestPersonaJudgeOrchestration:
    def test_run_returns_one_score_per_triple(self) -> None:
        stub = StubDispatcher(fixed_score=2)
        judge = PersonaJudge(dispatcher=stub, max_concurrency=4, max_retries=0)
        triples = [JudgeTriple(query="q", bibcode=f"B{i}", snippet="s") for i in range(5)]

        results = asyncio.run(judge.run(triples))
        assert len(results) == 5
        # Preserves triple-score association via order
        for i, (triple, score) in enumerate(zip(triples, results)):
            assert score.score == 2
            assert score.triple is triple

    def test_retry_recovers_from_transient_failures(self) -> None:
        flaky = _FlakyDispatcher(fail_count=2)
        judge = PersonaJudge(
            dispatcher=flaky,
            max_concurrency=2,
            max_retries=3,
            backoff_base_s=0.0,  # disable sleep in tests
        )
        triples = [JudgeTriple(query="q", bibcode="B1", snippet="s")]

        results = asyncio.run(judge.run(triples))
        assert len(results) == 1
        assert results[0].score == 1

    def test_exceeded_retries_returns_error_score(self) -> None:
        always_fail = _FlakyDispatcher(fail_count=999)
        judge = PersonaJudge(
            dispatcher=always_fail,
            max_concurrency=2,
            max_retries=2,
            backoff_base_s=0.0,
        )
        triples = [JudgeTriple(query="q", bibcode="B1", snippet="s")]

        results = asyncio.run(judge.run(triples))
        assert len(results) == 1
        assert results[0].score == -1  # sentinel for failure
        assert "error" in results[0].reason.lower() or "failure" in results[0].reason.lower()

    def test_concurrency_is_bounded(self) -> None:
        """StubDispatcher with tracking — verify we never exceed max_concurrency in flight."""

        max_concurrency = 3
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        class _TrackingDispatcher:
            async def judge(self, triple: JudgeTriple) -> JudgeScore:
                nonlocal in_flight, peak
                async with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                async with lock:
                    in_flight -= 1
                return JudgeScore(score=0, reason="ok")

        judge = PersonaJudge(
            dispatcher=_TrackingDispatcher(),
            max_concurrency=max_concurrency,
            max_retries=0,
        )
        triples = [JudgeTriple(query="q", bibcode=f"B{i}", snippet="s") for i in range(10)]
        asyncio.run(judge.run(triples))
        assert peak <= max_concurrency

    def test_empty_input_returns_empty(self) -> None:
        stub = StubDispatcher(fixed_score=2)
        judge = PersonaJudge(dispatcher=stub, max_concurrency=4, max_retries=0)
        assert asyncio.run(judge.run([])) == []
