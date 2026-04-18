"""Tests for the ``scripts/judge_triples.py`` CLI orchestrator.

All tests stub the dispatcher — no real subagent / subprocess invocation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import judge_triples  # noqa: E402
from scix.eval.persona_judge import JudgeScore, JudgeTriple, StubDispatcher  # noqa: E402


def _write_triples_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestReadTriples:
    def test_reads_jsonl_into_triples(self, tmp_path: Path) -> None:
        p = tmp_path / "triples.jsonl"
        _write_triples_jsonl(
            p,
            [
                {"query": "q1", "bibcode": "B1", "paper_snippet": "s1"},
                {"query": "q2", "bibcode": "B2", "paper_snippet": "s2"},
            ],
        )
        triples = judge_triples.read_triples(p)
        assert len(triples) == 2
        assert triples[0].query == "q1"
        assert triples[0].bibcode == "B1"
        assert triples[0].snippet == "s1"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "triples.jsonl"
        content = (
            json.dumps({"query": "q", "bibcode": "B1", "paper_snippet": "s"})
            + "\n\n  \n"
            + json.dumps({"query": "q", "bibcode": "B2", "paper_snippet": "s"})
            + "\n"
        )
        p.write_text(content, encoding="utf-8")
        triples = judge_triples.read_triples(p)
        assert len(triples) == 2

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "triples.jsonl"
        _write_triples_jsonl(p, [{"query": "q", "bibcode": "B1"}])
        with pytest.raises(ValueError):
            judge_triples.read_triples(p)


class TestWriteScores:
    def test_writes_one_line_per_score(self, tmp_path: Path) -> None:
        out = tmp_path / "scores.jsonl"
        triples = [
            JudgeTriple(query="q1", bibcode="B1", snippet="s"),
            JudgeTriple(query="q2", bibcode="B2", snippet="s"),
        ]
        scores = [
            JudgeScore(score=2, reason="relevant", triple=triples[0]),
            JudgeScore(score=0, reason="off-topic", triple=triples[1]),
        ]
        judge_triples.write_scores(out, scores)

        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert rec0["query"] == "q1"
        assert rec0["bibcode"] == "B1"
        assert rec0["score"] == 2
        assert rec0["reason"] == "relevant"


@pytest.mark.skipif(
    not __import__("os").environ.get("SCIX_JUDGE_LIVE"),
    reason="live test — requires SCIX_JUDGE_LIVE=1 and working claude CLI OAuth",
)
class TestLiveIntegration:
    def test_five_triple_end_to_end(self, tmp_path: Path) -> None:
        """5-triple live run — gated behind SCIX_JUDGE_LIVE=1 env var.

        Exercises ClaudeSubprocessDispatcher. Will actually dispatch to the
        Claude CLI, so we keep it tiny (5 triples) and skip in normal CI.
        """
        from scix.eval.persona_judge import ClaudeSubprocessDispatcher

        triples_path = tmp_path / "in.jsonl"
        out_path = tmp_path / "out.jsonl"
        _write_triples_jsonl(
            triples_path,
            [
                {
                    "query": "transformer models for protein structure",
                    "bibcode": f"LIVE_{i:02d}",
                    "paper_snippet": (
                        f"Title: Test paper {i}\n\nAbstract: A paper about "
                        f"{'protein folding' if i % 2 == 0 else 'coffee brewing'}."
                    ),
                }
                for i in range(5)
            ],
        )

        judge_triples.run(
            input_path=triples_path,
            output_path=out_path,
            dispatcher=ClaudeSubprocessDispatcher(),
            max_concurrency=2,
            max_retries=1,
        )
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line in lines:
            rec = json.loads(line)
            # score must be 0-3 or the error sentinel -1
            assert rec["score"] in {-1, 0, 1, 2, 3}


class TestEndToEnd:
    def test_run_with_stub_dispatcher(self, tmp_path: Path) -> None:
        triples_path = tmp_path / "in.jsonl"
        out_path = tmp_path / "out.jsonl"
        _write_triples_jsonl(
            triples_path,
            [
                {"query": "q", "bibcode": f"B{i}", "paper_snippet": "s"} for i in range(5)
            ],
        )

        dispatcher = StubDispatcher(fixed_score=1, reason="stub")
        judge_triples.run(
            input_path=triples_path,
            output_path=out_path,
            dispatcher=dispatcher,
            max_concurrency=2,
            max_retries=0,
        )

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line in lines:
            rec = json.loads(line)
            assert rec["score"] == 1
            assert rec["reason"] == "stub"
