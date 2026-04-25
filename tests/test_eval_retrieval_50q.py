"""Tests for the 50-query retrieval evaluation driver.

Covers:
    - nDCG@10 against known toy rankings
    - MRR@10 against known toy rankings
    - Recall@k against known toy rankings
    - aggregate_metrics excludes None scores rather than averaging them
    - --dry-run produces a fixed-shape JSON output (AC3)

These tests are pure-Python: no DB connection, no model load, no torch
import. They exercise the metric primitives directly and the CLI via
``main(["--dry-run", ...])``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable so we can pull the driver as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_retrieval_50q as drv  # noqa: E402


# ---------------------------------------------------------------------------
# nDCG@10
# ---------------------------------------------------------------------------


class TestNDCGAt10:
    def test_perfect_single_gold_at_top(self) -> None:
        assert drv.ndcg_at_10(["A", "B", "C"], ["A"]) == pytest.approx(1.0)

    def test_single_gold_at_rank_2(self) -> None:
        # gold=["A"], retrieved=["B","A","C"] -> dcg=1/log2(3); idcg=1/log2(2)=1
        assert drv.ndcg_at_10(["B", "A", "C"], ["A"]) == pytest.approx(1.0 / math.log2(3))

    def test_zero_when_no_overlap(self) -> None:
        assert drv.ndcg_at_10(["X", "Y", "Z"], ["A"]) == 0.0

    def test_excluded_when_gold_empty(self) -> None:
        # Per spec: empty gold -> exclude (None) rather than score zero.
        assert drv.ndcg_at_10(["A", "B"], []) is None

    def test_relevant_at_rank_11_is_zero(self) -> None:
        retrieved = [f"X{i}" for i in range(10)] + ["A"]
        assert drv.ndcg_at_10(retrieved, ["A"]) == 0.0

    def test_two_gold_perfect_top_two(self) -> None:
        # Both gold at top -> perfect.
        assert drv.ndcg_at_10(["A", "B", "X"], ["A", "B"]) == pytest.approx(1.0)

    def test_two_gold_swapped_with_distractor(self) -> None:
        # gold={A,B}, retrieved=[A,X,B]
        # dcg  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # idcg = 1/log2(2) + 1/log2(3)
        retrieved = ["A", "X", "B"]
        gold = ["A", "B"]
        expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
        assert drv.ndcg_at_10(retrieved, gold) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# MRR@10
# ---------------------------------------------------------------------------


class TestMRRAt10:
    def test_first_position(self) -> None:
        assert drv.mrr_at_10(["A", "B", "C"], ["A"]) == 1.0

    def test_second_position(self) -> None:
        assert drv.mrr_at_10(["B", "A", "C"], ["A"]) == 0.5

    def test_no_match(self) -> None:
        assert drv.mrr_at_10(["B", "C", "D"], ["A"]) == 0.0

    def test_excluded_when_gold_empty(self) -> None:
        assert drv.mrr_at_10(["A", "B"], []) is None

    def test_only_top_10_counted(self) -> None:
        retrieved = [f"X{i}" for i in range(10)] + ["A"]
        assert drv.mrr_at_10(retrieved, ["A"]) == 0.0

    def test_first_relevant_wins_when_multi_gold(self) -> None:
        # gold={A,B}, retrieved=[X,A,B] -> 1/2 from A.
        assert drv.mrr_at_10(["X", "A", "B"], ["A", "B"]) == 0.5


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_full_recall(self) -> None:
        retrieved = ["A", "B"] + [f"X{i}" for i in range(48)]
        assert drv.recall_at_k(retrieved, ["A", "B"], 50) == 1.0

    def test_half_recall(self) -> None:
        retrieved = ["A"] + [f"X{i}" for i in range(49)]
        assert drv.recall_at_k(retrieved, ["A", "B"], 50) == 0.5

    def test_zero_recall(self) -> None:
        assert drv.recall_at_k(["X", "Y", "Z"], ["A", "B"], 50) == 0.0

    def test_excluded_when_gold_empty(self) -> None:
        assert drv.recall_at_k(["A", "B"], [], 50) is None

    def test_truncates_at_k(self) -> None:
        retrieved = [f"X{i}" for i in range(50)] + ["A"]
        assert drv.recall_at_k(retrieved, ["A"], 50) == 0.0


# ---------------------------------------------------------------------------
# Aggregator exclusion semantics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_excludes_none(self) -> None:
        per_query = [
            {"ndcg_at_10": 1.0, "mrr_at_10": 1.0, "recall_at_50": 1.0},
            {"ndcg_at_10": None, "mrr_at_10": None, "recall_at_50": None},
            {"ndcg_at_10": 0.0, "mrr_at_10": 0.0, "recall_at_50": 0.0},
        ]
        agg = drv.aggregate_metrics(per_query)
        assert agg["ndcg_at_10"] == pytest.approx(0.5)
        assert agg["mrr_at_10"] == pytest.approx(0.5)
        assert agg["recall_at_50"] == pytest.approx(0.5)
        assert agg["n_queries"] == 3
        assert agg["n_scored_ndcg"] == 2
        assert agg["n_scored_mrr"] == 2
        assert agg["n_scored_recall"] == 2

    def test_empty_yields_zero_not_nan(self) -> None:
        agg = drv.aggregate_metrics([])
        assert agg["ndcg_at_10"] == 0.0
        assert agg["n_queries"] == 0


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


class TestRRFFuse:
    def test_basic_intersection(self) -> None:
        # Both rankings rank A first and B second; A should win deterministically.
        fused = drv.rrf_fuse_bibcodes([["A", "B"], ["A", "B"]])
        assert fused[:2] == ["A", "B"]

    def test_disjoint_rankings(self) -> None:
        # Each list has unique items at rank 1; fused should put both at the
        # top (tied) with deterministic alphabetical tiebreak.
        fused = drv.rrf_fuse_bibcodes([["A"], ["B"]])
        assert set(fused[:2]) == {"A", "B"}
        # Tiebreak by ascending bibcode -> A before B.
        assert fused == ["A", "B"]


# ---------------------------------------------------------------------------
# CLI dry-run schema (AC3)
# ---------------------------------------------------------------------------


class TestDryRunOutputSchema:
    def test_dry_run_schema(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        rc = drv.main(
            [
                "--dry-run",
                "--modes",
                "baseline,section,fused",
                "--queries",
                str(_REPO_ROOT / "eval" / "retrieval_50q.jsonl"),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        payload = json.loads(out.read_text())
        # Top-level required keys
        assert payload["dry_run"] is True
        assert "modes" in payload
        # Required modes present and in correct shape (AC3)
        assert set(payload["modes"].keys()) == {"baseline", "section", "fused"}
        for mode_name, block in payload["modes"].items():
            assert "overall" in block, f"{mode_name} missing 'overall'"
            assert "by_bucket" in block, f"{mode_name} missing 'by_bucket'"
            for metric in ("ndcg_at_10", "mrr_at_10", "recall_at_50"):
                assert metric in block["overall"], f"{mode_name}.overall missing {metric}"
            # All four buckets present in by_bucket
            for bucket in ("title_matchable", "concept", "method", "author_specific"):
                assert bucket in block["by_bucket"], f"{mode_name} missing bucket {bucket}"
                for metric in ("ndcg_at_10", "mrr_at_10", "recall_at_50"):
                    assert metric in block["by_bucket"][bucket]

    def test_dry_run_baseline_only(self, tmp_path: Path) -> None:
        # AC5: --dry-run --modes baseline must exit 0 even with no data.
        out = tmp_path / "out.json"
        rc = drv.main(
            [
                "--dry-run",
                "--modes",
                "baseline",
                "--output",
                str(out),
                "--queries",
                str(_REPO_ROOT / "eval" / "retrieval_50q.jsonl"),
            ]
        )
        assert rc == 0
        payload = json.loads(out.read_text())
        assert list(payload["modes"].keys()) == ["baseline"]


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------


class TestLoadQueries:
    def test_real_gold_set_loads(self) -> None:
        path = _REPO_ROOT / "eval" / "retrieval_50q.jsonl"
        if not path.exists():
            pytest.skip(f"gold set {path} missing in this checkout")
        queries = drv.load_queries(path)
        assert len(queries) == 50
        # Buckets only ever from the spec set.
        valid_buckets = {"title_matchable", "concept", "method", "author_specific"}
        assert {q.bucket for q in queries}.issubset(valid_buckets)

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            json.dumps({"query": "x", "bucket": "concept", "discipline": "earth"}) + "\n"
        )
        with pytest.raises(ValueError, match="gold_bibcodes"):
            drv.load_queries(bad)

    def test_skips_blank_and_comment_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "g.jsonl"
        rows = [
            "",
            "# a comment",
            json.dumps(
                {
                    "query": "q",
                    "bucket": "concept",
                    "discipline": "astrophysics",
                    "gold_bibcodes": ["A"],
                }
            ),
        ]
        f.write_text("\n".join(rows) + "\n")
        queries = drv.load_queries(f)
        assert len(queries) == 1
        assert queries[0].gold_bibcodes == ("A",)


# ---------------------------------------------------------------------------
# score_query end-to-end
# ---------------------------------------------------------------------------


class TestScoreQuery:
    def test_full_pipeline(self) -> None:
        retrieved = ["A", "B", "C"] + [f"X{i}" for i in range(47)]
        scores = drv.score_query(retrieved, ["A"], k=10)
        assert scores["ndcg_at_10"] == pytest.approx(1.0)
        assert scores["mrr_at_10"] == 1.0
        assert scores["recall_at_50"] == 1.0

    def test_empty_gold_yields_none_metrics(self) -> None:
        scores = drv.score_query(["A", "B"], [], k=10)
        assert scores["ndcg_at_10"] is None
        assert scores["mrr_at_10"] is None
        assert scores["recall_at_50"] is None
