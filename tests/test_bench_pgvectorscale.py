"""Unit tests for scripts/bench_pgvectorscale.py.

Covers the acceptance criteria:
  (a) production-DSN refusal
  (b) Wilcoxon + Bonferroni logic on synthetic input
  (c) nDCG/Recall/MRR implementations correct on hand-computed fixtures
  (d) argparse wiring (required flags present in --help)
  (e) schema-valid dry-run output
  (f) query loader reads unique seed bibcodes from eval JSON
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import bench_pgvectorscale as mod


# ---------------------------------------------------------------------------
# (a) Production-DSN refusal
# ---------------------------------------------------------------------------


class TestAssertPilotDsn:
    def test_refuses_production_dsn_kv(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            mod.assert_pilot_dsn("dbname=scix")
        assert "production" in str(excinfo.value).lower()

    def test_refuses_production_dsn_uri(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            mod.assert_pilot_dsn("postgresql://user@localhost/scix")
        assert "production" in str(excinfo.value).lower()

    def test_refuses_empty_dsn(self) -> None:
        with pytest.raises(ValueError):
            mod.assert_pilot_dsn("")
        with pytest.raises(ValueError):
            mod.assert_pilot_dsn(None)

    def test_allows_pilot_dsn(self) -> None:
        mod.assert_pilot_dsn("dbname=scix_pilot")
        mod.assert_pilot_dsn("dbname=scix_pgvs_pilot")
        mod.assert_pilot_dsn("dbname=scix_test")

    def test_main_refuses_production_dsn(self) -> None:
        rc = mod.main(["--dsn", "dbname=scix", "--dry-run"])
        assert rc == 2

    def test_main_requires_dsn(self) -> None:
        rc = mod.main(["--dry-run"])
        assert rc == 2


# ---------------------------------------------------------------------------
# (c) Metric correctness on hand-computed fixtures
# ---------------------------------------------------------------------------


class TestMetrics:
    """All fixtures are hand-computed and verified below."""

    def test_ndcg_empty_relevant(self) -> None:
        assert mod.ndcg_at_10(["a", "b"], set()) == 0.0

    def test_ndcg_perfect_ranking(self) -> None:
        # 3 relevant docs at positions 1,2,3 → actual DCG == ideal DCG → nDCG = 1.
        ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        relevant = {"a", "b", "c"}
        assert mod.ndcg_at_10(ranked, relevant) == pytest.approx(1.0, abs=1e-9)

    def test_ndcg_reversed_ranking(self) -> None:
        # 1 relevant doc at position 1 → actual DCG=1, ideal DCG=1 → nDCG=1.
        # Move the relevant doc to position 10 → actual DCG=1/log2(11) ≈ 0.2890.
        import math as _m

        ranked = [f"x{i}" for i in range(9)] + ["rel"]
        relevant = {"rel"}
        expected = (1 / _m.log2(10 + 1)) / 1.0
        assert mod.ndcg_at_10(ranked, relevant) == pytest.approx(expected, abs=1e-9)

    def test_ndcg_no_relevant_in_top_k(self) -> None:
        ranked = [f"x{i}" for i in range(10)]
        relevant = {"other_not_in_topk"}
        assert mod.ndcg_at_10(ranked, relevant) == 0.0

    def test_recall_at_k_basic(self) -> None:
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "z"}
        # top-5 has 2 of 3 relevant → recall = 2/3.
        assert mod.recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)

    def test_recall_at_k_zero_when_k_zero(self) -> None:
        # With k=0 we retrieve nothing.
        assert mod.recall_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_recall_empty_relevant(self) -> None:
        assert mod.recall_at_k(["a"], set(), 10) == 0.0

    def test_recall_at_10_vs_20(self) -> None:
        # Put relevant at position 15; only recall_20 sees it.
        ranked = [f"x{i}" for i in range(14)] + ["rel"] + ["y1", "y2", "y3", "y4", "y5"]
        relevant = {"rel"}
        assert mod.recall_at_k(ranked, relevant, 10) == 0.0
        assert mod.recall_at_k(ranked, relevant, 20) == 1.0

    def test_mrr_first_match_at_rank_3(self) -> None:
        ranked = ["a", "b", "c"]
        relevant = {"c"}
        assert mod.mrr(ranked, relevant) == pytest.approx(1 / 3)

    def test_mrr_no_match(self) -> None:
        assert mod.mrr(["a", "b"], {"z"}) == 0.0

    def test_mrr_empty_relevant(self) -> None:
        assert mod.mrr(["a"], set()) == 0.0


class TestPercentile:
    def test_p50_odd_length(self) -> None:
        assert mod.percentile([10, 20, 30, 40, 50], 50) == pytest.approx(30.0)

    def test_p50_even_length(self) -> None:
        # Linear interpolation between 2.0 and 3.0 → 2.5 at pct=50.
        assert mod.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p95_picks_near_top(self) -> None:
        vals = list(range(1, 101))  # 1..100
        p95 = mod.percentile(vals, 95)
        assert 95.0 <= p95 <= 96.0

    def test_empty(self) -> None:
        assert mod.percentile([], 50) == 0.0

    def test_single(self) -> None:
        assert mod.percentile([42.0], 95) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# (b) Wilcoxon + Bonferroni logic
# ---------------------------------------------------------------------------


class TestWilcoxonBonferroni:
    def test_clearly_different_paired_arrays_significant(self) -> None:
        # Variant strictly better than baseline on every one of 20 pairs.
        variant = [0.60 + 0.01 * i for i in range(20)]
        baseline = [0.30 + 0.01 * i for i in range(20)]
        out = mod.wilcoxon_compare(variant, baseline)
        assert out["n_pairs"] == 20
        assert out["p_value"] is not None
        assert out["p_value"] < 0.05
        # Mean diff should be ~+0.30
        assert out["mean_diff"] == pytest.approx(0.30, abs=1e-6)

    def test_bonferroni_caps_at_one(self) -> None:
        assert mod.bonferroni_adjust(0.9, 3) == 1.0

    def test_bonferroni_standard_case(self) -> None:
        # p=0.04, n=3 → adjusted=0.12
        assert mod.bonferroni_adjust(0.04, 3) == pytest.approx(0.12, abs=1e-9)

    def test_bonferroni_none_passthrough(self) -> None:
        assert mod.bonferroni_adjust(None, 3) is None

    def test_bonferroni_pushes_above_threshold(self) -> None:
        # A p-value at 0.02 with 3 comparisons → adjusted 0.06 (> 0.05).
        p_raw = 0.02
        adj = mod.bonferroni_adjust(p_raw, 3)
        assert p_raw < 0.05
        assert adj is not None and adj > 0.05

    def test_wilcoxon_all_zero_diffs(self) -> None:
        out = mod.wilcoxon_compare([0.5] * 10, [0.5] * 10)
        assert out["p_value"] == 1.0
        assert out["statistic"] == 0.0

    def test_wilcoxon_small_sample(self) -> None:
        out = mod.wilcoxon_compare([0.1, 0.2], [0.3, 0.1])
        assert out["n_pairs"] == 2
        assert out["p_value"] is None  # too small

    def test_wilcoxon_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            mod.wilcoxon_compare([0.1, 0.2], [0.3])

    def test_wilcoxon_empty(self) -> None:
        out = mod.wilcoxon_compare([], [])
        assert out["p_value"] is None
        assert out["n_pairs"] == 0


# ---------------------------------------------------------------------------
# Query loader
# ---------------------------------------------------------------------------


class TestLoadQueries:
    def test_loads_unique_ordered(self, tmp_path: Path) -> None:
        eval_json = {
            "per_query": [
                {"seed_bibcode": "A", "method": "x"},
                {"seed_bibcode": "B", "method": "x"},
                {"seed_bibcode": "A", "method": "y"},  # duplicate
                {"seed_bibcode": "C", "method": "x"},
            ]
        }
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(eval_json))
        assert mod.load_queries_from_eval(path) == ["A", "B", "C"]

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.json"
        path.write_text(json.dumps({}))
        assert mod.load_queries_from_eval(path) == []


# ---------------------------------------------------------------------------
# Index map parsing
# ---------------------------------------------------------------------------


class TestIndexMap:
    def test_default_used_when_none(self) -> None:
        out = mod._parse_index_map(None, mod.DEFAULT_INDEX_MAP)
        assert out == mod.DEFAULT_INDEX_MAP
        # Returned dict is a copy — mutations don't leak.
        out["hnsw"] = "mutated"
        assert mod.DEFAULT_INDEX_MAP["hnsw"] == "idx_hnsw_baseline_indus"

    def test_override_merges_with_defaults(self) -> None:
        out = mod._parse_index_map("hnsw=my_hnsw", mod.DEFAULT_INDEX_MAP)
        assert out["hnsw"] == "my_hnsw"
        assert out["v1"] == mod.DEFAULT_INDEX_MAP["v1"]  # untouched

    def test_malformed_token_raises(self) -> None:
        with pytest.raises(ValueError):
            mod._parse_index_map("just_a_name", mod.DEFAULT_INDEX_MAP)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


_REQUIRED_TOP_LEVEL = {
    "run_id",
    "timestamp",
    "dry_run",
    "eval_path",
    "exact_baseline_sample_size",
    "random_seed",
    "model_name",
    "env",
    "queries",
    "indexes",
    "pairwise_significance",
}

_REQUIRED_INDEX_KEYS = {"name", "index_name", "metrics", "per_query"}
_REQUIRED_METRIC_KEYS = {
    "ndcg_at_10",
    "recall_at_10",
    "recall_at_20",
    "mrr",
    "p50_ms",
    "p95_ms",
}
_REQUIRED_SIG_KEYS = {
    "compared",
    "p_value",
    "bonferroni_adjusted_p",
}


def _assert_schema(payload: dict[str, Any]) -> None:
    missing = _REQUIRED_TOP_LEVEL - set(payload.keys())
    assert not missing, f"missing top-level keys: {missing}"
    for entry in payload["indexes"]:
        assert _REQUIRED_INDEX_KEYS <= set(entry.keys())
        assert _REQUIRED_METRIC_KEYS <= set(entry["metrics"].keys())
    for sig in payload["pairwise_significance"]:
        assert _REQUIRED_SIG_KEYS <= set(sig.keys())


class TestDryRun:
    def test_payload_schema(self) -> None:
        payload = mod.dry_run_payload(
            index_names=["hnsw", "v1", "v2", "v3"],
            index_map=mod.DEFAULT_INDEX_MAP,
            sample_size=1_000_000,
            eval_path=Path("results/retrieval_eval_50q.json"),
        )
        assert payload["dry_run"] is True
        assert payload["exact_baseline_sample_size"] == 1_000_000
        _assert_schema(payload)
        # Four indexes, each with nulled-out metrics.
        names = [e["name"] for e in payload["indexes"]]
        assert names == ["hnsw", "v1", "v2", "v3"]
        # Three v? vs hnsw comparisons.
        compared = [s["compared"] for s in payload["pairwise_significance"]]
        assert compared == ["v1 vs hnsw", "v2 vs hnsw", "v3 vs hnsw"]

    def test_main_dry_run_writes_json_and_md(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_pilot",
                "--dry-run",
                "--out",
                str(out_dir),
                "--eval-path",
                "results/retrieval_eval_50q.json",
            ]
        )
        assert rc == 0
        json_path = out_dir / mod.DEFAULT_OUT_JSON_NAME
        md_path = out_dir / mod.DEFAULT_OUT_MD_NAME
        assert json_path.exists()
        assert md_path.exists()
        payload = json.loads(json_path.read_text())
        _assert_schema(payload)
        assert payload["dry_run"] is True
        md_text = md_path.read_text()
        assert "Retrieval Quality Benchmark" in md_text
        assert "Summary" in md_text

    def test_main_dry_run_custom_indexes(self, tmp_path: Path) -> None:
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_test",
                "--dry-run",
                "--indexes",
                "hnsw,v2",
                "--out",
                str(tmp_path),
            ]
        )
        assert rc == 0
        payload = json.loads((tmp_path / mod.DEFAULT_OUT_JSON_NAME).read_text())
        names = [e["name"] for e in payload["indexes"]]
        assert names == ["hnsw", "v2"]


# ---------------------------------------------------------------------------
# Markdown rendering — PASS/FAIL annotations
# ---------------------------------------------------------------------------


class TestMarkdownPassFail:
    def test_pass_and_fail_annotations(self) -> None:
        payload: dict[str, Any] = {
            "run_id": "r",
            "timestamp": "t",
            "dry_run": False,
            "model_name": "indus",
            "eval_path": "x",
            "exact_baseline_sample_size": 1000,
            "random_seed": 42,
            "indexes": [
                {
                    "name": "hnsw",
                    "index_name": "idx_hnsw_baseline_indus",
                    "metrics": {
                        "ndcg_at_10": 0.5000,
                        "recall_at_10": 0.3,
                        "recall_at_20": 0.5,
                        "mrr": 0.7,
                        "p50_ms": 10.0,
                        "p95_ms": 25.0,
                    },
                    "per_query": [],
                },
                {
                    "name": "v1",
                    "index_name": "paper_embeddings_diskann_v1",
                    "metrics": {
                        "ndcg_at_10": 0.4950,  # within 1% → PASS
                        "recall_at_10": 0.3,
                        "recall_at_20": 0.5,
                        "mrr": 0.7,
                        "p50_ms": 8.0,
                        "p95_ms": 22.0,
                    },
                    "per_query": [],
                },
                {
                    "name": "v2",
                    "index_name": "paper_embeddings_diskann_v2",
                    "metrics": {
                        "ndcg_at_10": 0.4500,  # outside 1% → FAIL
                        "recall_at_10": 0.3,
                        "recall_at_20": 0.5,
                        "mrr": 0.7,
                        "p50_ms": 6.0,
                        "p95_ms": 18.0,
                    },
                    "per_query": [],
                },
            ],
            "pairwise_significance": [
                {
                    "compared": "v1 vs hnsw",
                    "p_value": 0.8,
                    "bonferroni_adjusted_p": 1.0,
                    "n_pairs": 20,
                    "mean_diff": -0.005,
                },
                {
                    "compared": "v2 vs hnsw",
                    "p_value": 0.01,
                    "bonferroni_adjusted_p": 0.03,
                    "n_pairs": 20,
                    "mean_diff": -0.05,
                },
            ],
        }
        md = mod.render_markdown(payload)
        assert "PASS" in md
        assert "FAIL" in md
        # Per-variant annotation rows should be present.
        assert "v1 |" in md and "v2 |" in md
        # Summary table header present.
        assert "nDCG@10" in md
        assert "p95 (ms)" in md
        # Pairwise significance rows.
        assert "v1 vs hnsw" in md and "v2 vs hnsw" in md


# ---------------------------------------------------------------------------
# (d) Argparse wiring — all required flags in --help
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_help_exits_zero_and_lists_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/bench_pgvectorscale.py", "--help"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        for flag in ("--dsn", "--eval-path", "--indexes", "--out", "--dry-run", "--sample-size"):
            assert flag in out, f"missing {flag} in --help output"

    def test_build_parser_defaults(self) -> None:
        parser = mod.build_parser()
        args = parser.parse_args(["--dsn", "dbname=scix_pilot"])
        assert args.indexes == mod.DEFAULT_INDEXES
        assert args.sample_size == mod.DEFAULT_SAMPLE_SIZE
        assert args.dry_run is False
        assert str(args.eval_path) == str(mod.DEFAULT_EVAL_PATH)
        # Indexes parse to 4 names.
        names = [n.strip() for n in args.indexes.split(",")]
        assert names == ["hnsw", "v1", "v2", "v3"]
