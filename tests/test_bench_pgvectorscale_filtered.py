"""Unit tests for scripts/bench_pgvectorscale_filtered.py.

Covers the acceptance criteria for the filtered-query benchmark:

  (a) Production-DSN refusal (key=value + URI, empty DSN).
  (b) ``flag_p95_degradation`` logic on synthetic metrics dicts.
  (c) argparse wiring — ``--help`` lists ``--dsn``, ``--eval-path``,
      ``--filter``, ``--out`` and ``--filter`` accepts only
      ``{f1, f2, both}``.
  (d) ``FILTERS`` constant contains the exact ``year = 2024`` and
      ``astro-ph`` substrings required by the AC grep checks.
  (e) Dry-run produces schema-valid JSON + MD with one cell per
      (index, filter) pair and a ``>=2x`` degradation flag column.
  (f) Unfiltered baseline loader handles missing-file gracefully.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import bench_pgvectorscale_filtered as mod


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# (a) Production-DSN refusal
# ---------------------------------------------------------------------------


class TestAssertPilotDsn:
    def test_refuses_production_dsn_kv(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            mod.assert_pilot_dsn("dbname=scix")
        msg = str(excinfo.value).lower()
        assert "production" in msg or "refuse" in msg

    def test_refuses_production_dsn_uri(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            mod.assert_pilot_dsn("postgresql://user@localhost/scix")
        msg = str(excinfo.value).lower()
        assert "production" in msg or "refuse" in msg

    def test_refuses_empty_dsn(self) -> None:
        with pytest.raises(ValueError):
            mod.assert_pilot_dsn("")
        with pytest.raises(ValueError):
            mod.assert_pilot_dsn(None)

    def test_allows_pilot_dsn(self) -> None:
        # These must NOT raise.
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
# (b) Degradation flagging logic
# ---------------------------------------------------------------------------


class TestFlagP95Degradation:
    def test_two_point_five_x_flagged(self) -> None:
        # 100 / 40 = 2.5 >= 2.0 → True
        assert mod.flag_p95_degradation(100.0, 40.0) is True

    def test_just_under_two_x_not_flagged(self) -> None:
        # 79 / 40 = 1.975 < 2.0 → False
        assert mod.flag_p95_degradation(79.0, 40.0) is False

    def test_exact_two_x_flagged(self) -> None:
        # 80 / 40 = 2.0, boundary is inclusive (>=) → True
        assert mod.flag_p95_degradation(80.0, 40.0) is True

    def test_zero_baseline_returns_false(self) -> None:
        assert mod.flag_p95_degradation(100.0, 0.0) is False

    def test_negative_baseline_returns_false(self) -> None:
        assert mod.flag_p95_degradation(100.0, -1.0) is False

    def test_none_filtered_returns_false(self) -> None:
        assert mod.flag_p95_degradation(None, 40.0) is False

    def test_none_unfiltered_returns_false(self) -> None:
        assert mod.flag_p95_degradation(100.0, None) is False

    def test_custom_threshold_three_x(self) -> None:
        # 100 / 40 = 2.5, threshold=3.0 → False
        assert mod.flag_p95_degradation(100.0, 40.0, threshold=3.0) is False
        # 150 / 40 = 3.75, threshold=3.0 → True
        assert mod.flag_p95_degradation(150.0, 40.0, threshold=3.0) is True


# ---------------------------------------------------------------------------
# (c) argparse wiring
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_help_exits_zero_with_required_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/bench_pgvectorscale_filtered.py", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        help_text = proc.stdout
        assert "--dsn" in help_text
        assert "--eval-path" in help_text
        assert "--filter" in help_text
        assert "--out" in help_text

    def test_filter_choices_f1(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args(["--dsn", "dbname=scix_pilot", "--filter", "f1"])
        assert ns.filter_choice == "f1"

    def test_filter_choices_f2(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args(["--dsn", "dbname=scix_pilot", "--filter", "f2"])
        assert ns.filter_choice == "f2"

    def test_filter_choices_both(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args(["--dsn", "dbname=scix_pilot", "--filter", "both"])
        assert ns.filter_choice == "both"

    def test_filter_rejects_garbage(self) -> None:
        parser = mod.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--dsn", "dbname=scix_pilot", "--filter", "garbage"]
            )

    def test_filter_default_is_both(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args(["--dsn", "dbname=scix_pilot"])
        assert ns.filter_choice == "both"


# ---------------------------------------------------------------------------
# (d) FILTERS constant contents
# ---------------------------------------------------------------------------


class TestFilters:
    def test_has_f1_and_f2_keys(self) -> None:
        assert set(mod.FILTERS.keys()) == {"f1", "f2"}

    def test_f1_clause_year_2024(self) -> None:
        assert "year = 2024" in mod.FILTERS["f1"]["clause"]

    def test_f2_clause_astro_ph_classes(self) -> None:
        clause = mod.FILTERS["f2"]["clause"]
        assert "astro-ph.GA" in clause
        assert "astro-ph.SR" in clause
        assert "&&" in clause  # array-overlap operator

    def test_filter_choices_constant(self) -> None:
        assert mod.FILTER_CHOICES == ["f1", "f2", "both"]

    def test_resolve_filter_keys_both(self) -> None:
        assert mod._resolve_filter_keys("both") == ["f1", "f2"]

    def test_resolve_filter_keys_single(self) -> None:
        assert mod._resolve_filter_keys("f1") == ["f1"]
        assert mod._resolve_filter_keys("f2") == ["f2"]


# ---------------------------------------------------------------------------
# Filter strategy metadata
# ---------------------------------------------------------------------------


class TestFilterStrategy:
    def test_hnsw_uses_iterative_scan(self) -> None:
        strategy = mod.filter_strategy_for("hnsw")
        assert "iterative_scan" in strategy
        assert "relaxed_order" in strategy

    def test_diskann_v1_uses_native(self) -> None:
        assert "native" in mod.filter_strategy_for("v1")

    def test_diskann_v2_uses_native(self) -> None:
        assert "native" in mod.filter_strategy_for("v2")

    def test_diskann_v3_uses_native(self) -> None:
        assert "native" in mod.filter_strategy_for("v3")


# ---------------------------------------------------------------------------
# (e) Dry-run schema + markdown rendering
# ---------------------------------------------------------------------------


class TestDryRunSchema:
    def _dry_run_payload_both(self) -> dict:
        return mod.dry_run_payload(
            index_names=["hnsw", "v1", "v2", "v3"],
            filter_keys=["f1", "f2"],
            index_map=mod.DEFAULT_INDEX_MAP,
            eval_path=Path("results/retrieval_eval_50q.json"),
            unfiltered_baseline={},
            unfiltered_baseline_present=False,
        )

    def test_top_level_keys(self) -> None:
        p = self._dry_run_payload_both()
        for k in [
            "run_id",
            "timestamp",
            "dry_run",
            "eval_path",
            "model_name",
            "filters",
            "degradation_threshold",
            "unfiltered_baseline_path",
            "unfiltered_baseline_present",
            "results",
        ]:
            assert k in p, f"missing top-level key {k}"
        assert p["dry_run"] is True

    def test_eight_cells_when_filter_both(self) -> None:
        p = self._dry_run_payload_both()
        assert len(p["results"]) == 8  # 4 indexes × 2 filters
        pairs = {(c["index"], c["filter"]) for c in p["results"]}
        expected = {
            (idx, f) for idx in ["hnsw", "v1", "v2", "v3"] for f in ["f1", "f2"]
        }
        assert pairs == expected

    def test_cell_schema(self) -> None:
        p = self._dry_run_payload_both()
        cell = p["results"][0]
        for k in [
            "index",
            "index_name",
            "filter",
            "filter_clause",
            "filter_strategy",
            "metrics",
            "unfiltered_p95_ms",
            "p95_ratio_filtered_over_unfiltered",
            "p95_degradation_threshold",
            "p95_degradation_flag",
            "per_query",
        ]:
            assert k in cell, f"missing cell key {k}"
        # Metric values are all None in dry-run.
        for v in cell["metrics"].values():
            assert v is None
        # When baseline missing, flag is "unknown".
        assert cell["p95_degradation_flag"] == "unknown"

    def test_four_cells_when_filter_f1(self) -> None:
        p = mod.dry_run_payload(
            index_names=["hnsw", "v1", "v2", "v3"],
            filter_keys=["f1"],
            index_map=mod.DEFAULT_INDEX_MAP,
            eval_path=Path("results/retrieval_eval_50q.json"),
            unfiltered_baseline={},
            unfiltered_baseline_present=False,
        )
        assert len(p["results"]) == 4
        assert all(c["filter"] == "f1" for c in p["results"])


class TestMarkdown:
    def test_contains_degradation_flag_column(self) -> None:
        p = mod.dry_run_payload(
            index_names=["hnsw", "v1"],
            filter_keys=["f1", "f2"],
            index_map=mod.DEFAULT_INDEX_MAP,
            eval_path=Path("x.json"),
            unfiltered_baseline={},
            unfiltered_baseline_present=False,
        )
        md = mod.render_markdown(p)
        # AC: MD must flag any index with p95 under filter >= 2x unfiltered.
        assert ">=2x degradation under filter" in md
        assert "p95 filt" in md
        assert "p95 unfilt" in md
        # Filter clauses appear in the MD.
        assert "year = 2024" in md
        assert "astro-ph" in md

    def test_end_to_end_dry_run_writes_files(self, tmp_path: Path) -> None:
        out = tmp_path / "pgvs_benchmark"
        rc = mod.main(
            [
                "--dsn",
                "dbname=scix_pilot",
                "--dry-run",
                "--out",
                str(out),
                "--filter",
                "both",
            ]
        )
        assert rc == 0
        json_path = out / mod.DEFAULT_OUT_JSON_NAME
        md_path = out / mod.DEFAULT_OUT_MD_NAME
        assert json_path.exists()
        assert md_path.exists()
        payload = json.loads(json_path.read_text())
        assert payload["dry_run"] is True
        assert len(payload["results"]) == 8
        md = md_path.read_text()
        assert ">=2x degradation under filter" in md


# ---------------------------------------------------------------------------
# (f) Unfiltered baseline loader
# ---------------------------------------------------------------------------


class TestUnfilteredBaseline:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.json"
        assert mod.load_unfiltered_baseline(missing) == {}

    def test_loads_p95_per_index(self, tmp_path: Path) -> None:
        path = tmp_path / "retrieval_quality.json"
        path.write_text(
            json.dumps(
                {
                    "indexes": [
                        {"name": "hnsw", "metrics": {"p95_ms": 42.5}},
                        {"name": "v1", "metrics": {"p95_ms": 37.1}},
                        {"name": "v2", "metrics": {"p95_ms": None}},
                    ]
                }
            )
        )
        baseline = mod.load_unfiltered_baseline(path)
        assert baseline["hnsw"] == 42.5
        assert baseline["v1"] == 37.1
        assert baseline["v2"] is None

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        assert mod.load_unfiltered_baseline(path) == {}


class TestBuildCellUsesBaseline:
    def test_flag_true_when_filtered_p95_exceeds_threshold(self) -> None:
        p = mod.dry_run_payload(
            index_names=["hnsw"],
            filter_keys=["f1"],
            index_map=mod.DEFAULT_INDEX_MAP,
            eval_path=Path("x.json"),
            unfiltered_baseline={"hnsw": 40.0},
            unfiltered_baseline_present=True,
        )
        # Dry-run has None metrics so flag is False (not "unknown", since baseline present).
        assert p["results"][0]["unfiltered_p95_ms"] == 40.0
        assert p["results"][0]["p95_degradation_flag"] is False

    def test_build_cell_computes_ratio_and_flag(self) -> None:
        cell = mod._build_cell(
            logical_name="hnsw",
            physical_name="idx_hnsw_baseline_indus",
            filter_key="f1",
            metrics={
                "ndcg_at_10": 0.5,
                "recall_at_10": 0.3,
                "recall_at_20": 0.4,
                "mrr": 0.6,
                "p50_ms": 50.0,
                "p95_ms": 120.0,
            },
            per_query=[],
            unfiltered_p95=40.0,
            unfiltered_baseline_present=True,
        )
        assert cell["p95_ratio_filtered_over_unfiltered"] == pytest.approx(3.0)
        assert cell["p95_degradation_flag"] is True

    def test_build_cell_flag_unknown_when_no_baseline(self) -> None:
        cell = mod._build_cell(
            logical_name="v1",
            physical_name="paper_embeddings_diskann_v1",
            filter_key="f2",
            metrics={
                "ndcg_at_10": 0.5,
                "recall_at_10": 0.3,
                "recall_at_20": 0.4,
                "mrr": 0.6,
                "p50_ms": 50.0,
                "p95_ms": 120.0,
            },
            per_query=[],
            unfiltered_p95=None,
            unfiltered_baseline_present=False,
        )
        assert cell["p95_degradation_flag"] == "unknown"
        assert cell["p95_ratio_filtered_over_unfiltered"] is None


# ---------------------------------------------------------------------------
# Metric smoke tests (module-local copies — ensure helpers work)
# ---------------------------------------------------------------------------


class TestMetricsSmoke:
    def test_ndcg_empty_relevant(self) -> None:
        assert mod.ndcg_at_10(["a", "b"], set()) == 0.0

    def test_ndcg_perfect_ranking(self) -> None:
        ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        assert mod.ndcg_at_10(ranked, {"a", "b", "c"}) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_recall_at_k(self) -> None:
        assert mod.recall_at_k(["a", "b", "c"], {"a", "b"}, 10) == 1.0
        assert mod.recall_at_k(["x", "y"], {"a"}, 10) == 0.0

    def test_mrr(self) -> None:
        assert mod.mrr(["x", "a"], {"a"}) == pytest.approx(0.5)
        assert mod.mrr(["x", "y"], {"a"}) == 0.0

    def test_percentile_empty(self) -> None:
        assert mod.percentile([], 50.0) == 0.0

    def test_percentile_single(self) -> None:
        assert mod.percentile([42.0], 95.0) == 42.0

    def test_percentile_basic(self) -> None:
        assert mod.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Source-level grep guarantees (AC 4)
# ---------------------------------------------------------------------------


class TestSourceContainsFilterClauses:
    def test_source_contains_year_2024(self) -> None:
        src = (REPO_ROOT / "scripts" / "bench_pgvectorscale_filtered.py").read_text()
        assert src.count("year = 2024") >= 1

    def test_source_contains_astro_ph(self) -> None:
        src = (REPO_ROOT / "scripts" / "bench_pgvectorscale_filtered.py").read_text()
        assert src.count("astro-ph") >= 1
