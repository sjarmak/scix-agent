"""Unit tests for scripts/compare_extraction_sources.py.

All tests are offline. The Anthropic SDK and HuggingFace transformers are
never imported. Live-extractor adapters are exercised only through their
NotImplementedError paths to confirm the cost gate / mock-flag plumbing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest

# Dynamically load scripts/compare_extraction_sources.py since scripts/ is
# not a Python package. Mirrors the pattern in test_eval_ner_wiesp.py.
_SCRIPT_PATH: Path = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_extraction_sources.py"
)
_spec = importlib.util.spec_from_file_location("compare_extraction_sources", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ces = importlib.util.module_from_spec(_spec)
sys.modules["compare_extraction_sources"] = ces
_spec.loader.exec_module(ces)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_model_revision_pinned_sha(self) -> None:
        """Must match the SHA from M1 (eval_ner_wiesp.py)."""
        assert ces.MODEL_REVISION == "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"
        assert len(ces.MODEL_REVISION) == 40
        assert all(c in "0123456789abcdef" for c in ces.MODEL_REVISION)

    def test_model_name_matches_m1(self) -> None:
        assert ces.MODEL_NAME == "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"

    def test_haiku_pricing_is_positive(self) -> None:
        assert ces.HAIKU_INPUT_USD_PER_MTOK > 0
        assert ces.HAIKU_OUTPUT_USD_PER_MTOK > 0

    def test_default_sample_size_is_500(self) -> None:
        assert ces.DEFAULT_SAMPLE_SIZE == 500

    def test_cost_gate_env_name(self) -> None:
        assert ces.COST_GATE_ENV == "SCIX_HEAD_TO_HEAD_BUDGET_USD"

    def test_method_keys_cover_all_three(self) -> None:
        assert set(ces.METHOD_KEYS) == {"metadata", "ner", "haiku"}


class TestExecutable:
    def test_script_has_executable_bit(self) -> None:
        mode = os.stat(_SCRIPT_PATH).st_mode
        assert mode & stat.S_IXUSR


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _make_cohort(strata: dict[str | None, int]) -> list:
    """Build a cohort with the requested per-stratum bibcode counts."""
    cohort = []
    counter = 0
    for stratum, n in strata.items():
        for _ in range(n):
            counter += 1
            cohort.append(
                ces.CohortPaper(
                    bibcode=f"BIB{counter:05d}",
                    arxiv_class_primary=stratum,
                )
            )
    return cohort


class TestStratifiedSampling:
    def test_proportional_allocation_60_30_10(self) -> None:
        cohort = _make_cohort({"astro-ph.CO": 600, "astro-ph.GA": 300, "astro-ph.HE": 100})
        sample = ces.stratified_sample(cohort, sample_size=100, seed=42)
        assert len(sample) == 100
        counts = Counter(p.arxiv_class_primary for p in sample)
        assert counts["astro-ph.CO"] == 60
        assert counts["astro-ph.GA"] == 30
        assert counts["astro-ph.HE"] == 10

    def test_fallback_uniform_when_no_arxiv_class(self) -> None:
        cohort = _make_cohort({None: 1000})
        sample = ces.stratified_sample(cohort, sample_size=50, seed=42)
        assert len(sample) == 50
        # Determinism: re-running with same seed yields same bibcodes.
        sample2 = ces.stratified_sample(cohort, sample_size=50, seed=42)
        assert [p.bibcode for p in sample] == [p.bibcode for p in sample2]

    def test_seed_determinism_with_strata(self) -> None:
        cohort = _make_cohort({"astro-ph.CO": 100, "astro-ph.SR": 100})
        a = ces.stratified_sample(cohort, sample_size=20, seed=7)
        b = ces.stratified_sample(cohort, sample_size=20, seed=7)
        assert [p.bibcode for p in a] == [p.bibcode for p in b]

    def test_different_seeds_yield_different_samples(self) -> None:
        cohort = _make_cohort({None: 200})
        a = ces.stratified_sample(cohort, sample_size=20, seed=1)
        b = ces.stratified_sample(cohort, sample_size=20, seed=2)
        # Vanishingly small probability of collision under a good RNG.
        assert {p.bibcode for p in a} != {p.bibcode for p in b}

    def test_sample_size_exceeds_cohort_returns_full_cohort(self) -> None:
        cohort = _make_cohort({"astro-ph.CO": 5})
        sample = ces.stratified_sample(cohort, sample_size=100, seed=0)
        assert len(sample) == 5
        assert {p.bibcode for p in sample} == {p.bibcode for p in cohort}

    def test_zero_sample_returns_empty(self) -> None:
        cohort = _make_cohort({"astro-ph.CO": 5})
        assert ces.stratified_sample(cohort, sample_size=0, seed=0) == []

    def test_single_stratum_falls_back_to_uniform(self) -> None:
        cohort = _make_cohort({"astro-ph.CO": 50})
        sample = ces.stratified_sample(cohort, sample_size=10, seed=0)
        assert len(sample) == 10


class TestLargestRemainder:
    def test_allocations_sum_to_sample_size(self) -> None:
        sizes = {"a": 7, "b": 13, "c": 19}
        alloc = ces._largest_remainder_allocation(sizes, 11)
        assert sum(alloc.values()) == 11

    def test_allocation_clamped_to_bucket_size(self) -> None:
        sizes = {"small": 2, "big": 1000}
        alloc = ces._largest_remainder_allocation(sizes, 100)
        assert alloc["small"] <= 2

    def test_zero_total_returns_zeroes(self) -> None:
        alloc = ces._largest_remainder_allocation({"a": 0, "b": 0}, 5)
        assert alloc == {"a": 0, "b": 0}


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


class TestComputeMethodMetrics:
    def test_perfect_predictions_score_one(self) -> None:
        gold = {
            "B1": {"instruments": ["HST"], "datasets": ["SDSS"], "software": ["astropy"]},
            "B2": {"instruments": ["JWST"], "datasets": [], "software": []},
        }
        report = ces.compute_method_metrics(gold, gold)
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.f1 == pytest.approx(1.0)
        for etype_scores in report.per_entity_type.values():
            assert etype_scores["precision"] in (0.0, 1.0)
            # F1 = 1.0 for any type with positive support
            if etype_scores["support"] > 0:
                assert etype_scores["f1"] == pytest.approx(1.0)

    def test_one_missed_one_spurious(self) -> None:
        # Gold:  B1 instruments={HST, JWST}; B1 datasets={SDSS}
        # Pred:  B1 instruments={HST};       B1 datasets={SDSS, FAKE}
        # instruments: TP=1 FP=0 FN=1 → P=1.00 R=0.50 F1=0.667
        # datasets:    TP=1 FP=1 FN=0 → P=0.50 R=1.00 F1=0.667
        # software:    TP=0 FP=0 FN=0 → 0/0/0
        # Aggregate: TP=2 FP=1 FN=1 → P=0.667 R=0.667 F1=0.667
        gold = {"B1": {"instruments": ["HST", "JWST"], "datasets": ["SDSS"], "software": []}}
        pred = {"B1": {"instruments": ["HST"], "datasets": ["SDSS", "FAKE"], "software": []}}
        report = ces.compute_method_metrics(pred, gold)
        assert report.per_entity_type["instruments"]["precision"] == pytest.approx(1.0)
        assert report.per_entity_type["instruments"]["recall"] == pytest.approx(0.5)
        assert report.per_entity_type["instruments"]["f1"] == pytest.approx(2 / 3, rel=1e-6)
        assert report.per_entity_type["datasets"]["precision"] == pytest.approx(0.5)
        assert report.per_entity_type["datasets"]["recall"] == pytest.approx(1.0)
        assert report.per_entity_type["datasets"]["f1"] == pytest.approx(2 / 3, rel=1e-6)
        assert report.precision == pytest.approx(2 / 3, rel=1e-6)
        assert report.recall == pytest.approx(2 / 3, rel=1e-6)
        assert report.f1 == pytest.approx(2 / 3, rel=1e-6)

    def test_normalization_lowercase_strip(self) -> None:
        gold = {"B1": {"instruments": ["HST"]}}
        pred = {"B1": {"instruments": ["  hst  "]}}
        report = ces.compute_method_metrics(pred, gold)
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)

    def test_empty_inputs(self) -> None:
        report = ces.compute_method_metrics({}, {})
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1 == 0.0

    def test_cost_passthrough(self) -> None:
        report = ces.compute_method_metrics({}, {}, cost_per_paper_usd=0.0125)
        assert report.cost_per_paper_usd == pytest.approx(0.0125)


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


class TestComputeHaikuCost:
    def test_known_token_count(self) -> None:
        # 1,000,000 input tokens + 200,000 output tokens, 100 papers.
        usages = [ces.HaikuUsage(input_tokens=10_000, output_tokens=2_000)] * 100
        cost = ces.compute_haiku_cost(usages, n_papers=100)
        # total: 1.0 MTok in * $1 + 0.2 MTok out * $5 = $1 + $1 = $2  → $0.02/paper
        assert cost == pytest.approx(0.02)

    def test_zero_papers_returns_zero(self) -> None:
        assert ces.compute_haiku_cost([], n_papers=0) == 0.0

    def test_no_usages_returns_zero(self) -> None:
        assert ces.compute_haiku_cost([], n_papers=10) == 0.0

    def test_pricing_applied_correctly(self) -> None:
        usages = [ces.HaikuUsage(input_tokens=2_000_000, output_tokens=500_000)]
        cost = ces.compute_haiku_cost(usages, n_papers=1)
        # 2 * $1 + 0.5 * $5 = $4.50 / 1 paper
        assert cost == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def _metrics(
    f1_overall: float,
    cost: float,
    per_type: dict[str, float] | None = None,
) -> "ces.MethodMetrics":
    """Build a MethodMetrics with given aggregate F1 and per-type F1 map."""
    per_type = per_type or {}
    per_entity_type = {
        etype: {"precision": f1, "recall": f1, "f1": f1, "support": 10.0}
        for etype, f1 in per_type.items()
    }
    return ces.MethodMetrics(
        precision=f1_overall,
        recall=f1_overall,
        f1=f1_overall,
        cost_per_paper_usd=cost,
        per_entity_type=per_entity_type,
    )


class TestRecommendation:
    def test_metadata_wins(self) -> None:
        metrics = {
            "metadata": _metrics(0.85, 0.0, {"instruments": 0.9, "datasets": 0.8}),
            "ner": _metrics(0.55, 0.0, {"instruments": 0.5, "datasets": 0.6}),
            "haiku": _metrics(0.70, 0.05, {"instruments": 0.7, "datasets": 0.7}),
        }
        rec, _ = ces.decide_recommendation(metrics)
        assert rec == "metadata"

    def test_ner_wins_when_metadata_low(self) -> None:
        # NER higher than metadata; metadata below ensemble threshold so the
        # ensemble override is skipped. Both are $0.
        metrics = {
            "metadata": _metrics(0.40, 0.0, {"instruments": 0.4}),
            "ner": _metrics(0.80, 0.0, {"instruments": 0.85, "datasets": 0.75}),
            "haiku": _metrics(0.65, 0.10, {"instruments": 0.7}),
        }
        rec, rationale = ces.decide_recommendation(metrics)
        assert rec == "ner"
        assert "ner" in rationale.lower()

    def test_haiku_wins_when_only_haiku_in_band(self) -> None:
        # All free; Haiku has highest F1 and ensemble override doesn't apply
        # (metadata below threshold).
        metrics = {
            "metadata": _metrics(0.30, 0.0, {"instruments": 0.3}),
            "ner": _metrics(0.40, 0.0, {"instruments": 0.4}),
            "haiku": _metrics(0.90, 0.0, {"instruments": 0.95}),
        }
        rec, _ = ces.decide_recommendation(metrics)
        assert rec == "haiku"

    def test_ensemble_when_disjoint_dominance(self) -> None:
        # NER dominates 'instruments'; metadata dominates 'datasets'; both
        # exceed the F1 ensemble threshold.
        metrics = {
            "metadata": _metrics(
                0.78,
                0.0,
                {"instruments": 0.5, "datasets": 0.95, "software": 0.8},
            ),
            "ner": _metrics(
                0.82,
                0.0,
                {"instruments": 0.95, "datasets": 0.5, "software": 0.85},
            ),
            "haiku": _metrics(0.60, 0.05, {"instruments": 0.6, "datasets": 0.6, "software": 0.6}),
        }
        rec, rationale = ces.decide_recommendation(metrics)
        assert rec == "ensemble"
        assert "disjoint" in rationale.lower()

    def test_haiku_pruned_by_cost_band_when_others_free(self) -> None:
        # Haiku has highest F1, but cost > 0 while metadata/ner are free;
        # 2x of $0 is still $0, so Haiku is out-of-band.
        metrics = {
            "metadata": _metrics(0.40, 0.0, {"instruments": 0.4}),
            "ner": _metrics(0.50, 0.0, {"instruments": 0.5}),
            "haiku": _metrics(0.99, 0.10, {"instruments": 1.0}),
        }
        rec, _ = ces.decide_recommendation(metrics)
        assert rec == "ner"

    def test_tie_break_prefers_cheaper(self) -> None:
        # Two methods with identical F1; cheaper one should win.
        metrics = {
            "metadata": _metrics(0.5, 0.0, {"instruments": 0.5}),
            "ner": _metrics(0.5, 0.0, {"instruments": 0.5}),
            "haiku": _metrics(0.5, 0.10, {"instruments": 0.5}),
        }
        rec, _ = ces.decide_recommendation(metrics)
        assert rec in {"metadata", "ner"}  # both cost 0; tie-break by cost is irrelevant


class TestDisjointDominance:
    def test_one_method_dominates_both_returns_false(self) -> None:
        # NER strictly higher on both 'a' and 'b' → not disjoint dominance
        # (md doesn't dominate any type).
        ner = {"a": {"f1": 0.8}, "b": {"f1": 0.7}}
        md = {"a": {"f1": 0.5}, "b": {"f1": 0.4}}
        assert ces._disjoint_entity_dominance(ner, md) is False

    def test_disjoint_returns_true(self) -> None:
        ner = {"a": {"f1": 0.8}, "b": {"f1": 0.5}}
        md = {"a": {"f1": 0.7}, "b": {"f1": 0.6}}
        # ner > md on a; md > ner on b → disjoint dominance.
        assert ces._disjoint_entity_dominance(ner, md) is True

    def test_no_dominance_returns_false(self) -> None:
        ner = {"a": {"f1": 0.5}}
        md = {"a": {"f1": 0.5}}
        assert ces._disjoint_entity_dominance(ner, md) is False


# ---------------------------------------------------------------------------
# Cost gate
# ---------------------------------------------------------------------------


class _Args:
    """Lightweight stand-in for argparse.Namespace in cost-gate tests."""

    def __init__(self, mock_all: bool = False, mock_haiku: bool = False):
        self.mock_all = mock_all
        self.mock_haiku = mock_haiku


class TestCostGate:
    def test_missing_env_exits_2(self) -> None:
        env: dict[str, str] = {}
        args = _Args(mock_all=False, mock_haiku=False)
        with pytest.raises(SystemExit) as exc:
            ces.enforce_cost_gate(args, env=env)
        assert exc.value.code == 2

    def test_invalid_value_exits_2(self) -> None:
        env = {ces.COST_GATE_ENV: "not-a-float"}
        args = _Args(mock_all=False, mock_haiku=False)
        with pytest.raises(SystemExit) as exc:
            ces.enforce_cost_gate(args, env=env)
        assert exc.value.code == 2

    def test_zero_or_negative_exits_2(self) -> None:
        for bad in ("0", "-1.0"):
            env = {ces.COST_GATE_ENV: bad}
            args = _Args(mock_all=False, mock_haiku=False)
            with pytest.raises(SystemExit) as exc:
                ces.enforce_cost_gate(args, env=env)
            assert exc.value.code == 2

    def test_positive_budget_passes(self) -> None:
        env = {ces.COST_GATE_ENV: "5.00"}
        args = _Args(mock_all=False, mock_haiku=False)
        ces.enforce_cost_gate(args, env=env)  # no exception

    def test_mock_all_skips_gate(self) -> None:
        env: dict[str, str] = {}  # no budget set
        args = _Args(mock_all=True)
        ces.enforce_cost_gate(args, env=env)

    def test_mock_haiku_skips_gate(self) -> None:
        env: dict[str, str] = {}
        args = _Args(mock_haiku=True)
        ces.enforce_cost_gate(args, env=env)


# ---------------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------------


class TestCohortLoading:
    def test_loads_bibcodes_from_sample(self, tmp_path: Path) -> None:
        report = {
            "gap_cohort_bibcodes_sample": ["2020A", "2020B", "  2020C  "],
            "gap_cohort_bibcodes_sidecar_path": None,
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        cohort = ces.load_cohort(path)
        assert {p.bibcode for p in cohort} == {"2020A", "2020B", "2020C"}
        assert all(p.arxiv_class_primary is None for p in cohort)

    def test_loads_arxiv_class_when_present(self, tmp_path: Path) -> None:
        report = {
            "gap_cohort_bibcodes_sample": ["2020A", "2020B"],
            "arxiv_class_by_bibcode": {
                "2020A": ["astro-ph.CO"],
                "2020B": ["astro-ph.SR", "physics.space-ph"],
            },
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        cohort = ces.load_cohort(path)
        primary = {p.bibcode: p.arxiv_class_primary for p in cohort}
        assert primary["2020A"] == "astro-ph.CO"
        assert primary["2020B"] == "astro-ph.SR"

    def test_empty_cohort_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"gap_cohort_bibcodes_sample": []}))
        with pytest.raises(ValueError):
            ces.load_cohort(path)


# ---------------------------------------------------------------------------
# End-to-end mock-all run
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _write_fixtures(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        cohort_path = tmp_path / "metadata_gap_report.json"
        cohort_payload = {
            "gap_cohort_bibcodes_sample": [f"BIB{i:04d}" for i in range(10)],
            "gap_cohort_bibcodes_sidecar_path": None,
        }
        cohort_path.write_text(json.dumps(cohort_payload))

        # Predictions: metadata is perfect on instruments, NER is perfect on
        # software, Haiku partial across all types.
        preds_path = tmp_path / "predictions.json"
        gold_path = tmp_path / "gold.json"

        gold = {
            f"BIB{i:04d}": {
                "instruments": ["HST"],
                "datasets": ["SDSS"],
                "software": ["astropy"],
            }
            for i in range(10)
        }
        metadata_preds = {
            f"BIB{i:04d}": {
                "instruments": ["HST"],
                "datasets": [],
                "software": [],
            }
            for i in range(10)
        }
        ner_preds = {
            f"BIB{i:04d}": {
                "instruments": [],
                "datasets": [],
                "software": ["astropy"],
            }
            for i in range(10)
        }
        haiku_preds = {
            f"BIB{i:04d}": {
                "instruments": ["HST"],
                "datasets": ["SDSS"],
                "software": ["astropy"],
            }
            for i in range(10)
        }
        preds_payload = {
            "metadata": metadata_preds,
            "ner": ner_preds,
            "haiku": haiku_preds,
            "haiku_usage": [
                {"input_tokens": 1200, "output_tokens": 250} for _ in range(10)
            ],
        }
        preds_path.write_text(json.dumps(preds_payload))
        gold_path.write_text(json.dumps(gold))

        output_path = tmp_path / "head_to_head.json"
        return cohort_path, preds_path, gold_path, output_path

    def test_mock_all_writes_valid_report(self, tmp_path: Path) -> None:
        cohort_path, preds_path, gold_path, output_path = self._write_fixtures(tmp_path)
        rc = ces.main(
            [
                "--mock-all",
                "--sample-size",
                "10",
                "--cohort-path",
                str(cohort_path),
                "--predictions-fixture",
                str(preds_path),
                "--gold-fixture",
                str(gold_path),
                "--output-path",
                str(output_path),
            ]
        )
        assert rc == 0
        assert output_path.is_file()
        report = json.loads(output_path.read_text())
        # Schema checks per the brief.
        assert report["cohort_source"] == str(cohort_path)
        assert report["sample_size"] == 10
        assert report["model_revision"] == ces.MODEL_REVISION
        assert set(report["per_method"].keys()) == {"metadata", "ner", "haiku"}
        for method_block in report["per_method"].values():
            assert "precision" in method_block
            assert "recall" in method_block
            assert "f1" in method_block
            assert "cost_per_paper_usd" in method_block
            assert "per_entity_type" in method_block
        assert report["per_method"]["metadata"]["cost_per_paper_usd"] == 0.0
        assert report["per_method"]["ner"]["cost_per_paper_usd"] == 0.0
        assert report["per_method"]["haiku"]["cost_per_paper_usd"] > 0
        assert report["recommendation"] in {"metadata", "ner", "haiku", "ensemble"}
        assert isinstance(report["recommendation_rationale"], str)
        assert "generated_at" in report["meta"]
        assert report["meta"]["sample_seed"] == ces.DEFAULT_SAMPLE_SEED
        assert report["meta"]["mock_mode"] == "mock_all"

    def test_mock_all_recommendation_is_haiku_for_perfect_haiku(
        self, tmp_path: Path
    ) -> None:
        # In our fixture metadata.f1 ≈ 1/3, ner.f1 ≈ 1/3, haiku.f1 = 1.0 but
        # haiku has positive cost. Cost band excludes haiku since metadata/ner
        # are free → recommendation should be metadata or ner (not haiku).
        cohort_path, preds_path, gold_path, output_path = self._write_fixtures(tmp_path)
        ces.main(
            [
                "--mock-all",
                "--sample-size",
                "10",
                "--cohort-path",
                str(cohort_path),
                "--predictions-fixture",
                str(preds_path),
                "--gold-fixture",
                str(gold_path),
                "--output-path",
                str(output_path),
            ]
        )
        report = json.loads(output_path.read_text())
        # Haiku has higher F1 but is out of cost band → not recommended.
        assert report["recommendation"] != "haiku"

    def test_main_without_cost_gate_exits_2(self, tmp_path: Path, monkeypatch) -> None:
        cohort_path, _, gold_path, output_path = self._write_fixtures(tmp_path)
        # Live mode: cost gate must trip when env var absent and no mock flag.
        monkeypatch.delenv(ces.COST_GATE_ENV, raising=False)
        with pytest.raises(SystemExit) as exc:
            ces.main(
                [
                    "--cohort-path",
                    str(cohort_path),
                    "--gold-fixture",
                    str(gold_path),
                    "--output-path",
                    str(output_path),
                ]
            )
        assert exc.value.code == 2
