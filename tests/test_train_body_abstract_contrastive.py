"""Smoke + unit tests for scripts/train_body_abstract_contrastive.py (S1).

The full 100K pilot is GPU-window deferred. These tests prove the
training loop converges on a tiny model + synthetic pairs, that cohort
selection honours --cohort-strategy, and that the cohort planner
correctly reads M1's coverage-bias JSON output.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest

# scripts/ is on the pytest pythonpath (see pyproject.toml).
tbac = importlib.import_module("train_body_abstract_contrastive")


# ---------------------------------------------------------------------------
# (a) Script imports cleanly
# ---------------------------------------------------------------------------


class TestImports:
    def test_module_exposes_public_api(self) -> None:
        for name in (
            "parse_args",
            "build_cohort_plan",
            "load_coverage_bias",
            "make_loss_capture",
            "make_synthetic_pairs",
            "run_smoke_training",
            "load_model",
            "train",
            "BASE_MODELS",
            "COHORT_STRATEGIES",
        ):
            assert hasattr(tbac, name), f"missing public symbol: {name}"

    def test_required_cli_flags_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            tbac.parse_args(["--help"])
        out = capsys.readouterr().out
        for flag in (
            "--base-model",
            "--cohort-size",
            "--cohort-strategy",
            "--output-model-dir",
            "--dry-run",
            "--max-steps",
        ):
            assert flag in out, f"--help output missing {flag}"


# ---------------------------------------------------------------------------
# (c) Cohort-strategy stratified-by-arxiv-class
# ---------------------------------------------------------------------------


def _fake_coverage_payload() -> dict[str, object]:
    """Three-stratum mock at 0.50 / 0.30 / 0.20 corpus prior."""
    return {
        "corpus_total": 1_000_000,
        "fulltext_total": 500_000,
        "facets": {
            "arxiv_class": {
                "kl_divergence_vs_corpus_prior": 0.0,
                "rows": [
                    {"label": "cs.LG", "q_corpus": 0.50, "p_fulltext": 0.50},
                    {"label": "hep-ph", "q_corpus": 0.30, "p_fulltext": 0.30},
                    {"label": "astro-ph.SR", "q_corpus": 0.20, "p_fulltext": 0.20},
                ],
            }
        },
    }


class TestCohortPlan:
    def test_random_strategy_returns_single_bucket(self) -> None:
        plan = tbac.build_cohort_plan(
            coverage=None,
            cohort_size=100,
            strategy="random",
            seed=0,
        )
        assert len(plan) == 1
        assert plan[0].label == "random"
        assert plan[0].n == 100

    def test_stratified_strategy_proportional_to_corpus_prior(self) -> None:
        plan = tbac.build_cohort_plan(
            coverage=_fake_coverage_payload(),
            cohort_size=100,
            strategy="stratified-by-arxiv-class",
            seed=0,
        )
        labels = {a.label: a.n for a in plan}
        assert sum(labels.values()) == 100, labels
        assert labels["cs.LG"] == 50
        assert labels["hep-ph"] == 30
        assert labels["astro-ph.SR"] == 20

    def test_stratified_strategy_handles_rounding_drift(self) -> None:
        # 7 over 3 buckets at 0.5/0.3/0.2 -> 4/2/1 with drift=0,
        # but a non-trivial cohort_size=11 forces 6/3/2 with drift compensated.
        plan = tbac.build_cohort_plan(
            coverage=_fake_coverage_payload(),
            cohort_size=11,
            strategy="stratified-by-arxiv-class",
            seed=0,
        )
        assert sum(a.n for a in plan) == 11

    def test_stratified_strategy_requires_coverage(self) -> None:
        with pytest.raises(ValueError):
            tbac.build_cohort_plan(
                coverage=None,
                cohort_size=10,
                strategy="stratified-by-arxiv-class",
                seed=0,
            )

    def test_unknown_strategy_rejected(self) -> None:
        with pytest.raises(ValueError):
            tbac.build_cohort_plan(
                coverage=None,
                cohort_size=10,
                strategy="bogus",
                seed=0,
            )


# ---------------------------------------------------------------------------
# (5) Cohort-selection helper reads M1 JSON  (mocked filesystem)
# ---------------------------------------------------------------------------


class TestCoverageBiasIO:
    def test_reads_m1_json_from_mocked_path(self, tmp_path: Path) -> None:
        json_path = tmp_path / "full_text_coverage_bias.json"
        payload = _fake_coverage_payload()
        json_path.write_text(json.dumps(payload))

        loaded = tbac.load_coverage_bias(json_path)
        assert "facets" in loaded
        rows = loaded["facets"]["arxiv_class"]["rows"]
        assert {row["label"] for row in rows} == {"cs.LG", "hep-ph", "astro-ph.SR"}

    def test_missing_path_raises_actionable_error(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            tbac.load_coverage_bias(bogus)

    def test_planner_uses_loaded_payload(self, tmp_path: Path) -> None:
        json_path = tmp_path / "cov.json"
        json_path.write_text(json.dumps(_fake_coverage_payload()))
        coverage = tbac.load_coverage_bias(json_path)
        plan = tbac.build_cohort_plan(
            coverage=coverage,
            cohort_size=100,
            strategy="stratified-by-arxiv-class",
            seed=0,
        )
        assert sum(a.n for a in plan) == 100
        # Confirms the planner consulted q_corpus from the JSON.
        assert any(a.label == "cs.LG" and a.n == 50 for a in plan)


# ---------------------------------------------------------------------------
# (b) Loss decreases on the smoke loop
# ---------------------------------------------------------------------------


sentence_transformers = pytest.importorskip("sentence_transformers")


def test_smoke_training_decreases_loss(tmp_path: Path) -> None:
    """End-to-end smoke: 32 synthetic pairs, MiniLM, < 60 s on CPU."""
    start = time.monotonic()
    avg_loss_first, avg_loss_last = tbac.run_smoke_training(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        n_pairs=32,
        batch_size=4,
        epochs=5,
        output_dir=tmp_path / "smoke",
        seed=13,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 60.0, f"Smoke test took {elapsed:.1f}s; budget is 60 s"
    assert avg_loss_last < avg_loss_first, (
        "Average loss did not decrease over training: "
        f"first_window={avg_loss_first:.4f} last_window={avg_loss_last:.4f}"
    )


def test_synthetic_pair_count_matches_request() -> None:
    examples = tbac.make_synthetic_pairs(7, seed=0)
    assert len(examples) == 7
    # Each InputExample carries exactly two texts (anchor, positive).
    for ex in examples:
        assert len(ex.texts) == 2
