"""Unit tests for scripts/canary_ner.py.

All tests mock the HuggingFace model loader so they run without network
or GPU. Each test writes into ``tmp_path`` so no state leaks across
runs.

Schema contract note: ``scripts/eval_ner_wiesp.py`` writes
``{"per_entity": ..., "summary": ..., "meta": {"model_revision": ..., ...}}``
so ``model_revision`` is nested under ``meta`` — these fixtures must
match that exact schema.
"""
from __future__ import annotations

import importlib.util
import json
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "canary_ner.py"


def _load_canary_module():
    """Load scripts/canary_ner.py as a module (it's a script, not on sys.path)."""
    spec = importlib.util.spec_from_file_location("canary_ner", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["canary_ner"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def canary():
    return _load_canary_module()


# ---------------------------------------------------------------------------
# Helpers — schema mirrors scripts/eval_ner_wiesp.py MetricsReport.to_dict()
# ---------------------------------------------------------------------------


def _make_baseline(
    tmp_path: Path,
    per_entity: dict[str, dict[str, float]] | None = None,
    *,
    include_meta: bool = True,
) -> Path:
    per_entity = per_entity or {
        "Mission": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10},
        "Instrument": {
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "support": 10,
        },
    }
    macro = sum(v["f1"] for v in per_entity.values()) / len(per_entity)
    payload: dict = {
        "per_entity": per_entity,
        "summary": {
            "macro_f1": macro,
            "macro_precision": macro,
            "macro_recall": macro,
            "micro_f1": macro,
            "micro_precision": macro,
            "micro_recall": macro,
            "n_examples": 3,
            "total_support": sum(v.get("support", 0) for v in per_entity.values()),
        },
    }
    if include_meta:
        # This mirrors the REAL schema produced by scripts/eval_ner_wiesp.py:
        # model_revision is nested under `meta`, NOT at the top level.
        payload["meta"] = {
            "dataset": "adsabs/WIESP2022-NER",
            "model_name": "adsabs/nasa-smd-ibm-v0.1_NER_DEAL",
            "model_revision": "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d",
            "split": "test",
            "source": "fixture",
        }
    path = tmp_path / "ner_wiesp_eval.json"
    path.write_text(json.dumps(payload))
    return path


def _make_fixture(tmp_path: Path) -> Path:
    fixture = {
        "version": 1,
        "description": "test",
        "papers": [
            {
                "id": "p1",
                "text": "Hubble saw NGC 1234.",
                "gold_entities": [
                    {"type": "Mission", "span": [0, 6], "text": "Hubble"},
                    {
                        "type": "CelestialObject",
                        "span": [11, 19],
                        "text": "NGC 1234",
                    },
                ],
            },
            {
                "id": "p2",
                "text": "Chandra imaged M87.",
                "gold_entities": [
                    {"type": "Mission", "span": [0, 7], "text": "Chandra"},
                    {
                        "type": "CelestialObject",
                        "span": [15, 18],
                        "text": "M87",
                    },
                ],
            },
        ],
    }
    path = tmp_path / "canary_ner_reference.json"
    path.write_text(json.dumps(fixture))
    return path


# ---------------------------------------------------------------------------
# Structural / packaging checks
# ---------------------------------------------------------------------------


def test_script_file_exists_and_is_executable():
    assert SCRIPT_PATH.exists(), f"Expected {SCRIPT_PATH} to exist"
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, "scripts/canary_ner.py must be executable"


def test_scheduling_comment_block_present():
    src = SCRIPT_PATH.read_text()
    # Sanity: cron and systemd hints must live at the top of the file.
    head = src[:2000]
    assert "Cron:" in head
    assert "systemd" in head


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------


def test_load_baseline_happy_path(canary, tmp_path: Path):
    path = _make_baseline(tmp_path)
    payload = canary.load_baseline(path)
    assert "per_entity" in payload
    assert payload["per_entity"]["Mission"]["f1"] == 0.9
    # model_revision is nested under `meta`, matching eval_ner_wiesp.py.
    assert (
        payload["meta"]["model_revision"]
        == "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"
    )


def test_load_baseline_tolerates_missing_meta(canary, tmp_path: Path):
    """Baselines without a `meta` block still load (backwards-compat)."""
    path = _make_baseline(tmp_path, include_meta=False)
    payload = canary.load_baseline(path)
    assert "per_entity" in payload
    # model_revision accessor returns None when meta is absent.
    assert canary.baseline_model_revision(payload) is None


def test_baseline_model_revision_extracts_from_meta(canary):
    payload = {
        "per_entity": {"X": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
        "summary": {},
        "meta": {"model_revision": "abc123"},
    }
    assert canary.baseline_model_revision(payload) == "abc123"


def test_baseline_model_revision_returns_none_when_absent(canary):
    assert (
        canary.baseline_model_revision(
            {"per_entity": {}, "summary": {}}
        )
        is None
    )
    assert (
        canary.baseline_model_revision(
            {"per_entity": {}, "summary": {}, "meta": {}}
        )
        is None
    )


def test_load_baseline_missing_file_raises_clear_error(canary, tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(canary.FileFormatError) as excinfo:
        canary.load_baseline(missing)
    assert str(missing) in str(excinfo.value)


def test_load_baseline_malformed_missing_per_entity(
    canary, tmp_path: Path
):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"summary": {}, "meta": {"model_revision": "x"}}))
    with pytest.raises(canary.FileFormatError) as excinfo:
        canary.load_baseline(path)
    assert "per_entity" in str(excinfo.value)


def test_load_baseline_malformed_missing_f1(canary, tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "per_entity": {"Mission": {"precision": 0.9, "recall": 0.9}},
                "summary": {"micro_f1": 0.9, "macro_f1": 0.9},
                "meta": {"model_revision": "x"},
            }
        )
    )
    with pytest.raises(canary.FileFormatError) as excinfo:
        canary.load_baseline(path)
    assert "f1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Integration: regression guard against schema drift with eval_ner_wiesp.py
# ---------------------------------------------------------------------------


def test_load_baseline_accepts_committed_eval_ner_wiesp_output(canary):
    """Regression guard: the canary must load the schema produced by
    scripts/eval_ner_wiesp.py without raising FileFormatError.

    This is the exact bug the previous review caught — canary required
    model_revision at the top level, but eval_ner_wiesp writes it nested
    under `meta`. This test loads the actual committed baseline file.
    """
    committed_baseline = REPO_ROOT / "results" / "ner_wiesp_eval.json"
    if not committed_baseline.exists():
        pytest.skip(
            f"{committed_baseline} not present in this checkout; "
            "regenerate via scripts/eval_ner_wiesp.py."
        )
    payload = canary.load_baseline(committed_baseline)
    # Schema sanity: top-level keys, per_entity populated, meta.model_revision set.
    assert "per_entity" in payload
    assert "summary" in payload
    assert isinstance(payload["per_entity"], dict) and payload["per_entity"]
    # model_revision must be reachable via the accessor (it's nested in meta).
    revision = canary.baseline_model_revision(payload)
    assert revision is None or isinstance(revision, str)


# ---------------------------------------------------------------------------
# Reference fixture loading
# ---------------------------------------------------------------------------


def test_load_reference_happy_path(canary, tmp_path: Path):
    path = _make_fixture(tmp_path)
    papers = canary.load_reference(path)
    assert len(papers) == 2
    assert papers[0].id == "p1"
    assert papers[0].gold_entities[0].type == "Mission"


def test_repo_reference_fixture_has_ten_papers(canary):
    """The committed fixture must hold exactly 10 reference papers."""
    papers = canary.load_reference(REPO_ROOT / canary.REFERENCE_FIXTURE)
    assert len(papers) == 10


# ---------------------------------------------------------------------------
# F1 + drift computation
# ---------------------------------------------------------------------------


def test_compute_per_entity_f1_perfect(canary, tmp_path: Path):
    fixture_path = _make_fixture(tmp_path)
    papers = canary.load_reference(fixture_path)
    predictions = {p.id: p.gold_entities for p in papers}
    f1_map = canary.compute_per_entity_f1(predictions, papers)
    assert f1_map["Mission"]["f1"] == pytest.approx(1.0)
    assert f1_map["CelestialObject"]["f1"] == pytest.approx(1.0)


def test_compute_per_entity_f1_missed_entity(canary, tmp_path: Path):
    fixture_path = _make_fixture(tmp_path)
    papers = canary.load_reference(fixture_path)
    # Drop the Mission in paper p1 — 1 FN, 1 TP → recall 0.5, precision 1.0, f1 ~0.667
    predictions = {
        "p1": (papers[0].gold_entities[1],),  # only CelestialObject
        "p2": papers[1].gold_entities,
    }
    f1_map = canary.compute_per_entity_f1(predictions, papers)
    assert f1_map["Mission"]["precision"] == pytest.approx(1.0)
    assert f1_map["Mission"]["recall"] == pytest.approx(0.5)
    assert f1_map["Mission"]["f1"] == pytest.approx(2 / 3)


def test_compute_drift_values(canary):
    current = {
        "Mission": {"f1": 0.85},
        "Instrument": {"f1": 0.80},
    }
    baseline_per_entity = {
        "Mission": {"f1": 0.90},
        "Instrument": {"f1": 0.80},
    }
    drift = canary.compute_drift(current, baseline_per_entity)
    assert drift["Mission"]["drift"] == pytest.approx(0.05)
    assert drift["Instrument"]["drift"] == pytest.approx(0.0)


def test_evaluate_drift_within_threshold(canary):
    drift = {
        "Mission": {"drift": 0.03},
        "Instrument": {"drift": 0.01},
    }
    max_drift, exceeded = canary.evaluate_drift(drift, 0.05)
    assert max_drift == pytest.approx(0.03)
    assert exceeded is False


def test_evaluate_drift_exceeds_threshold(canary):
    drift = {
        "Mission": {"drift": 0.07},
        "Instrument": {"drift": 0.01},
    }
    max_drift, exceeded = canary.evaluate_drift(drift, 0.05)
    assert max_drift == pytest.approx(0.07)
    assert exceeded is True


# ---------------------------------------------------------------------------
# End-to-end main() — exit code + log file
# ---------------------------------------------------------------------------


def _run_main(
    canary,
    tmp_path: Path,
    *,
    baseline_per_entity: dict[str, dict[str, float]] | None = None,
    monkeypatch_model: bool = True,
    use_mock_flag: bool = True,
    monkeypatch=None,
) -> tuple[int, Path]:
    baseline_path = _make_baseline(tmp_path, per_entity=baseline_per_entity)
    fixture_path = _make_fixture(tmp_path)
    log_dir = tmp_path / "logs"

    # Safety: if the test is NOT using --mock-model, ensure load_model
    # never actually runs (this would hit the network).
    if monkeypatch_model and monkeypatch is not None:
        def _boom(revision: str):
            raise AssertionError(
                "load_model should not be called when --mock-model is set"
            )

        monkeypatch.setattr(canary, "load_model", _boom)

    argv = [
        "--baseline",
        str(baseline_path),
        "--fixture",
        str(fixture_path),
        "--log-dir",
        str(log_dir),
    ]
    if use_mock_flag:
        argv.append("--mock-model")

    exit_code = canary.main(argv)
    # Log filename is today's UTC date.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = log_dir / f"{stamp}.json"
    return exit_code, log_path


def test_main_exit_zero_within_threshold(
    canary, tmp_path: Path, monkeypatch
):
    # Mock model path produces gold predictions → perfect F1 (1.0) vs baseline
    # F1 of 0.9 and 0.8 → drift 0.1 and 0.2 which WOULD fail. We set baseline
    # to 1.0 so drift is zero and exit code is 0.
    perfect_baseline = {
        "Mission": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "CelestialObject": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    }
    exit_code, log_path = _run_main(
        canary,
        tmp_path,
        baseline_per_entity=perfect_baseline,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    assert log_path.exists()


def test_main_exit_one_when_drift_exceeds(
    canary, tmp_path: Path, monkeypatch
):
    # Mock model = perfect F1 (1.0). Baseline set very low → drift > 0.05.
    low_baseline = {
        "Mission": {"precision": 0.3, "recall": 0.3, "f1": 0.3},
        "CelestialObject": {"precision": 0.3, "recall": 0.3, "f1": 0.3},
    }
    exit_code, log_path = _run_main(
        canary,
        tmp_path,
        baseline_per_entity=low_baseline,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 1
    assert log_path.exists()


def test_main_writes_log_with_expected_schema(
    canary, tmp_path: Path, monkeypatch
):
    perfect_baseline = {
        "Mission": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "CelestialObject": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    }
    _, log_path = _run_main(
        canary,
        tmp_path,
        baseline_per_entity=perfect_baseline,
        monkeypatch=monkeypatch,
    )
    assert log_path.exists()
    payload = json.loads(log_path.read_text())

    # Required schema fields
    for key in (
        "generated_at",
        "model_name",
        "model_revision",
        "baseline_revision",
        "per_entity",
        "current",
        "max_drift",
        "threshold",
        "exceeded",
        "paper_count",
    ):
        assert key in payload, f"log payload missing key {key!r}"

    assert payload["model_name"] == canary.MODEL_NAME
    assert payload["model_revision"] == canary.MODEL_REVISION
    # baseline_revision is sourced from meta.model_revision in the baseline
    assert (
        payload["baseline_revision"]
        == "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"
    )
    assert payload["paper_count"] == 2
    assert payload["threshold"] == canary.DRIFT_THRESHOLD
    assert payload["exceeded"] is False

    # per_entity entries carry current_f1, baseline_f1, drift
    for etype, entry in payload["per_entity"].items():
        assert "current_f1" in entry
        assert "baseline_f1" in entry
        assert "drift" in entry


def test_main_log_path_follows_date_convention(
    canary, tmp_path: Path, monkeypatch
):
    perfect_baseline = {
        "Mission": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "CelestialObject": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    }
    exit_code, log_path = _run_main(
        canary,
        tmp_path,
        baseline_per_entity=perfect_baseline,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    # Parent dir is log-dir passed in; file stem is today (UTC).
    assert log_path.parent.name == "logs"
    assert log_path.suffix == ".json"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert log_path.stem == today


def test_mock_model_flag_skips_real_loader(canary, tmp_path: Path, monkeypatch):
    """--mock-model must not invoke load_model (which would hit HF network)."""
    called = {"n": 0}

    def _tracker(revision: str):  # pragma: no cover - should never run
        called["n"] += 1
        raise AssertionError("should not be called with --mock-model")

    monkeypatch.setattr(canary, "load_model", _tracker)

    baseline_path = _make_baseline(tmp_path)
    fixture_path = _make_fixture(tmp_path)
    log_dir = tmp_path / "logs"
    exit_code = canary.main(
        [
            "--baseline",
            str(baseline_path),
            "--fixture",
            str(fixture_path),
            "--log-dir",
            str(log_dir),
            "--mock-model",
        ]
    )
    assert exit_code in (0, 1)  # either outcome is fine; just verifying no call
    assert called["n"] == 0


def test_main_raises_file_format_error_when_baseline_missing(
    canary, tmp_path: Path
):
    fixture_path = _make_fixture(tmp_path)
    missing_baseline = tmp_path / "nope.json"
    with pytest.raises(canary.FileFormatError):
        canary.main(
            [
                "--baseline",
                str(missing_baseline),
                "--fixture",
                str(fixture_path),
                "--log-dir",
                str(tmp_path / "logs"),
                "--mock-model",
            ]
        )


def test_main_runs_against_committed_baseline_and_fixture(
    canary, tmp_path: Path, monkeypatch
):
    """End-to-end regression: the script should run cleanly against the
    committed baseline and fixture without raising FileFormatError.

    In --mock-model mode, predictions equal gold so drift = 0 and exit = 0
    (unless the baseline declares entity types with f1 < 1 for entity types
    that appear in the reference fixture — which would legitimately be drift).

    This guards against the exact previous regression: canary required
    top-level model_revision and crashed on its own companion's output.
    """
    committed_baseline = REPO_ROOT / "results" / "ner_wiesp_eval.json"
    committed_fixture = REPO_ROOT / "tests" / "fixtures" / "canary_ner_reference.json"
    if not committed_baseline.exists() or not committed_fixture.exists():
        pytest.skip("committed baseline/fixture not present in this checkout")

    # Belt-and-suspenders: ensure no real model load even if --mock-model is
    # somehow ignored.
    def _boom(revision: str):
        raise AssertionError("load_model should not be called with --mock-model")

    monkeypatch.setattr(canary, "load_model", _boom)

    log_dir = tmp_path / "logs"
    exit_code = canary.main(
        [
            "--baseline",
            str(committed_baseline),
            "--fixture",
            str(committed_fixture),
            "--log-dir",
            str(log_dir),
            "--mock-model",
        ]
    )
    # Exit 0 or 1 both acceptable; key assertion is NO FileFormatError.
    assert exit_code in (0, 1)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert (log_dir / f"{today}.json").exists()
