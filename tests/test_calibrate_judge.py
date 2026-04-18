"""Tests for ``scripts/calibrate_judge.py``.

No DB access — ``fetch_snippets_for`` is stubbed. Dispatcher is stubbed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import calibrate_judge  # noqa: E402
from scix.eval.persona_judge import JudgeScore, JudgeTriple, StubDispatcher  # noqa: E402


def _write_seed_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "bibcode", "human_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestReadSeedCsv:
    def test_reads_and_parses(self, tmp_path: Path) -> None:
        p = tmp_path / "seed.csv"
        _write_seed_csv(
            p,
            [
                {"query": "q1", "bibcode": "B1", "human_score": "3"},
                {"query": "q2", "bibcode": "B2", "human_score": "0"},
            ],
        )
        rows = calibrate_judge.read_seed_csv(p)
        assert len(rows) == 2
        assert rows[0].human_score == 3
        assert rows[1].human_score == 0

    def test_rejects_out_of_range_human_score(self, tmp_path: Path) -> None:
        p = tmp_path / "seed.csv"
        _write_seed_csv(p, [{"query": "q", "bibcode": "B", "human_score": "7"}])
        with pytest.raises(ValueError):
            calibrate_judge.read_seed_csv(p)

    def test_rejects_missing_column(self, tmp_path: Path) -> None:
        p = tmp_path / "seed.csv"
        with p.open("w", encoding="utf-8") as f:
            f.write("query,bibcode\n")
            f.write("q,B\n")
        with pytest.raises(ValueError):
            calibrate_judge.read_seed_csv(p)


class TestDriftWatchLog:
    def test_appends_entry_and_creates_parent(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "judge_calibration_log.jsonl"
        calibrate_judge.append_drift_entry(
            log_path=log_path,
            prompt_version="v1",
            kappa=0.73,
            spearman=0.81,
            n_triples=50,
        )
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["prompt_version"] == "v1"
        assert rec["kappa"] == 0.73
        assert rec["spearman"] == 0.81
        assert rec["n_triples"] == 50
        assert "run_date" in rec

    def test_appends_do_not_overwrite(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        calibrate_judge.append_drift_entry(
            log_path=log_path, prompt_version="v1", kappa=0.5, spearman=0.6, n_triples=10
        )
        calibrate_judge.append_drift_entry(
            log_path=log_path, prompt_version="v2", kappa=0.7, spearman=0.8, n_triples=20
        )
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2


class TestCalibrationRun:
    def test_reports_kappa_and_spearman_with_stub(self, tmp_path: Path) -> None:
        seed = tmp_path / "seed.csv"
        _write_seed_csv(
            seed,
            [
                {"query": "q1", "bibcode": "B1", "human_score": "2"},
                {"query": "q2", "bibcode": "B2", "human_score": "2"},
                {"query": "q3", "bibcode": "B3", "human_score": "2"},
                {"query": "q4", "bibcode": "B4", "human_score": "2"},
            ],
        )
        log_path = tmp_path / "log.jsonl"

        # Fake snippet fetcher — never touches DB
        def fake_fetch(bibcodes: list[str]) -> dict[str, str]:
            return {b: f"title for {b}\nabstract" for b in bibcodes}

        dispatcher = StubDispatcher(fixed_score=2, reason="stub")

        report = calibrate_judge.run_calibration(
            seed_path=seed,
            log_path=log_path,
            snippet_fetcher=fake_fetch,
            dispatcher=dispatcher,
            prompt_version="test-v1",
            max_concurrency=2,
            max_retries=0,
        )

        # All humans rated 2, stub rates 2 → perfect agreement
        assert report.kappa == pytest.approx(1.0, abs=1e-9)
        # Drift entry written
        assert log_path.exists()

    def test_warns_when_kappa_below_threshold(self, tmp_path: Path) -> None:
        seed = tmp_path / "seed.csv"
        _write_seed_csv(
            seed,
            [
                {"query": "q1", "bibcode": "B1", "human_score": "0"},
                {"query": "q2", "bibcode": "B2", "human_score": "3"},
                {"query": "q3", "bibcode": "B3", "human_score": "0"},
                {"query": "q4", "bibcode": "B4", "human_score": "3"},
            ],
        )
        log_path = tmp_path / "log.jsonl"

        def fake_fetch(bibcodes: list[str]) -> dict[str, str]:
            return {b: "snippet" for b in bibcodes}

        # Stub that picks opposite extreme
        class _WrongDispatcher:
            async def judge(self, triple: JudgeTriple) -> JudgeScore:
                return JudgeScore(score=3 if triple.bibcode in {"B1", "B3"} else 0,
                                  reason="wrong")

        report = calibrate_judge.run_calibration(
            seed_path=seed,
            log_path=log_path,
            snippet_fetcher=fake_fetch,
            dispatcher=_WrongDispatcher(),
            prompt_version="test-v1",
            max_concurrency=2,
            max_retries=0,
        )
        assert report.kappa < 0.6
        assert report.trustworthy is False
