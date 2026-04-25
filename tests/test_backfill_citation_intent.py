"""Tests for scripts/backfill_citation_intent.py.

Covers PRD MH-1 work-unit acceptance criterion (7):
  (a) dry-run plan
  (b) resume logic skips already-classified rows
  (c) batch failure does not advance offset
  (d) validation-sample export shape

DB is mocked with MagicMock — no live PostgreSQL. The classifier is a
deterministic fake — no transformers/model load.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "backfill_citation_intent.py"
)


def _load_script_module() -> Any:
    """Import the script as a module so its functions are testable.

    The script is not a package; pyproject.toml puts ``scripts`` on
    pythonpath, but loading via a fresh spec lets us be explicit and
    import-isolate.
    """
    spec = importlib.util.spec_from_file_location(
        "backfill_citation_intent_script", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backfill = _load_script_module()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeClassifier:
    """Deterministic classifier — returns a fixed cycle of intents."""

    intents: list[str] = field(
        default_factory=lambda: ["background", "method", "result_comparison"]
    )
    calls: list[list[str]] = field(default_factory=list)
    fail_on_call: int | None = None

    def classify_intent(self, text: str) -> str:  # pragma: no cover — unused here
        return self.intents[0]

    def classify_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("Simulated classifier failure")
        out: list[str] = []
        for i, _ in enumerate(texts):
            out.append(self.intents[i % len(self.intents)])
        return out


class FakeCursor:
    """Minimal cursor mock supporting .execute/.fetchall and the `with` protocol."""

    def __init__(self, table_rows: list[tuple], updates_log: list[tuple]) -> None:
        self._table_rows = table_rows
        self._updates_log = updates_log
        self._last_fetch: list[tuple] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        sql_norm = " ".join(sql.lower().split())
        params = params or ()
        if sql_norm.startswith("select id, source_bibcode"):
            limit = params[0] if params else len(self._table_rows)
            unclassified = [
                r for r in self._table_rows if r[5] is None  # intent slot
            ]
            unclassified.sort(key=lambda r: r[0])
            self._last_fetch = [
                (r[0], r[1], r[2], r[3], r[4]) for r in unclassified[:limit]
            ]
        elif sql_norm.startswith("update citation_contexts set intent"):
            new_intent, row_id = params
            self._updates_log.append((row_id, new_intent))
            for i, r in enumerate(self._table_rows):
                if r[0] == row_id:
                    self._table_rows[i] = (r[0], r[1], r[2], r[3], r[4], new_intent)
                    break
        elif sql_norm.startswith("insert into ingest_log"):
            self._last_fetch = []
        elif sql_norm.startswith("update ingest_log"):
            self._last_fetch = []
        elif sql_norm.startswith("select count(*) from citation_contexts"):
            n = sum(1 for r in self._table_rows if r[5] is not None)
            self._last_fetch = [(n,)]
        elif sql_norm.startswith("select context_text, intent"):
            limit = params[0] if params else 100
            classified = [r for r in self._table_rows if r[5] is not None]
            # Per-class row_number cap, mirroring the SQL the script issues.
            seen: dict[str, int] = {}
            picked: list[tuple] = []
            for r in classified:
                seen[r[5]] = seen.get(r[5], 0) + 1
                if seen[r[5]] <= limit:
                    picked.append((r[4], r[5], 0.9))
            self._last_fetch = picked
        else:
            self._last_fetch = []

    def fetchall(self) -> list[tuple]:
        return list(self._last_fetch)

    def fetchone(self) -> tuple | None:
        return self._last_fetch[0] if self._last_fetch else None


class FakeConn:
    """Minimal connection mock for the backfill loop."""

    def __init__(self, rows: list[tuple]) -> None:
        # rows: (id, source_bibcode, target_bibcode, char_offset, context_text, intent)
        self._rows = rows
        self.updates: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.autocommit = False

    @property
    def rows(self) -> list[tuple]:
        return self._rows

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._rows, self.updates)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        # On rollback, drop updates that were issued since the last commit.
        # For our purposes it's enough to record the count — the failing-batch
        # test checks that the *underlying rows* are not advanced via re-fetch.
        self.rollbacks += 1
        # Revert any rows that were modified but not yet committed:
        # in practice the FakeCursor mutates in place, so we restore from
        # the updates log since the last commit. The simplest correct
        # behavior is to revert all unclassified-at-start rows whose intent
        # was just set in this batch — we use the updates list as the
        # authoritative log of writes since last commit.
        if not self.updates:
            return
        last_batch_ids = {row_id for row_id, _ in self.updates}
        self.updates.clear()
        for i, r in enumerate(self._rows):
            if r[0] in last_batch_ids:
                self._rows[i] = (r[0], r[1], r[2], r[3], r[4], None)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# (a) dry-run plan
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Guarantee no DB call: get_connection should never be invoked.
    sentinel = MagicMock(side_effect=AssertionError("DB must not be opened in dry-run"))
    monkeypatch.setattr(backfill, "get_connection", sentinel)

    rc = backfill.main(
        ["--dry-run", "--limit", "100", "--dsn", "dbname=scix_test"]
    )
    assert rc == 0

    captured = capsys.readouterr()
    out = captured.out
    assert "Plan: backfill citation_contexts.intent" in out
    assert "limit          : 100" in out
    assert "ingest_log marker" in out
    sentinel.assert_not_called()


def test_dry_run_via_function(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = backfill._parse_args(
        ["--dry-run", "--smoke-test", "--dsn", "dbname=scix_test"]
    )
    plan = backfill._format_plan(cfg)
    assert "limit          : 100" in plan
    assert cfg.dry_run is True
    assert cfg.limit == 100


# ---------------------------------------------------------------------------
# (b) resume logic skips already-classified rows
# ---------------------------------------------------------------------------


def test_resume_skips_already_classified_rows() -> None:
    rows = [
        (1, "src1", "tgt1", 0, "ctx 1", "background"),  # already classified
        (2, "src2", "tgt2", 5, "ctx 2", None),  # NULL — should be picked up
        (3, "src3", "tgt3", 9, "ctx 3", "method"),  # already classified
        (4, "src4", "tgt4", 12, "ctx 4", None),  # NULL — should be picked up
    ]
    conn = FakeConn(rows)
    classifier = FakeClassifier(intents=["result_comparison", "background"])
    cfg = backfill._parse_args(["--dsn", "dbname=scix_test", "--batch-size", "10"])

    total = backfill.run_backfill(cfg, classifier, conn=conn)

    assert total == 2
    classified_ids = {row_id for row_id, _ in conn.updates}
    assert classified_ids == {2, 4}
    # Already-classified rows must remain untouched.
    final_by_id = {r[0]: r[5] for r in conn.rows}
    assert final_by_id[1] == "background"
    assert final_by_id[3] == "method"
    # Newly-classified rows now have the fake intents.
    assert final_by_id[2] == "result_comparison"
    assert final_by_id[4] == "background"
    # Connection was not closed (caller-owned).
    assert conn.closed is False


def test_resume_idempotent_on_second_call() -> None:
    rows = [
        (1, "src1", "tgt1", 0, "ctx 1", None),
        (2, "src2", "tgt2", 5, "ctx 2", None),
    ]
    conn = FakeConn(rows)
    classifier = FakeClassifier(intents=["background"])
    cfg = backfill._parse_args(["--dsn", "dbname=scix_test", "--batch-size", "10"])

    first = backfill.run_backfill(cfg, classifier, conn=conn)
    second = backfill.run_backfill(cfg, classifier, conn=conn)

    assert first == 2
    assert second == 0  # nothing left with NULL intent


# ---------------------------------------------------------------------------
# (c) batch failure does not advance offset
# ---------------------------------------------------------------------------


def test_batch_failure_does_not_advance_offset() -> None:
    rows = [
        (1, "s", "t", 0, "ctx 1", None),
        (2, "s", "t", 1, "ctx 2", None),
        (3, "s", "t", 2, "ctx 3", None),
        (4, "s", "t", 3, "ctx 4", None),
    ]
    conn = FakeConn(rows)
    # Fail on the very first classify_batch call — no rows advance.
    classifier = FakeClassifier(fail_on_call=1)
    cfg = backfill._parse_args(["--dsn", "dbname=scix_test", "--batch-size", "2"])

    with pytest.raises(RuntimeError, match="Simulated classifier failure"):
        backfill.run_backfill(cfg, classifier, conn=conn)

    # All rows must still have NULL intent (offset has not advanced).
    assert all(r[5] is None for r in conn.rows)
    assert conn.rollbacks >= 1


def test_db_failure_mid_batch_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        (1, "s", "t", 0, "ctx 1", None),
        (2, "s", "t", 1, "ctx 2", None),
    ]
    conn = FakeConn(rows)
    classifier = FakeClassifier(intents=["background"])
    cfg = backfill._parse_args(["--dsn", "dbname=scix_test", "--batch-size", "2"])

    def explode(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("Simulated UPDATE failure")

    monkeypatch.setattr(backfill, "_update_batch_in_transaction", explode)

    with pytest.raises(RuntimeError, match="Simulated UPDATE failure"):
        backfill.run_backfill(cfg, classifier, conn=conn)

    assert all(r[5] is None for r in conn.rows)
    assert conn.rollbacks >= 1


# ---------------------------------------------------------------------------
# (d) validation-sample export shape
# ---------------------------------------------------------------------------


def test_validate_sample_export_shape(tmp_path: Path) -> None:
    rows = [
        (1, "s", "t", 0, "ctx background 1", "background"),
        (2, "s", "t", 1, "ctx background 2", "background"),
        (3, "s", "t", 2, "ctx background 3", "background"),
        (4, "s", "t", 3, "ctx method 1", "method"),
        (5, "s", "t", 4, "ctx method 2", "method"),
        (6, "s", "t", 5, "ctx method 3", "method"),
        (7, "s", "t", 6, "ctx result_comparison 1", "result_comparison"),
        (8, "s", "t", 7, "ctx result_comparison 2", "result_comparison"),
        (9, "s", "t", 8, "ctx result_comparison 3", "result_comparison"),
    ]
    conn = FakeConn(rows)
    report_path = tmp_path / "mh1_intent_validation.md"
    cfg = backfill._parse_args(
        [
            "--dsn",
            "dbname=scix_test",
            "--validate-sample",
            "6",
            "--report-path",
            str(report_path),
        ]
    )

    out = backfill.export_validation_sample(cfg, conn=conn)

    assert out == report_path
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    # Required column header (per acceptance criterion 5).
    assert "snippet" in text
    assert "predicted_intent" in text
    assert "confidence" in text
    assert "manual_label_placeholder" in text
    # Methodology + throughput sections (per acceptance criterion 8).
    assert "## Methodology" in text
    assert "Throughput" in text
    assert "GPU-day" in text or "GPU-hours" in text
    # All three intent classes appear in the table — stratification worked.
    assert text.count("background") >= 1
    assert text.count("method") >= 1
    assert text.count("result_comparison") >= 1


def test_stratified_sample_distributes_evenly() -> None:
    raw = [
        ("s1", "background", 0.9),
        ("s2", "background", 0.8),
        ("s3", "method", 0.7),
        ("s4", "method", 0.6),
        ("s5", "result_comparison", 0.5),
        ("s6", "result_comparison", 0.4),
    ]
    samples = backfill._stratified_sample(raw, n=3, seed=42)
    classes = [s.predicted_intent for s in samples]
    # 3 across 3 classes → exactly one of each.
    assert sorted(classes) == ["background", "method", "result_comparison"]


def test_stratified_sample_handles_empty() -> None:
    assert backfill._stratified_sample([], n=10, seed=0) == []


def test_stratified_sample_remainder_goes_to_first_classes() -> None:
    raw = [
        ("a", "background", 0.9),
        ("b", "background", 0.8),
        ("c", "method", 0.7),
        ("d", "method", 0.6),
        ("e", "result_comparison", 0.5),
        ("f", "result_comparison", 0.4),
    ]
    # 4 across 3 classes → 2 in 'background' (alpha-first), 1 each in others.
    samples = backfill._stratified_sample(raw, n=4, seed=42)
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.predicted_intent] = counts.get(s.predicted_intent, 0) + 1
    assert counts["background"] == 2
    assert counts["method"] == 1
    assert counts["result_comparison"] == 1


# ---------------------------------------------------------------------------
# CLI parsing — flag coverage
# ---------------------------------------------------------------------------


def test_all_required_flags_parse() -> None:
    cfg = backfill._parse_args(
        [
            "--batch-size",
            "128",
            "--limit",
            "1000",
            "--resume",
            "--validate-sample",
            "500",
            "--dry-run",
            "--dsn",
            "dbname=scix_test",
        ]
    )
    assert cfg.batch_size == 128
    assert cfg.limit == 1000
    assert cfg.resume is True
    assert cfg.validate_sample == 500
    assert cfg.dry_run is True


def test_smoke_test_flag_caps_at_100() -> None:
    cfg = backfill._parse_args(["--smoke-test", "--dsn", "dbname=scix_test"])
    assert cfg.limit == 100


def test_smoke_test_does_not_raise_existing_lower_limit() -> None:
    cfg = backfill._parse_args(
        ["--smoke-test", "--limit", "50", "--dsn", "dbname=scix_test"]
    )
    assert cfg.limit == 50
