"""Unit tests for the MH-0 Day-1 latency probe.

Covers acceptance criteria #5 of the work unit:

(a) Report format parses (markdown sections present, p50/p95 surfaced).
(b) Verdict logic produces PASS for fixtures with p50≈2s, p95≈5s.
(c) Verdict logic produces FAIL for fixtures with p50≈4s.
(d) Pivot block is emitted on FAIL but suppressed on PASS.

Tests use a fake dispatcher with pre-canned per-turn durations and a fake
clock so no real ``claude -p`` subprocess is invoked. No OAuth, no
network, no paid SDK.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loader — the probe lives under ``scripts/`` (not in the package),
# so we import it by file path. Keeping this in one place isolates test
# files from sys.path manipulation.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROBE_PATH = _REPO_ROOT / "scripts" / "mh0_latency_probe.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("mh0_latency_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mh0_latency_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe_mod():
    return _load_probe_module()


# ---------------------------------------------------------------------------
# Fakes — fake dispatcher + fake clock so timings are deterministic
# ---------------------------------------------------------------------------


@dataclass
class FakeClock:
    """Deterministic monotonic clock backed by a list of durations.

    Each call to ``__call__`` advances the clock. The probe calls the
    clock twice per turn (start + end), so we feed pairs.
    """

    durations: list[float]
    _now: float = 0.0
    _idx: int = 0

    def __call__(self) -> float:
        if self._idx == 0:
            self._idx += 1
            return self._now
        # End of the (start, end) pair: advance by next duration.
        d = self.durations[(self._idx // 2) % len(self.durations)]
        self._idx += 1
        self._now += d
        return self._now


@dataclass
class FakeDispatcher:
    """Records calls; returns a stub JudgeScore-shaped object.

    The probe doesn't introspect the return value — it only times the
    awaited call — so a sentinel object is fine.
    """

    calls: list = field(default_factory=list)

    async def judge(self, triple):  # noqa: ANN001 — duck-typed
        self.calls.append(triple)
        return object()


def _build_clock_for_durations(durations: list[float]):
    """Build a clock callable that yields (start, end) pairs producing
    the requested elapsed values."""
    state = {"i": 0, "t": 0.0}

    def clock() -> float:
        i = state["i"]
        if i % 2 == 0:
            # Start of pair: return current time.
            state["i"] += 1
            return state["t"]
        # End of pair: advance by the next duration.
        d = durations[(i // 2) % len(durations)]
        state["t"] += d
        state["i"] += 1
        return state["t"]

    return clock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_percentile_basic(probe_mod):
    # Sanity check the percentile helper before relying on it elsewhere.
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert probe_mod.compute_percentile(vals, 50.0) == pytest.approx(3.0)
    assert probe_mod.compute_percentile(vals, 95.0) == pytest.approx(4.8)


def test_decide_verdict_pass_boundary(probe_mod):
    # Exactly at threshold counts as PASS (≤, not <).
    assert probe_mod.decide_verdict(3.0, 6.0) == "PASS"
    assert probe_mod.decide_verdict(2.999, 5.999) == "PASS"


def test_decide_verdict_fail_p50(probe_mod):
    assert probe_mod.decide_verdict(3.01, 5.0) == "FAIL"


def test_decide_verdict_fail_p95(probe_mod):
    assert probe_mod.decide_verdict(2.0, 6.01) == "FAIL"


def test_pass_fixture_median_2s_p95_5s(probe_mod, tmp_path):
    """Acceptance #5(b): median 2s, p95 5s → PASS."""
    # 20 timings: 18 values around 2s, 2 spike values to push p95 to ~5s.
    durations = [2.0] * 18 + [5.0, 5.0]

    dispatcher = FakeDispatcher()
    probe = probe_mod.LatencyProbe(
        dispatcher=dispatcher,
        clock=_build_clock_for_durations(durations),
    )
    timings = asyncio.run(probe.measure(runs=4, turns=5))
    assert len(timings) == 20
    assert len(dispatcher.calls) == 20

    result = probe_mod.aggregate(timings, runs=4, turns=5, dry_run=False)
    assert result.p50_s == pytest.approx(2.0)
    assert result.p95_s == pytest.approx(5.0, abs=0.2)
    assert result.verdict == "PASS"


def test_fail_fixture_median_4s(probe_mod):
    """Acceptance #5(c): median 4s → FAIL."""
    durations = [4.0] * 5

    dispatcher = FakeDispatcher()
    probe = probe_mod.LatencyProbe(
        dispatcher=dispatcher,
        clock=_build_clock_for_durations(durations),
    )
    timings = asyncio.run(probe.measure(runs=1, turns=5))
    result = probe_mod.aggregate(timings, runs=1, turns=5, dry_run=False)
    assert result.p50_s == pytest.approx(4.0)
    assert result.verdict == "FAIL"


def test_report_format_parses_pass(probe_mod, tmp_path):
    """Acceptance #5(a) + (d): markdown sections present; PASS suppresses pivot."""
    durations = [2.0] * 18 + [5.0, 5.0]
    dispatcher = FakeDispatcher()
    probe = probe_mod.LatencyProbe(
        dispatcher=dispatcher, clock=_build_clock_for_durations(durations)
    )
    timings = asyncio.run(probe.measure(runs=4, turns=5))
    result = probe_mod.aggregate(timings, runs=4, turns=5, dry_run=False)

    report_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    probe_mod.write_report(result, report_path, json_path)

    text = report_path.read_text(encoding="utf-8")
    # All four mandated sections present.
    assert "## Methodology" in text
    assert "## Measurements" in text
    assert "## Verdict" in text
    assert "## Pivot Recommendation" in text
    # Verdict surfaced.
    assert "**PASS**" in text
    # Pivot block suppressed on PASS — the FAIL-specific text from
    # ``PIVOT_TEXT`` (e.g. "Pivot v1 to standalone MCP tools") must NOT
    # appear when the verdict is PASS.
    assert "Pivot v1 to standalone MCP tools" not in text
    assert "Not triggered" in text

    # JSON sidecar parses and carries the same numbers.
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["p50_threshold_s"] == 3.0
    assert payload["p95_threshold_s"] == 6.0
    assert len(payload["timings"]) == 20


def test_report_emits_pivot_block_on_fail(probe_mod, tmp_path):
    """Acceptance #5(d): FAIL report emits the pivot block."""
    durations = [4.0] * 5
    dispatcher = FakeDispatcher()
    probe = probe_mod.LatencyProbe(
        dispatcher=dispatcher, clock=_build_clock_for_durations(durations)
    )
    timings = asyncio.run(probe.measure(runs=1, turns=5))
    result = probe_mod.aggregate(timings, runs=1, turns=5, dry_run=False)

    report_path = tmp_path / "fail.md"
    json_path = tmp_path / "fail.json"
    probe_mod.write_report(result, report_path, json_path)

    text = report_path.read_text(encoding="utf-8")
    assert "**FAIL**" in text
    # Pivot block content (from PIVOT_TEXT) must be present on FAIL.
    assert "Pivot v1 to standalone MCP tools" in text
    assert "claim_blame" in text
    assert "find_replications" in text
    # The "Not triggered" PASS-only stub must not be present.
    assert "Not triggered" not in text


def test_dry_run_synthesizes_pass_report(probe_mod, tmp_path):
    """The dry-run path must emit a PASS-shaped report without invoking
    the dispatcher (acceptance #6 — supports CI verification of report
    shape without OAuth)."""
    timings = probe_mod._synthesize_dry_run_timings(runs=5, turns=5)
    result = probe_mod.aggregate(timings, runs=5, turns=5, dry_run=True)
    assert result.verdict == "PASS"

    report_path = tmp_path / "dry.md"
    json_path = tmp_path / "dry.json"
    probe_mod.write_report(result, report_path, json_path)
    text = report_path.read_text(encoding="utf-8")
    assert "Mode: dry-run" in text
    assert "**PASS**" in text


def test_main_dry_run_writes_files(probe_mod, tmp_path, monkeypatch):
    """End-to-end: the CLI dry-run path produces both report + JSON."""
    report = tmp_path / "out" / "mh0.md"
    json_out = tmp_path / "out" / "mh0.json"
    rc = probe_mod.main(
        [
            "--dry-run",
            "--runs",
            "5",
            "--turns",
            "5",
            "--report",
            str(report),
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0  # PASS exits 0
    assert report.exists()
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["dry_run"] is True


def test_measure_rejects_zero_runs(probe_mod):
    probe = probe_mod.LatencyProbe(dispatcher=FakeDispatcher())
    with pytest.raises(ValueError):
        asyncio.run(probe.measure(runs=0, turns=5))
    with pytest.raises(ValueError):
        asyncio.run(probe.measure(runs=5, turns=0))
