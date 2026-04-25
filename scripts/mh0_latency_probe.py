#!/usr/bin/env python3
"""MH-0 Day-1 latency probe (PRD: scix_deep_search_v1, amendment A1).

Measures per-turn dispatcher overhead for ``ClaudeSubprocessDispatcher``
(see :mod:`scix.eval.persona_judge` line 354) — i.e. ``claude -p``
subprocess startup + OAuth + persona context replay — *excluding* actual
tool execution. The premortem amendment A1 mandates this experiment as a
blocking gate before Phase 1: if the dispatcher overhead alone is too
costly to support a 25-turn investigation loop, MH-7's harness shape is
not viable and v1 must pivot to standalone MCP tools (``claim_blame`` /
``find_replications``) without the agent loop.

Usage::

    # Real probe (requires ``claude`` CLI + OAuth session):
    python scripts/mh0_latency_probe.py --runs 5 --turns 5 \\
        --report docs/eval/mh0_latency_probe.md

    # Dry-run (no OAuth; emits sample-shaped report and JSON):
    python scripts/mh0_latency_probe.py --dry-run \\
        --report docs/eval/mh0_latency_probe.md

The probe writes:

- A markdown report (``--report``) with sections Methodology,
  Measurements, Verdict, Pivot Recommendation.
- A JSON sidecar (``--json-out``, defaults to ``<report>.json``) with the
  raw per-turn timings and computed p50 / p95.

Acceptance (amendment A1): PASS iff p50 per-turn overhead <= 3.0s AND p95
per-turn overhead <= 6.0s. Otherwise FAIL — and the report explicitly
states v1 must pivot to a non-agent shape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

# Ensure the repo's ``src/`` layout is importable when invoked as
# ``python scripts/mh0_latency_probe.py`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger("mh0_latency_probe")

# ---------------------------------------------------------------------------
# Acceptance thresholds (amendment A1)
# ---------------------------------------------------------------------------

P50_THRESHOLD_S: float = 3.0
P95_THRESHOLD_S: float = 6.0

# ---------------------------------------------------------------------------
# Fixture: a 5-turn replay of representative judge triples
# ---------------------------------------------------------------------------

# The fixture content is intentionally small and synthetic — what matters
# for amendment A1 is *dispatcher overhead*, not what the persona returns.
# Each entry is (query, bibcode, snippet).
DEFAULT_FIXTURE: tuple[tuple[str, str, str], ...] = (
    (
        "What is the local Hubble constant from Cepheid distance ladder?",
        "2022ApJ...934L...7R",
        "Title: A Comprehensive Measurement of the Local Value of the "
        "Hubble Constant. Abstract: We present a new measurement ...",
    ),
    (
        "Which papers report tensions in H0 from CMB vs distance ladder?",
        "2020A&A...641A...6P",
        "Title: Planck 2018 results. VI. Cosmological parameters. "
        "Abstract: We present cosmological parameter results ...",
    ),
    (
        "Have BICEP2 primordial gravitational wave claims been retracted?",
        "2014PhRvL.112x1101B",
        "Title: Detection of B-Mode Polarization at Degree Angular "
        "Scales by BICEP2. Abstract: We report on B-mode polarization ...",
    ),
    (
        "What corrections were issued for the SH0ES LMC Cepheid sample?",
        "2019ApJ...876...85R",
        "Title: Large Magellanic Cloud Cepheid Standards Provide a 1% "
        "Foundation. Abstract: We present a recalibration ...",
    ),
    (
        "Which JWST early-universe galaxy candidates were later contested?",
        "2023Natur.616..266L",
        "Title: A population of red candidate massive galaxies "
        "~600 Myr after the Big Bang. Abstract: ...",
    ),
)


# ---------------------------------------------------------------------------
# Dispatcher protocol — narrow contract the probe relies on
# ---------------------------------------------------------------------------


class _DispatcherLike(Protocol):
    """Subset of :class:`ClaudeSubprocessDispatcher` used by the probe.

    Any object with an awaitable ``judge(triple)`` method works — the
    probe only times the call boundary. Tests pass a fake implementation;
    production passes the real :class:`ClaudeSubprocessDispatcher`.
    """

    async def judge(self, triple): ...  # noqa: ANN001 — duck-typed for tests


# ---------------------------------------------------------------------------
# Probe core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnTiming:
    """One per-turn measurement.

    Attributes:
        run_index: Which run (0-based) this turn came from.
        turn_index: Which turn within the run (0-based).
        elapsed_s: Wall-clock seconds for ``await dispatcher.judge(...)``.
            This is the dispatcher overhead we are gating on — subprocess
            startup, OAuth handshake, and persona context replay.
    """

    run_index: int
    turn_index: int
    elapsed_s: float


@dataclass(frozen=True)
class ProbeResult:
    """Aggregate result of a probe run.

    Attributes:
        timings: Every measured turn (run × turn flattened).
        p50_s: 50th-percentile per-turn overhead.
        p95_s: 95th-percentile per-turn overhead.
        verdict: ``"PASS"`` or ``"FAIL"``.
        runs: Number of runs in the probe.
        turns: Number of turns per run.
        dry_run: ``True`` if the probe synthesized timings instead of
            invoking a dispatcher.
    """

    timings: tuple[TurnTiming, ...]
    p50_s: float
    p95_s: float
    verdict: str
    runs: int
    turns: int
    dry_run: bool

    def to_jsonable(self) -> dict:
        return {
            "p50_s": self.p50_s,
            "p95_s": self.p95_s,
            "verdict": self.verdict,
            "runs": self.runs,
            "turns": self.turns,
            "dry_run": self.dry_run,
            "p50_threshold_s": P50_THRESHOLD_S,
            "p95_threshold_s": P95_THRESHOLD_S,
            "timings": [
                {
                    "run": t.run_index,
                    "turn": t.turn_index,
                    "elapsed_s": t.elapsed_s,
                }
                for t in self.timings
            ],
        }


def compute_percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile.

    Wraps ``statistics.quantiles`` for the common p50/p95 case but
    handles the small-sample edge cases (n < 2) explicitly. ``pct`` is in
    the range ``(0, 100]``.
    """
    if not values:
        raise ValueError("compute_percentile: empty values")
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    # Linear interpolation between the two surrounding ranks.
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_vals[int(rank)])
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def decide_verdict(p50_s: float, p95_s: float) -> str:
    """Apply the amendment A1 acceptance rule."""
    if p50_s <= P50_THRESHOLD_S and p95_s <= P95_THRESHOLD_S:
        return "PASS"
    return "FAIL"


@dataclass
class LatencyProbe:
    """Measure per-turn dispatcher overhead against a fixture.

    This wraps any :class:`_DispatcherLike` (real or fake). The class
    deliberately does *not* import :class:`ClaudeSubprocessDispatcher` at
    module scope so the probe script and the unit tests can run on hosts
    without ``claude`` installed.

    Attributes:
        dispatcher: The dispatcher under test. Its ``judge`` method is
            awaited once per turn.
        fixture: Sequence of (query, bibcode, snippet) tuples.
        clock: Monotonic-clock callable (defaults to
            :func:`time.perf_counter`). Tests inject a fake clock here.
    """

    dispatcher: _DispatcherLike
    fixture: Sequence[tuple[str, str, str]] = field(default_factory=lambda: DEFAULT_FIXTURE)
    clock: callable = time.perf_counter  # type: ignore[type-arg]

    async def measure(self, *, runs: int, turns: int) -> tuple[TurnTiming, ...]:
        """Run ``runs`` × ``turns`` measurements and return raw timings.

        Reuses fixture entries cyclically when ``turns`` exceeds the
        fixture length — the dispatcher overhead is what's measured, so
        repeating a triple does not bias the result.
        """
        if runs < 1 or turns < 1:
            raise ValueError("runs and turns must both be >= 1")
        # Build a JudgeTriple lazily so this module imports cleanly when
        # ``scix.eval.persona_judge`` is unavailable (e.g. in dry-run on a
        # bare CI worker without project deps installed).
        from scix.eval.persona_judge import JudgeTriple  # local import

        timings: list[TurnTiming] = []
        for run_idx in range(runs):
            for turn_idx in range(turns):
                q, bib, snip = self.fixture[turn_idx % len(self.fixture)]
                triple = JudgeTriple(query=q, bibcode=bib, snippet=snip)
                start = self.clock()
                await self.dispatcher.judge(triple)
                elapsed = self.clock() - start
                timings.append(
                    TurnTiming(
                        run_index=run_idx, turn_index=turn_idx, elapsed_s=float(elapsed)
                    )
                )
        return tuple(timings)


def aggregate(timings: Sequence[TurnTiming], *, runs: int, turns: int, dry_run: bool) -> ProbeResult:
    """Compute p50/p95 + verdict from raw timings."""
    elapsed = [t.elapsed_s for t in timings]
    p50 = compute_percentile(elapsed, 50.0)
    p95 = compute_percentile(elapsed, 95.0)
    return ProbeResult(
        timings=tuple(timings),
        p50_s=p50,
        p95_s=p95,
        verdict=decide_verdict(p50, p95),
        runs=runs,
        turns=turns,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Dry-run synthetic timings — sized to PASS so CI can verify report shape
# ---------------------------------------------------------------------------


def _synthesize_dry_run_timings(*, runs: int, turns: int) -> tuple[TurnTiming, ...]:
    """Generate a deterministic, PASSing fixture timings array.

    The dry-run is for CI / report-shape validation — not measurement.
    We hard-code values that satisfy the A1 thresholds so the dry-run
    report shows what a PASS looks like; FAIL shape is exercised by
    ``tests/test_mh0_latency_probe.py``.
    """
    # Median ~1.8s, p95 ~4.5s — comfortably under thresholds.
    base = [1.5, 1.8, 2.0, 2.2, 4.5]
    timings: list[TurnTiming] = []
    for run_idx in range(runs):
        for turn_idx in range(turns):
            timings.append(
                TurnTiming(
                    run_index=run_idx,
                    turn_index=turn_idx,
                    elapsed_s=base[turn_idx % len(base)],
                )
            )
    return tuple(timings)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


PIVOT_TEXT = (
    "**Pivot Recommendation (per amendment A1):** the per-turn dispatcher "
    "overhead exceeds the acceptance budget. v1 MUST NOT proceed with the "
    "MH-7 30-turn investigation harness as currently shaped. Two options "
    "are pre-authorized in the PRD:\n\n"
    "1. **Replan MH-7** with a persistent dispatcher / batched tool calls / "
    "shorter turn budget (each carries the build-stall risk surfaced in "
    "premortem narrative N2).\n"
    "2. **Pivot v1 to standalone MCP tools** — ship `claim_blame` and "
    "`find_replications` as direct MCP tools without the investigation "
    "loop; the deep-search persona defers to v2.\n\n"
    "This pivot is pre-authorized; no Phase-4 negotiation under sunk-cost "
    "pressure is required. Update `docs/prd/scix_deep_search_v1.md` and "
    "open a follow-up PRD for the chosen path."
)


def render_report(
    result: ProbeResult,
    *,
    pivot_text: str = PIVOT_TEXT,
) -> str:
    """Render the markdown report body."""
    lines: list[str] = []
    lines.append("# MH-0 Day-1 Latency Probe")
    lines.append("")
    lines.append(
        "Per amendment A1 of `docs/prd/scix_deep_search_v1.md` — measures "
        "per-turn dispatcher overhead for `ClaudeSubprocessDispatcher` "
        "(`src/scix/eval/persona_judge.py:354`)."
    )
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"- Fixture: {result.turns}-turn replay of representative "
        "(query, bibcode, snippet) triples."
    )
    lines.append(f"- Runs: {result.runs} (total measured turns: {len(result.timings)}).")
    lines.append(
        "- Each turn awaits `dispatcher.judge(triple)` and records "
        "`time.perf_counter()` delta — i.e. subprocess startup + OAuth "
        "handshake + persona context replay. *Tool execution is excluded "
        "from this probe by construction* (no tool calls inside the "
        "judge subagent)."
    )
    lines.append(
        f"- Acceptance (amendment A1): PASS iff p50 ≤ {P50_THRESHOLD_S:.1f}s "
        f"AND p95 ≤ {P95_THRESHOLD_S:.1f}s."
    )
    if result.dry_run:
        lines.append(
            "- **Mode: dry-run.** Timings are synthesized; no `claude -p` "
            "subprocess was invoked. This artifact exists to verify report "
            "shape on CI without OAuth."
        )
    lines.append("")

    lines.append("## Measurements")
    lines.append("")
    lines.append("| Metric | Value (s) | Threshold (s) | Status |")
    lines.append("|---|---|---|---|")
    p50_status = "OK" if result.p50_s <= P50_THRESHOLD_S else "OVER"
    p95_status = "OK" if result.p95_s <= P95_THRESHOLD_S else "OVER"
    lines.append(
        f"| p50 per-turn overhead | {result.p50_s:.3f} | "
        f"{P50_THRESHOLD_S:.1f} | {p50_status} |"
    )
    lines.append(
        f"| p95 per-turn overhead | {result.p95_s:.3f} | "
        f"{P95_THRESHOLD_S:.1f} | {p95_status} |"
    )
    lines.append("")

    # Per-turn detail (compact).
    if result.timings:
        lines.append("### Per-turn timings (seconds)")
        lines.append("")
        lines.append("| run | turn | elapsed_s |")
        lines.append("|---|---|---|")
        for t in result.timings:
            lines.append(f"| {t.run_index} | {t.turn_index} | {t.elapsed_s:.3f} |")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{result.verdict}**")
    lines.append("")
    if result.verdict == "PASS":
        lines.append(
            "Per-turn dispatcher overhead is within the amendment A1 "
            "acceptance budget. MH-7's investigation-loop shape is "
            "viable on this hardware/OAuth path; proceed to Phase 1."
        )
    else:
        lines.append(
            "Per-turn dispatcher overhead exceeds the amendment A1 "
            "acceptance budget. See *Pivot Recommendation* below."
        )
    lines.append("")

    lines.append("## Pivot Recommendation")
    lines.append("")
    if result.verdict == "FAIL":
        lines.append(pivot_text)
    else:
        lines.append(
            "_Not triggered — verdict is PASS. The pivot block is emitted "
            "only on FAIL, per amendment A1._"
        )
    lines.append("")

    return "\n".join(lines)


def write_report(result: ProbeResult, report_path: Path, json_path: Path) -> None:
    """Write both the markdown report and JSON sidecar."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result.to_jsonable(), indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mh0_latency_probe",
        description=(
            "MH-0 Day-1 latency probe (PRD scix_deep_search_v1, "
            "amendment A1). Measures per-turn dispatcher overhead for "
            "ClaudeSubprocessDispatcher and writes a PASS/FAIL report."
        ),
    )
    p.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs (each = one full sweep of --turns turns). Default: 5.",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=5,
        help="Number of turns per run. Amendment A1 specifies 5. Default: 5.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("docs/eval/mh0_latency_probe.md"),
        help="Path to the markdown report.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help=(
            "Path to the JSON sidecar with raw timings + p50/p95. Defaults "
            "to <report>.json."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Synthesize PASS-shaped timings without invoking the dispatcher. "
            "Used by CI to verify report shape on hosts without OAuth."
        ),
    )
    p.add_argument(
        "--claude-binary",
        type=str,
        default="claude",
        help="Path or name of the ``claude`` CLI (real-mode only).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )
    return p


def _resolve_json_path(report: Path, json_out: Path | None) -> Path:
    if json_out is not None:
        return json_out
    return report.with_suffix(report.suffix + ".json") if report.suffix else (
        report.parent / (report.name + ".json")
    )


async def _run_real_probe(
    *, runs: int, turns: int, claude_binary: str
) -> ProbeResult:
    """Run the real probe against ``ClaudeSubprocessDispatcher``."""
    from scix.eval.persona_judge import ClaudeSubprocessDispatcher

    dispatcher = ClaudeSubprocessDispatcher(claude_binary=claude_binary)
    probe = LatencyProbe(dispatcher=dispatcher)
    timings = await probe.measure(runs=runs, turns=turns)
    return aggregate(timings, runs=runs, turns=turns, dry_run=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    json_path = _resolve_json_path(args.report, args.json_out)

    if args.dry_run:
        timings = _synthesize_dry_run_timings(runs=args.runs, turns=args.turns)
        result = aggregate(timings, runs=args.runs, turns=args.turns, dry_run=True)
    else:
        result = asyncio.run(
            _run_real_probe(
                runs=args.runs, turns=args.turns, claude_binary=args.claude_binary
            )
        )

    write_report(result, args.report, json_path)
    logger.info(
        "MH-0 probe complete: verdict=%s p50=%.3fs p95=%.3fs report=%s json=%s",
        result.verdict,
        result.p50_s,
        result.p95_s,
        args.report,
        json_path,
    )
    # Non-zero exit on FAIL so CI/orchestrators can branch on the verdict.
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
