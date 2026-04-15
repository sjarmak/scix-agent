#!/usr/bin/env python3
"""M11 JIT load test — gate before enabling enrich_entities=True by default.

Proves that concurrent JIT entity enrichment (with fault-injected Anthropic
endpoint: 2.5s p99, 5% error rate) does not degrade unrelated MCP tools:
keyword search, vector search, citation chain.

Acceptance criteria:
  - p95 drift < 10% for all three tools
  - asyncio.debug traces clean (zero slow-callback warnings)
  - Report written to build-artifacts/m11_load_test.md

Usage:
    python scripts/m11_load_test.py [--skip-vector] [--iterations N]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import itertools
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — fault injection model
# ---------------------------------------------------------------------------

# Lognormal calibration: p50=300ms, p99=2.5s
# p50 = exp(mu) = 0.300 => mu = ln(0.300)
# p99 = exp(mu + sigma * 2.326) = 2.500 => sigma = (ln(2.5) - mu) / 2.326
LATENCY_MU: float = math.log(0.300)  # -1.2040
LATENCY_SIGMA: float = 0.9115
ERROR_RATE: float = 0.05

# Test traffic
N_JIT_WORKERS: int = 8
DRIFT_THRESHOLD: float = 0.10
DEFAULT_KEYWORD_ITERS: int = 200
DEFAULT_VECTOR_ITERS: int = 100
DEFAULT_CITATION_ITERS: int = 100
DEFAULT_WARMUP: int = 10

# Known-good test bibcodes from smoke_test_mcp.py
CITE_PAIR_SRC = "2018AdSpR..62.2773B"
CITE_PAIR_TGT = "1990BAICz..41..137P"

VECTOR_QUERIES = [
    "gravitational wave detection",
    "exoplanet atmospheres",
    "dark matter halo",
    "stellar evolution",
    "galaxy formation",
    "cosmic microwave background",
    "supernova nucleosynthesis",
    "black hole accretion",
    "neutron star merger",
    "solar wind plasma",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class JITStats:
    """Mutable counters for JIT load loop statistics."""

    total_calls: int = 0
    bulkhead_degrades: int = 0
    errors_injected: int = 0
    local_ner_fallbacks: int = 0
    mean_latency_ms: float = 0.0
    # Accumulator for computing mean_latency_ms at loop end.
    # All mutations happen on the asyncio event loop thread (no to_thread),
    # so concurrent coroutine access is safe under the GIL.
    latency_sum_ms: float = field(default=0.0, repr=False)


@dataclass
class DriftResult:
    """p95 drift result for a single tool."""

    tool: str
    baseline_p95: float
    loaded_p95: float
    drift_pct: float
    passed: bool


@dataclass
class LoadTestConfig:
    """Configuration for the load test run."""

    keyword_iters: int = DEFAULT_KEYWORD_ITERS
    vector_iters: int = DEFAULT_VECTOR_ITERS
    citation_iters: int = DEFAULT_CITATION_ITERS
    warmup: int = DEFAULT_WARMUP
    n_jit_workers: int = N_JIT_WORKERS
    drift_threshold: float = DRIFT_THRESHOLD
    bulkhead_concurrency: int = 4
    bulkhead_budget_ms: int = 400
    skip_vector: bool = False


@dataclass
class LoadTestReport:
    """Complete load test results."""

    baseline: dict[str, list[float]]
    loaded: dict[str, list[float]]
    drifts: list[DriftResult]
    jit_stats: JITStats
    asyncio_warnings: list[str]
    config: LoadTestConfig


# ---------------------------------------------------------------------------
# Fault injection primitives
# ---------------------------------------------------------------------------


def sample_latency() -> float:
    """Draw a latency sample from the calibrated lognormal distribution."""
    return random.lognormvariate(LATENCY_MU, LATENCY_SIGMA)


def should_inject_error() -> bool:
    """Return True with ERROR_RATE probability."""
    return random.random() < ERROR_RATE


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a sorted copy of data.

    Uses nearest-rank method.
    """
    if not data:
        raise ValueError("percentile requires non-empty data")
    s = sorted(data)
    idx = max(0, min(int(math.ceil(pct * len(s))) - 1, len(s) - 1))
    return s[idx]


def compute_drift(
    tool: str,
    baseline: list[float],
    loaded: list[float],
    threshold: float = DRIFT_THRESHOLD,
) -> DriftResult:
    """Compute p95 drift between baseline and loaded latency distributions."""
    bp95 = percentile(baseline, 0.95)
    lp95 = percentile(loaded, 0.95)
    if bp95 == 0:
        drift = 0.0
    else:
        drift = (lp95 - bp95) / bp95
    return DriftResult(
        tool=tool,
        baseline_p95=bp95,
        loaded_p95=lp95,
        drift_pct=drift,
        passed=drift <= threshold,
    )


# ---------------------------------------------------------------------------
# JIT load loop
# ---------------------------------------------------------------------------


async def jit_load_loop(
    *,
    stop_event: asyncio.Event,
    bulkhead: Any | None,
    stats: JITStats,
    candidate_set: frozenset[int],
    live_jit_fn: Callable | None = None,
) -> None:
    """Continuously call route_jit with fault injection until stop_event is set."""
    from scix.jit.bulkhead import DEGRADED, JITBulkhead
    from scix.jit.router import LiveJITResult, route_jit

    if bulkhead is None:
        bulkhead = JITBulkhead(concurrency=4, budget_ms=400)

    if live_jit_fn is None:

        async def _fault_injected(
            bibcode: str,
            text: str,
            cset: frozenset[int],
            **kwargs: Any,
        ) -> LiveJITResult:
            if should_inject_error():
                stats.errors_injected += 1
                raise RuntimeError("simulated Anthropic 500")
            delay = sample_latency()
            await asyncio.sleep(delay)
            confidences = frozenset((eid, 0.95) for eid in cset)
            return LiveJITResult(
                bibcode=bibcode,
                entity_ids=frozenset(cset),
                confidences=confidences,
                model_version="haiku-v1-faultinject",
            )

        live_jit_fn = _fault_injected

    bibcode_counter = 0
    while not stop_event.is_set():
        bibcode_counter += 1
        fake_bibcode = f"LOAD{bibcode_counter:08d}"
        t0 = time.monotonic()
        try:
            result = await route_jit(
                bibcode=fake_bibcode,
                text="synthetic load test text for entity enrichment",
                candidate_set=candidate_set,
                bulkhead=bulkhead,
                live_jit_fn=live_jit_fn,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            stats.total_calls += 1
            stats.latency_sum_ms += elapsed_ms

            if result is DEGRADED:
                stats.bulkhead_degrades += 1
            elif hasattr(result, "lane") and result.lane == "local_ner":
                stats.local_ner_fallbacks += 1
        except Exception:
            elapsed_ms = (time.monotonic() - t0) * 1000
            stats.total_calls += 1
            stats.latency_sum_ms += elapsed_ms
            stats.bulkhead_degrades += 1

        # Yield control briefly to avoid starving the event loop
        await asyncio.sleep(0)

    if stats.total_calls > 0:
        stats.mean_latency_ms = stats.latency_sum_ms / stats.total_calls


# ---------------------------------------------------------------------------
# Tool measurement
# ---------------------------------------------------------------------------


async def measure_tool(
    *,
    name: str,
    tool_fn: Callable,
    queries: list[Any],
    conn: Any,
    iterations: int,
    warmup: int = 0,
    tool_kwargs: dict[str, Any] | None = None,
) -> list[float]:
    """Measure latencies for a tool over multiple iterations.

    Runs warmup iterations first (not recorded), then measured iterations.
    Each call is dispatched via asyncio.to_thread to avoid blocking the loop.
    """
    kwargs = tool_kwargs or {}
    query_cycle = itertools.cycle(queries)
    latencies: list[float] = []

    for i in range(warmup + iterations):
        q = next(query_cycle)
        t0 = time.monotonic()
        await asyncio.to_thread(tool_fn, conn, q, **kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    logger.info(
        "measure_tool %s: %d samples, p50=%.1fms p95=%.1fms",
        name,
        len(latencies),
        percentile(latencies, 0.50),
        percentile(latencies, 0.95),
    )
    return latencies


# ---------------------------------------------------------------------------
# Traffic generators
# ---------------------------------------------------------------------------


def load_keyword_queries(conn: Any, limit: int = 200) -> list[str]:
    """Load keyword search queries from query_log."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT query FROM query_log
            WHERE tool = 'keyword_search' AND query IS NOT NULL
            ORDER BY random()
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    queries = [r[0] for r in rows]
    if not queries:
        # Fallback to hardcoded astronomy terms
        queries = ["dark matter halo", "stellar evolution", "exoplanet transit"]
    logger.info("Loaded %d keyword queries from query_log", len(queries))
    return queries


def load_vector_embeddings() -> list[list[float]]:
    """Pre-embed the fixed vector query set using INDUS."""
    from scix.embed import embed_batch, load_model

    logger.info("Loading INDUS model for vector query embedding...")
    model, tokenizer = load_model("indus")
    embeddings = embed_batch(model, tokenizer, VECTOR_QUERIES, pooling="mean")
    logger.info("Embedded %d vector queries", len(embeddings))
    return embeddings


def load_citation_pairs(conn: Any, target: int = 10) -> list[tuple[str, str]]:
    """Load citation chain bibcode pairs, validated for non-empty paths."""
    from scix.search import citation_chain

    pairs = [(CITE_PAIR_SRC, CITE_PAIR_TGT)]

    # Pull additional pairs from DB
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT source_bibcode, target_bibcode
            FROM citation_edges
            TABLESAMPLE SYSTEM(0.001)
            LIMIT %s
            """,
            (target * 5,),
        )
        candidates = cur.fetchall()

    # Validate pairs — keep only those with a path
    for src, tgt in candidates:
        if len(pairs) >= target:
            break
        try:
            result = citation_chain(conn, src, tgt, max_depth=3)
            path = result.metadata.get("path_bibcodes", []) if result.metadata else []
            if len(path) >= 2:
                pairs.append((src, tgt))
        except Exception as exc:
            logger.debug("citation_chain validation failed for (%s, %s): %s", src, tgt, exc)
            continue

    # If we still don't have enough, repeat the known-good pair
    while len(pairs) < target:
        pairs.append((CITE_PAIR_SRC, CITE_PAIR_TGT))

    logger.info("Loaded %d citation pairs (%d unique)", len(pairs), len(set(pairs)))
    return pairs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(report: LoadTestReport) -> str:
    """Generate the markdown load test report."""
    cfg = report.config
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tools = ["bm25_search", "vector_search", "citation_chain"]

    def _latency_table(heading: str, data: dict[str, list[float]]) -> list[str]:
        """Render a markdown latency table for one phase (baseline or loaded)."""
        rows = [
            heading,
            "",
            "| tool | p50_ms | p95_ms | p99_ms | mean_ms |",
            "|------|--------|--------|--------|---------|",
        ]
        for tool in tools:
            if tool not in data:
                rows.append(f"| {tool} | skipped | — | — | — |")
                continue
            lats = data[tool]
            rows.append(
                f"| {tool} | {percentile(lats, 0.50):.1f} | {percentile(lats, 0.95):.1f} "
                f"| {percentile(lats, 0.99):.1f} | {sum(lats)/len(lats):.1f} |"
            )
        rows.append("")
        return rows

    lines = [
        "# M11 JIT Load Test Report",
        "",
        f"**Date**: {now}",
        "**DB**: dbname=scix (32.4M papers, 299M edges)",
        f"**JIT fault model**: lognormal(mu={LATENCY_MU:.3f}, sigma={LATENCY_SIGMA:.3f}), "
        f"p50=300ms, p99=2.5s, {ERROR_RATE*100:.0f}% error",
        f"**JIT concurrency**: {cfg.n_jit_workers} workers, "
        f"bulkhead({cfg.bulkhead_concurrency}, {cfg.bulkhead_budget_ms}ms)",
        f"**Tool iterations**: keyword={cfg.keyword_iters}, "
        f"vector={cfg.vector_iters}, citation_chain={cfg.citation_iters}",
        f"**Warmup**: {cfg.warmup} iterations (discarded)",
        "",
    ]

    lines.extend(_latency_table("## Baseline (no JIT load)", report.baseline))
    lines.extend(
        _latency_table(
            f"## Loaded ({cfg.n_jit_workers} concurrent JIT enrichment tasks)", report.loaded
        )
    )

    # Drift analysis
    lines.append("## P95 Drift Analysis")
    lines.append("")
    lines.append("| tool | baseline_p95 | loaded_p95 | drift_pct | status |")
    lines.append("|------|-------------|------------|-----------|--------|")
    for d in report.drifts:
        status = "PASS" if d.passed else "**FAIL**"
        lines.append(
            f"| {d.tool} | {d.baseline_p95:.1f} | {d.loaded_p95:.1f} "
            f"| {d.drift_pct*100:+.1f}% | {status} |"
        )
    lines.append("")
    lines.append(f"**Threshold**: drift <= {cfg.drift_threshold*100:.0f}%")
    lines.append("")

    # JIT stats
    lines.append("## JIT Load Statistics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|--------|-------|")
    js = report.jit_stats
    lines.append(f"| total_jit_calls | {js.total_calls} |")
    lines.append(f"| bulkhead_degrades | {js.bulkhead_degrades} |")
    lines.append(f"| errors_injected | {js.errors_injected} |")
    lines.append(f"| local_ner_fallbacks | {js.local_ner_fallbacks} |")
    lines.append(f"| mean_jit_latency_ms | {js.mean_latency_ms:.1f} |")
    lines.append("")

    # asyncio traces
    lines.append("## asyncio.debug Traces")
    lines.append("")
    if report.asyncio_warnings:
        lines.append(f"Status: **FAIL** ({len(report.asyncio_warnings)} warnings)")
        lines.append("")
        lines.append("```")
        for w in report.asyncio_warnings:
            lines.append(w)
        lines.append("```")
    else:
        lines.append("Status: CLEAN (0 warnings)")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    all_passed = all(d.passed for d in report.drifts)
    traces_clean = len(report.asyncio_warnings) == 0
    if all_passed and traces_clean:
        lines.append(
            "**PASS** — all tools maintain p95 within "
            f"{cfg.drift_threshold*100:.0f}% of baseline under concurrent JIT enrichment load."
        )
    else:
        failures = [d.tool for d in report.drifts if not d.passed]
        reasons = []
        if failures:
            reasons.append(f"p95 drift exceeded for: {', '.join(failures)}")
        if not traces_clean:
            reasons.append(f"{len(report.asyncio_warnings)} asyncio slow-callback warnings")
        lines.append(f"**FAIL** — {'; '.join(reasons)}.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run_load_test(
    conn: Any,
    config: LoadTestConfig,
) -> LoadTestReport:
    """Execute the four-phase load test."""
    # --- Setup: asyncio debug mode ---
    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    loop.slow_callback_duration = 0.1  # 100ms

    asyncio_log_buffer = io.StringIO()
    asyncio_handler = logging.StreamHandler(asyncio_log_buffer)
    asyncio_handler.setLevel(logging.WARNING)
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.addHandler(asyncio_handler)

    try:
        return await _run_phases(conn, config, loop, asyncio_log_buffer, asyncio_handler)
    finally:
        asyncio_handler.flush()
        asyncio_logger.removeHandler(asyncio_handler)


async def _run_phases(
    conn: Any,
    config: LoadTestConfig,
    loop: asyncio.AbstractEventLoop,
    asyncio_log_buffer: io.StringIO,
    asyncio_handler: logging.Handler,
) -> LoadTestReport:
    """Inner implementation — separated so the handler cleanup is in a finally."""
    from scix.jit.bulkhead import JITBulkhead
    from scix.search import citation_chain, lexical_search, vector_search

    # --- Setup: load traffic ---
    logger.info("Loading test traffic...")
    keyword_queries = load_keyword_queries(conn, limit=config.keyword_iters)

    vector_embeddings: list[list[float]] = []
    if not config.skip_vector:
        vector_embeddings = load_vector_embeddings()

    citation_pairs = load_citation_pairs(conn)

    # --- Tool wrappers (adapt signatures for measure_tool) ---
    def kw_fn(c: Any, query: str, **kw: Any) -> Any:
        return lexical_search(c, query, limit=10)

    def vec_fn(c: Any, embedding: list[float], **kw: Any) -> Any:
        return vector_search(c, embedding, model_name="indus", limit=10)

    def cite_fn(c: Any, pair: tuple[str, str], **kw: Any) -> Any:
        return citation_chain(c, pair[0], pair[1], max_depth=3)

    # --- Phase 1: Baseline ---
    logger.info("=== Phase 1: Baseline (no JIT load) ===")

    baseline: dict[str, list[float]] = {}

    baseline["bm25_search"] = await measure_tool(
        name="bm25_search",
        tool_fn=kw_fn,
        queries=keyword_queries,
        conn=conn,
        iterations=config.keyword_iters,
        warmup=config.warmup,
    )

    if not config.skip_vector:
        baseline["vector_search"] = await measure_tool(
            name="vector_search",
            tool_fn=vec_fn,
            queries=vector_embeddings,
            conn=conn,
            iterations=config.vector_iters,
            warmup=config.warmup,
        )

    baseline["citation_chain"] = await measure_tool(
        name="citation_chain",
        tool_fn=cite_fn,
        queries=citation_pairs,
        conn=conn,
        iterations=config.citation_iters,
        warmup=config.warmup,
    )

    # --- Phase 2+3: Loaded measurement ---
    logger.info("=== Phase 2+3: Loaded measurement (%d JIT workers) ===", config.n_jit_workers)

    jit_stats = JITStats()
    stop_event = asyncio.Event()
    bulkhead = JITBulkhead(
        concurrency=config.bulkhead_concurrency, budget_ms=config.bulkhead_budget_ms
    )
    candidate_set = frozenset({1, 2, 3, 42, 99})

    loaded: dict[str, list[float]] = {}

    async def _measure_all() -> None:
        loaded["bm25_search"] = await measure_tool(
            name="bm25_search (loaded)",
            tool_fn=kw_fn,
            queries=keyword_queries,
            conn=conn,
            iterations=config.keyword_iters,
            warmup=config.warmup,
        )

        if not config.skip_vector:
            loaded["vector_search"] = await measure_tool(
                name="vector_search (loaded)",
                tool_fn=vec_fn,
                queries=vector_embeddings,
                conn=conn,
                iterations=config.vector_iters,
                warmup=config.warmup,
            )

        loaded["citation_chain"] = await measure_tool(
            name="citation_chain (loaded)",
            tool_fn=cite_fn,
            queries=citation_pairs,
            conn=conn,
            iterations=config.citation_iters,
            warmup=config.warmup,
        )

    jit_tasks = [
        asyncio.create_task(
            jit_load_loop(
                stop_event=stop_event,
                bulkhead=bulkhead,
                stats=jit_stats,
                candidate_set=candidate_set,
            )
        )
        for _ in range(config.n_jit_workers)
    ]

    try:
        await _measure_all()
    finally:
        stop_event.set()
        await asyncio.gather(*jit_tasks, return_exceptions=True)

    # --- Phase 4: Analysis ---
    logger.info("=== Phase 4: Report ===")

    drifts: list[DriftResult] = []
    for tool in ["bm25_search", "vector_search", "citation_chain"]:
        if tool in baseline and tool in loaded:
            drifts.append(compute_drift(tool, baseline[tool], loaded[tool]))

    # Collect asyncio warnings (handler cleanup is in run_load_test's finally)
    warning_text = asyncio_log_buffer.getvalue().strip()
    asyncio_warnings = [line for line in warning_text.split("\n") if line.strip()]

    return LoadTestReport(
        baseline=baseline,
        loaded=loaded,
        drifts=drifts,
        jit_stats=jit_stats,
        asyncio_warnings=asyncio_warnings,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="M11 JIT load test")
    parser.add_argument(
        "--skip-vector", action="store_true", help="Skip vector search (avoids INDUS model load)"
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override iteration count for all tools"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations (discarded)"
    )
    parser.add_argument(
        "--jit-workers", type=int, default=N_JIT_WORKERS, help="Number of concurrent JIT workers"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = LoadTestConfig(
        keyword_iters=args.iterations or DEFAULT_KEYWORD_ITERS,
        vector_iters=args.iterations or DEFAULT_VECTOR_ITERS,
        citation_iters=args.iterations or DEFAULT_CITATION_ITERS,
        warmup=args.warmup,
        n_jit_workers=args.jit_workers,
        skip_vector=args.skip_vector,
    )

    from scix.db import get_connection

    # This load test is designed for production (read-only DB access).
    # Require explicit --dsn or default to SCIX_DSN / dbname=scix.
    dsn = os.environ.get("SCIX_DSN", "dbname=scix")
    logger.info("Connecting to DSN: %s", dsn)
    conn = get_connection(dsn)
    conn.autocommit = True

    try:
        report = asyncio.run(run_load_test(conn, config))
    finally:
        conn.close()

    md = generate_report(report)

    out_path = Path(__file__).resolve().parent.parent / "build-artifacts" / "m11_load_test.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(md)

    # Print summary to stdout
    print("\n" + "=" * 70)
    for d in report.drifts:
        status = "PASS" if d.passed else "FAIL"
        print(f"  {d.tool:<20s} p95 drift={d.drift_pct*100:+.1f}%  [{status}]")
    all_pass = all(d.passed for d in report.drifts) and not report.asyncio_warnings
    print(f"\n  Verdict: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Report: {out_path}")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
