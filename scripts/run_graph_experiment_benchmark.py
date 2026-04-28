#!/usr/bin/env python3
"""Run the graph-experiment benchmark and write the go/no-go report.

Tracks bead scix_experiments-vdtd. Loads the slice snapshot, materializes
BENCHMARK_TEMPLATES with graph-aware pickers, runs the harness in control
and treatment for each question, summarizes the trace JSONL, and writes
``results/graph_experiment_<timestamp>.{json,md}``.

The treatment trace summary IS the experiment data — depth distribution,
tool-call patterns, and freeform-query emergence determine the Apache AGE
go/no-go recommendation.

Usage::

    scix-batch python scripts/run_graph_experiment_benchmark.py \\
        --snapshot data/graph_experiment/astronomy_1hop.pkl.gz \\
        --max-questions 6                       # sub-sample for cheap runs

Reads the local production MCP via env vars ``SCIX_MCP_URL`` +
``SCIX_MCP_TOKEN`` (set in ``deploy/.env``). For fully-offline tests pass
``--no-production-mcp``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from scix.graph_experiment.analysis import compare_summaries
from scix.graph_experiment.bench_runner import (
    go_no_go,
    materialize_questions,
    render_markdown,
    run_all,
    summaries_by_variant,
)
from scix.graph_experiment.benchmark import write_jsonl
from scix.graph_experiment.harness import HarnessConfig
from scix.graph_experiment.loader import load_snapshot

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/graph_experiment/astronomy_1hop.pkl.gz"),
        help="Path to the slice snapshot",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("data/graph_experiment/traces"),
        help="Where to write per-session trace JSONL files",
    )
    parser.add_argument(
        "--out-stem",
        type=str,
        default=None,
        help="Filename stem (default: graph_experiment_<unix_ts>)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap question count (for quick runs)",
    )
    parser.add_argument(
        "--budget-per-run",
        type=float,
        default=0.50,
        help="Max USD per (question, variant) — passed to claude -p",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model used by the agent harness",
    )
    parser.add_argument(
        "--no-production-mcp",
        action="store_true",
        help="Skip wiring the production MCP into the harness config",
    )
    parser.add_argument(
        "--production-stdio",
        action="store_true",
        help="Run the production MCP via stdio (python -m scix.mcp_server) "
        "instead of HTTP. Useful when the Docker container isn't running.",
    )
    parser.add_argument(
        "--questions-out",
        type=Path,
        default=Path("data/graph_experiment/benchmark_questions.jsonl"),
        help="Where to persist the materialized questions",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    if not args.snapshot.exists():
        logger.error(
            "snapshot not found: %s — run scripts/build_graph_experiment_slice.py first",
            args.snapshot,
        )
        return 2

    logger.info("loading snapshot %s", args.snapshot)
    graph = load_snapshot(args.snapshot)
    logger.info("loaded graph: |V|=%d |E|=%d", graph.vcount(), graph.ecount())

    questions = materialize_questions(graph)
    logger.info("materialized %d questions", len(questions))
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
        logger.info("capped to %d questions", len(questions))
    if not questions:
        logger.error(
            "no questions materialized — slice too thin or pickers misconfigured"
        )
        return 3

    write_jsonl(args.questions_out, questions)
    logger.info("questions written to %s", args.questions_out)

    if args.no_production_mcp:
        production_url = None
        production_token = None
        production_stdio = False
    elif args.production_stdio:
        production_url = None
        production_token = None
        production_stdio = True
    else:
        production_url = os.environ.get("SCIX_MCP_URL")
        production_token = os.environ.get("SCIX_MCP_TOKEN")
        production_stdio = False

    config = HarnessConfig(
        snapshot_path=args.snapshot,
        trace_dir=args.trace_dir,
        production_mcp_url=production_url,
        production_mcp_token=production_token,
        production_mcp_stdio=production_stdio,
        budget_usd=args.budget_per_run,
        model=args.model,
    )

    results = run_all(questions, config)

    summaries = summaries_by_variant(results, args.trace_dir)
    comparison = compare_summaries(summaries["control"], summaries["treatment"])
    verdict, rationale = go_no_go(comparison)

    stem = args.out_stem or f"graph_experiment_{int(time.time())}"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{stem}.json"
    md_path = results_dir / f"{stem}.md"

    json_payload = {
        "verdict": verdict,
        "rationale": rationale,
        "comparison": comparison,
        "questions": [q.to_dict() for q in questions],
        "results": [r.to_dict() for r in results],
        "config": {
            "snapshot": str(args.snapshot),
            "trace_dir": str(args.trace_dir),
            "model": args.model,
            "budget_per_run_usd": args.budget_per_run,
            "production_mcp_url": production_url,
            "production_mcp_stdio": production_stdio,
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=str) + "\n")
    md_path.write_text(
        render_markdown(
            verdict=verdict,
            rationale=rationale,
            comparison=comparison,
            questions=questions,
            results=results,
        )
    )
    logger.info("verdict=%s — wrote %s and %s", verdict, json_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
