#!/usr/bin/env python3
"""Benchmark CPU rerank latency for ms-marco-MiniLM-L-12-v2 and BAAI/bge-reranker-large.

Forces CPU inference even on GPU hosts so the numbers reflect the no-GPU
deployment path. Runs a warm-up phase to absorb JIT/lazy-load overhead,
then times ``--runs`` rerank calls of one query against ``--top-n``
synthetic candidate documents and reports p50/p95 wall-clock latency.

Output is written to ``results/rerank_cpu_bench.md``. The file's
``Verdict`` line records whether MiniLM lands under the 400 ms p95
threshold that we treat as the no-GPU deployment cutoff.

OPERATOR NOTE: bge-reranker-large uses ~2-3 GB resident RAM during
inference. On the shared gascity host run this bench inside a memory
scope so it can't take the supervisor down with it::

    scix-batch python scripts/bench_rerank_cpu.py

For pure CPU smoke tests on a workstation the wrapper is optional.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Force CPU BEFORE importing torch / transformers / sentence_transformers.
# CUDA_VISIBLE_DEVICES='' hides every GPU from the CUDA driver, so even if
# downstream code calls .to('cuda') or trusts torch.cuda.is_available() the
# device will resolve to CPU. We additionally pass device='cpu' to the
# CrossEncoder constructor below as belt-and-suspenders.
# ---------------------------------------------------------------------------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("bench_rerank_cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "rerank_cpu_bench.md"

# Local snapshot directory for the bge weights — keeps the ~1.3 GB model
# off the network for repeat runs and matches the layout the production
# CrossEncoderReranker prefers.
BGE_MODEL_NAME = "BAAI/bge-reranker-large"
BGE_LOCAL_DIR = REPO_ROOT / "models" / "bge-reranker-large"

MINILM_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Threshold for the "MiniLM is suitable as no-GPU default" verdict. Tied to
# the rerank latency budget for the deep-search persona harness.
P95_THRESHOLD_MS = 400.0


@dataclass(frozen=True)
class BenchResult:
    model_label: str
    model_id: str
    runs: int
    top_n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


def _build_synthetic_pairs(top_n: int) -> tuple[str, list[tuple[str, str]]]:
    """Construct one query plus ``top_n`` representative (query, doc) pairs.

    Doc text is roughly 200 tokens of astrophysics-flavoured prose so the
    cross-encoder receives an input distribution close to the production
    title+abstract snippet payload.
    """
    query = (
        "How does dark matter halo concentration evolve with redshift in "
        "Lambda-CDM cosmological N-body simulations across a range of halo masses?"
    )
    base_doc = (
        "We present a detailed analysis of dark matter halo concentration as a "
        "function of redshift, halo mass, and cosmological parameters using a "
        "suite of high-resolution Lambda-CDM N-body simulations spanning a "
        "dynamic range of more than four decades in mass. Halo concentrations "
        "are measured via the standard Navarro-Frenk-White profile fit to "
        "spherically averaged density profiles, and we explore the impact of "
        "relaxation criteria, fitting radius, and substructure removal on the "
        "inferred concentration values. Our results confirm the well-known "
        "decline of concentration with mass at fixed redshift and the overall "
        "decrease of concentration with increasing redshift, while highlighting "
        "deviations from simple power-law scaling at the highest masses and "
        "lowest redshifts probed by the simulations."
    )
    pairs = [(query, f"Document {i}. {base_doc}") for i in range(top_n)]
    return query, pairs


def _ensure_bge_local_snapshot() -> str:
    """Download the bge weights to ``BGE_LOCAL_DIR`` if missing.

    Returns the path/identifier to hand to ``CrossEncoder``. Prefers the
    local directory when present so repeat benchmark runs do not hit the
    network.
    """
    if BGE_LOCAL_DIR.exists() and any(BGE_LOCAL_DIR.iterdir()):
        log.info("bge-reranker-large already present at %s", BGE_LOCAL_DIR)
        return str(BGE_LOCAL_DIR)

    log.info("Downloading %s into %s (~1.3 GB)…", BGE_MODEL_NAME, BGE_LOCAL_DIR)
    from huggingface_hub import snapshot_download

    BGE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BGE_MODEL_NAME,
        local_dir=str(BGE_LOCAL_DIR),
    )
    log.info("Download complete: %s", BGE_LOCAL_DIR)
    return str(BGE_LOCAL_DIR)


def _load_cross_encoder(model_id: str):
    """Instantiate a sentence_transformers CrossEncoder pinned to CPU."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, device="cpu")


def _percentile(values: list[float], pct: float) -> float:
    """Inclusive nearest-rank percentile in milliseconds."""
    if not values:
        raise ValueError("cannot take percentile of empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # Nearest-rank: ceil(pct/100 * N) - 1, clamped into bounds.
    import math

    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _bench_one(
    model_label: str,
    model_id: str,
    runs: int,
    top_n: int,
    warmup: int,
) -> BenchResult:
    log.info("Loading %s (%s) on CPU…", model_label, model_id)
    encoder = _load_cross_encoder(model_id)
    query, pairs = _build_synthetic_pairs(top_n)

    log.info("Warming up %s for %d untimed runs…", model_label, warmup)
    for _ in range(warmup):
        encoder.predict(pairs)

    log.info("Timing %s for %d runs at top_n=%d…", model_label, runs, top_n)
    samples_ms: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        encoder.predict(pairs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        samples_ms.append(elapsed_ms)
        if (i + 1) % 10 == 0:
            log.info(
                "  %s run %d/%d: %.1f ms (running median %.1f ms)",
                model_label,
                i + 1,
                runs,
                elapsed_ms,
                statistics.median(samples_ms),
            )

    return BenchResult(
        model_label=model_label,
        model_id=model_id,
        runs=runs,
        top_n=top_n,
        p50_ms=_percentile(samples_ms, 50),
        p95_ms=_percentile(samples_ms, 95),
        mean_ms=statistics.mean(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def _verify_cpu_only() -> str:
    """Sanity check: confirm torch sees no CUDA device."""
    import torch  # imported here so the env var above takes effect first.

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    if cuda_available or device_count > 0:
        # Hard fail: the whole point of this bench is the no-GPU number.
        raise RuntimeError(
            "Expected CUDA to be hidden by CUDA_VISIBLE_DEVICES='', but "
            f"torch reports cuda_available={cuda_available}, "
            f"device_count={device_count}."
        )
    return f"torch {torch.__version__} | cuda_available=False | threads={torch.get_num_threads()}"


def _format_report(results: list[BenchResult], torch_info: str) -> str:
    minilm = next(r for r in results if r.model_id == MINILM_MODEL_NAME)
    suitable = minilm.p95_ms < P95_THRESHOLD_MS

    lines: list[str] = []
    lines.append("# Cross-encoder rerank CPU latency benchmark")
    lines.append("")
    lines.append(
        f"Synthetic single-query rerank, top_n={results[0].top_n}, "
        f"runs={results[0].runs}, warmup=3 untimed runs."
    )
    lines.append(
        "CPU forced via `CUDA_VISIBLE_DEVICES=''` and `device='cpu'` on the "
        "CrossEncoder. Run on the project RTX 5090 host with the GPU hidden."
    )
    lines.append("")
    lines.append(f"Runtime: {torch_info}")
    lines.append("")
    lines.append("| Model | p50 (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| `{r.model_id}` | {r.p50_ms:.1f} | {r.p95_ms:.1f} | "
            f"{r.mean_ms:.1f} | {r.min_ms:.1f} | {r.max_ms:.1f} |"
        )
    lines.append("")
    lines.append(f"Threshold for no-GPU default: p95 < {P95_THRESHOLD_MS:.0f} ms.")
    lines.append("")
    if suitable:
        verdict = (
            f"Verdict: ms-marco-MiniLM-L-12-v2 IS suitable as the no-GPU "
            f"deployment default — measured p95 {minilm.p95_ms:.1f} ms is "
            f"under the {P95_THRESHOLD_MS:.0f} ms threshold."
        )
    else:
        verdict = (
            f"Verdict: ms-marco-MiniLM-L-12-v2 is NOT suitable as the no-GPU "
            f"deployment default — measured p95 {minilm.p95_ms:.1f} ms exceeds "
            f"the {P95_THRESHOLD_MS:.0f} ms threshold."
        )
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of candidate documents per query (default: 20).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed rerank runs per model (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of untimed warmup runs per model (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help=f"Markdown output path (default: {RESULTS_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.top_n < 1:
        parser.error("--top-n must be >= 1")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    torch_info = _verify_cpu_only()
    log.info("CPU-only confirmed: %s", torch_info)

    bge_model_id = _ensure_bge_local_snapshot()

    results: list[BenchResult] = []
    results.append(
        _bench_one(
            model_label="MiniLM",
            model_id=MINILM_MODEL_NAME,
            runs=args.runs,
            top_n=args.top_n,
            warmup=args.warmup,
        )
    )
    results.append(
        _bench_one_with_loader(
            model_label="bge-reranker-large",
            model_id=bge_model_id,
            # Report the canonical HF identifier in the output even when the
            # weights load from a local snapshot directory.
            display_id=BGE_MODEL_NAME,
            runs=args.runs,
            top_n=args.top_n,
            warmup=args.warmup,
        )
    )

    report = _format_report(results, torch_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    log.info("Wrote %s", args.output)

    # Echo the verdict to stdout so cron / RFC pipelines can see it without
    # re-reading the file.
    for line in report.splitlines():
        if line.startswith("Verdict:"):
            print(line)
            break

    return 0


def _bench_one_with_loader(
    model_label: str,
    model_id: str,
    display_id: str,
    runs: int,
    top_n: int,
    warmup: int,
) -> BenchResult:
    """Variant of _bench_one that records ``display_id`` in the report.

    Lets us load bge from a local snapshot directory while still labeling
    the row with the canonical HuggingFace identifier.
    """
    raw = _bench_one(
        model_label=model_label,
        model_id=model_id,
        runs=runs,
        top_n=top_n,
        warmup=warmup,
    )
    return BenchResult(
        model_label=raw.model_label,
        model_id=display_id,
        runs=raw.runs,
        top_n=raw.top_n,
        p50_ms=raw.p50_ms,
        p95_ms=raw.p95_ms,
        mean_ms=raw.mean_ms,
        min_ms=raw.min_ms,
        max_ms=raw.max_ms,
    )


if __name__ == "__main__":
    sys.exit(main())
