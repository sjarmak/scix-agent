#!/usr/bin/env python3
"""Benchmark harness for ``chunk_search`` against ``scix_chunks_v1``.

Implements PRD ``chunk-embeddings-build`` Phase 5 §P1-1 latency bench.

Encodes a workload of representative queries with INDUS (mean pooling, 768d)
and measures end-to-end latency for two scenarios:

* ``no_filter`` — pure ANN against ``scix_chunks_v1``.
* ``filtered`` — ANN with ``year`` (year_min/year_max) + ``arxiv_class``
  payload filters applied at query time.

Reports p50 / p95 / p99 and mean / min / max for each scenario.

Output JSON schema (per PRD §P1-1)::

    {
      "schema_version": 1,
      "tool": "chunk_search",
      "collection": "scix_chunks_v1",
      "embedder": {"model": "indus", "device": "cpu", "dim": 768},
      "ran_at": "2026-04-29T12:00:00Z",
      "git_sha": "abc1234",
      "qdrant_url": "http://127.0.0.1:6333",
      "iterations": 1000,
      "limit": 20,
      "scenarios": {
        "no_filter":   {"n": 1000, "p50_ms": ..., "p95_ms": ..., "p99_ms": ...,
                        "mean_ms": ..., "min_ms": ..., "max_ms": ...,
                        "encode_p50_ms": ..., "ann_p50_ms": ...},
        "filtered":    {... same shape ...}
      },
      "p1_1_pass": {"no_filter_p95_lt_50ms": bool,
                    "filtered_p95_lt_100ms": bool},
      "queries": ["...", ...]
    }

Usage
-----

    # Default: 1000 iterations, 20 results, write JSON to stdout
    QDRANT_URL=http://127.0.0.1:6333 python scripts/bench_chunk_search.py

    # Custom iterations, save to file
    QDRANT_URL=http://127.0.0.1:6333 python scripts/bench_chunk_search.py \\
        --iterations 1000 --output docs/eval/chunk_search_v1_bench.json

    # Smoke test without Qdrant — uses a stub client to validate plumbing
    python scripts/bench_chunk_search.py --dry-run --iterations 5

The bench is read-only and idempotent. It does not write to Postgres or Qdrant.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

# 50 representative query stems. Cycled with deterministic year/class filters
# to reach the 1000-iteration target without inflating the source-of-truth
# question pool.
BENCH_QUERIES: tuple[str, ...] = (
    # Astrophysics — methods
    "MCMC sampling of cosmological parameters",
    "Bayesian hierarchical modeling exoplanet transits",
    "kernel density estimation stellar populations",
    "Gaussian process radial velocity detrending",
    "wavelet decomposition pulsar timing residuals",
    # Astrophysics — datasets/instruments
    "SDSS DR16 photometric calibration",
    "Gaia DR3 astrometry parallax",
    "JWST NIRSpec exoplanet atmosphere",
    "Chandra X-ray observatory cluster mass",
    "LIGO O3 binary neutron star",
    # Heliophysics
    "solar coronal mass ejection forecast",
    "Parker Solar Probe magnetic field reconnection",
    "interplanetary scintillation solar wind",
    "EUV imager Carrington event",
    "global magnetohydrodynamic simulation magnetosphere",
    # Planetary
    "Mars Science Laboratory Curiosity rover regolith",
    "Cassini Saturn ring particle size distribution",
    "exomoon detection transit timing variation",
    "asteroid spectral classification SMASS",
    "lunar regolith volatile inventory LCROSS",
    # Earth science
    "MODIS aerosol optical depth retrieval",
    "GPM precipitation algorithm validation",
    "ICESat-2 ice sheet elevation change",
    "ENSO teleconnection precipitation anomaly",
    "Sentinel-1 InSAR subsidence",
    # Biology / bioastronomy
    "extremophile microbial community Atacama",
    "biosignature methane Mars atmosphere",
    "tardigrade radiation resistance",
    "endolithic cyanobacteria desert",
    "panspermia interstellar dust transport",
    # Generic methods
    "convolutional neural network image classification",
    "graph neural network molecular property prediction",
    "transformer language model fine-tuning",
    "diffusion model image synthesis",
    "self-supervised contrastive representation learning",
    # Generic dataset queries
    "ImageNet benchmark transfer learning",
    "Common Crawl web text",
    "Pan-STARRS1 photometric survey",
    "WISE all-sky infrared catalog",
    "Spitzer IRAC mid-infrared survey",
    # Software / tools
    "emcee affine invariant ensemble sampler",
    "dynesty nested sampling",
    "stan no-u-turn sampler",
    "scikit-learn random forest hyperparameter",
    "pytorch lightning multi-gpu training",
    # Cross-discipline
    "bayesian model selection evidence",
    "principal component analysis dimensionality reduction",
    "k-nearest neighbors classification",
    "autoencoder anomaly detection",
    "time series forecasting recurrent neural network",
)


@dataclass(frozen=True)
class FilterSpec:
    """A single query's payload filter, sampled from a deterministic cycle."""

    year_min: int | None
    year_max: int | None
    arxiv_class: tuple[str, ...] | None


# Year windows + arxiv classes cycled deterministically across iterations so
# the filtered scenario exercises the indexed payload fields described in
# PRD P0-2 (year + arxiv_class AND-combination is the canonical case).
_YEAR_WINDOWS: tuple[tuple[int | None, int | None], ...] = (
    (2020, 2024),
    (2018, 2022),
    (2015, 2020),
    (2010, 2015),
    (2022, 2026),
)
_ARXIV_CLASSES: tuple[tuple[str, ...], ...] = (
    ("astro-ph.CO",),
    ("astro-ph.EP",),
    ("astro-ph.GA",),
    ("astro-ph.IM",),
    ("astro-ph.SR",),
)


def _filter_for_iteration(i: int) -> FilterSpec:
    """Deterministic filter cycle so re-runs hit the same payload paths."""
    year_min, year_max = _YEAR_WINDOWS[i % len(_YEAR_WINDOWS)]
    arxiv = _ARXIV_CLASSES[(i // len(_YEAR_WINDOWS)) % len(_ARXIV_CLASSES)]
    return FilterSpec(year_min=year_min, year_max=year_max, arxiv_class=arxiv)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile, no external deps."""
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def _summary(
    values: list[float], encode_values: list[float], ann_values: list[float]
) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": round(_percentile(values, 50), 3),
        "p95_ms": round(_percentile(values, 95), 3),
        "p99_ms": round(_percentile(values, 99), 3),
        "mean_ms": round(statistics.fmean(values), 3) if values else float("nan"),
        "min_ms": round(min(values), 3) if values else float("nan"),
        "max_ms": round(max(values), 3) if values else float("nan"),
        "encode_p50_ms": round(_percentile(encode_values, 50), 3),
        "ann_p50_ms": round(_percentile(ann_values, 50), 3),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _StubEmbedder:
    """Deterministic stand-in for INDUS used by --dry-run.

    Returns a 768d unit-norm vector seeded by query hash so dry-run timings are
    representative of pure ANN cost without the encoder load.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            seed = abs(hash(t)) % (10**9)
            vec = [((seed >> (i % 30)) & 0xFF) / 255.0 for i in range(768)]
            mag = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / mag for v in vec])
        return out


class _StubQdrant:
    """Stand-in for the chunks-collection client used by --dry-run."""

    def chunk_search_by_text(self, vector, **kwargs):  # noqa: ARG002
        from scix.qdrant_tools import ChunkHit

        # 50 µs of fake work — emulates network call without leaving the process.
        time.sleep(0.00005)
        return [
            ChunkHit(
                bibcode=f"2024STUB.{i:03d}",
                chunk_id=i,
                section_idx=0,
                section_heading_norm="methods",
                section_heading="Methods",
                score=1.0 - i * 0.01,
                snippet=None,
                char_offset=0,
                n_tokens=512,
            )
            for i in range(min(20, kwargs.get("limit", 20)))
        ]


def _load_real_embedder(device: str):
    """Load the real INDUS encoder. Returns an object with ``encode(texts)``."""
    from scix import embed as _embed

    model, tokenizer = _embed.load_model("indus", device=device)

    class _IndusEncoder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return _embed.embed_batch(
                model, tokenizer, texts, batch_size=len(texts), pooling="mean"
            )

    return _IndusEncoder()


def _resolve_clients(*, dry_run: bool, device: str):
    """Return (embedder, qdrant_module) honouring --dry-run."""
    if dry_run:
        return _StubEmbedder(), _StubQdrant()
    if not os.environ.get("QDRANT_URL"):
        raise RuntimeError(
            "QDRANT_URL is not set. Either start the bench with "
            "QDRANT_URL=http://... or pass --dry-run."
        )
    from scix import qdrant_tools as _qt

    return _load_real_embedder(device), _qt


def _run_iteration(
    *,
    embedder: Any,
    qdrant: Any,
    query: str,
    filter_spec: FilterSpec | None,
    limit: int,
) -> tuple[float, float, float]:
    """Run one bench iteration; return (total_ms, encode_ms, ann_ms)."""
    t_encode_start = time.perf_counter()
    vectors = embedder.encode([query])
    t_encode_end = time.perf_counter()
    if not vectors:
        raise RuntimeError("encoder returned no vectors")
    vector = vectors[0]

    kwargs: dict[str, Any] = {"limit": limit}
    if filter_spec is not None:
        kwargs["year_min"] = filter_spec.year_min
        kwargs["year_max"] = filter_spec.year_max
        if filter_spec.arxiv_class is not None:
            kwargs["arxiv_class"] = list(filter_spec.arxiv_class)

    t_ann_start = time.perf_counter()
    qdrant.chunk_search_by_text(vector, **kwargs)
    t_ann_end = time.perf_counter()

    encode_ms = (t_encode_end - t_encode_start) * 1000.0
    ann_ms = (t_ann_end - t_ann_start) * 1000.0
    total_ms = encode_ms + ann_ms
    return total_ms, encode_ms, ann_ms


def run_bench(
    *,
    iterations: int,
    limit: int,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Execute the bench and return a JSON-serialisable result dict."""
    embedder, qdrant = _resolve_clients(dry_run=dry_run, device=device)

    scenarios: dict[str, dict[str, list[float]]] = {
        "no_filter": {"total": [], "encode": [], "ann": []},
        "filtered": {"total": [], "encode": [], "ann": []},
    }

    for scenario_name, filter_fn in (
        ("no_filter", lambda i: None),
        ("filtered", _filter_for_iteration),
    ):
        logger.info("Running scenario %r (%d iterations)", scenario_name, iterations)
        for i in range(iterations):
            query = BENCH_QUERIES[i % len(BENCH_QUERIES)]
            spec = filter_fn(i)
            total_ms, encode_ms, ann_ms = _run_iteration(
                embedder=embedder,
                qdrant=qdrant,
                query=query,
                filter_spec=spec,
                limit=limit,
            )
            scenarios[scenario_name]["total"].append(total_ms)
            scenarios[scenario_name]["encode"].append(encode_ms)
            scenarios[scenario_name]["ann"].append(ann_ms)
            if (i + 1) % 100 == 0:
                logger.info("  iter %d / %d", i + 1, iterations)

    summaries = {
        name: _summary(buckets["total"], buckets["encode"], buckets["ann"])
        for name, buckets in scenarios.items()
    }

    p1_1 = {
        "no_filter_p95_lt_50ms": summaries["no_filter"]["p95_ms"] < 50.0,
        "filtered_p95_lt_100ms": summaries["filtered"]["p95_ms"] < 100.0,
    }

    return {
        "schema_version": 1,
        "tool": "chunk_search",
        "collection": "scix_chunks_v1",
        "embedder": {"model": "indus", "device": device, "dim": 768},
        "ran_at": _utc_now(),
        "git_sha": _git_sha(),
        "qdrant_url": os.environ.get("QDRANT_URL", "<dry-run>"),
        "iterations": iterations,
        "limit": limit,
        "scenarios": summaries,
        "p1_1_pass": p1_1,
        "queries": list(BENCH_QUERIES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Iterations per scenario (PRD target: 1000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Top-K per ANN call (PRD target: 20).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Encoder device. CPU is the production deployment shape.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report to this path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use stub embedder + stub Qdrant client (no network, no model load).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.limit < 1 or args.limit > 100:
        parser.error("--limit must be in [1, 100]")

    report = run_bench(
        iterations=args.iterations,
        limit=args.limit,
        device=args.device,
        dry_run=args.dry_run,
    )

    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        logger.info("Wrote %s", args.output)
    else:
        print(payload)

    no_filter_pass = report["p1_1_pass"]["no_filter_p95_lt_50ms"]
    filtered_pass = report["p1_1_pass"]["filtered_p95_lt_100ms"]
    if not (no_filter_pass and filtered_pass):
        logger.warning(
            "P1-1 latency target NOT met: no_filter_p95<50ms=%s filtered_p95<100ms=%s",
            no_filter_pass,
            filtered_pass,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
