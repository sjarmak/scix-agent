#!/usr/bin/env python3
"""Run the pgvectorscale Tier 2 scale-validation benchmark on 32M halfvec.

Tier 2 answers the question Tier 1 cannot: does pgvectorscale actually fit
the 32M halfvec(768) regime where HNSW physically does not (per
docs/runbooks/halfvec_migration_outcome.md)? There is **no HNSW comparison
at 32M** — three independent build attempts at 62 GB RAM established that
HNSW at this scale is not buildable on this host.

This script runs DiskANN V1/V2/V3 only. Each variant is built, measured
(build time, peak working memory, on-disk size), benched (unfiltered +
filtered), and **dropped before the next variant** so the pilot DB never
holds more than one DiskANN index at once. Disk budget per build/bench
cycle: ~50 GB heap + ~10-50 GB index = ≤100 GB. Verified against
/dev/nvme1n1p2 free-space target ≥120 GB.

Outputs (all in results/pgvs_benchmark/):
  - tier2_builds.json
  - tier2_retrieval_quality.json + tier2_retrieval_quality.md
  - tier2_<variant>/{retrieval_quality,filtered_queries}.{json,md}

Usage:
    python scripts/bench_pgvs_tier2.py --dsn dbname=scix_pilot \\
        --variants v1,v2,v3 --sample-size 1000000

The --sample-size is the EXACT-baseline sample for ground-truth recall;
it is independent of the 32M index. 1M is plenty.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bench_pgvs_tier2")

OUT_DIR = Path("results/pgvs_benchmark")
BUILDS_PATH = OUT_DIR / "tier2_builds.json"
RETRIEVAL_PATH = OUT_DIR / "tier2_retrieval_quality.json"
RETRIEVAL_MD_PATH = OUT_DIR / "tier2_retrieval_quality.md"

VARIANTS: dict[str, dict[str, Any]] = {
    "v1": {
        "ddl": (
            "CREATE INDEX paper_embeddings_diskann_v1 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WHERE model_name='indus'"
        ),
        "index_name": "paper_embeddings_diskann_v1",
        "params": {"storage_layout": "plain", "sbq": False, "type": "halfvec(768)"},
    },
    "v2": {
        "ddl": (
            "CREATE INDEX paper_embeddings_diskann_v2 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WITH (num_bits_per_dimension = 2) "
            "WHERE model_name='indus'"
        ),
        "index_name": "paper_embeddings_diskann_v2",
        "params": {"storage_layout": "plain", "sbq": True, "num_bits_per_dimension": 2, "type": "halfvec(768)"},
    },
    "v3": {
        "ddl": (
            "CREATE INDEX paper_embeddings_diskann_v3 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WITH (storage_layout = 'memory_optimized', num_neighbors = 64, "
            "num_bits_per_dimension = 2) "
            "WHERE model_name='indus'"
        ),
        "index_name": "paper_embeddings_diskann_v3",
        "params": {"storage_layout": "memory_optimized", "num_neighbors": 64, "sbq": True, "num_bits_per_dimension": 2, "type": "halfvec(768)"},
    },
}


def _drop_dense_indexes(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname='public' AND tablename='paper_embeddings'
              AND indexname != 'paper_embeddings_pkey'
        """)
        existing = [r[0] for r in cur.fetchall()]
        for idx in existing:
            logger.info("Dropping existing index: %s", idx)
            cur.execute(f"DROP INDEX IF EXISTS {idx}")


def _index_size(conn: psycopg.Connection, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size(%s::regclass)", (name,))
        return int(cur.fetchone()[0] or 0)


def build_one(dsn: str, name: str) -> dict[str, Any]:
    spec = VARIANTS[name]
    rec: dict[str, Any] = {
        "variant": name,
        "kind": "diskann",
        "index_name": spec["index_name"],
        "params": spec["params"],
        "ddl": spec["ddl"],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        _drop_dense_indexes(conn)
        with conn.cursor() as cur:
            cur.execute(spec["ddl"])
        wall = time.perf_counter() - t0
        rec["build_wall_seconds"] = round(wall, 3)
        rec["index_size_bytes"] = _index_size(conn, spec["index_name"])
    rec["peak_rss_kb_self"] = int(max(rss_before, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    rec["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Built %s in %.0fs — size=%dGB",
        spec["index_name"], wall, rec["index_size_bytes"] // (1024**3),
    )
    return rec


def run_unfiltered_bench(dsn: str, name: str, sample_size: int) -> dict[str, Any]:
    spec = VARIANTS[name]
    out_dir = OUT_DIR / f"tier2_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/bench_pgvectorscale.py",
        "--dsn", dsn,
        "--indexes", name,
        "--index-map", f"{name}={spec['index_name']}",
        "--sample-size", str(sample_size),
        "--out", str(out_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stderr.strip():
        logger.info("stderr: %s", res.stderr.strip()[:400])
    bench_json = out_dir / "retrieval_quality.json"
    return json.loads(bench_json.read_text())


def run_filtered_bench(dsn: str, name: str) -> dict[str, Any]:
    spec = VARIANTS[name]
    out_dir = OUT_DIR / f"tier2_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/bench_pgvectorscale_filtered.py",
        "--dsn", dsn,
        "--indexes", name,
        "--index-map", f"{name}={spec['index_name']}",
        "--filter", "both",
        "--out", str(out_dir),
    ]
    logger.info("Running filtered: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stderr.strip():
        logger.info("stderr: %s", res.stderr.strip()[:400])
    return json.loads((out_dir / "filtered_queries.json").read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default="dbname=scix_pilot")
    parser.add_argument("--variants", default="v1,v2,v3")
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--skip-filtered", action="store_true")
    args = parser.parse_args(argv)

    if "scix" in args.dsn and "scix_pilot" not in args.dsn and "scix_test" not in args.dsn:
        print(f"REFUSING — DSN looks like prod: {args.dsn}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            print(f"unknown variant: {v}", file=sys.stderr)
            return 2

    builds: list[dict[str, Any]] = []
    bench_runs: dict[str, dict[str, Any]] = {}
    filtered_runs: dict[str, dict[str, Any]] = {}
    for v in variants:
        logger.info("\n=== TIER 2 VARIANT %s ===", v)
        b = build_one(args.dsn, v)
        builds.append(b)
        BUILDS_PATH.write_text(json.dumps(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "tier": 2,
             "sample_size": args.sample_size, "builds": builds}, indent=2))
        bench_runs[v] = run_unfiltered_bench(args.dsn, v, args.sample_size)
        if not args.skip_filtered:
            filtered_runs[v] = run_filtered_bench(args.dsn, v)

    merged: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "tier": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dsn_sanitized": args.dsn,
        "sample_size": args.sample_size,
        "variants": [],
        "filtered": [],
    }
    for v in variants:
        bench = bench_runs[v]
        per_index = bench.get("indexes") or []
        if not per_index:
            continue
        ix = per_index[0]
        merged["variants"].append({
            "name": v,
            "index_name": VARIANTS[v]["index_name"],
            "metrics": ix["metrics"],
            "n_per_query": len(ix.get("per_query") or []),
        })
        if v in filtered_runs:
            for cell in filtered_runs[v].get("results", []):
                merged["filtered"].append({
                    "variant": v,
                    "filter_key": cell.get("filter_key"),
                    "metrics": cell.get("metrics"),
                    "p95_unfiltered_ms": ix["metrics"].get("p95_ms"),
                })
    RETRIEVAL_PATH.write_text(json.dumps(merged, indent=2))
    RETRIEVAL_MD_PATH.write_text(_render(merged, builds))
    logger.info("Wrote %s + %s", RETRIEVAL_PATH, RETRIEVAL_MD_PATH)
    return 0


def _render(merged: dict[str, Any], builds: list[dict[str, Any]]) -> str:
    lines = [
        "# Tier 2 — pgvectorscale 32M scale-validation",
        "",
        "**No HNSW baseline:** HNSW on 32M halfvec(768) does not fit on this host's",
        "62 GB physical RAM (per docs/runbooks/halfvec_migration_outcome.md). The",
        "decision question Tier 2 answers is *whether pgvectorscale fits where HNSW",
        "did not*, not *whether pgvectorscale beats HNSW at 32M*.",
        "",
        f"- Run ID: `{merged['run_id']}`",
        f"- Timestamp: `{merged['timestamp']}`",
        f"- Exact-baseline sample size: `{merged['sample_size']}`",
        "",
        "## Build metrics (32M halfvec rows)",
        "",
        "| Variant | Index | Build wall (s) | Index size |",
        "|---------|-------|---------------:|-----------:|",
    ]
    for b in builds:
        lines.append(
            f"| {b['variant']} | `{b['index_name']}` | "
            f"{b['build_wall_seconds']:.0f} | "
            f"{_pretty(b['index_size_bytes'])} |"
        )
    lines.append("")
    lines.append("## Unfiltered retrieval quality")
    lines.append("")
    lines.append("| Variant | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 (ms) | p95 (ms) |")
    lines.append("|---------|--------:|----------:|----------:|----:|---------:|---------:|")
    for v in merged["variants"]:
        m = v["metrics"]
        lines.append(
            f"| {v['name']} | "
            f"{_fmt(m.get('ndcg_at_10'))} | "
            f"{_fmt(m.get('recall_at_10'))} | "
            f"{_fmt(m.get('recall_at_20'))} | "
            f"{_fmt(m.get('mrr'))} | "
            f"{_fmt(m.get('p50_ms'), 1)} | "
            f"{_fmt(m.get('p95_ms'), 1)} |"
        )
    lines.append("")
    if merged.get("filtered"):
        lines.append("## Filtered queries")
        lines.append("")
        lines.append(
            "| Variant | Filter | nDCG@10 | p50 (ms) | p95 (ms) | p95 unfilt (ms) |"
        )
        lines.append(
            "|---------|--------|--------:|---------:|---------:|----------------:|"
        )
        for f in merged["filtered"]:
            m = f.get("metrics") or {}
            lines.append(
                f"| {f['variant']} | {f['filter_key']} | "
                f"{_fmt(m.get('ndcg_at_10'))} | "
                f"{_fmt(m.get('p50_ms'), 1)} | "
                f"{_fmt(m.get('p95_ms'), 1)} | "
                f"{_fmt(f.get('p95_unfiltered_ms'), 1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _pretty(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(n)
    i = 0
    while val >= 1024 and i < len(units) - 1:
        val /= 1024
        i += 1
    return f"{val:.1f} {units[i]}"


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
