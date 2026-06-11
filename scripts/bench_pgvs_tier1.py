#!/usr/bin/env python3
"""Run the pgvectorscale Tier 1 benchmark on the pilot DB.

Tier 1 is the quality-preservation gate at a scale where HNSW works
comfortably (1M sample). Goal: prove that DiskANN variants V1/V2/V3 stay
within 1% nDCG@10 of an HNSW baseline. The pilot DB at 32M is out of
reach for HNSW on this host (per docs/runbooks/halfvec_migration_outcome.md),
so the head-to-head HNSW comparison only happens at 1M.

For each variant we:
  1. drop any existing dense index (paper_embeddings_pkey is kept)
  2. build the variant via DDL with timing
  3. record build wall-clock + on-disk size
  4. run the unfiltered bench script restricted to that variant
  5. parse the per-query metrics out of the bench JSON
  6. drop the variant before the next iteration

Outputs:
  - results/pgvs_benchmark/tier1_builds.json
  - results/pgvs_benchmark/tier1_retrieval_quality.json
  - results/pgvs_benchmark/tier1_retrieval_quality.md

Usage:
    python scripts/bench_pgvs_tier1.py --dsn dbname=scix_pilot
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
logger = logging.getLogger("bench_pgvs_tier1")

OUT_DIR = Path("results/pgvs_benchmark")
BUILDS_PATH = OUT_DIR / "tier1_builds.json"
RETRIEVAL_PATH = OUT_DIR / "tier1_retrieval_quality.json"
RETRIEVAL_MD_PATH = OUT_DIR / "tier1_retrieval_quality.md"

# All variants build the index on the real `embedding halfvec(768)` column
# with `halfvec_cosine_ops`. This avoids the pgvectorscale
# `assertion failed: attnum > 0` error that strikes when DiskANN is built on
# a cast expression (verified on pgvectorscale 0.9.0 / pgvector 0.8.2).
VARIANTS: dict[str, dict[str, Any]] = {
    "hnsw": {
        "kind": "hnsw",
        "ddl": (
            "CREATE INDEX paper_embeddings_hnsw_indus ON paper_embeddings "
            "USING hnsw (embedding halfvec_cosine_ops) "
            "WITH (m = 16, ef_construction = 64) "
            "WHERE model_name='indus'"
        ),
        "index_name": "paper_embeddings_hnsw_indus",
        "params": {"m": 16, "ef_construction": 64, "type": "halfvec(768)"},
    },
    "v1": {
        "kind": "diskann",
        "ddl": (
            "CREATE INDEX paper_embeddings_diskann_v1 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WHERE model_name='indus'"
        ),
        "index_name": "paper_embeddings_diskann_v1",
        "params": {"storage_layout": "plain", "sbq": False, "type": "halfvec(768)"},
    },
    "v2": {
        "kind": "diskann",
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
        "kind": "diskann",
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


def _drop_all_dense_indexes(conn: psycopg.Connection) -> None:
    """Drop all variant indexes — leave only the primary key in place."""
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
    conn.commit()


def _index_size_bytes(conn: psycopg.Connection, index_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size(%s::regclass)", (index_name,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _index_total_size_bytes(conn: psycopg.Connection, index_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_total_relation_size(%s::regclass)", (index_name,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def build_one_variant(dsn: str, name: str) -> dict[str, Any]:
    spec = VARIANTS[name]
    rec: dict[str, Any] = {
        "variant": name,
        "kind": spec["kind"],
        "index_name": spec["index_name"],
        "params": spec["params"],
        "ddl": spec["ddl"],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        _drop_all_dense_indexes(conn)
        with conn.cursor() as cur:
            cur.execute(spec["ddl"])
        wall = time.perf_counter() - t0
        rec["build_wall_seconds"] = round(wall, 3)
        rec["index_size_bytes"] = _index_size_bytes(conn, spec["index_name"])
        rec["index_total_size_bytes"] = _index_total_size_bytes(
            conn, spec["index_name"]
        )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["peak_rss_kb_self"] = int(max(rss_before, rss_after))
    rec["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Built %s in %.1fs — size=%dMB total=%dMB",
        spec["index_name"],
        wall,
        rec["index_size_bytes"] // (1024 * 1024),
        rec["index_total_size_bytes"] // (1024 * 1024),
    )
    return rec


def run_bench_for_variant(dsn: str, name: str, sample_size: int) -> dict[str, Any]:
    spec = VARIANTS[name]
    out_dir = OUT_DIR / f"_tier1_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/bench_pgvectorscale.py",
        "--dsn", dsn,
        "--indexes", name,
        "--index-map", f"{name}={spec['index_name']}",
        "--sample-size", str(sample_size),
        "--out", str(out_dir),
    ]
    logger.info("Running bench: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stderr.strip():
        logger.info("bench stderr: %s", res.stderr.strip()[:400])
    bench_json = out_dir / "retrieval_quality.json"
    if not bench_json.exists():
        raise FileNotFoundError(bench_json)
    return json.loads(bench_json.read_text())


def run_filtered_bench_for_variant(dsn: str, name: str) -> dict[str, Any]:
    spec = VARIANTS[name]
    out_dir = OUT_DIR / f"_tier1_{name}_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/bench_pgvectorscale_filtered.py",
        "--dsn", dsn,
        "--indexes", name,
        "--index-map", f"{name}={spec['index_name']}",
        "--filter", "both",
        "--out", str(out_dir),
    ]
    logger.info("Running filtered bench: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stderr.strip():
        logger.info("filtered bench stderr: %s", res.stderr.strip()[:400])
    bench_json = out_dir / "filtered_queries.json"
    if not bench_json.exists():
        raise FileNotFoundError(bench_json)
    return json.loads(bench_json.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default="dbname=scix_pilot")
    parser.add_argument("--variants", default="hnsw,v1,v2,v3")
    parser.add_argument("--sample-size", type=int, default=500_000,
                        help="Exact-baseline sample size (default 500K — keep < pilot row count)")
    parser.add_argument("--skip-filtered", action="store_true",
                        help="Skip filtered (year, arxiv_class) benches; unfiltered only.")
    args = parser.parse_args(argv)

    if "scix" in args.dsn and "scix_pilot" not in args.dsn:
        if "scix_test" not in args.dsn:
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
    run_filtered = not args.skip_filtered
    for v in variants:
        logger.info("\n=== TIER 1 VARIANT %s ===", v)
        b = build_one_variant(args.dsn, v)
        builds.append(b)
        # Persist incrementally so a later crash doesn't lose earlier work.
        BUILDS_PATH.write_text(json.dumps(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "tier": 1,
             "sample_size": args.sample_size, "builds": builds}, indent=2))
        bench = run_bench_for_variant(args.dsn, v, args.sample_size)
        bench_runs[v] = bench
        if run_filtered:
            filtered = run_filtered_bench_for_variant(args.dsn, v)
            filtered_runs[v] = filtered

    # Write merged retrieval quality artifact.
    merged: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "tier": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dsn_sanitized": args.dsn,
        "sample_size": args.sample_size,
        "variants": [],
        "filtered": [],
    }
    for v in variants:
        bench = bench_runs[v]
        per_index = bench.get("indexes", [])
        # bench script with --indexes <one> emits one entry; collect its metrics.
        if not per_index:
            continue
        ix = per_index[0]
        merged["variants"].append({
            "name": v,
            "kind": VARIANTS[v]["kind"],
            "index_name": VARIANTS[v]["index_name"],
            "metrics": ix["metrics"],
            "n_per_query": len(ix.get("per_query") or []),
            "true_recall_at_10": _mean_true_recall(ix.get("per_query") or []),
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
    RETRIEVAL_MD_PATH.write_text(_render_md(merged, builds))
    logger.info("Wrote %s and %s", RETRIEVAL_PATH, RETRIEVAL_MD_PATH)
    return 0


def _mean_true_recall(per_query: list[dict[str, Any]]) -> float | None:
    vals = [
        float(r["true_recall_10"])
        for r in per_query
        if isinstance(r.get("true_recall_10"), (int, float))
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _render_md(merged: dict[str, Any], builds: list[dict[str, Any]]) -> str:
    lines = [
        "# Tier 1 — pgvectorscale vs HNSW retrieval-quality bench (1M sample)",
        "",
        f"- Run ID: `{merged['run_id']}`",
        f"- Timestamp: `{merged['timestamp']}`",
        f"- DSN: `{merged['dsn_sanitized']}`",
        f"- Exact-baseline sample size: `{merged['sample_size']}`",
        "",
        "## Build metrics",
        "",
        "| Variant | Index | Build wall (s) | Index size | Total relation size |",
        "|---------|-------|---------------:|-----------:|---------------------:|",
    ]
    for b in builds:
        lines.append(
            f"| {b['variant']} | `{b['index_name']}` | "
            f"{b['build_wall_seconds']:.1f} | "
            f"{_pretty_bytes(b['index_size_bytes'])} | "
            f"{_pretty_bytes(b['index_total_size_bytes'])} |"
        )
    lines.append("")
    lines.append("## Retrieval quality")
    lines.append("")
    lines.append(
        "| Variant | nDCG@10 | Recall@10 | Recall@20 | MRR | true_recall@10 vs exact | p50 (ms) | p95 (ms) |"
    )
    lines.append(
        "|---------|--------:|----------:|----------:|----:|------------------------:|---------:|---------:|"
    )
    for v in merged["variants"]:
        m = v["metrics"]
        lines.append(
            f"| {v['name']} | "
            f"{_fmt(m.get('ndcg_at_10'))} | "
            f"{_fmt(m.get('recall_at_10'))} | "
            f"{_fmt(m.get('recall_at_20'))} | "
            f"{_fmt(m.get('mrr'))} | "
            f"{_fmt(v.get('true_recall_at_10'))} | "
            f"{_fmt(m.get('p50_ms'), digits=1)} | "
            f"{_fmt(m.get('p95_ms'), digits=1)} |"
        )
    lines.append("")

    # PASS/FAIL on C1 — within 1% nDCG@10 vs hnsw.
    hnsw = next((v for v in merged["variants"] if v["name"] == "hnsw"), None)
    if hnsw is not None and hnsw["metrics"].get("ndcg_at_10") is not None:
        h = hnsw["metrics"]["ndcg_at_10"]
        lines.append("## C1 PASS/FAIL — |Δ nDCG@10 vs HNSW| ≤ 0.01")
        lines.append("")
        lines.append("| Variant | nDCG@10 | Δ vs HNSW | Verdict |")
        lines.append("|---------|--------:|----------:|---------|")
        for v in merged["variants"]:
            if v["name"] == "hnsw":
                continue
            x = v["metrics"].get("ndcg_at_10")
            if x is None:
                lines.append(f"| {v['name']} | N/A | N/A | N/A |")
                continue
            d = x - h
            verdict = "PASS" if abs(d) <= 0.01 else "FAIL"
            lines.append(f"| {v['name']} | {x:.4f} | {d:+.4f} | {verdict} |")
        lines.append("")

    if merged.get("filtered"):
        lines.append("## Filtered queries (F1 year=2024 ~10%, F2 arxiv_class astro-ph.{GA,SR} ~20%)")
        lines.append("")
        lines.append(
            "| Variant | Filter | nDCG@10 | Recall@10 | p50 (ms) | p95 (ms) | "
            "p95 unfilt (ms) | p95 filt/unfilt |"
        )
        lines.append(
            "|---------|--------|--------:|----------:|---------:|---------:|----------------:|----------------:|"
        )
        for f in merged["filtered"]:
            m = f.get("metrics") or {}
            p95_unf = f.get("p95_unfiltered_ms")
            p95_f = m.get("p95_ms")
            ratio = (p95_f / p95_unf) if (p95_unf and p95_f) else None
            lines.append(
                f"| {f['variant']} | {f['filter_key']} | "
                f"{_fmt(m.get('ndcg_at_10'))} | "
                f"{_fmt(m.get('recall_at_10'))} | "
                f"{_fmt(m.get('p50_ms'), digits=1)} | "
                f"{_fmt(p95_f, digits=1)} | "
                f"{_fmt(p95_unf, digits=1)} | "
                f"{_fmt(ratio, digits=2)} |"
            )
        lines.append("")

        # C2 PASS/FAIL — DiskANN p95 ≥30% lower than HNSW p95 on each filter.
        if hnsw is not None:
            lines.append("## C2 PASS/FAIL — filtered p95 ≥30% lower than HNSW")
            lines.append("")
            lines.append("| Variant | Filter | p95 (ms) | HNSW p95 (ms) | Δ% | Verdict |")
            lines.append("|---------|--------|---------:|--------------:|---:|---------|")
            hnsw_p95: dict[str, float | None] = {}
            for f in merged["filtered"]:
                if f["variant"] == "hnsw":
                    hnsw_p95[f["filter_key"]] = (f.get("metrics") or {}).get("p95_ms")
            for f in merged["filtered"]:
                if f["variant"] == "hnsw":
                    continue
                fk = f["filter_key"]
                p95_v = (f.get("metrics") or {}).get("p95_ms")
                p95_h = hnsw_p95.get(fk)
                if p95_v is None or p95_h is None:
                    lines.append(f"| {f['variant']} | {fk} | N/A | N/A | N/A | N/A |")
                    continue
                pct = (p95_h - p95_v) / p95_h * 100 if p95_h else 0.0
                verdict = "PASS" if pct >= 30.0 else "FAIL"
                lines.append(
                    f"| {f['variant']} | {fk} | {p95_v:.1f} | {p95_h:.1f} | "
                    f"{pct:+.1f}% | {verdict} |"
                )
            lines.append("")
    return "\n".join(lines)


def _pretty_bytes(n: int | None) -> str:
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
