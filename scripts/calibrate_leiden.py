#!/usr/bin/env python3
"""Leiden resolution calibration: sweep resolutions and compare partition quality.

Runs Leiden community detection at a range of resolutions on a subgraph or
the full giant component, collecting quality metrics (community count,
coverage, conductance, NMI vs arXiv taxonomy) at each point.

Supports both modularity (RBConfigurationVertexPartition) and CPM
(CPMVertexPartition, recommended by CWTS) partition types.

Usage:
    # Sweep modularity resolutions on giant component (reads from DB)
    python scripts/calibrate_leiden.py --partition-type modularity

    # Sweep CPM resolutions
    python scripts/calibrate_leiden.py --partition-type CPM

    # Limit to N nodes for faster iteration
    python scripts/calibrate_leiden.py --max-nodes 100000

    # Custom resolution range (log-spaced)
    python scripts/calibrate_leiden.py --res-min 0.0001 --res-max 1.0 --res-steps 20

Outputs:
    results/leiden_calibration.json   — structured sweep results
    results/leiden_calibration.md     — formatted comparison table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.graph_metrics import compare_partitions, sweep_resolutions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate_leiden")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_subgraph(
    conn: Any,
    max_nodes: int | None = None,
) -> tuple[Any, list[int] | None]:
    """Load giant component from DB into igraph, optionally limited to max_nodes.

    Returns (graph, reference_labels) where reference_labels are arXiv taxonomy
    IDs (or None if unavailable).
    """
    import igraph
    from psycopg.rows import dict_row

    logger.info("Loading giant component from paper_metrics...")
    t0 = time.perf_counter()

    # Get giant component bibcodes (community_id_coarse IS NOT NULL and != -1)
    sql = """
        SELECT bibcode, community_taxonomic
        FROM paper_metrics
        WHERE community_id_coarse IS NOT NULL
          AND community_id_coarse != -1
        ORDER BY bibcode
    """
    params: tuple[Any, ...] = ()
    if max_nodes:
        sql += " LIMIT %s"
        params = (max_nodes,)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        logger.error("No giant-component papers found in paper_metrics.")
        sys.exit(1)

    bibcodes = [r["bibcode"] for r in rows]
    bibcode_to_id = {b: i for i, b in enumerate(bibcodes)}
    n_nodes = len(bibcodes)
    logger.info("Loaded %d giant-component nodes in %.1fs", n_nodes, time.perf_counter() - t0)

    # Build reference labels from arXiv taxonomy
    tax_labels = [r["community_taxonomic"] for r in rows]
    has_taxonomy = any(t is not None for t in tax_labels)
    reference_labels: list[int] | None = None
    if has_taxonomy:
        unique_labels = sorted({t for t in tax_labels if t is not None})
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        # Assign -1 to papers without taxonomy (will be excluded from NMI)
        reference_labels = [label_to_id.get(t, -1) if t else -1 for t in tax_labels]
        n_with_tax = sum(1 for r in reference_labels if r >= 0)
        logger.info(
            "Reference taxonomy: %d classes, %d/%d papers labeled",
            len(unique_labels),
            n_with_tax,
            n_nodes,
        )

    # Stream edges
    logger.info("Streaming edges for giant component subgraph...")
    t_edge = time.perf_counter()

    src_list: list[int] = []
    tgt_list: list[int] = []

    with conn.cursor(name="cal_edge_cursor") as cur:
        cur.itersize = 500_000
        cur.execute("SELECT source_bibcode, target_bibcode FROM citation_edges")
        for source, target in cur:
            src_id = bibcode_to_id.get(source)
            tgt_id = bibcode_to_id.get(target)
            if src_id is not None and tgt_id is not None:
                src_list.append(src_id)
                tgt_list.append(tgt_id)

    logger.info("Loaded %d edges in %.1fs", len(src_list), time.perf_counter() - t_edge)

    # Build undirected igraph
    edges = list(zip(src_list, tgt_list))
    graph = igraph.Graph(n=n_nodes, edges=edges, directed=True)
    graph = graph.as_undirected(mode="collapse")
    logger.info("Graph: %d nodes, %d edges", graph.vcount(), graph.ecount())

    return graph, reference_labels


def _write_report(
    sweep_results: list[dict[str, Any]],
    comparison: dict[str, Any],
    partition_type: str,
    elapsed: float,
) -> None:
    """Write a markdown report of the calibration results."""
    md_path = RESULTS_DIR / "leiden_calibration.md"
    lines: list[str] = []
    a = lines.append

    a(f"# Leiden Calibration Report ({partition_type})\n")
    a(f"*Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}*\n")
    a(f"*Elapsed: {elapsed:.1f}s*\n")

    # Sweep table
    a("## Resolution Sweep\n")
    has_nmi = any("nmi" in r for r in sweep_results)
    header = (
        "| Resolution | Communities | Singletons | Max Size | Mean Size | Coverage | Cond. Mean |"
    )
    sep = "|-----------|------------|-----------|---------|----------|----------|-----------|"
    if has_nmi:
        header += " NMI |"
        sep += "-----|"
    a(header)
    a(sep)

    for r in sweep_results:
        s = r["size_stats"]
        row = (
            f"| {r['resolution']:.6f} | {r['n_communities']:,} | "
            f"{s['singletons']:,} | {s['max_size']:,} | {s['mean_size']} | "
            f"{r['coverage']:.4f} | {r['conductance']['mean']:.4f} |"
        )
        if has_nmi:
            nmi_val = r.get("nmi", "N/A")
            if isinstance(nmi_val, float):
                nmi_val = f"{nmi_val:.4f}"
            row += f" {nmi_val} |"
        a(row)
    a("")

    # Pairwise NMI
    if comparison and len(comparison.get("partition_names", [])) > 1:
        a("## Pairwise NMI Between Resolutions\n")
        names = comparison["partition_names"]
        header_row = "| |" + " | ".join(names) + " |"
        sep_row = "|---|" + " | ".join(["---"] * len(names)) + " |"
        a(header_row)
        a(sep_row)
        for name_a in names:
            cells = []
            for name_b in names:
                val = comparison["nmi_matrix"][name_a][name_b]
                cells.append(f"{val:.4f}")
            a(f"| {name_a} | " + " | ".join(cells) + " |")
        a("")

    md_path.write_text("\n".join(lines))
    logger.info("Report saved to %s", md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leiden resolution calibration sweep")
    parser.add_argument(
        "--partition-type",
        choices=["modularity", "CPM"],
        default="CPM",
        help="Partition quality function (default: CPM per CWTS recommendation)",
    )
    parser.add_argument(
        "--res-min",
        type=float,
        default=None,
        help="Minimum resolution (default: 0.0001 for modularity, 0.001 for CPM)",
    )
    parser.add_argument(
        "--res-max",
        type=float,
        default=None,
        help="Maximum resolution (default: 10.0 for modularity, 1.0 for CPM)",
    )
    parser.add_argument(
        "--res-steps",
        type=int,
        default=10,
        help="Number of log-spaced resolution steps (default: 10)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Limit giant component to N nodes for faster iteration",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    t_total = time.perf_counter()

    # Set default resolution ranges based on partition type
    if args.res_min is None:
        args.res_min = 0.001 if args.partition_type == "CPM" else 0.0001
    if args.res_max is None:
        args.res_max = 1.0 if args.partition_type == "CPM" else 10.0

    resolutions = list(np.logspace(np.log10(args.res_min), np.log10(args.res_max), args.res_steps))
    logger.info(
        "Sweeping %d resolutions [%.6f .. %.6f] with %s partition",
        len(resolutions),
        resolutions[0],
        resolutions[-1],
        args.partition_type,
    )

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SET application_name = 'leiden_calibration'")
        cur.execute("SET work_mem = '512MB'")
    conn.commit()

    graph, reference_labels = _load_subgraph(conn, max_nodes=args.max_nodes)
    conn.close()

    # Run sweep
    sweep_results = sweep_resolutions(
        graph,
        resolutions=resolutions,
        seed=args.seed,
        partition_type=args.partition_type,
        reference_labels=reference_labels,
    )

    # Compare partitions pairwise
    partitions = {f"res_{r['resolution']:.6f}": r["membership"] for r in sweep_results}
    comparison = compare_partitions(partitions)

    # Strip memberships for JSON serialization (too large)
    json_results = []
    for r in sweep_results:
        entry = {k: v for k, v in r.items() if k != "membership"}
        json_results.append(entry)

    elapsed = round(time.perf_counter() - t_total, 1)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "partition_type": args.partition_type,
        "resolutions": resolutions,
        "max_nodes": args.max_nodes,
        "seed": args.seed,
        "sweep": json_results,
        "pairwise_nmi": comparison,
        "elapsed_seconds": elapsed,
    }

    json_path = RESULTS_DIR / "leiden_calibration.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", json_path)

    _write_report(sweep_results, comparison, args.partition_type, elapsed)
    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
