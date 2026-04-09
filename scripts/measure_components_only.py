#!/usr/bin/env python3
"""Measure giant component and graph structure for paper Section 5.4.

Lightweight Phase A only — uses scipy sparse matrix for connected components.
Peak memory: ~4-5GB (vs ~20GB for igraph + Leiden).

Outputs:
  results/graph_quality_metrics.json  — structured results (Phase A only)
  results/graph_quality_report.md     — formatted for paper
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("measure_components")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    t_total = time.perf_counter()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SET application_name = 'graph_components_measurement'")
        cur.execute("SET work_mem = '512MB'")
    conn.commit()

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # --- Get node count ---
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM papers")
        (n_nodes,) = cur.fetchone()
    logger.info("Total papers: %d", n_nodes)

    # --- Create bibcode→int mapping in temp table ---
    logger.info("Creating node ID mapping...")
    t_map = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE node_ids (
                bibcode TEXT PRIMARY KEY,
                nid INT NOT NULL
            ) ON COMMIT PRESERVE ROWS
        """)
        cur.execute("""
            INSERT INTO node_ids (bibcode, nid)
            SELECT bibcode, ROW_NUMBER() OVER (ORDER BY bibcode)::INT - 1
            FROM papers
        """)
        cur.execute("CREATE INDEX ON node_ids (nid)")
        cur.execute("ANALYZE node_ids")
    conn.commit()
    logger.info("Node mapping created in %.1fs", time.perf_counter() - t_map)

    # --- Count edges ---
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM citation_edges")
        (edge_count_total,) = cur.fetchone()
    logger.info("Total edges in DB: %d", edge_count_total)

    # --- Stream edges as integer pairs ---
    logger.info("Streaming edges (SQL JOIN, no Python dicts)...")
    t_edge = time.perf_counter()

    src_arr = np.empty(edge_count_total, dtype=np.int32)
    tgt_arr = np.empty(edge_count_total, dtype=np.int32)
    valid = 0

    with conn.cursor(name="edge_cursor") as cur:
        cur.itersize = 1_000_000
        cur.execute("""
            SELECT n1.nid, n2.nid
            FROM citation_edges ce
            JOIN node_ids n1 ON ce.source_bibcode = n1.bibcode
            JOIN node_ids n2 ON ce.target_bibcode = n2.bibcode
        """)
        for src_id, tgt_id in cur:
            src_arr[valid] = src_id
            tgt_arr[valid] = tgt_id
            valid += 1

    src_arr = src_arr[:valid]
    tgt_arr = tgt_arr[:valid]
    skipped = edge_count_total - valid
    logger.info(
        "Loaded %d edges (skipped %d dangling) in %.1fs",
        valid, skipped, time.perf_counter() - t_edge,
    )

    results["full_graph"] = {
        "nodes": n_nodes,
        "edges": valid,
        "edges_total": edge_count_total,
        "edges_dangling": skipped,
        "edge_resolution_pct": round(valid / edge_count_total * 100, 2),
    }

    # --- Build scipy sparse + connected components ---
    logger.info("Building CSR matrix...")
    data = np.ones(valid, dtype=np.int8)
    graph_csr = csr_matrix((data, (src_arr, tgt_arr)), shape=(n_nodes, n_nodes))
    del data
    graph_sym = graph_csr + graph_csr.T
    del graph_csr, src_arr, tgt_arr
    gc.collect()

    logger.info("Computing connected components...")
    t_comp = time.perf_counter()
    n_components, labels = connected_components(graph_sym, directed=False, return_labels=True)
    del graph_sym
    gc.collect()
    logger.info("Found %d components in %.1fs", n_components, time.perf_counter() - t_comp)

    # --- Analyze ---
    label_counts = Counter(labels.tolist())
    comp_sizes = sorted(label_counts.values(), reverse=True)
    giant_label = max(label_counts, key=label_counts.get)
    n_giant = label_counts[giant_label]
    isolated_count = sum(1 for s in label_counts.values() if s == 1)

    # Size distribution
    sizes_arr = np.array(comp_sizes)
    size_distribution = {
        "1 (isolated)": int(np.sum(sizes_arr == 1)),
        "2-10": int(np.sum((sizes_arr >= 2) & (sizes_arr <= 10))),
        "11-100": int(np.sum((sizes_arr >= 11) & (sizes_arr <= 100))),
        "101-1000": int(np.sum((sizes_arr >= 101) & (sizes_arr <= 1000))),
        "1001-10000": int(np.sum((sizes_arr >= 1001) & (sizes_arr <= 10000))),
        "10001+": int(np.sum(sizes_arr > 10000)),
    }

    results["isolated_nodes"] = isolated_count
    results["components"] = {
        "total_components": n_components,
        "giant_component_nodes": n_giant,
        "giant_component_pct_of_total": round(n_giant / n_nodes * 100, 2),
        "giant_component_pct_of_connected": round(
            n_giant / max(n_nodes - isolated_count, 1) * 100, 2
        ),
        "top_10_sizes": comp_sizes[:10],
        "components_gt_100": sum(1 for s in comp_sizes if s > 100),
        "components_gt_1000": sum(1 for s in comp_sizes if s > 1000),
        "small_component_papers": n_nodes - n_giant - isolated_count,
        "size_distribution": size_distribution,
    }

    logger.info("Giant component: %d nodes (%.1f%% of total)", n_giant, n_giant / n_nodes * 100)
    logger.info("Isolated nodes: %d", isolated_count)

    # --- Degree distribution stats (from DB, memory-efficient) ---
    logger.info("Computing degree statistics from DB...")
    with conn.cursor() as cur:
        cur.execute("""
            WITH degree AS (
                SELECT source_bibcode as bibcode, COUNT(*) as out_deg FROM citation_edges GROUP BY 1
            )
            SELECT
                MIN(out_deg), MAX(out_deg),
                AVG(out_deg)::NUMERIC(10,2),
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY out_deg),
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY out_deg)
            FROM degree
        """)
        row = cur.fetchone()
        results["out_degree"] = {
            "min": int(row[0]), "max": int(row[1]),
            "mean": float(row[2]), "median": float(row[3]),
            "p99": float(row[4]),
        }

    del labels
    gc.collect()

    # --- Save ---
    elapsed = round(time.perf_counter() - t_total, 1)
    results["elapsed_seconds"] = elapsed
    results["note"] = "Phase A only (components). Leiden communities not computed (requires separate run with more memory)."

    json_path = RESULTS_DIR / "graph_quality_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics to %s", json_path)

    _write_report(results)

    conn.close()
    logger.info("Done in %.0fs", elapsed)


def _write_report(results: dict[str, Any]) -> None:
    md_path = RESULTS_DIR / "graph_quality_report.md"
    lines: list[str] = []
    a = lines.append

    a("# Graph Quality Metrics — Section 5.4\n")
    a(f"*Generated: {results['timestamp']}*\n")

    fg = results["full_graph"]
    a("## Full Citation Graph\n")
    a(f"- **Nodes (papers):** {fg['nodes']:,}")
    a(f"- **Edges (resolved citations):** {fg['edges']:,}")
    a(f"- **Total edges in DB:** {fg['edges_total']:,}")
    a(f"- **Dangling edges:** {fg['edges_dangling']:,} ({100 - fg['edge_resolution_pct']:.1f}%)")
    a(f"- **Edge resolution:** {fg['edge_resolution_pct']}%")
    a(f"- **Isolated nodes (degree=0):** {results['isolated_nodes']:,}")
    a("")

    comp = results["components"]
    a("## Connected Components\n")
    a(f"- **Total components:** {comp['total_components']:,}")
    a(f"- **Giant component:** {comp['giant_component_nodes']:,} nodes "
      f"({comp['giant_component_pct_of_total']}% of total, "
      f"{comp['giant_component_pct_of_connected']}% of connected)")
    a(f"- **Components > 100 nodes:** {comp['components_gt_100']:,}")
    a(f"- **Components > 1,000 nodes:** {comp['components_gt_1000']:,}")
    a(f"- **Small-component papers:** {comp['small_component_papers']:,}")
    a(f"- **Top-10 component sizes:** {comp['top_10_sizes']}")
    a("")

    if "size_distribution" in comp:
        a("### Component Size Distribution\n")
        a("| Size Range | Count |")
        a("|-----------|-------|")
        for rng, cnt in comp["size_distribution"].items():
            a(f"| {rng} | {cnt:,} |")
        a("")

    if "out_degree" in results:
        d = results["out_degree"]
        a("## Out-Degree Statistics (citations made)\n")
        a(f"- **Min:** {d['min']}, **Max:** {d['max']}")
        a(f"- **Mean:** {d['mean']}, **Median:** {d['median']}")
        a(f"- **P99:** {d['p99']}")
        a("")

    a("## Leiden Community Detection\n")
    if any(f"leiden_{rn}" in results for rn in ("coarse", "medium", "fine")):
        a("| Resolution | Value | Communities | Singletons | Max Size | Mean Size | Top-10 % |")
        a("|-----------|-------|------------|-----------|---------|----------|---------|")
        for rn in ("coarse", "medium", "fine"):
            key = f"leiden_{rn}"
            if key not in results:
                continue
            s = results[key]["stats"]
            a(f"| {rn} | {results[key]['resolution']} | {s['n_communities']:,} | "
              f"{s['singletons']:,} | {s['max_size']:,} | {s['mean_size']} | "
              f"{s['pct_in_top10']}% |")
    else:
        a("*Not yet computed — requires separate run with more memory.*\n")
    a("")

    if "nmi_vs_arxiv" in results:
        nmi = results["nmi_vs_arxiv"]
        a("## NMI: Leiden vs arXiv Taxonomy\n")
        a(f"- **Papers evaluated:** {nmi['n_papers_evaluated']:,}")
        a(f"- **arXiv classes:** {nmi['n_arxiv_classes']}")
        a("")
        a("| Resolution | NMI |")
        a("|-----------|-----|")
        for rn, val in nmi["scores"].items():
            a(f"| {rn} | {val} |")
        a("")

    a(f"\n*Total elapsed: {results.get('elapsed_seconds', 'N/A')}s*\n")

    md_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", md_path)


if __name__ == "__main__":
    main()
