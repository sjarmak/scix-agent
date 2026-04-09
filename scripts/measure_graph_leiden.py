#!/usr/bin/env python3
"""Phase B+C: Leiden community detection + quality metrics for Section 5.4.

Resumes from Phase A state in the database (non-giant papers already marked
with community_id_coarse = -1).  Designed to run AFTER measure_components_only.py.

Phase B: Leiden at 3 resolutions on giant component (~10GB peak)
Phase C: NMI and purity vs arXiv taxonomy (<2GB)

Memory optimisations vs measure_graph_quality.py:
  - Giant node mapping lives in a PostgreSQL temp table (not Python dict)
  - Build igraph directed, convert to undirected, delete directed (sequential)
  - gc.collect() at every stage gate

Outputs:
  results/graph_quality_metrics.json  — merged Phase A + B + C
  results/graph_quality_report.md     — formatted for paper
"""

from __future__ import annotations

import gc
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("measure_leiden")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nmi(labels_a: list[int], labels_b: list[int]) -> float:
    """NMI with arithmetic mean normalization: 2*MI / (H(A) + H(B))."""
    n = len(labels_a)
    if n == 0:
        return 0.0

    contingency: dict[tuple[int, int], int] = Counter(zip(labels_a, labels_b))
    counts_a: dict[int, int] = Counter(labels_a)
    counts_b: dict[int, int] = Counter(labels_b)

    mi = 0.0
    for (a, b), nij in contingency.items():
        if nij == 0:
            continue
        ni = counts_a[a]
        nj = counts_b[b]
        mi += (nij / n) * math.log(n * nij / (ni * nj))

    h_a = -sum((c / n) * math.log(c / n) for c in counts_a.values() if c > 0)
    h_b = -sum((c / n) * math.log(c / n) for c in counts_b.values() if c > 0)

    if h_a + h_b == 0:
        return 1.0 if mi == 0 else 0.0
    return 2.0 * mi / (h_a + h_b)


def _community_size_stats(membership: list[int]) -> dict[str, Any]:
    sizes = list(Counter(membership).values())
    arr = np.array(sizes)
    return {
        "n_communities": len(sizes),
        "min_size": int(arr.min()),
        "max_size": int(arr.max()),
        "mean_size": round(float(arr.mean()), 1),
        "median_size": round(float(np.median(arr)), 1),
        "std_size": round(float(arr.std()), 1),
        "singletons": int(np.sum(arr == 1)),
        "pct_in_top10": round(float(np.sort(arr)[-10:].sum() / arr.sum() * 100), 1),
    }


def _percentile_distribution(membership: list[int]) -> list[dict[str, Any]]:
    counts = Counter(membership)
    total = len(membership)
    return [
        {"community_id": cid, "size": cnt, "pct": round(cnt / total * 100, 2)}
        for cid, cnt in counts.most_common(20)
    ]


# ---------------------------------------------------------------------------
# Phase B: Leiden on giant component
# ---------------------------------------------------------------------------


def phase_b_leiden(conn: psycopg.Connection, results: dict[str, Any]) -> None:
    """Run Leiden at 3 resolutions on the giant component.

    Identifies giant-component papers from paper_metrics:
      - community_id_coarse IS NULL  → giant component (not yet assigned)
      - community_id_coarse = -1     → non-giant (marked by Phase A)
    """
    import igraph
    import leidenalg

    t0 = time.perf_counter()

    # --- Create giant-component node mapping in temp table ---
    logger.info("Creating giant node mapping from paper_metrics state...")
    t_map = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE giant_node_map (
                bibcode TEXT PRIMARY KEY,
                new_nid INT NOT NULL
            ) ON COMMIT PRESERVE ROWS
        """)
        # Papers in giant component: community_id_coarse IS NULL
        # (Phase A set non-giant to -1; giant papers were never touched)
        cur.execute("""
            INSERT INTO giant_node_map (bibcode, new_nid)
            SELECT bibcode, ROW_NUMBER() OVER (ORDER BY bibcode)::INT - 1
            FROM paper_metrics
            WHERE community_id_coarse IS NULL
        """)
        cur.execute("CREATE INDEX ON giant_node_map (new_nid)")
        cur.execute("ANALYZE giant_node_map")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM giant_node_map")
        (n_giant,) = cur.fetchone()
    logger.info("Giant component: %d nodes (mapping built in %.1fs)", n_giant, time.perf_counter() - t_map)

    results["giant_component_nodes"] = n_giant

    # --- Stream edges for giant component ---
    logger.info("Streaming giant component edges (SQL JOIN)...")
    t_edge = time.perf_counter()

    # Estimate upper bound from full edge count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM citation_edges")
        (total_edges,) = cur.fetchone()

    src_arr = np.empty(total_edges, dtype=np.int32)
    tgt_arr = np.empty(total_edges, dtype=np.int32)
    valid = 0

    with conn.cursor(name="giant_edge_cursor") as cur:
        cur.itersize = 1_000_000
        cur.execute("""
            SELECT g1.new_nid, g2.new_nid
            FROM citation_edges ce
            JOIN giant_node_map g1 ON ce.source_bibcode = g1.bibcode
            JOIN giant_node_map g2 ON ce.target_bibcode = g2.bibcode
        """)
        for src_id, tgt_id in cur:
            src_arr[valid] = src_id
            tgt_arr[valid] = tgt_id
            valid += 1

    src_arr = src_arr[:valid]
    tgt_arr = tgt_arr[:valid]
    logger.info("Loaded %d giant component edges in %.1fs", valid, time.perf_counter() - t_edge)
    results["giant_component_edges"] = valid

    # --- Build igraph: directed → undirected ---
    logger.info("Building directed igraph (%d nodes, %d edges)...", n_giant, valid)
    t_graph = time.perf_counter()

    edges = np.column_stack([src_arr, tgt_arr])
    del src_arr, tgt_arr
    gc.collect()

    graph_dir = igraph.Graph(n=n_giant, edges=edges.tolist(), directed=True)
    del edges
    gc.collect()
    logger.info("Directed igraph built in %.1fs (%d V, %d E)",
                time.perf_counter() - t_graph, graph_dir.vcount(), graph_dir.ecount())

    logger.info("Converting to undirected (collapse mode)...")
    t_undir = time.perf_counter()
    graph = graph_dir.as_undirected(mode="collapse")
    del graph_dir
    gc.collect()
    logger.info("Undirected: %d V, %d E in %.1fs",
                graph.vcount(), graph.ecount(), time.perf_counter() - t_undir)

    # --- Run Leiden at 3 resolutions ---
    resolution_map = {"coarse": 0.001, "medium": 0.01, "fine": 0.1}

    for res_name, res_val in resolution_map.items():
        logger.info("Running Leiden at %s (res=%.4f)...", res_name, res_val)
        t_leiden = time.perf_counter()
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=res_val,
            seed=42,
        )
        membership = list(partition.membership)
        elapsed_leiden = time.perf_counter() - t_leiden
        logger.info("Leiden %s: %d communities in %.1fs",
                     res_name, len(set(membership)), elapsed_leiden)

        stats = _community_size_stats(membership)
        dist = _percentile_distribution(membership)
        results[f"leiden_{res_name}"] = {
            "resolution": res_val,
            "elapsed_seconds": round(elapsed_leiden, 1),
            "stats": stats,
            "top_20_communities": dist,
        }

        # Store community assignments
        _store_community_assignments(conn, res_name, membership)

        del membership, partition
        gc.collect()

    del graph
    gc.collect()

    results["phase_b_seconds"] = round(time.perf_counter() - t0, 1)
    logger.info("Phase B complete in %.1fs", results["phase_b_seconds"])


def _store_community_assignments(
    conn: psycopg.Connection,
    res_name: str,
    membership: list[int],
) -> None:
    """Store community assignments via temp table + UPDATE JOIN."""
    col = f"community_id_{res_name}"
    chunk_size = 100_000
    n = len(membership)
    stored = 0

    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _comm_assign (new_nid INT, comm_id INT)")
        cur.execute("TRUNCATE _comm_assign")
    conn.commit()

    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        with conn.cursor() as cur:
            with cur.copy("COPY _comm_assign (new_nid, comm_id) FROM STDIN") as copy:
                for vid in range(i, chunk_end):
                    copy.write_row((vid, membership[vid]))
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE paper_metrics pm
                SET {col} = ca.comm_id
                FROM _comm_assign ca
                JOIN giant_node_map gnm ON ca.new_nid = gnm.new_nid
                WHERE pm.bibcode = gnm.bibcode
            """)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("TRUNCATE _comm_assign")
        conn.commit()

        stored += chunk_end - i
        if stored % 1_000_000 == 0 or stored == n:
            logger.info("  Stored %d/%d %s assignments", stored, n, res_name)

    logger.info("Stored %d %s community assignments", stored, res_name)


# ---------------------------------------------------------------------------
# Phase C: NMI and purity
# ---------------------------------------------------------------------------


def phase_c_quality(conn: psycopg.Connection, results: dict[str, Any]) -> None:
    """Compute NMI and purity from stored community assignments."""
    t0 = time.perf_counter()
    logger.info("=== Phase C: NMI vs arXiv taxonomy ===")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT community_id_coarse, community_id_medium, community_id_fine,
                   community_taxonomic
            FROM paper_metrics
            WHERE community_taxonomic IS NOT NULL
              AND community_id_coarse IS NOT NULL
              AND community_id_coarse != -1
        """)
        tax_rows = cur.fetchall()

    logger.info("Papers with both Leiden and arXiv labels: %d", len(tax_rows))

    if tax_rows:
        tax_labels_str = [r["community_taxonomic"] for r in tax_rows]
        tax_unique = sorted(set(tax_labels_str))
        tax_label_to_id = {lbl: i for i, lbl in enumerate(tax_unique)}
        tax_ids = [tax_label_to_id[lbl] for lbl in tax_labels_str]

        nmi_results: dict[str, float] = {}
        for res_name in ("coarse", "medium", "fine"):
            leiden_ids = [r[f"community_id_{res_name}"] for r in tax_rows]
            nmi_val = _nmi(leiden_ids, tax_ids)
            nmi_results[res_name] = round(nmi_val, 4)
            logger.info("NMI(%s vs arXiv): %.4f", res_name, nmi_val)

        results["nmi_vs_arxiv"] = {
            "n_papers_evaluated": len(tax_rows),
            "n_arxiv_classes": len(tax_unique),
            "arxiv_classes": tax_unique,
            "scores": nmi_results,
        }

    # --- Community purity ---
    logger.info("=== Phase C: Community purity ===")

    for res_name in ("coarse", "medium", "fine"):
        col = f"community_id_{res_name}"
        comm_to_tax: dict[int, list[str]] = {}
        for row in tax_rows:
            cid = row[col]
            if cid is not None and cid != -1:
                comm_to_tax.setdefault(cid, []).append(row["community_taxonomic"])

        purities = []
        for cid, labels in comm_to_tax.items():
            if len(labels) < 10:
                continue
            dominant_count = Counter(labels).most_common(1)[0][1]
            purities.append(dominant_count / len(labels))

        if purities:
            arr = np.array(purities)
            results[f"purity_{res_name}"] = {
                "n_communities_evaluated": len(purities),
                "mean_purity": round(float(arr.mean()), 4),
                "median_purity": round(float(np.median(arr)), 4),
                "std_purity": round(float(arr.std()), 4),
                "min_purity": round(float(arr.min()), 4),
                "max_purity": round(float(arr.max()), 4),
            }
            logger.info("%s purity: mean=%.3f, median=%.3f (%d communities)",
                        res_name, arr.mean(), np.median(arr), len(purities))

    results["phase_c_seconds"] = round(time.perf_counter() - t0, 1)
    logger.info("Phase C complete in %.1fs", results["phase_c_seconds"])


# ---------------------------------------------------------------------------
# Report writer — merges Phase A (from JSON) + B + C
# ---------------------------------------------------------------------------


def _write_report(results: dict[str, Any]) -> None:
    md_path = RESULTS_DIR / "graph_quality_report.md"
    lines: list[str] = []
    a = lines.append

    a("# Graph Quality Metrics — Section 5.4\n")
    a(f"*Generated: {results['timestamp']}*\n")

    # Full graph stats (from Phase A JSON if available)
    if "full_graph" in results:
        fg = results["full_graph"]
        a("## Full Citation Graph\n")
        a(f"- **Nodes (papers):** {fg['nodes']:,}")
        a(f"- **Edges (resolved citations):** {fg['edges']:,}")
        a(f"- **Total edges in DB:** {fg['edges_total']:,}")
        dangling = fg.get("edges_dangling", fg["edges_total"] - fg["edges"])
        a(f"- **Dangling edges:** {dangling:,} "
          f"({dangling / fg['edges_total'] * 100:.1f}%)")
        a(f"- **Edge resolution:** {fg.get('edge_resolution_pct', round(fg['edges'] / fg['edges_total'] * 100, 2))}%")
        a(f"- **Isolated nodes (degree=0):** {results.get('isolated_nodes', 'N/A'):,}")
        a("")

    if "components" in results:
        comp = results["components"]
        a("## Connected Components\n")
        a(f"- **Total components:** {comp['total_components']:,}")
        a(f"- **Giant component:** {comp['giant_component_nodes']:,} nodes "
          f"({comp['giant_component_pct_of_total']}% of total, "
          f"{comp['giant_component_pct_of_connected']}% of connected)")
        if "giant_component_edges" in comp:
            a(f"- **Giant component edges:** {comp['giant_component_edges']:,}")
        elif "giant_component_edges" in results:
            a(f"- **Giant component edges:** {results['giant_component_edges']:,}")
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
    has_leiden = any(f"leiden_{rn}" in results for rn in ("coarse", "medium", "fine"))
    if has_leiden:
        a("| Resolution | Value | Communities | Singletons | Max Size | Mean Size | Top-10 % | Time |")
        a("|-----------|-------|------------|-----------|---------|----------|---------|------|")
        for rn in ("coarse", "medium", "fine"):
            key = f"leiden_{rn}"
            if key not in results:
                continue
            s = results[key]["stats"]
            elapsed = results[key].get("elapsed_seconds", "—")
            a(f"| {rn} | {results[key]['resolution']} | {s['n_communities']:,} | "
              f"{s['singletons']:,} | {s['max_size']:,} | {s['mean_size']} | "
              f"{s['pct_in_top10']}% | {elapsed}s |")
    else:
        a("*Not yet computed.*\n")
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

    has_purity = any(f"purity_{rn}" in results for rn in ("coarse", "medium", "fine"))
    if has_purity:
        a("## Community Purity (arXiv-class dominant fraction, communities ≥ 10 papers)\n")
        a("| Resolution | Mean | Median | Std | Min | Max | Communities |")
        a("|-----------|------|--------|-----|-----|-----|------------|")
        for rn in ("coarse", "medium", "fine"):
            key = f"purity_{rn}"
            if key not in results:
                continue
            p = results[key]
            a(f"| {rn} | {p['mean_purity']} | {p['median_purity']} | "
              f"{p['std_purity']} | {p['min_purity']} | {p['max_purity']} | "
              f"{p['n_communities_evaluated']} |")
        a("")

    a(f"\n*Total elapsed: {results.get('elapsed_seconds', 'N/A')}s*\n")

    md_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_total = time.perf_counter()

    # Load Phase A results if available
    phase_a_path = RESULTS_DIR / "graph_quality_metrics.json"
    if phase_a_path.exists():
        with open(phase_a_path) as f:
            results = json.load(f)
        logger.info("Loaded Phase A results from %s", phase_a_path)
    else:
        results = {}
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SET application_name = 'graph_leiden_measurement'")
        cur.execute("SET work_mem = '512MB'")
    conn.commit()

    # Verify Phase A state in DB
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(CASE WHEN community_id_coarse = -1 THEN 1 END) AS non_giant,
                COUNT(CASE WHEN community_id_coarse IS NULL THEN 1 END) AS giant_candidates
            FROM paper_metrics
        """)
        row = cur.fetchone()
        total, non_giant, giant_candidates = row

    logger.info("DB state: %d total, %d non-giant (-1), %d giant candidates (NULL)",
                total, non_giant, giant_candidates)

    if giant_candidates == 0:
        logger.error("No giant-component candidates found (community_id_coarse IS NULL). "
                      "Run measure_components_only.py or measure_graph_quality.py Phase A first.")
        sys.exit(1)

    # Phase B: Leiden
    logger.info("========== Phase B: Leiden community detection ==========")
    phase_b_leiden(conn, results)
    gc.collect()

    # Phase C: NMI + purity
    logger.info("========== Phase C: Quality metrics ==========")
    phase_c_quality(conn, results)

    # Save merged results
    elapsed = round(time.perf_counter() - t_total, 1)
    results["elapsed_seconds"] = elapsed

    json_path = RESULTS_DIR / "graph_quality_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics to %s", json_path)

    _write_report(results)

    conn.close()
    logger.info("Done in %.0fs", elapsed)


if __name__ == "__main__":
    main()
