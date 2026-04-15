#!/usr/bin/env python3
"""Measure giant component metrics and community quality for paper Section 5.4.

Two-phase approach to fit within available memory (~17GB):
  Phase A: scipy.sparse for connected components (~4GB peak)
  Phase B: igraph for Leiden community detection (~11GB peak)

Key optimization: bibcode→int mapping pushed to PostgreSQL temp tables,
eliminating ~9GB of Python dict overhead.

Outputs:
  results/graph_quality_metrics.json  — structured results
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
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.graph_metrics import community_size_stats, compute_nmi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("measure_graph_quality")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# NMI and community_size_stats are now in src/scix/graph_metrics.py
_nmi = compute_nmi
_community_size_stats = community_size_stats


def _percentile_distribution(membership: list[int]) -> list[dict[str, Any]]:
    counts = Counter(membership)
    total = len(membership)
    return [
        {"community_id": cid, "size": cnt, "pct": round(cnt / total * 100, 2)}
        for cid, cnt in counts.most_common(20)
    ]


# ---------------------------------------------------------------------------
# Phase A: Connected components via scipy (low memory)
# ---------------------------------------------------------------------------


def phase_a_components(conn: psycopg.Connection) -> dict[str, Any]:
    """Compute connected components using scipy sparse matrix.

    Memory profile: ~4GB peak (vs ~20GB with igraph + Python dicts).
    Uses PostgreSQL temp table for bibcode→int mapping.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    t0 = time.perf_counter()
    results: dict[str, Any] = {}

    # --- Create temp table with integer node IDs ---
    logger.info("Creating temp table for node ID mapping...")
    t_map = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE node_ids (
                bibcode TEXT PRIMARY KEY,
                nid INT NOT NULL
            ) ON COMMIT PRESERVE ROWS
        """)
        # Populate via COPY from a subquery for speed
        cur.execute("""
            INSERT INTO node_ids (bibcode, nid)
            SELECT bibcode, ROW_NUMBER() OVER (ORDER BY bibcode)::INT - 1
            FROM papers
        """)
        cur.execute("CREATE INDEX ON node_ids (nid)")
        cur.execute("ANALYZE node_ids")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM node_ids")
        (n_nodes,) = cur.fetchone()
    logger.info("Node ID mapping: %d nodes in %.1fs", n_nodes, time.perf_counter() - t_map)

    # --- Stream edges as integer pairs via SQL JOIN ---
    logger.info("Streaming edges as integer pairs (SQL JOIN)...")
    t_edge = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM citation_edges")
        (edge_count_total,) = cur.fetchone()
    logger.info("Total edges in DB: %d", edge_count_total)

    # Pre-allocate at estimated size
    src_arr = np.empty(edge_count_total, dtype=np.int32)
    tgt_arr = np.empty(edge_count_total, dtype=np.int32)
    valid = 0

    with conn.cursor(name="edge_int_cursor") as cur:
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
        valid,
        skipped,
        time.perf_counter() - t_edge,
    )

    results["full_graph"] = {"nodes": n_nodes, "edges": valid, "edges_total": edge_count_total}

    # --- Build scipy sparse matrix ---
    logger.info("Building scipy CSR matrix...")
    t_csr = time.perf_counter()
    data = np.ones(valid, dtype=np.int8)
    graph_csr = csr_matrix((data, (src_arr, tgt_arr)), shape=(n_nodes, n_nodes))
    del data
    # Make symmetric for undirected component detection
    graph_sym = graph_csr + graph_csr.T
    del graph_csr, src_arr, tgt_arr
    gc.collect()
    logger.info("CSR matrix built in %.1fs", time.perf_counter() - t_csr)

    # --- Connected components ---
    logger.info("Computing connected components (scipy)...")
    t_comp = time.perf_counter()
    n_components, labels = connected_components(graph_sym, directed=False, return_labels=True)
    del graph_sym
    gc.collect()
    logger.info("Found %d components in %.1fs", n_components, time.perf_counter() - t_comp)

    # --- Analyze components ---
    label_counts = Counter(labels.tolist())
    comp_sizes = sorted(label_counts.values(), reverse=True)
    giant_label = max(label_counts, key=label_counts.get)
    n_giant = label_counts[giant_label]

    # Isolated nodes: degree 0 means they appear in no edges
    # In scipy, we already filtered edges via JOIN, so unresolved edges are gone.
    # Isolated = nodes that appear in no resolved edge.
    # Simplest: use label_counts — components of size 1 are isolated.
    isolated_count = sum(1 for s in label_counts.values() if s == 1)

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
    }
    logger.info(
        "Giant component: %d nodes (%.1f%% of total)",
        n_giant,
        n_giant / n_nodes * 100,
    )

    # --- Store giant component membership in paper_metrics ---
    logger.info("Storing giant component membership in paper_metrics...")
    t_store = time.perf_counter()

    giant_mask = labels == giant_label
    giant_nids = np.where(giant_mask)[0]

    # Create temp table for giant component nids
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE giant_nids (
                nid INT PRIMARY KEY
            ) ON COMMIT PRESERVE ROWS
        """)

    # Bulk insert giant nids via COPY
    chunk_size = 500_000
    for i in range(0, len(giant_nids), chunk_size):
        chunk = giant_nids[i : i + chunk_size]
        with conn.cursor() as cur:
            with cur.copy("COPY giant_nids (nid) FROM STDIN") as copy:
                for nid in chunk:
                    copy.write_row((int(nid),))
        conn.commit()

    # Mark non-giant-component papers with community_id = -1 (LEFT JOIN, no NOT IN)
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE paper_metrics pm
            SET community_id_coarse = -1,
                community_id_medium = -1,
                community_id_fine = -1
            FROM node_ids ni
            LEFT JOIN giant_nids gn ON ni.nid = gn.nid
            WHERE pm.bibcode = ni.bibcode
              AND gn.nid IS NULL
              AND pm.community_id_coarse IS DISTINCT FROM -1
        """)
        non_giant_updated = cur.rowcount
    conn.commit()
    logger.info(
        "Marked %d non-giant-component papers, stored giant membership in %.1fs",
        non_giant_updated,
        time.perf_counter() - t_store,
    )

    # --- Create giant component node mapping for Phase B ---
    logger.info("Creating giant component node mapping table...")
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE giant_node_map (
                bibcode TEXT PRIMARY KEY,
                new_nid INT NOT NULL
            ) ON COMMIT PRESERVE ROWS
        """)
        cur.execute("""
            INSERT INTO giant_node_map (bibcode, new_nid)
            SELECT ni.bibcode, ROW_NUMBER() OVER (ORDER BY ni.nid)::INT - 1
            FROM node_ids ni
            JOIN giant_nids gn ON ni.nid = gn.nid
        """)
        cur.execute("CREATE INDEX ON giant_node_map (new_nid)")
        cur.execute("ANALYZE giant_node_map")
    conn.commit()
    logger.info("Giant node mapping created for %d nodes", n_giant)

    del labels, giant_mask, giant_nids
    gc.collect()

    results["phase_a_seconds"] = round(time.perf_counter() - t0, 1)
    logger.info("Phase A complete in %.1fs", results["phase_a_seconds"])
    return results


# ---------------------------------------------------------------------------
# Phase B: Leiden community detection via igraph (giant component only)
# ---------------------------------------------------------------------------


def phase_b_leiden(
    conn: psycopg.Connection,
    n_giant: int,
    results: dict[str, Any],
) -> None:
    """Run Leiden at 3 resolutions on giant component.

    Memory profile: ~11GB peak (undirected igraph for ~30M nodes).
    Uses SQL temp table for node mapping — no Python dicts.
    """
    import igraph
    import leidenalg

    t0 = time.perf_counter()

    # --- Load giant component edges as integer pairs ---
    logger.info("Streaming giant component edges (SQL JOIN)...")
    t_edge = time.perf_counter()

    # Use total resolved edges as upper bound (giant component is ~99% of resolved)
    est_edges = results["full_graph"]["edges"]
    logger.info("Allocating for up to %d edges (upper bound)", est_edges)

    src_arr = np.empty(est_edges, dtype=np.int32)
    tgt_arr = np.empty(est_edges, dtype=np.int32)
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
    results["components"]["giant_component_edges"] = valid

    # --- Build undirected igraph directly ---
    logger.info("Building undirected igraph for giant component...")
    t_graph = time.perf_counter()
    edges = np.column_stack([src_arr, tgt_arr])
    del src_arr, tgt_arr
    gc.collect()

    # Build directed first, then convert (igraph deduplicates in as_undirected)
    graph_dir = igraph.Graph(n=n_giant, edges=edges, directed=True)
    del edges
    gc.collect()

    graph = graph_dir.as_undirected(mode="collapse")
    del graph_dir
    gc.collect()
    logger.info(
        "Undirected igraph: %d nodes, %d edges in %.1fs",
        graph.vcount(),
        graph.ecount(),
        time.perf_counter() - t_graph,
    )

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
        logger.info(
            "Leiden %s done in %.1fs",
            res_name,
            time.perf_counter() - t_leiden,
        )

        stats = _community_size_stats(membership)
        dist = _percentile_distribution(membership)
        results[f"leiden_{res_name}"] = {
            "resolution": res_val,
            "stats": stats,
            "top_20_communities": dist,
        }
        logger.info("%s: %d communities", res_name, stats["n_communities"])

        # Store community assignments to paper_metrics via chunked UPDATE
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
    """Store community assignments via SQL, using giant_node_map for bibcode lookup."""
    col = f"community_id_{res_name}"
    chunk_size = 100_000
    stored = 0
    n = len(membership)

    # Create temp table for bulk assignment
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
        if stored % 500_000 == 0 or stored == n:
            logger.info("  Stored %d/%d %s assignments", stored, n, res_name)

    logger.info("Stored %d %s community assignments", stored, res_name)


# ---------------------------------------------------------------------------
# Phase C: NMI and purity (SQL-driven, minimal memory)
# ---------------------------------------------------------------------------


def phase_c_quality(conn: psycopg.Connection, results: dict[str, Any]) -> None:
    """Compute NMI and purity metrics entirely from paper_metrics.

    Memory profile: <2GB (only loads label arrays for papers with both labels).
    """
    t0 = time.perf_counter()

    # --- NMI: Leiden vs arXiv taxonomy ---
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
        # Encode taxonomic labels as integers
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
            logger.info(
                "%s purity: mean=%.3f, median=%.3f",
                res_name,
                arr.mean(),
                np.median(arr),
            )

    results["phase_c_seconds"] = round(time.perf_counter() - t0, 1)
    logger.info("Phase C complete in %.1fs", results["phase_c_seconds"])


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


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
    a(
        f"- **Dangling edges (unresolved):** {fg['edges_total'] - fg['edges']:,} "
        f"({(fg['edges_total'] - fg['edges']) / fg['edges_total'] * 100:.1f}%)"
    )
    a(f"- **Isolated nodes (degree=0):** {results['isolated_nodes']:,}")
    a("")

    comp = results["components"]
    a("## Connected Components\n")
    a(f"- **Total components:** {comp['total_components']:,}")
    a(
        f"- **Giant component:** {comp['giant_component_nodes']:,} nodes "
        f"({comp['giant_component_pct_of_total']}% of total, "
        f"{comp['giant_component_pct_of_connected']}% of connected)"
    )
    if "giant_component_edges" in comp:
        a(f"- **Giant component edges:** {comp['giant_component_edges']:,}")
    a(f"- **Components > 100 nodes:** {comp['components_gt_100']:,}")
    a(f"- **Components > 1,000 nodes:** {comp['components_gt_1000']:,}")
    a(f"- **Small-component papers:** {comp['small_component_papers']:,}")
    a(f"- **Top-10 component sizes:** {comp['top_10_sizes']}")
    a("")

    a("## Leiden Community Detection\n")
    a("| Resolution | Value | Communities | Singletons | Max Size | " "Mean Size | Top-10 % |")
    a("|-----------|-------|------------|-----------|---------|----------|---------|")
    for rn in ("coarse", "medium", "fine"):
        key = f"leiden_{rn}"
        if key not in results:
            continue
        s = results[key]["stats"]
        a(
            f"| {rn} | {results[key]['resolution']} | {s['n_communities']:,} | "
            f"{s['singletons']:,} | {s['max_size']:,} | {s['mean_size']} | "
            f"{s['pct_in_top10']}% |"
        )
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

    a("## Community Purity (arXiv-class dominant fraction)\n")
    a("| Resolution | Mean | Median | Std | Min | Max | Communities |")
    a("|-----------|------|--------|-----|-----|-----|------------|")
    for rn in ("coarse", "medium", "fine"):
        key = f"purity_{rn}"
        if key not in results:
            continue
        p = results[key]
        a(
            f"| {rn} | {p['mean_purity']} | {p['median_purity']} | "
            f"{p['std_purity']} | {p['min_purity']} | {p['max_purity']} | "
            f"{p['n_communities_evaluated']} |"
        )
    a("")
    a(f"\n*Total elapsed: {results.get('elapsed_seconds', 'N/A')}s*\n")

    md_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_total = time.perf_counter()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SET application_name = 'graph_quality_measurement'")
        cur.execute("SET work_mem = '512MB'")
    conn.commit()

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Phase A: Connected components via scipy
    logger.info("========== Phase A: Connected components ==========")
    phase_a_results = phase_a_components(conn)
    results.update(phase_a_results)
    gc.collect()

    n_giant = results["components"]["giant_component_nodes"]

    # Phase B: Leiden community detection via igraph
    logger.info("========== Phase B: Leiden community detection ==========")
    phase_b_leiden(conn, n_giant, results)
    gc.collect()

    # Phase C: NMI and purity
    logger.info("========== Phase C: Quality metrics ==========")
    phase_c_quality(conn, results)

    # Save results
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
