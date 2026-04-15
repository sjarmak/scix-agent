"""Precompute graph metrics (PageRank, HITS, Leiden communities) on the citation graph.

Also provides community quality metrics (NMI, conductance, coverage) for
evaluating partition quality against external labels or structural criteria.
"""

from __future__ import annotations

import io
import logging
import math
import time
from collections import Counter
from typing import Any

import psycopg
from psycopg.rows import dict_row

from scix.db import get_connection

logger = logging.getLogger(__name__)

_VALID_RESOLUTIONS = frozenset({"coarse", "medium", "fine"})

_ASTRO_PH_PREFIX = "astro-ph"

_CHUNK_SIZE = 100_000


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------


def _elapsed_ms(t0: float) -> float:
    """Milliseconds elapsed since t0 (from time.perf_counter)."""
    return round((time.perf_counter() - t0) * 1000, 2)


# ---------------------------------------------------------------------------
# Community quality metrics
# ---------------------------------------------------------------------------


def compute_nmi(labels_a: list[int], labels_b: list[int]) -> float:
    """Normalized Mutual Information with arithmetic mean normalization.

    Computes 2*MI(A,B) / (H(A) + H(B)) where MI is mutual information and
    H is Shannon entropy. Returns a value in [0, 1] where 1 means perfect
    agreement and 0 means independence.

    Args:
        labels_a: Integer partition labels for each element.
        labels_b: Integer partition labels for each element (same length).

    Returns:
        NMI score in [0, 1].
    """
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


def community_size_stats(membership: list[int]) -> dict[str, Any]:
    """Compute descriptive statistics for community sizes.

    Args:
        membership: Community ID for each node (list indexed by vertex ID).

    Returns:
        Dictionary with keys: n_communities, min_size, max_size, mean_size,
        median_size, std_size, singletons, pct_in_top10.
    """
    if not membership:
        return {
            "n_communities": 0,
            "min_size": 0,
            "max_size": 0,
            "mean_size": 0.0,
            "median_size": 0.0,
            "std_size": 0.0,
            "singletons": 0,
            "pct_in_top10": 0.0,
        }

    import numpy as np

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


def compute_conductance(
    graph: Any,
    membership: list[int],
) -> dict[str, float]:
    """Compute conductance for each community and return summary statistics.

    Conductance of community S = |edges leaving S| / min(vol(S), vol(V-S))
    where vol(S) is the sum of degrees of nodes in S.

    Lower conductance means a better-separated community.

    Args:
        graph: Undirected igraph.Graph.
        membership: Community ID for each vertex.

    Returns:
        Dictionary with mean, median, max conductance and n_communities evaluated.
    """
    import numpy as np

    comm_to_nodes: dict[int, list[int]] = {}
    for vid, cid in enumerate(membership):
        comm_to_nodes.setdefault(cid, []).append(vid)

    degrees = graph.degree()
    total_vol = sum(degrees)

    # Single O(E) pass: count cut edges per community
    cut_edges_per_comm: dict[int, int] = Counter()
    for edge in graph.es:
        cs, ct = membership[edge.source], membership[edge.target]
        if cs != ct:
            cut_edges_per_comm[cs] += 1
            cut_edges_per_comm[ct] += 1

    conductances: list[float] = []
    for cid, nodes in comm_to_nodes.items():
        vol_s = sum(degrees[v] for v in nodes)
        vol_complement = total_vol - vol_s

        if vol_s == 0 or vol_complement == 0:
            conductances.append(0.0)
            continue

        denominator = min(vol_s, vol_complement)
        conductances.append(cut_edges_per_comm.get(cid, 0) / denominator)

    if not conductances:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "n_communities": 0}

    arr = np.array(conductances)
    return {
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "max": round(float(arr.max()), 6),
        "n_communities": len(conductances),
    }


def compute_coverage(graph: Any, membership: list[int]) -> float:
    """Compute coverage: fraction of edges that are intra-community.

    Coverage = (intra-community edges) / (total edges).
    A value of 1.0 means all edges are within communities (perfect partition).
    A value of 0.0 means all edges cross community boundaries.

    Args:
        graph: Undirected igraph.Graph.
        membership: Community ID for each vertex.

    Returns:
        Coverage fraction in [0, 1].
    """
    total_edges = graph.ecount()
    if total_edges == 0:
        return 0.0

    intra = 0
    for edge in graph.es:
        if membership[edge.source] == membership[edge.target]:
            intra += 1

    return intra / total_edges


# ---------------------------------------------------------------------------
# Import guards for optional dependencies
# ---------------------------------------------------------------------------


def _import_igraph() -> Any:
    """Lazy-import igraph with a helpful error message."""
    try:
        import igraph

        return igraph
    except ImportError:
        raise ImportError(
            "python-igraph is required for graph metrics. "
            "Install with: pip install python-igraph"
        )


def _import_leidenalg() -> Any:
    """Lazy-import leidenalg with a helpful error message."""
    try:
        import leidenalg

        return leidenalg
    except ImportError:
        raise ImportError(
            "leidenalg is required for community detection. " "Install with: pip install leidenalg"
        )


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------


def load_graph(
    conn: psycopg.Connection,
) -> tuple[Any, dict[str, int], dict[int, str]]:
    """Load citation graph from PostgreSQL into an igraph.Graph.

    Streams nodes and edges via server-side cursors to handle millions of rows
    without exhausting memory.

    Returns:
        (graph, bibcode_to_id, id_to_bibcode) where graph is an igraph.Graph
        and the dicts map between bibcodes and integer vertex IDs.
    """
    igraph = _import_igraph()

    t0 = time.perf_counter()

    # --- Stream nodes ---
    bibcode_to_id: dict[str, int] = {}
    id_to_bibcode: dict[int, str] = {}
    node_idx = 0

    with conn.cursor(name="node_cursor") as cur:
        cur.itersize = 500_000
        cur.execute("SELECT bibcode FROM papers")
        for (bibcode,) in cur:
            bibcode_to_id[bibcode] = node_idx
            id_to_bibcode[node_idx] = bibcode
            node_idx += 1

    node_count = len(bibcode_to_id)
    logger.info("Loaded %d nodes in %.1fs", node_count, (time.perf_counter() - t0))

    # --- Stream edges ---
    t_edges = time.perf_counter()
    edge_list: list[tuple[int, int]] = []
    skipped = 0

    with conn.cursor(name="edge_cursor") as cur:
        cur.itersize = 500_000
        cur.execute("SELECT source_bibcode, target_bibcode FROM citation_edges")
        for source, target in cur:
            src_id = bibcode_to_id.get(source)
            tgt_id = bibcode_to_id.get(target)
            if src_id is None or tgt_id is None:
                skipped += 1
                continue
            edge_list.append((src_id, tgt_id))

    edge_count = len(edge_list)
    logger.info(
        "Loaded %d edges (skipped %d) in %.1fs",
        edge_count,
        skipped,
        (time.perf_counter() - t_edges),
    )

    # --- Build graph ---
    t_build = time.perf_counter()
    graph = igraph.Graph(n=node_count, edges=edge_list, directed=True)
    logger.info("Built igraph in %.1fs", (time.perf_counter() - t_build))

    total_ms = _elapsed_ms(t0)
    logger.info(
        "Graph loaded: %d nodes, %d edges, %d skipped (%.1fms total)",
        node_count,
        edge_count,
        skipped,
        total_ms,
    )

    return graph, bibcode_to_id, id_to_bibcode


# ---------------------------------------------------------------------------
# Isolated node filtering
# ---------------------------------------------------------------------------


def filter_isolated_nodes(
    graph: Any,
    bibcode_to_id: dict[str, int],
    id_to_bibcode: dict[int, str],
) -> tuple[Any, dict[str, int], dict[int, str], set[str]]:
    """Extract subgraph of nodes with degree >= 1 (at least one edge).

    Isolated nodes (degree 0) are meaningless for community detection —
    Leiden always assigns each to its own singleton community.

    Args:
        graph: Full igraph.Graph (directed).
        bibcode_to_id: Full bibcode→vertex-id mapping.
        id_to_bibcode: Full vertex-id→bibcode mapping.

    Returns:
        (subgraph, sub_bibcode_to_id, sub_id_to_bibcode, isolated_bibcodes)
        where subgraph contains only connected vertices with new contiguous IDs.
    """
    t0 = time.perf_counter()

    degrees = graph.degree(mode="all")
    connected_vids: list[int] = []
    isolated_bibcodes: set[str] = set()
    for vid, deg in enumerate(degrees):
        if deg > 0:
            connected_vids.append(vid)
        else:
            isolated_bibcodes.add(id_to_bibcode[vid])

    subgraph = graph.induced_subgraph(connected_vids)

    # Build new contiguous ID mappings for the subgraph
    sub_bibcode_to_id: dict[str, int] = {}
    sub_id_to_bibcode: dict[int, str] = {}
    for new_id, old_id in enumerate(connected_vids):
        bib = id_to_bibcode[old_id]
        sub_bibcode_to_id[bib] = new_id
        sub_id_to_bibcode[new_id] = bib

    logger.info(
        "Filtered graph: %d connected nodes, %d isolated (%.1fms)",
        len(connected_vids),
        len(isolated_bibcodes),
        _elapsed_ms(t0),
    )
    return subgraph, sub_bibcode_to_id, sub_id_to_bibcode, isolated_bibcodes


# ---------------------------------------------------------------------------
# Giant component extraction
# ---------------------------------------------------------------------------


def extract_giant_component(
    graph: Any,
    bibcode_to_id: dict[str, int],
    id_to_bibcode: dict[int, str],
) -> tuple[Any, dict[str, int], dict[int, str], set[str]]:
    """Extract the largest connected component (giant component) from a graph.

    After filtering isolated nodes, the remaining graph may still contain
    multiple disconnected components.  Leiden should run only on the giant
    component; papers in smaller components are assigned to communities
    separately via embedding distance.

    Uses weak (undirected) connectivity — appropriate for citation graphs
    where an edge in either direction implies relatedness.

    Args:
        graph: igraph.Graph (directed), typically the output of filter_isolated_nodes().
        bibcode_to_id: bibcode→vertex-id mapping for *graph*.
        id_to_bibcode: vertex-id→bibcode mapping for *graph*.

    Returns:
        (giant_graph, giant_b2i, giant_i2b, small_bibcodes)
        where giant_graph is the induced subgraph of the largest component,
        giant_b2i / giant_i2b are the new contiguous ID mappings, and
        small_bibcodes contains bibcodes from all other (smaller) components.
    """
    t0 = time.perf_counter()

    components = graph.connected_components(mode="weak")

    if len(components) <= 1:
        # Entire graph is one component (or empty) — nothing to extract
        logger.info(
            "Graph has %d component(s); skipping giant-component extraction (%.1fms)",
            len(components),
            _elapsed_ms(t0),
        )
        return graph, bibcode_to_id, id_to_bibcode, set()

    # Find the largest component by vertex count
    giant_idx = max(range(len(components)), key=lambda i: len(components[i]))
    giant_vids: list[int] = components[giant_idx]
    giant_vid_set = set(giant_vids)

    # Collect bibcodes in small components
    small_bibcodes: set[str] = set()
    for vid in range(graph.vcount()):
        if vid not in giant_vid_set:
            small_bibcodes.add(id_to_bibcode[vid])

    # Extract giant component subgraph
    giant_graph = graph.induced_subgraph(giant_vids)

    # Build new contiguous ID mappings
    giant_b2i: dict[str, int] = {}
    giant_i2b: dict[int, str] = {}
    for new_id, old_id in enumerate(giant_vids):
        bib = id_to_bibcode[old_id]
        giant_b2i[bib] = new_id
        giant_i2b[new_id] = bib

    logger.info(
        "Giant component: %d nodes (%d small-component nodes in %d other components) (%.1fms)",
        len(giant_vids),
        len(small_bibcodes),
        len(components) - 1,
        _elapsed_ms(t0),
    )
    return giant_graph, giant_b2i, giant_i2b, small_bibcodes


# ---------------------------------------------------------------------------
# Small-component community assignment by embedding distance
# ---------------------------------------------------------------------------


def assign_small_component_communities(
    small_bibcodes: set[str],
    giant_membership: dict[str, int],
    giant_embeddings: dict[str, Any],
    small_embeddings: dict[str, Any],
) -> dict[str, int | None]:
    """Assign small-component papers to the nearest giant-component community.

    Computes community centroids from giant-component embeddings, then assigns
    each small-component paper to the community with the highest cosine
    similarity to its embedding.

    Args:
        small_bibcodes: Bibcodes of papers in small components.
        giant_membership: {bibcode: community_id} for giant-component papers.
        giant_embeddings: {bibcode: np.ndarray} for giant-component papers.
        small_embeddings: {bibcode: np.ndarray} for small-component papers.

    Returns:
        {bibcode: community_id} for each small-component paper.
        Papers without embeddings get None.
    """
    import numpy as np

    if not small_bibcodes:
        return {}

    t0 = time.perf_counter()

    # Build community centroids from giant-component embeddings
    community_sums: dict[int, Any] = {}
    community_counts: dict[int, int] = {}
    for bib, cid in giant_membership.items():
        vec = giant_embeddings.get(bib)
        if vec is None:
            continue
        if cid not in community_sums:
            community_sums[cid] = np.zeros_like(vec, dtype=np.float64)
            community_counts[cid] = 0
        community_sums[cid] += vec
        community_counts[cid] += 1

    # Normalize centroids
    centroids: dict[int, Any] = {}
    for cid, s in community_sums.items():
        centroid = s / community_counts[cid]
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cid] = centroid

    if not centroids:
        logger.warning("No community centroids available — cannot assign small components")
        return {bib: None for bib in small_bibcodes}

    # Assign each small-component paper to nearest centroid
    result: dict[str, int | None] = {}
    assigned = 0
    for bib in small_bibcodes:
        vec = small_embeddings.get(bib)
        if vec is None:
            result[bib] = None
            continue

        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            vec_normalized = vec / vec_norm
        else:
            result[bib] = None
            continue

        best_cid = None
        best_sim = -2.0
        for cid, centroid in centroids.items():
            sim = float(np.dot(vec_normalized, centroid))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        result[bib] = best_cid
        assigned += 1

    logger.info(
        "Assigned %d/%d small-component papers to communities (%.1fms)",
        assigned,
        len(small_bibcodes),
        _elapsed_ms(t0),
    )
    return result


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_pagerank(graph: Any) -> list[float]:
    """Compute PageRank scores for all vertices.

    Returns a list of floats indexed by vertex ID, summing to ~1.0.
    """
    t0 = time.perf_counter()
    scores: list[float] = graph.pagerank(directed=True)
    logger.info("PageRank computed in %.1fms", _elapsed_ms(t0))
    return scores


def compute_hits(graph: Any) -> tuple[list[float], list[float]]:
    """Compute HITS hub and authority scores for all vertices.

    Returns (hub_scores, authority_scores), each a list of floats indexed
    by vertex ID with non-negative values.
    """
    t0 = time.perf_counter()
    hub_scores: list[float] = graph.hub_score()
    authority_scores: list[float] = graph.authority_score()
    logger.info("HITS computed in %.1fms", _elapsed_ms(t0))
    return hub_scores, authority_scores


_PARTITION_TYPES = frozenset({"modularity", "CPM"})


def _resolve_partition_class(partition_type: str) -> Any:
    """Map a partition type name to a leidenalg partition class.

    Args:
        partition_type: One of 'modularity' (RBConfigurationVertexPartition)
            or 'CPM' (CPMVertexPartition). CPM is recommended by CWTS for
            resolution-limit-free community detection.

    Returns:
        The leidenalg partition class.

    Raises:
        ValueError: If partition_type is not recognized.
    """
    if partition_type not in _PARTITION_TYPES:
        raise ValueError(
            f"Invalid partition_type: {partition_type!r}. "
            f"Must be one of {sorted(_PARTITION_TYPES)}"
        )
    leidenalg = _import_leidenalg()
    if partition_type == "CPM":
        return leidenalg.CPMVertexPartition
    return leidenalg.RBConfigurationVertexPartition


def compute_leiden(
    graph: Any,
    resolution: float,
    seed: int = 42,
    partition_type: str = "modularity",
) -> list[int]:
    """Run Leiden community detection on an undirected projection of the graph.

    Args:
        graph: igraph.Graph (directed or undirected; directed graphs are
            converted to undirected via edge collapse).
        resolution: Resolution parameter. For 'modularity'
            (RBConfigurationVertexPartition), higher values yield more, smaller
            communities. For 'CPM' (CPMVertexPartition), the resolution is an
            absolute edge-density threshold — communities must have internal
            density above this value.
        seed: Random seed for reproducibility.
        partition_type: Partition quality function. 'modularity' uses
            RBConfigurationVertexPartition; 'CPM' uses CPMVertexPartition
            (recommended by CWTS for resolution-limit-free detection).

    Returns:
        Membership list (list[int]) indexed by vertex ID.
    """
    partition_class = _resolve_partition_class(partition_type)

    t0 = time.perf_counter()
    if graph.is_directed():
        undirected = graph.as_undirected(mode="collapse")
    else:
        undirected = graph
    leidenalg = _import_leidenalg()
    partition = leidenalg.find_partition(
        undirected,
        partition_class,
        resolution_parameter=resolution,
        seed=seed,
    )
    membership: list[int] = list(partition.membership)
    n_communities = len(set(membership))
    logger.info(
        "Leiden (%s, res=%.4f) found %d communities in %.1fms",
        partition_type,
        resolution,
        n_communities,
        _elapsed_ms(t0),
    )
    return membership


# ---------------------------------------------------------------------------
# Resolution calibration
# ---------------------------------------------------------------------------


def calibrate_resolution(
    graph: Any,
    target_communities: int,
    tolerance: float = 0.3,
    low: float = 1e-7,
    high: float = 1.0,
    max_iter: int = 8,
    seed: int = 42,
    partition_type: str = "modularity",
) -> float:
    """Binary search (log-scale) for a Leiden resolution yielding ~target_communities.

    Searches the [low, high] interval on a logarithmic scale for a resolution
    parameter that produces a community count within (1 +/- tolerance) * target.

    Args:
        graph: igraph.Graph (directed; will be converted internally).
        target_communities: Desired number of communities.
        tolerance: Fractional tolerance (e.g. 0.3 = within 30% of target).
        low: Lower bound of resolution search space.
        high: Upper bound of resolution search space.
        max_iter: Maximum binary search iterations.
        seed: Random seed for Leiden.
        partition_type: 'modularity' or 'CPM'.

    Returns:
        Best resolution found (float).
    """
    import math

    log_low = math.log10(low)
    log_high = math.log10(high)
    best_res = 10 ** ((log_low + log_high) / 2)
    best_diff = float("inf")

    for i in range(max_iter):
        log_mid = (log_low + log_high) / 2
        mid = 10**log_mid
        membership = compute_leiden(graph, resolution=mid, seed=seed, partition_type=partition_type)
        n_communities = len(set(membership))
        diff = abs(n_communities - target_communities)

        logger.info(
            "Calibration iter %d/%d: res=%.6f -> %d communities " "(target=%d, diff=%d)",
            i + 1,
            max_iter,
            mid,
            n_communities,
            target_communities,
            diff,
        )

        if diff < best_diff:
            best_diff = diff
            best_res = mid

        # Check if within tolerance
        lower_bound = target_communities * (1 - tolerance)
        upper_bound = target_communities * (1 + tolerance)
        if lower_bound <= n_communities <= upper_bound:
            logger.info(
                "Calibration converged at res=%.6f (%d communities)",
                mid,
                n_communities,
            )
            return mid

        # Binary search direction (log-scale): more communities -> lower resolution
        if n_communities > target_communities:
            log_high = log_mid
        else:
            log_low = log_mid

    logger.warning(
        "Calibration did not converge after %d iterations. " "Best resolution=%.6f (diff=%d)",
        max_iter,
        best_res,
        best_diff,
    )
    return best_res


# ---------------------------------------------------------------------------
# Resolution sweep and partition comparison
# ---------------------------------------------------------------------------


def sweep_resolutions(
    graph: Any,
    resolutions: list[float],
    seed: int = 42,
    partition_type: str = "modularity",
    reference_labels: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run Leiden at multiple resolutions and collect quality metrics at each.

    For each resolution, computes: community count, size statistics,
    conductance, coverage, and optionally NMI against reference labels.

    The graph is converted to undirected once and reused across all runs.

    Args:
        graph: igraph.Graph (directed or undirected).
        resolutions: List of resolution parameter values to sweep.
        seed: Random seed for Leiden.
        partition_type: 'modularity' (RBConfigurationVertexPartition) or
            'CPM' (CPMVertexPartition).
        reference_labels: Optional ground-truth labels (one per vertex).
            When provided, NMI is computed for each partition.

    Returns:
        List of dicts, one per resolution, each containing:
            resolution, n_communities, membership, size_stats,
            conductance, coverage, and optionally nmi.
    """
    if partition_type not in _PARTITION_TYPES:
        raise ValueError(
            f"Invalid partition_type: {partition_type!r}. "
            f"Must be one of {sorted(_PARTITION_TYPES)}"
        )

    if not resolutions:
        return []

    t0 = time.perf_counter()

    # Convert to undirected once for all runs
    if graph.is_directed():
        undirected = graph.as_undirected(mode="collapse")
    else:
        undirected = graph

    partition_class = _resolve_partition_class(partition_type)
    leidenalg = _import_leidenalg()

    results: list[dict[str, Any]] = []
    for res_val in resolutions:
        t_run = time.perf_counter()
        partition = leidenalg.find_partition(
            undirected,
            partition_class,
            resolution_parameter=res_val,
            seed=seed,
        )
        membership: list[int] = list(partition.membership)
        n_communities = len(set(membership))

        size_stats = community_size_stats(membership)
        conductance = compute_conductance(undirected, membership)
        coverage = compute_coverage(undirected, membership)

        entry: dict[str, Any] = {
            "resolution": res_val,
            "n_communities": n_communities,
            "membership": membership,
            "size_stats": size_stats,
            "conductance": conductance,
            "coverage": coverage,
            "elapsed_ms": round(_elapsed_ms(t_run), 1),
        }

        if reference_labels is not None:
            entry["nmi"] = compute_nmi(membership, reference_labels)

        results.append(entry)
        logger.info(
            "Sweep (%s, res=%.6f): %d communities, coverage=%.4f, "
            "conductance_mean=%.4f in %.1fms",
            partition_type,
            res_val,
            n_communities,
            coverage,
            conductance["mean"],
            entry["elapsed_ms"],
        )

    logger.info(
        "Resolution sweep complete: %d resolutions in %.1fms",
        len(resolutions),
        _elapsed_ms(t0),
    )
    return results


def compare_partitions(
    partitions: dict[str, list[int]],
) -> dict[str, Any]:
    """Compute pairwise NMI between named partitions.

    Useful for understanding how community structure changes across
    resolutions or partition types (e.g., modularity vs CPM).

    Args:
        partitions: {name: membership_list} for each partition to compare.
            All membership lists must have the same length.

    Returns:
        Dictionary with:
            partition_names: list of partition names (sorted by insertion order).
            community_counts: {name: n_communities} for each partition.
            nmi_matrix: {name_a: {name_b: nmi_score}} for all pairs.

    Raises:
        ValueError: If partitions dict is empty.
    """
    if not partitions:
        raise ValueError("partitions dict must not be empty")

    names = list(partitions.keys())
    community_counts = {name: len(set(m)) for name, m in partitions.items()}

    nmi_matrix: dict[str, dict[str, float]] = {}
    for name_a in names:
        nmi_matrix[name_a] = {}
        for name_b in names:
            if name_a == name_b:
                nmi_matrix[name_a][name_b] = 1.0
            elif name_b in nmi_matrix and name_a in nmi_matrix[name_b]:
                # Reuse symmetric result
                nmi_matrix[name_a][name_b] = nmi_matrix[name_b][name_a]
            else:
                nmi_matrix[name_a][name_b] = compute_nmi(partitions[name_a], partitions[name_b])

    return {
        "partition_names": names,
        "community_counts": community_counts,
        "nmi_matrix": nmi_matrix,
    }


# ---------------------------------------------------------------------------
# Community label generation (TF-IDF on keywords)
# ---------------------------------------------------------------------------


def generate_community_labels(
    conn: psycopg.Connection,
    resolution_name: str,
    top_k: int = 5,
) -> dict[int, tuple[str, list[str]]]:
    """Generate descriptive labels for communities using TF-IDF on paper keywords.

    Args:
        conn: Database connection.
        resolution_name: One of 'coarse', 'medium', 'fine'.
        top_k: Number of top keywords per community for the label.

    Returns:
        {community_id: ("kw1 / kw2 / ...", [kw1, kw2, ...])}
    """
    if resolution_name not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution_name: {resolution_name!r}. "
            f"Must be one of {sorted(_VALID_RESOLUTIONS)}"
        )

    t0 = time.perf_counter()
    col = f"community_id_{resolution_name}"

    # Count total communities for IDF denominator
    count_sql = f"SELECT COUNT(DISTINCT {col}) FROM paper_metrics WHERE {col} IS NOT NULL"
    with conn.cursor() as cur:
        cur.execute(count_sql)
        (total_communities,) = cur.fetchone()  # type: ignore[misc]

    if total_communities == 0:
        logger.warning("No communities found for resolution %s", resolution_name)
        return {}

    # TF-IDF query: term frequency within community * inverse document frequency
    tfidf_sql = f"""
        WITH community_kw AS (
            SELECT pm.{col} AS cid, kw, COUNT(*) AS tf
            FROM paper_metrics pm
            JOIN papers p ON p.bibcode = pm.bibcode,
                 LATERAL unnest(p.keywords) AS kw
            WHERE p.keywords IS NOT NULL
            GROUP BY cid, kw
        ),
        global_kw AS (
            SELECT kw, COUNT(DISTINCT pm.{col}) AS df
            FROM paper_metrics pm
            JOIN papers p ON p.bibcode = pm.bibcode,
                 LATERAL unnest(p.keywords) AS kw
            WHERE p.keywords IS NOT NULL
            GROUP BY kw
        )
        SELECT ck.cid, ck.kw,
               ck.tf * ln((%s::float) / GREATEST(gk.df, 1)) AS tfidf
        FROM community_kw ck
        JOIN global_kw gk ON gk.kw = ck.kw
        ORDER BY ck.cid, tfidf DESC
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(tfidf_sql, (total_communities,))
        rows = cur.fetchall()

    # Take top_k keywords per community
    labels: dict[int, tuple[str, list[str]]] = {}
    current_cid: int | None = None
    current_kws: list[str] = []

    for row in rows:
        cid = row["cid"]
        if cid != current_cid:
            if current_cid is not None and current_kws:
                label_str = " / ".join(current_kws[:top_k])
                labels[current_cid] = (label_str, current_kws[:top_k])
            current_cid = cid
            current_kws = []
        if len(current_kws) < top_k:
            current_kws.append(row["kw"])

    # Flush last community
    if current_cid is not None and current_kws:
        label_str = " / ".join(current_kws[:top_k])
        labels[current_cid] = (label_str, current_kws[:top_k])

    logger.info(
        "Generated labels for %d communities (res=%s) in %.1fms",
        len(labels),
        resolution_name,
        _elapsed_ms(t0),
    )
    return labels


# ---------------------------------------------------------------------------
# Metrics storage (COPY via staging table)
# ---------------------------------------------------------------------------


def store_metrics(
    conn: psycopg.Connection,
    pagerank: list[float],
    hub_scores: list[float],
    authority_scores: list[float],
    communities_coarse: list[int | None],
    communities_medium: list[int | None],
    communities_fine: list[int | None],
    id_to_bibcode: dict[int, str],
) -> int:
    """Store computed metrics into paper_metrics using COPY via a staging table.

    Writes in chunks to control memory. Uses INSERT ... ON CONFLICT for upsert.

    Returns:
        Number of rows written.
    """
    t0 = time.perf_counter()
    n = len(id_to_bibcode)

    staging_ddl = (
        "CREATE TEMP TABLE IF NOT EXISTS _metrics_staging "
        "(LIKE paper_metrics INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
    )
    copy_sql = (
        "COPY _metrics_staging (bibcode, pagerank, hub_score, authority_score, "
        "community_id_coarse, community_id_medium, community_id_fine) FROM STDIN"
    )
    merge_sql = """
        INSERT INTO paper_metrics (bibcode, pagerank, hub_score, authority_score,
                                   community_id_coarse, community_id_medium,
                                   community_id_fine, updated_at)
        SELECT bibcode, pagerank, hub_score, authority_score,
               community_id_coarse, community_id_medium, community_id_fine, NOW()
        FROM _metrics_staging
        ON CONFLICT (bibcode) DO UPDATE SET
            pagerank = EXCLUDED.pagerank,
            hub_score = EXCLUDED.hub_score,
            authority_score = EXCLUDED.authority_score,
            community_id_coarse = EXCLUDED.community_id_coarse,
            community_id_medium = EXCLUDED.community_id_medium,
            community_id_fine = EXCLUDED.community_id_fine,
            updated_at = EXCLUDED.updated_at
    """

    total_written = 0

    for chunk_start in range(0, n, _CHUNK_SIZE):
        chunk_end = min(chunk_start + _CHUNK_SIZE, n)

        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(staging_ddl)

            _null = "\\N"
            buf = io.StringIO()
            for vid in range(chunk_start, chunk_end):
                bibcode = id_to_bibcode[vid]
                cc = communities_coarse[vid]
                cm = communities_medium[vid]
                cf = communities_fine[vid]
                row = (
                    f"{bibcode}\t"
                    f"{pagerank[vid]}\t"
                    f"{hub_scores[vid]}\t"
                    f"{authority_scores[vid]}\t"
                    f"{_null if cc is None else cc}\t"
                    f"{_null if cm is None else cm}\t"
                    f"{_null if cf is None else cf}\n"
                )
                buf.write(row)

            buf.seek(0)
            with conn.cursor() as cur:
                with cur.copy(copy_sql) as copy:
                    while chunk := buf.read(8192):
                        copy.write(chunk.encode("utf-8"))

            with conn.cursor() as cur:
                cur.execute(merge_sql)

        chunk_written = chunk_end - chunk_start
        total_written += chunk_written
        logger.debug(
            "Stored metrics chunk %d-%d (%d rows)",
            chunk_start,
            chunk_end,
            chunk_written,
        )

    logger.info("Stored %d metric rows in %.1fms", total_written, _elapsed_ms(t0))
    return total_written


# ---------------------------------------------------------------------------
# Community metadata storage
# ---------------------------------------------------------------------------


def store_community_metadata(
    conn: psycopg.Connection,
    resolution_name: str,
    membership_by_bibcode: dict[str, int],
    labels: dict[int, tuple[str, list[str]]],
) -> int:
    """Store community metadata (labels, paper counts) into the communities table.

    Args:
        conn: Database connection.
        resolution_name: One of 'coarse', 'medium', 'fine'.
        membership_by_bibcode: {bibcode: community_id} mapping.
        labels: {community_id: ("label string", [kw1, kw2, ...])} from
            generate_community_labels().

    Returns:
        Number of community rows upserted.
    """
    if resolution_name not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution_name: {resolution_name!r}. "
            f"Must be one of {sorted(_VALID_RESOLUTIONS)}"
        )

    t0 = time.perf_counter()

    # Count papers per community
    community_counts: dict[int, int] = {}
    for cid in membership_by_bibcode.values():
        community_counts[cid] = community_counts.get(cid, 0) + 1

    upsert_sql = """
        INSERT INTO communities (community_id, resolution, label, paper_count,
                                 top_keywords, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (community_id, resolution) DO UPDATE SET
            label = EXCLUDED.label,
            paper_count = EXCLUDED.paper_count,
            top_keywords = EXCLUDED.top_keywords,
            updated_at = EXCLUDED.updated_at
    """

    rows_written = 0
    with conn.cursor() as cur:
        for cid, count in community_counts.items():
            label_str, top_kws = labels.get(cid, ("", []))
            cur.execute(upsert_sql, (cid, resolution_name, label_str, count, top_kws))
            rows_written += 1
    conn.commit()

    logger.info(
        "Stored %d community metadata rows (res=%s) in %.1fms",
        rows_written,
        resolution_name,
        _elapsed_ms(t0),
    )
    return rows_written


# ---------------------------------------------------------------------------
# Taxonomic community assignment (from arxiv_class)
# ---------------------------------------------------------------------------


def extract_taxonomic_community(arxiv_class: list[str] | None) -> str | None:
    """Extract the primary taxonomic community from a paper's arxiv_class list.

    Prefers astro-ph subcategories (astro-ph.CO, astro-ph.EP, etc.) over
    non-astrophysics categories. If no astro-ph entry exists, returns the
    first category in the list.

    Args:
        arxiv_class: List of arXiv classification strings, or None.

    Returns:
        Primary category string, or None if the list is empty/None.
    """
    if not arxiv_class:
        return None

    # Prefer the first astro-ph entry
    for cat in arxiv_class:
        if cat.startswith(_ASTRO_PH_PREFIX):
            return cat

    # Fall back to first category
    return arxiv_class[0]


def populate_taxonomic_communities(conn: psycopg.Connection) -> int:
    """Bulk-populate community_taxonomic in paper_metrics from papers.arxiv_class.

    Uses a single SQL UPDATE joining paper_metrics to papers, picking the first
    astro-ph.* subcategory (or the first arxiv_class entry if no astro-ph match).

    Args:
        conn: Database connection.

    Returns:
        Number of rows updated.
    """
    t0 = time.perf_counter()

    update_sql = """
        UPDATE paper_metrics pm
        SET community_taxonomic = (
            SELECT COALESCE(
                (SELECT unnest FROM unnest(p.arxiv_class) WHERE unnest LIKE 'astro-ph%%' LIMIT 1),
                p.arxiv_class[1]
            )
        ),
        updated_at = NOW()
        FROM papers p
        WHERE p.bibcode = pm.bibcode
          AND p.arxiv_class IS NOT NULL
          AND array_length(p.arxiv_class, 1) > 0
          AND pm.community_taxonomic IS NULL
    """

    with conn.cursor() as cur:
        cur.execute(update_sql)
        rows_updated = cur.rowcount
    conn.commit()

    logger.info(
        "Populated community_taxonomic for %d papers in %.1fms",
        rows_updated,
        _elapsed_ms(t0),
    )
    return rows_updated


# ---------------------------------------------------------------------------
# Embedding loader for small-component assignment
# ---------------------------------------------------------------------------


def _load_embeddings(
    conn: psycopg.Connection,
    bibcodes: set[str],
) -> dict[str, Any]:
    """Load embedding vectors from paper_embeddings for the given bibcodes.

    Returns {bibcode: numpy array} for all bibcodes that have embeddings.
    """
    import numpy as np

    if not bibcodes:
        return {}

    t0 = time.perf_counter()
    result: dict[str, Any] = {}

    # Use batched IN queries to avoid overly long parameter lists
    bibcode_list = list(bibcodes)
    batch_size = 10_000
    for i in range(0, len(bibcode_list), batch_size):
        batch = bibcode_list[i : i + batch_size]
        placeholders = ",".join(["%s"] * len(batch))
        sql = (
            f"SELECT bibcode, embedding::text FROM paper_embeddings "
            f"WHERE bibcode IN ({placeholders}) "
            f"AND embedding IS NOT NULL "
            f"ORDER BY bibcode "
            f"LIMIT 1"  # one embedding per bibcode (latest model)
        )
        # Actually we want one per bibcode, use DISTINCT ON
        sql = (
            f"SELECT DISTINCT ON (bibcode) bibcode, embedding::text "
            f"FROM paper_embeddings "
            f"WHERE bibcode IN ({placeholders}) "
            f"AND embedding IS NOT NULL "
            f"ORDER BY bibcode"
        )
        with conn.cursor() as cur:
            cur.execute(sql, batch)
            for bibcode, emb_text in cur.fetchall():
                # pgvector text format: "[0.1,0.2,...]"
                vec = np.fromstring(emb_text.strip("[]"), sep=",", dtype=np.float32)
                result[bibcode] = vec

    logger.info(
        "Loaded %d/%d embeddings in %.1fms",
        len(result),
        len(bibcodes),
        _elapsed_ms(t0),
    )
    return result


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    dsn: str | None = None,
    calibrate: bool = True,
    res_coarse: float = 5.0,
    res_medium: float = 1.0,
    res_fine: float = 0.1,
    seed: int = 42,
    skip_labels: bool = False,
) -> dict[str, float]:
    """Run the full graph metrics pipeline.

    Steps:
        1. Load citation graph from PostgreSQL into igraph.
        2. Compute PageRank and HITS scores.
        3. Filter isolated nodes, extract giant component.
        4. Run Leiden community detection at 3 resolutions on giant component.
        5. Assign small-component papers to nearest community by embedding distance.
        6. Store all metrics into paper_metrics.
        7. Generate and store community labels.

    Args:
        dsn: PostgreSQL DSN (defaults to SCIX_DSN env var).
        calibrate: If True, use binary search to calibrate Leiden resolutions
            targeting ~20 (coarse), ~200 (medium), ~2000 (fine) communities.
        res_coarse: Resolution for coarse communities (used if calibrate=False).
        res_medium: Resolution for medium communities (used if calibrate=False).
        res_fine: Resolution for fine communities (used if calibrate=False).
        seed: Random seed for Leiden.

    Returns:
        Timing dict with milliseconds for each step.
    """
    timing: dict[str, float] = {}
    pipeline_t0 = time.perf_counter()

    # --- Connect ---
    logger.info("Connecting to database...")
    conn = get_connection(dsn)

    # --- Load graph ---
    logger.info("Loading citation graph...")
    t0 = time.perf_counter()
    graph, bibcode_to_id, id_to_bibcode = load_graph(conn)
    timing["load_graph_ms"] = _elapsed_ms(t0)

    # --- PageRank ---
    logger.info("Computing PageRank...")
    t0 = time.perf_counter()
    pagerank = compute_pagerank(graph)
    timing["pagerank_ms"] = _elapsed_ms(t0)

    # --- HITS ---
    logger.info("Computing HITS...")
    t0 = time.perf_counter()
    hub_scores, authority_scores = compute_hits(graph)
    timing["hits_ms"] = _elapsed_ms(t0)

    # --- Filter isolated nodes for Leiden ---
    logger.info("Filtering isolated nodes for Leiden...")
    t0 = time.perf_counter()
    subgraph, sub_b2i, sub_i2b, isolated_bibcodes = filter_isolated_nodes(
        graph, bibcode_to_id, id_to_bibcode
    )
    timing["filter_isolates_ms"] = _elapsed_ms(t0)
    timing["isolated_node_count"] = len(isolated_bibcodes)
    del isolated_bibcodes  # free memory — count is recorded in timing

    # --- Extract giant component ---
    logger.info("Extracting giant component...")
    t0 = time.perf_counter()
    giant, giant_b2i, giant_i2b, small_component_bibcodes = extract_giant_component(
        subgraph, sub_b2i, sub_i2b
    )
    timing["extract_giant_ms"] = _elapsed_ms(t0)
    timing["small_component_node_count"] = len(small_component_bibcodes)

    # --- Leiden community detection (3 resolutions) on giant component ---
    if giant.vcount() == 0:
        logger.warning("No connected nodes — skipping Leiden entirely")
        giant_coarse: list[int] = []
        giant_medium: list[int] = []
        giant_fine: list[int] = []
    else:
        if calibrate:
            logger.info("Calibrating Leiden resolutions on giant component...")

            t0 = time.perf_counter()
            res_coarse = calibrate_resolution(
                giant,
                target_communities=20,
                seed=seed,
            )
            timing["calibrate_coarse_ms"] = _elapsed_ms(t0)

            t0 = time.perf_counter()
            res_medium = calibrate_resolution(
                giant,
                target_communities=200,
                seed=seed,
            )
            timing["calibrate_medium_ms"] = _elapsed_ms(t0)

            t0 = time.perf_counter()
            res_fine = calibrate_resolution(
                giant,
                target_communities=2000,
                seed=seed,
            )
            timing["calibrate_fine_ms"] = _elapsed_ms(t0)

        logger.info(
            "Running Leiden on %d giant-component nodes: coarse=%.4f, medium=%.4f, fine=%.4f",
            giant.vcount(),
            res_coarse,
            res_medium,
            res_fine,
        )

        t0 = time.perf_counter()
        giant_coarse = compute_leiden(giant, resolution=res_coarse, seed=seed)
        timing["leiden_coarse_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        giant_medium = compute_leiden(giant, resolution=res_medium, seed=seed)
        timing["leiden_medium_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        giant_fine = compute_leiden(giant, resolution=res_fine, seed=seed)
        timing["leiden_fine_ms"] = _elapsed_ms(t0)

    # --- Assign small-component papers by embedding distance ---
    if small_component_bibcodes and giant.vcount() > 0:
        logger.info(
            "Assigning %d small-component papers by embedding distance...",
            len(small_component_bibcodes),
        )
        t0 = time.perf_counter()

        # Load embeddings for giant-component and small-component papers
        all_bibcodes_needing_embeddings = set(giant_b2i.keys()) | small_component_bibcodes
        embeddings = _load_embeddings(conn, all_bibcodes_needing_embeddings)

        giant_embeddings = {b: embeddings[b] for b in giant_b2i if b in embeddings}
        small_embeddings = {b: embeddings[b] for b in small_component_bibcodes if b in embeddings}

        # Build giant membership dicts for each resolution
        giant_membership_coarse = {giant_i2b[vid]: giant_coarse[vid] for vid in giant_i2b}
        giant_membership_medium = {giant_i2b[vid]: giant_medium[vid] for vid in giant_i2b}
        giant_membership_fine = {giant_i2b[vid]: giant_fine[vid] for vid in giant_i2b}

        small_assign_coarse = assign_small_component_communities(
            small_component_bibcodes, giant_membership_coarse, giant_embeddings, small_embeddings
        )
        small_assign_medium = assign_small_component_communities(
            small_component_bibcodes, giant_membership_medium, giant_embeddings, small_embeddings
        )
        small_assign_fine = assign_small_component_communities(
            small_component_bibcodes, giant_membership_fine, giant_embeddings, small_embeddings
        )
        timing["assign_small_components_ms"] = _elapsed_ms(t0)
    else:
        small_assign_coarse = {}
        small_assign_medium = {}
        small_assign_fine = {}

    # --- Expand giant-component communities back to full node list ---
    # Giant-component nodes get their Leiden community ID.
    # Small-component nodes get their embedding-assigned community ID (or None).
    # Isolated nodes get None.
    n = len(id_to_bibcode)
    communities_coarse: list[int | None] = [None] * n
    communities_medium: list[int | None] = [None] * n
    communities_fine: list[int | None] = [None] * n

    for giant_vid, bib in giant_i2b.items():
        full_vid = bibcode_to_id[bib]
        communities_coarse[full_vid] = giant_coarse[giant_vid]
        communities_medium[full_vid] = giant_medium[giant_vid]
        communities_fine[full_vid] = giant_fine[giant_vid]

    for bib in small_component_bibcodes:
        full_vid = bibcode_to_id[bib]
        communities_coarse[full_vid] = small_assign_coarse.get(bib)
        communities_medium[full_vid] = small_assign_medium.get(bib)
        communities_fine[full_vid] = small_assign_fine.get(bib)

    # --- Store metrics ---
    # Switch to autocommit so each chunk commits immediately
    conn.commit()
    conn.autocommit = True
    logger.info("Storing metrics...")
    t0 = time.perf_counter()
    store_metrics(
        conn,
        pagerank,
        hub_scores,
        authority_scores,
        communities_coarse,
        communities_medium,
        communities_fine,
        id_to_bibcode,
    )
    timing["store_metrics_ms"] = _elapsed_ms(t0)

    # --- Generate and store community labels ---
    if skip_labels:
        logger.info("Skipping community label generation (--skip-labels)")
    else:
        for res_name, membership in [
            ("coarse", communities_coarse),
            ("medium", communities_medium),
            ("fine", communities_fine),
        ]:
            logger.info("Generating labels for %s communities...", res_name)
            t0 = time.perf_counter()

            membership_by_bibcode = {
                id_to_bibcode[vid]: cid for vid, cid in enumerate(membership) if cid is not None
            }

            labels = generate_community_labels(conn, res_name)
            store_community_metadata(conn, res_name, membership_by_bibcode, labels)
            timing[f"labels_{res_name}_ms"] = _elapsed_ms(t0)

    # --- Populate taxonomic communities from arxiv_class ---
    logger.info("Populating taxonomic communities from arxiv_class...")
    conn.autocommit = False
    t0 = time.perf_counter()
    populate_taxonomic_communities(conn)
    timing["taxonomic_communities_ms"] = _elapsed_ms(t0)

    conn.close()

    timing["total_ms"] = _elapsed_ms(pipeline_t0)
    logger.info("Pipeline complete in %.1fs", timing["total_ms"] / 1000)
    for step, ms in sorted(timing.items()):
        logger.info("  %-30s %10.1fms", step, ms)

    return timing
