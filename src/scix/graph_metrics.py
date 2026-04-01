"""Precompute graph metrics (PageRank, HITS, Leiden communities) on the citation graph."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from scix.db import get_connection

logger = logging.getLogger(__name__)

_VALID_RESOLUTIONS = frozenset({"coarse", "medium", "fine"})

_CHUNK_SIZE = 100_000


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------


def _elapsed_ms(t0: float) -> float:
    """Milliseconds elapsed since t0 (from time.perf_counter)."""
    return round((time.perf_counter() - t0) * 1000, 2)


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


def compute_leiden(
    graph: Any,
    resolution: float,
    seed: int = 42,
) -> list[int]:
    """Run Leiden community detection on an undirected projection of the graph.

    Args:
        graph: Directed igraph.Graph (will be converted to undirected).
        resolution: Resolution parameter for RBConfigurationVertexPartition.
            Higher values yield more, smaller communities.
        seed: Random seed for reproducibility.

    Returns:
        Membership list (list[int]) indexed by vertex ID.
    """
    leidenalg = _import_leidenalg()

    t0 = time.perf_counter()
    undirected = graph.as_undirected(mode="collapse")
    partition = leidenalg.find_partition(
        undirected,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    membership: list[int] = list(partition.membership)
    n_communities = len(set(membership))
    logger.info(
        "Leiden (res=%.4f) found %d communities in %.1fms",
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
        membership = compute_leiden(graph, resolution=mid, seed=seed)
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
    communities_coarse: list[int],
    communities_medium: list[int],
    communities_fine: list[int],
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
        "CREATE TEMP TABLE _metrics_staging "
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

            buf = io.StringIO()
            for vid in range(chunk_start, chunk_end):
                bibcode = id_to_bibcode[vid]
                row = (
                    f"{bibcode}\t"
                    f"{pagerank[vid]}\t"
                    f"{hub_scores[vid]}\t"
                    f"{authority_scores[vid]}\t"
                    f"{communities_coarse[vid]}\t"
                    f"{communities_medium[vid]}\t"
                    f"{communities_fine[vid]}\n"
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
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    dsn: str | None = None,
    calibrate: bool = True,
    res_coarse: float = 5.0,
    res_medium: float = 1.0,
    res_fine: float = 0.1,
    seed: int = 42,
) -> dict[str, float]:
    """Run the full graph metrics pipeline.

    Steps:
        1. Load citation graph from PostgreSQL into igraph.
        2. Compute PageRank and HITS scores.
        3. Run Leiden community detection at 3 resolutions.
        4. Store all metrics into paper_metrics.
        5. Generate and store community labels.

    Args:
        dsn: PostgreSQL DSN (defaults to SCIX_DSN env var).
        calibrate: If True, use binary search to calibrate Leiden resolutions
            targeting ~50 (coarse), ~500 (medium), ~5000 (fine) communities.
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

    # --- Leiden community detection (3 resolutions) ---
    if calibrate:
        logger.info("Calibrating Leiden resolutions...")

        t0 = time.perf_counter()
        res_coarse = calibrate_resolution(
            graph,
            target_communities=50,
            seed=seed,
        )
        timing["calibrate_coarse_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        res_medium = calibrate_resolution(
            graph,
            target_communities=500,
            seed=seed,
        )
        timing["calibrate_medium_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        res_fine = calibrate_resolution(
            graph,
            target_communities=5000,
            seed=seed,
        )
        timing["calibrate_fine_ms"] = _elapsed_ms(t0)

    logger.info(
        "Running Leiden: coarse=%.4f, medium=%.4f, fine=%.4f",
        res_coarse,
        res_medium,
        res_fine,
    )

    t0 = time.perf_counter()
    communities_coarse = compute_leiden(graph, resolution=res_coarse, seed=seed)
    timing["leiden_coarse_ms"] = _elapsed_ms(t0)

    t0 = time.perf_counter()
    communities_medium = compute_leiden(graph, resolution=res_medium, seed=seed)
    timing["leiden_medium_ms"] = _elapsed_ms(t0)

    t0 = time.perf_counter()
    communities_fine = compute_leiden(graph, resolution=res_fine, seed=seed)
    timing["leiden_fine_ms"] = _elapsed_ms(t0)

    # --- Store metrics ---
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
    for res_name, membership in [
        ("coarse", communities_coarse),
        ("medium", communities_medium),
        ("fine", communities_fine),
    ]:
        logger.info("Generating labels for %s communities...", res_name)
        t0 = time.perf_counter()

        # Build bibcode -> community_id mapping
        membership_by_bibcode = {id_to_bibcode[vid]: cid for vid, cid in enumerate(membership)}

        labels = generate_community_labels(conn, res_name)
        store_community_metadata(conn, res_name, membership_by_bibcode, labels)
        timing[f"labels_{res_name}_ms"] = _elapsed_ms(t0)

    conn.close()

    timing["total_ms"] = _elapsed_ms(pipeline_t0)
    logger.info("Pipeline complete in %.1fs", timing["total_ms"] / 1000)
    for step, ms in sorted(timing.items()):
        logger.info("  %-30s %10.1fms", step, ms)

    return timing
