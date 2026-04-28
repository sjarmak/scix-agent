"""Slice loader: Postgres → in-memory igraph snapshot.

Builds the slice in five phases:
  1. Seed selection (bibcodes matching ``SliceConfig.seed_filter_sql``).
  2. Neighborhood expansion (``hop_depth`` rounds via ``citation_edges``).
  3. Edge materialization (``COPY`` of in-slice edges).
  4. Vertex attribute attachment (title, year, citation_count).
  5. Pickle to ``.pkl.gz``.

For the default astronomy_1hop slice this writes ~5-8 GB of igraph state to
disk in roughly 10-20 minutes. Wrap invocations in ``scix-batch`` per
CLAUDE.md storage/memory guidance.
"""

from __future__ import annotations

import gc
import gzip
import logging
import pickle
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg

from scix.db import DEFAULT_DSN, get_connection
from scix.graph_experiment.slice_config import SliceConfig

logger = logging.getLogger(__name__)

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


@dataclass
class SliceStats:
    name: str
    seed_count: int
    node_count: int
    edge_count: int
    expansion_seconds: float
    edge_fetch_seconds: float
    graph_build_seconds: float
    snapshot_bytes: int


def build_slice(
    config: SliceConfig,
    *,
    dsn: str | None = None,
    write_snapshot: bool = True,
) -> tuple[object, SliceStats]:
    """Materialize the slice as an igraph.Graph and (optionally) snapshot it.

    Returns (graph, stats). ``graph`` is an ``igraph.Graph`` directed graph
    with vertex attribute ``name`` (bibcode) plus ``title``, ``year``,
    ``citation_count`` where present in ``papers``.
    """
    import igraph as ig

    dsn = dsn or DEFAULT_DSN
    logger.info("building slice %s against dsn=%s", config.name, dsn)

    with get_connection(dsn) as conn:
        seed_count = _populate_seed_table(conn, config)
        logger.info("seed_count=%d", seed_count)

        expansion_start = time.time()
        node_count = _expand_neighborhood(conn, config.hop_depth)
        expansion_seconds = time.time() - expansion_start
        logger.info(
            "expansion done: nodes=%d (+%.2f×) in %.1fs",
            node_count,
            node_count / max(seed_count, 1),
            expansion_seconds,
        )

        edge_fetch_start = time.time()
        bibcode_to_id, id_to_bibcode, sources, targets = _fetch_edges(conn)
        edge_fetch_seconds = time.time() - edge_fetch_start
        logger.info(
            "fetched %d in-slice edges over %d nodes in %.1fs",
            len(sources),
            len(bibcode_to_id),
            edge_fetch_seconds,
        )

        graph_build_start = time.time()
        # Build numpy edge matrix in-place from the array.array buffers; this
        # avoids materialising 235M Python tuples (the original OOM cause —
        # peak 30+ GB at this step). column_stack copies once into a single
        # 2-column int32 array (~2 GB for 235M edges) which igraph then
        # ingests directly.
        edge_array = np.column_stack(
            [
                np.frombuffer(sources, dtype=np.int32),
                np.frombuffer(targets, dtype=np.int32),
            ]
        )
        node_count_local = len(id_to_bibcode)
        # Free the source/target arrays + bibcode→id dict before graph
        # construction — igraph's internal copy adds ~4 GB and we don't need
        # the lookup dict any more (id_to_bibcode is enough for vertex names).
        del sources, targets, bibcode_to_id
        gc.collect()
        graph = ig.Graph(n=node_count_local, edges=edge_array, directed=True)
        del edge_array
        gc.collect()
        graph.vs["name"] = id_to_bibcode
        _attach_paper_attrs_by_name(conn, graph)
        graph_build_seconds = time.time() - graph_build_start
        logger.info("graph built in %.1fs", graph_build_seconds)

    snapshot_bytes = 0
    if write_snapshot:
        snapshot_bytes = _write_snapshot(graph, config.snapshot_path)
        logger.info(
            "snapshot written to %s (%.1f MB)",
            config.snapshot_path,
            snapshot_bytes / 1e6,
        )

    stats = SliceStats(
        name=config.name,
        seed_count=seed_count,
        node_count=graph.vcount(),
        edge_count=graph.ecount(),
        expansion_seconds=expansion_seconds,
        edge_fetch_seconds=edge_fetch_seconds,
        graph_build_seconds=graph_build_seconds,
        snapshot_bytes=snapshot_bytes,
    )
    return graph, stats


def load_snapshot(path: Path):
    """Load a pickled igraph snapshot. Returns igraph.Graph."""
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 — snapshot is local-only artifact


def _populate_seed_table(conn: psycopg.Connection, config: SliceConfig) -> int:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS tmp_slice_seed")
        cur.execute(
            "CREATE TEMP TABLE tmp_slice_seed (bibcode TEXT PRIMARY KEY) "
            "ON COMMIT PRESERVE ROWS"
        )
        sql = (
            "INSERT INTO tmp_slice_seed (bibcode) "
            f"SELECT bibcode FROM papers WHERE {config.seed_filter_sql} "
            "ON CONFLICT DO NOTHING"
        )
        cur.execute(sql, config.seed_filter_params)
        cur.execute("SELECT count(*) FROM tmp_slice_seed")
        row = cur.fetchone()
        seed_count = int(row[0]) if row else 0
        cur.execute("ANALYZE tmp_slice_seed")
    conn.commit()
    return seed_count


def _expand_neighborhood(conn: psycopg.Connection, hop_depth: int) -> int:
    """Expand seed set ``hop_depth`` times via citation_edges, both directions.

    Result lives in temp table ``tmp_slice_nodes``.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS tmp_slice_nodes")
        cur.execute(
            "CREATE TEMP TABLE tmp_slice_nodes (bibcode TEXT PRIMARY KEY) "
            "ON COMMIT PRESERVE ROWS"
        )
        cur.execute(
            "INSERT INTO tmp_slice_nodes (bibcode) SELECT bibcode FROM tmp_slice_seed"
        )

        for hop in range(1, hop_depth + 1):
            t0 = time.time()
            cur.execute(
                "INSERT INTO tmp_slice_nodes (bibcode) "
                "SELECT DISTINCT ce.target_bibcode FROM citation_edges ce "
                "JOIN tmp_slice_nodes n ON n.bibcode = ce.source_bibcode "
                "ON CONFLICT DO NOTHING"
            )
            forward_added = cur.rowcount
            cur.execute(
                "INSERT INTO tmp_slice_nodes (bibcode) "
                "SELECT DISTINCT ce.source_bibcode FROM citation_edges ce "
                "JOIN tmp_slice_nodes n ON n.bibcode = ce.target_bibcode "
                "ON CONFLICT DO NOTHING"
            )
            backward_added = cur.rowcount
            cur.execute("ANALYZE tmp_slice_nodes")
            logger.info(
                "hop %d: +%d forward, +%d backward in %.1fs",
                hop,
                forward_added,
                backward_added,
                time.time() - t0,
            )

        cur.execute("SELECT count(*) FROM tmp_slice_nodes")
        row = cur.fetchone()
        node_count = int(row[0]) if row else 0

    conn.commit()
    return node_count


def _fetch_edges(
    conn: psycopg.Connection,
) -> tuple[dict[str, int], list[str], "array[int]", "array[int]"]:
    """Stream in-slice edges via COPY, returning (bibcode→id, id→bibcode, src, dst)."""
    bibcode_to_id: dict[str, int] = {}
    id_to_bibcode: list[str] = []
    with conn.cursor() as cur:
        with cur.copy("COPY (SELECT bibcode FROM tmp_slice_nodes) TO STDOUT") as copy:
            for row in copy.rows():
                bibcode = row[0]
                bibcode_to_id[bibcode] = len(id_to_bibcode)
                id_to_bibcode.append(bibcode)

    sources: array = array("i")
    targets: array = array("i")
    with conn.cursor() as cur:
        with cur.copy(
            "COPY (SELECT ce.source_bibcode, ce.target_bibcode FROM citation_edges ce "
            "JOIN tmp_slice_nodes s1 ON s1.bibcode = ce.source_bibcode "
            "JOIN tmp_slice_nodes s2 ON s2.bibcode = ce.target_bibcode) TO STDOUT"
        ) as copy:
            for row in copy.rows():
                src_bib, tgt_bib = row[0], row[1]
                src_id = bibcode_to_id.get(src_bib)
                tgt_id = bibcode_to_id.get(tgt_bib)
                if src_id is None or tgt_id is None:
                    continue
                sources.append(src_id)
                targets.append(tgt_id)

    return bibcode_to_id, id_to_bibcode, sources, targets


def _attach_paper_attrs_by_name(conn: psycopg.Connection, graph) -> None:
    """Attach title, year, citation_count to vertices for papers we have data on.

    Uses the graph's built-in name index (``graph.vs.find(name=...)`` is O(1)
    after the first lookup builds the index) rather than holding a separate
    ``bibcode_to_id`` dict in Python — the dict is ~1.5 GB at full slice
    size, and we already freed it before reaching this step.
    """
    titles: list[str | None] = [None] * graph.vcount()
    years: list[int | None] = [None] * graph.vcount()
    citation_counts: list[int | None] = [None] * graph.vcount()

    name_index = {name: i for i, name in enumerate(graph.vs["name"])}

    with conn.cursor(name="attr_stream") as cur:
        cur.itersize = 50_000
        cur.execute(
            "SELECT p.bibcode, p.title, p.year, p.citation_count "
            "FROM papers p JOIN tmp_slice_nodes n ON n.bibcode = p.bibcode"
        )
        for bibcode, title, year, cc in cur:
            idx = name_index.get(bibcode)
            if idx is None:
                continue
            titles[idx] = title
            years[idx] = year
            citation_counts[idx] = cc

    del name_index
    gc.collect()

    graph.vs["title"] = titles
    graph.vs["year"] = years
    graph.vs["citation_count"] = citation_counts


def _write_snapshot(graph, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wb", compresslevel=3) as fh:
        pickle.dump(graph, fh, protocol=_PICKLE_PROTOCOL)
    tmp_path.replace(path)
    return path.stat().st_size
