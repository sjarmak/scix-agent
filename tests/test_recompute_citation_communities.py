"""Tests for scripts/recompute_citation_communities.py (PRD M3).

Builds a synthetic citation graph with:
  - a 100-node connected giant component, and
  - 10 disconnected pairs (20 satellite nodes)

then runs the script and verifies:
  (a) giant-component rows receive non-NULL community assignments;
  (b) satellite rows receive NULL;
  (c) re-running with the same seed produces deterministic memberships
      (identical up to canonical community-ID relabeling);
  (d) the largest coarse-level community is ≤ 10% of the giant component.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from tests.helpers import get_test_dsn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "recompute_citation_communities.py"

BIB_PREFIX = "TEST_M3_"
GIANT_N = 100
SATELLITE_PAIRS = 10  # → 20 bibcodes


# ---------------------------------------------------------------------------
# Fixture: synthetic citation graph in scix_test
# ---------------------------------------------------------------------------


def _giant_bibcodes() -> list[str]:
    return [f"{BIB_PREFIX}g_{i:03d}" for i in range(GIANT_N)]


def _satellite_bibcodes() -> list[str]:
    return [f"{BIB_PREFIX}s_{i:03d}" for i in range(2 * SATELLITE_PAIRS)]


def _build_synthetic_edges(rng: random.Random) -> list[tuple[str, str]]:
    """Build edges: 10 dense clusters of 10 nodes each, sparsely bridged, plus 10 pairs.

    Structure is deliberately clustered so Leiden recovers ~10 communities
    of ~10 nodes each — no community exceeds 10% of the 100-node giant
    component at any resolution. Bridge edges (one per adjacent cluster
    pair) keep everything in a single connected component.
    """
    giant = _giant_bibcodes()
    edges: list[tuple[str, str]] = []

    cluster_size = 10
    n_clusters = GIANT_N // cluster_size  # 10

    # Dense intra-cluster edges (near-complete bipartite pairing).
    for c in range(n_clusters):
        base = c * cluster_size
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                edges.append((giant[base + i], giant[base + j]))

    # Single bridge edge between consecutive clusters keeps the whole
    # thing in one connected component but doesn't overwhelm the
    # intra-cluster density.
    for c in range(n_clusters):
        src = giant[c * cluster_size]
        tgt = giant[((c + 1) % n_clusters) * cluster_size]
        edges.append((src, tgt))

    # A handful of extra random inter-cluster edges for realism — not
    # enough to dissolve cluster boundaries.
    for _ in range(n_clusters):
        a = rng.randrange(GIANT_N)
        b = rng.randrange(GIANT_N)
        if a != b:
            edges.append((giant[a], giant[b]))

    # Satellite pairs — each pair is its own 2-node component.
    sat = _satellite_bibcodes()
    for i in range(SATELLITE_PAIRS):
        edges.append((sat[2 * i], sat[2 * i + 1]))

    return edges


def _cleanup(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM citation_edges "
            "WHERE source_bibcode LIKE %s OR target_bibcode LIKE %s",
            (f"{BIB_PREFIX}%", f"{BIB_PREFIX}%"),
        )
        cur.execute(
            "DELETE FROM paper_metrics WHERE bibcode LIKE %s",
            (f"{BIB_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode LIKE %s",
            (f"{BIB_PREFIX}%",),
        )
    conn.commit()


@pytest.fixture(scope="module")
def test_dsn() -> str:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set — skipping integration test")
    return dsn


@pytest.fixture(scope="module")
def synthetic_graph(test_dsn: str) -> Iterator[dict[str, list[str]]]:
    """Insert synthetic papers / paper_metrics / citation_edges rows.

    Yields a dict with keys ``giant`` (list of giant bibcodes) and
    ``satellite`` (list of satellite bibcodes). Cleans up on teardown.
    """
    giant = _giant_bibcodes()
    satellite = _satellite_bibcodes()
    all_bibs = giant + satellite
    rng = random.Random(17)
    edges = _build_synthetic_edges(rng)

    with psycopg.connect(test_dsn) as conn:
        _cleanup(conn)
        with conn.cursor() as cur:
            # papers: only bibcode is NOT NULL.
            cur.executemany(
                "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
                [(b,) for b in all_bibs],
            )
            # paper_metrics: one row per bibcode, all community cols NULL.
            cur.executemany(
                "INSERT INTO paper_metrics (bibcode) VALUES (%s) "
                "ON CONFLICT (bibcode) DO NOTHING",
                [(b,) for b in all_bibs],
            )
            # Deduplicate edges to respect citation_edges PK.
            unique_edges = list({(s, t) for s, t in edges if s != t})
            cur.executemany(
                "INSERT INTO citation_edges (source_bibcode, target_bibcode) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                unique_edges,
            )
        conn.commit()

    try:
        yield {"giant": giant, "satellite": satellite}
    finally:
        with psycopg.connect(test_dsn) as conn:
            _cleanup(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_script(
    test_dsn: str,
    seed: int,
    log_dir: Path,
    resolution_coarse: float = 1.0,
    resolution_medium: float = 2.0,
    resolution_fine: float = 5.0,
) -> dict[str, object]:
    """Run the recompute script as a subprocess. Returns parsed run_meta.json.

    The default PRD resolutions (0.001/0.01/0.1) are calibrated for the
    full 32M-paper graph; on a 100-node synthetic graph modularity
    collapses to 1 community at those values. Callers pass test-scale
    resolutions (≥1.0) that actually resolve the 10-cluster structure.
    """
    env = os.environ.copy()
    # The script uses SCIX_TEST_DSN as a default, but pass --dsn explicitly
    # for clarity and to avoid leaking env vars into subprocess.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dsn",
            test_dsn,
            "--resolution-coarse",
            str(resolution_coarse),
            "--resolution-medium",
            str(resolution_medium),
            "--resolution-fine",
            str(resolution_fine),
            "--seed",
            str(seed),
            "--log-dir",
            str(log_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"script exited {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    meta_path = log_dir / "run_meta.json"
    assert meta_path.is_file(), f"run_meta.json not written to {meta_path}"
    return json.loads(meta_path.read_text())


def _fetch_memberships(
    dsn: str, bibcodes: list[str]
) -> dict[str, tuple[int | None, int | None, int | None]]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, community_id_coarse, community_id_medium, "
            "community_id_fine FROM paper_metrics WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        return {row[0]: (row[1], row[2], row[3]) for row in cur.fetchall()}


def _canonical_relabel(mapping: dict[str, int]) -> dict[str, int]:
    """Relabel community IDs in first-seen-smallest order for comparison."""
    seen: dict[int, int] = {}
    out: dict[str, int] = {}
    for bib in sorted(mapping):
        cid = mapping[bib]
        if cid not in seen:
            seen[cid] = len(seen)
        out[bib] = seen[cid]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecomputeCitationCommunities:
    def test_giant_assigned_satellites_null(
        self,
        test_dsn: str,
        synthetic_graph: dict[str, list[str]],
        tmp_path: Path,
    ) -> None:
        """Giant-component papers get non-NULL IDs; satellite papers get NULL."""
        log_dir = tmp_path / "run1"
        run_meta = _invoke_script(test_dsn, seed=42, log_dir=log_dir)

        assert run_meta["giant_component_size"] == len(synthetic_graph["giant"])
        assert run_meta["small_component_node_count"] == len(
            synthetic_graph["satellite"]
        )

        all_bibs = synthetic_graph["giant"] + synthetic_graph["satellite"]
        rows = _fetch_memberships(test_dsn, all_bibs)

        # (a) Every giant bibcode is non-NULL at all three resolutions.
        for bib in synthetic_graph["giant"]:
            coarse, medium, fine = rows[bib]
            assert coarse is not None, f"{bib} coarse must be non-NULL"
            assert medium is not None, f"{bib} medium must be non-NULL"
            assert fine is not None, f"{bib} fine must be non-NULL"

        # (b) Every satellite bibcode is NULL at all three resolutions.
        for bib in synthetic_graph["satellite"]:
            coarse, medium, fine = rows[bib]
            assert coarse is None, f"{bib} coarse must be NULL"
            assert medium is None, f"{bib} medium must be NULL"
            assert fine is None, f"{bib} fine must be NULL"

        # (c/d) Coarse invariant: largest ≤ 10% of giant.
        assert run_meta["invariants"]["largest_coarse_le_10pct"] is True
        assert run_meta["largest_community_size"]["coarse"] <= GIANT_N // 10

    def test_deterministic_rerun_same_seed(
        self,
        test_dsn: str,
        synthetic_graph: dict[str, list[str]],
        tmp_path: Path,
    ) -> None:
        """Re-running with the same seed yields identical memberships (up to relabel)."""
        log_dir_a = tmp_path / "det_a"
        log_dir_b = tmp_path / "det_b"

        _invoke_script(test_dsn, seed=42, log_dir=log_dir_a)
        rows_a = _fetch_memberships(test_dsn, synthetic_graph["giant"])

        _invoke_script(test_dsn, seed=42, log_dir=log_dir_b)
        rows_b = _fetch_memberships(test_dsn, synthetic_graph["giant"])

        for idx in (0, 1, 2):
            map_a = {bib: rows_a[bib][idx] for bib in synthetic_graph["giant"]}
            map_b = {bib: rows_b[bib][idx] for bib in synthetic_graph["giant"]}
            assert _canonical_relabel(map_a) == _canonical_relabel(map_b), (
                f"non-deterministic at resolution index {idx}"
            )
