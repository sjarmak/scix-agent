"""Tests for graph metrics computation.

Unit tests verify PageRank, HITS, Leiden, and calibration using small synthetic
graphs. No database required.
"""

from __future__ import annotations

import pytest

# Import guards for optional dependencies
igraph = pytest.importorskip("igraph")
leidenalg = pytest.importorskip("leidenalg")

from scix.graph_metrics import (
    _VALID_RESOLUTIONS,
    calibrate_resolution,
    compute_hits,
    compute_leiden,
    compute_pagerank,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _star_graph(n: int = 5) -> igraph.Graph:
    """Create a directed star graph: node 0 is the center, edges from 1..n-1 to 0."""
    edges = [(i, 0) for i in range(1, n)]
    return igraph.Graph(n=n, edges=edges, directed=True)


def _two_clique_graph(clique_size: int = 10) -> igraph.Graph:
    """Create two dense cliques connected by a single edge.

    Clique A: nodes 0..clique_size-1 (fully connected, directed both ways)
    Clique B: nodes clique_size..2*clique_size-1 (fully connected, directed both ways)
    Bridge: single edge from node 0 to node clique_size
    """
    n = 2 * clique_size
    edges: list[tuple[int, int]] = []

    # Clique A
    for i in range(clique_size):
        for j in range(clique_size):
            if i != j:
                edges.append((i, j))

    # Clique B
    for i in range(clique_size, n):
        for j in range(clique_size, n):
            if i != j:
                edges.append((i, j))

    # Single bridge
    edges.append((0, clique_size))

    return igraph.Graph(n=n, edges=edges, directed=True)


# ---------------------------------------------------------------------------
# PageRank tests
# ---------------------------------------------------------------------------


class TestComputePagerank:
    def test_pagerank_sums_to_one(self) -> None:
        graph = _star_graph(5)
        scores = compute_pagerank(graph)
        assert len(scores) == 5
        assert pytest.approx(sum(scores), abs=0.01) == 1.0

    def test_center_has_highest_pagerank(self) -> None:
        graph = _star_graph(6)
        scores = compute_pagerank(graph)
        center_score = scores[0]
        for i in range(1, 6):
            assert center_score > scores[i], (
                f"Center node (score={center_score}) should have higher "
                f"PageRank than node {i} (score={scores[i]})"
            )

    def test_pagerank_all_positive(self) -> None:
        graph = _star_graph(5)
        scores = compute_pagerank(graph)
        for s in scores:
            assert s >= 0.0

    def test_single_node_graph(self) -> None:
        graph = igraph.Graph(n=1, edges=[], directed=True)
        scores = compute_pagerank(graph)
        assert len(scores) == 1
        assert pytest.approx(scores[0], abs=0.01) == 1.0


# ---------------------------------------------------------------------------
# HITS tests
# ---------------------------------------------------------------------------


class TestComputeHits:
    def test_hits_returns_correct_length(self) -> None:
        graph = _star_graph(5)
        hubs, authorities = compute_hits(graph)
        assert len(hubs) == 5
        assert len(authorities) == 5

    def test_hits_non_negative(self) -> None:
        graph = _star_graph(5)
        hubs, authorities = compute_hits(graph)
        for h in hubs:
            assert h >= 0.0
        for a in authorities:
            assert a >= 0.0

    def test_center_is_top_authority(self) -> None:
        """In a star graph where all leaves point to center, center is the authority."""
        graph = _star_graph(6)
        _hubs, authorities = compute_hits(graph)
        center_auth = authorities[0]
        for i in range(1, 6):
            assert center_auth >= authorities[i]

    def test_leaves_are_hubs(self) -> None:
        """In a star graph where leaves point to center, leaves are hubs."""
        graph = _star_graph(6)
        hubs, _authorities = compute_hits(graph)
        # Leaves should have higher hub scores than center (center has no outgoing edges)
        for i in range(1, 6):
            assert hubs[i] >= hubs[0]


# ---------------------------------------------------------------------------
# Leiden tests
# ---------------------------------------------------------------------------


class TestComputeLeiden:
    def test_two_cliques_found(self) -> None:
        """Two well-separated cliques connected by one edge should yield 2 communities."""
        graph = _two_clique_graph(clique_size=10)
        membership = compute_leiden(graph, resolution=1.0, seed=42)
        n_communities = len(set(membership))
        assert (
            n_communities == 2
        ), f"Expected 2 communities for two-clique graph, got {n_communities}"

    def test_membership_length_matches_nodes(self) -> None:
        graph = _two_clique_graph(clique_size=8)
        membership = compute_leiden(graph, resolution=1.0, seed=42)
        assert len(membership) == 16

    def test_same_clique_same_community(self) -> None:
        """All nodes within a clique should be in the same community."""
        graph = _two_clique_graph(clique_size=10)
        membership = compute_leiden(graph, resolution=1.0, seed=42)

        # Nodes 0..9 should share a community
        clique_a_communities = set(membership[:10])
        assert (
            len(clique_a_communities) == 1
        ), f"Clique A should be in one community, found {clique_a_communities}"

        # Nodes 10..19 should share a community
        clique_b_communities = set(membership[10:])
        assert (
            len(clique_b_communities) == 1
        ), f"Clique B should be in one community, found {clique_b_communities}"

    def test_different_cliques_different_communities(self) -> None:
        graph = _two_clique_graph(clique_size=10)
        membership = compute_leiden(graph, resolution=1.0, seed=42)
        assert (
            membership[0] != membership[10]
        ), "Nodes from different cliques should be in different communities"

    def test_deterministic_with_seed(self) -> None:
        graph = _two_clique_graph(clique_size=10)
        m1 = compute_leiden(graph, resolution=1.0, seed=42)
        m2 = compute_leiden(graph, resolution=1.0, seed=42)
        assert m1 == m2

    def test_higher_resolution_more_communities(self) -> None:
        """Higher resolution should yield at least as many communities."""
        graph = _two_clique_graph(clique_size=15)
        m_low = compute_leiden(graph, resolution=0.1, seed=42)
        m_high = compute_leiden(graph, resolution=10.0, seed=42)
        assert len(set(m_high)) >= len(set(m_low))


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibrateResolution:
    def test_finds_target_within_tolerance(self) -> None:
        """Calibration should find a resolution giving ~2 communities for two-clique graph."""
        graph = _two_clique_graph(clique_size=10)
        res = calibrate_resolution(
            graph,
            target_communities=2,
            tolerance=0.3,
            low=0.001,
            high=10.0,
            max_iter=8,
            seed=42,
        )
        membership = compute_leiden(graph, resolution=res, seed=42)
        n_communities = len(set(membership))
        assert (
            1 <= n_communities <= 3
        ), f"Expected ~2 communities (tolerance=0.3), got {n_communities} at res={res}"

    def test_returns_float(self) -> None:
        graph = _two_clique_graph(clique_size=10)
        res = calibrate_resolution(graph, target_communities=2, max_iter=3, seed=42)
        assert isinstance(res, float)
        assert res > 0

    def test_max_iter_respected(self) -> None:
        """Should return a result even if it doesn't converge."""
        graph = _star_graph(5)
        # Asking for 100 communities from a 5-node graph is impossible,
        # but should not raise
        res = calibrate_resolution(
            graph,
            target_communities=100,
            max_iter=3,
            seed=42,
        )
        assert isinstance(res, float)
        assert res > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidResolutions:
    def test_valid_resolutions_set(self) -> None:
        assert _VALID_RESOLUTIONS == {"coarse", "medium", "fine"}
