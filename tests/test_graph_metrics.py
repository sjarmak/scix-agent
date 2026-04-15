"""Tests for graph metrics computation.

Unit tests verify PageRank, HITS, Leiden, calibration, and community quality
metrics using small synthetic graphs. No database required.
"""

from __future__ import annotations

import pytest

# Import guards for optional dependencies
igraph = pytest.importorskip("igraph")
leidenalg = pytest.importorskip("leidenalg")

from scix.graph_metrics import (
    _VALID_RESOLUTIONS,
    assign_small_component_communities,
    calibrate_resolution,
    community_size_stats,
    compare_partitions,
    compute_conductance,
    compute_coverage,
    compute_hits,
    compute_leiden,
    compute_nmi,
    compute_pagerank,
    extract_giant_component,
    extract_taxonomic_community,
    filter_isolated_nodes,
    sweep_resolutions,
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


def _two_clique_with_isolates(clique_size: int = 10, n_isolates: int = 5) -> tuple:
    """Two cliques + isolated nodes. Returns (graph, bibcode_to_id, id_to_bibcode).

    Nodes 0..clique_size-1: Clique A
    Nodes clique_size..2*clique_size-1: Clique B (bridge from 0 to clique_size)
    Nodes 2*clique_size..2*clique_size+n_isolates-1: Isolated (no edges)
    """
    base = _two_clique_graph(clique_size)
    graph = base.copy()
    graph.add_vertices(n_isolates)

    n = graph.vcount()
    bibcode_to_id = {f"BIB{i:04d}": i for i in range(n)}
    id_to_bibcode = {i: f"BIB{i:04d}" for i in range(n)}

    return graph, bibcode_to_id, id_to_bibcode


# ---------------------------------------------------------------------------
# Filter isolated nodes tests
# ---------------------------------------------------------------------------


class TestFilterIsolatedNodes:
    def test_removes_degree_zero_nodes(self) -> None:
        """Isolated nodes should not appear in the subgraph."""
        graph, b2i, i2b = _two_clique_with_isolates(clique_size=5, n_isolates=3)
        subgraph, sub_b2i, sub_i2b, isolates = filter_isolated_nodes(graph, b2i, i2b)

        assert subgraph.vcount() == 10  # 2 cliques of 5
        assert len(sub_b2i) == 10
        assert len(sub_i2b) == 10

    def test_preserves_connected_nodes(self) -> None:
        """All connected nodes should appear in the subgraph with correct edges."""
        graph, b2i, i2b = _two_clique_with_isolates(clique_size=5, n_isolates=3)
        subgraph, sub_b2i, sub_i2b, isolates = filter_isolated_nodes(graph, b2i, i2b)

        # Subgraph should have edges from both cliques + bridge
        # Each clique of 5: 5*4=20 directed edges, bridge: 1
        assert subgraph.ecount() == 41

        # Isolated bibcodes must not appear in subgraph mappings
        for bib in isolates:
            assert bib not in sub_b2i

    def test_returns_isolated_bibcodes(self) -> None:
        """Isolated bibcodes should be tracked in the returned set."""
        graph, b2i, i2b = _two_clique_with_isolates(clique_size=5, n_isolates=3)
        _subgraph, _sub_b2i, _sub_i2b, isolates = filter_isolated_nodes(graph, b2i, i2b)

        expected = {f"BIB{i:04d}" for i in range(10, 13)}
        assert isolates == expected

    def test_no_isolates_returns_same_size(self) -> None:
        """Graph with no isolated nodes returns a subgraph of same size."""
        graph = _two_clique_graph(clique_size=5)
        b2i = {f"BIB{i:04d}": i for i in range(10)}
        i2b = {i: f"BIB{i:04d}" for i in range(10)}
        subgraph, sub_b2i, sub_i2b, isolates = filter_isolated_nodes(graph, b2i, i2b)

        assert subgraph.vcount() == 10
        assert len(isolates) == 0

    def test_all_isolates_returns_empty_graph(self) -> None:
        """Graph of only isolated nodes returns an empty subgraph."""
        graph = igraph.Graph(n=5, edges=[], directed=True)
        b2i = {f"BIB{i:04d}": i for i in range(5)}
        i2b = {i: f"BIB{i:04d}" for i in range(5)}
        subgraph, sub_b2i, sub_i2b, isolates = filter_isolated_nodes(graph, b2i, i2b)

        assert subgraph.vcount() == 0
        assert len(isolates) == 5

    def test_subgraph_id_mapping_is_contiguous(self) -> None:
        """Subgraph vertex IDs should be 0..n-1 with correct bibcode mapping."""
        graph, b2i, i2b = _two_clique_with_isolates(clique_size=5, n_isolates=3)
        _subgraph, sub_b2i, sub_i2b, _isolates = filter_isolated_nodes(graph, b2i, i2b)

        # IDs should be contiguous 0..9
        assert set(sub_i2b.keys()) == set(range(10))
        # Round-trip: every bibcode in sub_b2i maps to an id that maps back
        for bib, vid in sub_b2i.items():
            assert sub_i2b[vid] == bib


class TestLeidenOnFilteredGraph:
    def test_no_singleton_communities_after_filtering(self) -> None:
        """Leiden on filtered graph should not produce singleton communities from isolates."""
        graph, b2i, i2b = _two_clique_with_isolates(clique_size=10, n_isolates=20)
        subgraph, sub_b2i, sub_i2b, isolates = filter_isolated_nodes(graph, b2i, i2b)

        membership = compute_leiden(subgraph, resolution=1.0, seed=42)
        n_communities = len(set(membership))
        # Should find 2 communities (the two cliques), not 22
        assert n_communities == 2
        assert len(membership) == 20  # only connected nodes


# ---------------------------------------------------------------------------
# Giant component extraction helpers
# ---------------------------------------------------------------------------


def _two_clique_with_small_component(clique_size: int = 10, small_size: int = 3) -> tuple:
    """Two large cliques (bridged) + one small disconnected component.

    Returns (graph, bibcode_to_id, id_to_bibcode).

    Layout:
      Nodes 0..clique_size-1: Clique A (giant component part 1)
      Nodes clique_size..2*clique_size-1: Clique B (giant component part 2, bridge from 0)
      Nodes 2*clique_size..2*clique_size+small_size-1: Small component (cycle)
    """
    base = _two_clique_graph(clique_size)
    graph = base.copy()
    graph.add_vertices(small_size)

    # Connect small component as a cycle
    offset = 2 * clique_size
    for i in range(small_size):
        j = (i + 1) % small_size
        graph.add_edge(offset + i, offset + j)
        graph.add_edge(offset + j, offset + i)

    n = graph.vcount()
    bibcode_to_id = {f"BIB{i:04d}": i for i in range(n)}
    id_to_bibcode = {i: f"BIB{i:04d}" for i in range(n)}

    return graph, bibcode_to_id, id_to_bibcode


# ---------------------------------------------------------------------------
# Giant component extraction tests
# ---------------------------------------------------------------------------


class TestExtractGiantComponent:
    def test_extracts_largest_component(self) -> None:
        """Giant component should contain both cliques, not the small component."""
        graph, b2i, i2b = _two_clique_with_small_component(clique_size=10, small_size=3)
        # First filter isolates (there are none here, but follow pipeline flow)
        subgraph, sub_b2i, sub_i2b, _isolates = filter_isolated_nodes(graph, b2i, i2b)

        giant, giant_b2i, giant_i2b, small_bibs = extract_giant_component(
            subgraph, sub_b2i, sub_i2b
        )

        assert giant.vcount() == 20  # two cliques of 10
        assert len(small_bibs) == 3  # small cycle component

    def test_small_component_bibcodes_returned(self) -> None:
        """Small component bibcodes should be returned as a set."""
        graph, b2i, i2b = _two_clique_with_small_component(clique_size=5, small_size=3)
        subgraph, sub_b2i, sub_i2b, _isolates = filter_isolated_nodes(graph, b2i, i2b)

        _giant, _giant_b2i, _giant_i2b, small_bibs = extract_giant_component(
            subgraph, sub_b2i, sub_i2b
        )

        expected = {f"BIB{i:04d}" for i in range(10, 13)}
        assert small_bibs == expected

    def test_giant_component_id_mapping_contiguous(self) -> None:
        """Giant component should have contiguous vertex IDs 0..n-1."""
        graph, b2i, i2b = _two_clique_with_small_component(clique_size=5, small_size=3)
        subgraph, sub_b2i, sub_i2b, _isolates = filter_isolated_nodes(graph, b2i, i2b)

        giant, giant_b2i, giant_i2b, _small_bibs = extract_giant_component(
            subgraph, sub_b2i, sub_i2b
        )

        assert set(giant_i2b.keys()) == set(range(giant.vcount()))
        for bib, vid in giant_b2i.items():
            assert giant_i2b[vid] == bib

    def test_no_small_components_returns_full_graph(self) -> None:
        """If the entire graph is one component, small_bibs should be empty."""
        graph = _two_clique_graph(clique_size=5)
        b2i = {f"BIB{i:04d}": i for i in range(10)}
        i2b = {i: f"BIB{i:04d}" for i in range(10)}

        giant, giant_b2i, giant_i2b, small_bibs = extract_giant_component(graph, b2i, i2b)

        assert giant.vcount() == 10
        assert len(small_bibs) == 0

    def test_leiden_on_giant_component(self) -> None:
        """Leiden on giant component should find 2 communities (the two cliques)."""
        graph, b2i, i2b = _two_clique_with_small_component(clique_size=10, small_size=3)
        subgraph, sub_b2i, sub_i2b, _isolates = filter_isolated_nodes(graph, b2i, i2b)
        giant, giant_b2i, giant_i2b, _small_bibs = extract_giant_component(
            subgraph, sub_b2i, sub_i2b
        )

        membership = compute_leiden(giant, resolution=1.0, seed=42)
        n_communities = len(set(membership))
        assert n_communities == 2
        assert len(membership) == 20


# ---------------------------------------------------------------------------
# Small component community assignment tests
# ---------------------------------------------------------------------------


class TestAssignSmallComponentCommunities:
    def test_assigns_by_nearest_embedding(self) -> None:
        """Papers should be assigned to the community whose centroid is nearest."""
        # Two communities with known centroids
        import numpy as np

        # Community 0 centroid near [1,0,0], community 1 near [0,1,0]
        giant_membership = {
            "BIB0000": 0,
            "BIB0001": 0,
            "BIB0002": 1,
            "BIB0003": 1,
        }
        giant_embeddings = {
            "BIB0000": np.array([1.0, 0.0, 0.0]),
            "BIB0001": np.array([0.9, 0.1, 0.0]),
            "BIB0002": np.array([0.0, 1.0, 0.0]),
            "BIB0003": np.array([0.1, 0.9, 0.0]),
        }
        # Small-component paper near community 0
        small_embeddings = {
            "BIB0010": np.array([0.8, 0.2, 0.0]),
        }

        result = assign_small_component_communities(
            small_bibcodes={"BIB0010"},
            giant_membership=giant_membership,
            giant_embeddings=giant_embeddings,
            small_embeddings=small_embeddings,
        )

        assert result["BIB0010"] == 0

    def test_assigns_to_nearer_community(self) -> None:
        """A paper closer to community 1 should be assigned to community 1."""
        import numpy as np

        giant_membership = {
            "BIB0000": 0,
            "BIB0001": 1,
        }
        giant_embeddings = {
            "BIB0000": np.array([1.0, 0.0]),
            "BIB0001": np.array([0.0, 1.0]),
        }
        small_embeddings = {
            "BIB0010": np.array([0.1, 0.9]),
        }

        result = assign_small_component_communities(
            small_bibcodes={"BIB0010"},
            giant_membership=giant_membership,
            giant_embeddings=giant_embeddings,
            small_embeddings=small_embeddings,
        )

        assert result["BIB0010"] == 1

    def test_missing_embedding_returns_none(self) -> None:
        """Papers without embeddings should get None (unassigned)."""
        import numpy as np

        giant_membership = {"BIB0000": 0}
        giant_embeddings = {"BIB0000": np.array([1.0, 0.0])}
        small_embeddings: dict = {}  # BIB0010 has no embedding

        result = assign_small_component_communities(
            small_bibcodes={"BIB0010"},
            giant_membership=giant_membership,
            giant_embeddings=giant_embeddings,
            small_embeddings=small_embeddings,
        )

        assert result["BIB0010"] is None

    def test_empty_small_bibcodes(self) -> None:
        """No small-component papers should return empty dict."""
        result = assign_small_component_communities(
            small_bibcodes=set(),
            giant_membership={},
            giant_embeddings={},
            small_embeddings={},
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidResolutions:
    def test_valid_resolutions_set(self) -> None:
        assert _VALID_RESOLUTIONS == {"coarse", "medium", "fine"}


# ---------------------------------------------------------------------------
# Taxonomic community extraction tests
# ---------------------------------------------------------------------------


class TestExtractTaxonomicCommunity:
    def test_single_astroph_subcategory(self) -> None:
        assert extract_taxonomic_community(["astro-ph.SR"]) == "astro-ph.SR"

    def test_multiple_astroph_returns_first(self) -> None:
        assert extract_taxonomic_community(["astro-ph.GA", "astro-ph.CO"]) == "astro-ph.GA"

    def test_mixed_categories_prefers_astroph(self) -> None:
        assert extract_taxonomic_community(["gr-qc", "astro-ph.HE"]) == "astro-ph.HE"

    def test_no_astroph_returns_first(self) -> None:
        assert extract_taxonomic_community(["hep-th", "gr-qc"]) == "hep-th"

    def test_empty_list_returns_none(self) -> None:
        assert extract_taxonomic_community([]) is None

    def test_none_returns_none(self) -> None:
        assert extract_taxonomic_community(None) is None

    def test_all_six_subcategories(self) -> None:
        """Each of the 6 astro-ph subcategories should be recognized."""
        for sub in [
            "astro-ph.CO",
            "astro-ph.EP",
            "astro-ph.GA",
            "astro-ph.HE",
            "astro-ph.IM",
            "astro-ph.SR",
        ]:
            assert extract_taxonomic_community([sub]) == sub

    def test_bare_astroph(self) -> None:
        """Plain 'astro-ph' (no subcategory) should still be returned."""
        assert extract_taxonomic_community(["astro-ph"]) == "astro-ph"

    def test_non_astroph_single(self) -> None:
        assert extract_taxonomic_community(["cs.AI"]) == "cs.AI"


# ---------------------------------------------------------------------------
# NMI tests
# ---------------------------------------------------------------------------


class TestComputeNmi:
    def test_identical_labels(self) -> None:
        """NMI of identical partitions should be 1.0."""
        labels = [0, 0, 1, 1, 2, 2]
        assert pytest.approx(compute_nmi(labels, labels), abs=1e-6) == 1.0

    def test_independent_labels(self) -> None:
        """NMI of independent partitions should be close to 0."""
        # Large enough to converge toward 0
        a = [i % 3 for i in range(300)]
        b = [i % 5 for i in range(300)]
        nmi = compute_nmi(a, b)
        assert nmi < 0.1, f"Expected near-zero NMI for independent partitions, got {nmi}"

    def test_symmetric(self) -> None:
        """NMI(A, B) == NMI(B, A)."""
        a = [0, 0, 1, 1, 2, 2]
        b = [0, 1, 1, 2, 2, 0]
        assert pytest.approx(compute_nmi(a, b), abs=1e-10) == compute_nmi(b, a)

    def test_single_cluster(self) -> None:
        """If one partition has a single cluster, NMI should be 0."""
        a = [0, 0, 0, 0]
        b = [0, 1, 2, 3]
        nmi = compute_nmi(a, b)
        assert nmi == pytest.approx(0.0, abs=1e-10)

    def test_empty_labels(self) -> None:
        """Empty label lists should return 0."""
        assert compute_nmi([], []) == 0.0

    def test_returns_float_in_range(self) -> None:
        """NMI should always be in [0, 1]."""
        a = [0, 0, 1, 1, 2, 2, 3, 3]
        b = [0, 1, 0, 1, 2, 3, 2, 3]
        nmi = compute_nmi(a, b)
        assert 0.0 <= nmi <= 1.0


# ---------------------------------------------------------------------------
# Community size stats tests
# ---------------------------------------------------------------------------


class TestCommunitySizeStats:
    def test_basic_stats(self) -> None:
        """Two equal communities of 5 each."""
        membership = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        stats = community_size_stats(membership)
        assert stats["n_communities"] == 2
        assert stats["min_size"] == 5
        assert stats["max_size"] == 5
        assert stats["mean_size"] == 5.0
        assert stats["median_size"] == 5.0
        assert stats["singletons"] == 0

    def test_singleton_communities(self) -> None:
        """Each node in its own community."""
        membership = [0, 1, 2, 3, 4]
        stats = community_size_stats(membership)
        assert stats["n_communities"] == 5
        assert stats["singletons"] == 5
        assert stats["min_size"] == 1
        assert stats["max_size"] == 1

    def test_unequal_communities(self) -> None:
        """One large community and one small one."""
        membership = [0] * 90 + [1] * 10
        stats = community_size_stats(membership)
        assert stats["n_communities"] == 2
        assert stats["max_size"] == 90
        assert stats["min_size"] == 10
        assert stats["pct_in_top10"] == 100.0

    def test_empty_membership(self) -> None:
        """Empty membership list."""
        stats = community_size_stats([])
        assert stats["n_communities"] == 0
        assert stats["singletons"] == 0


# ---------------------------------------------------------------------------
# Conductance tests
# ---------------------------------------------------------------------------


class TestComputeConductance:
    def test_two_cliques_low_conductance(self) -> None:
        """Two well-separated cliques should have low conductance."""
        graph = _two_clique_graph(clique_size=10)
        undirected = graph.as_undirected(mode="collapse")
        membership = compute_leiden(graph, resolution=1.0, seed=42)
        conductance = compute_conductance(undirected, membership)
        # Well-separated cliques => very low mean conductance
        assert conductance["mean"] < 0.1

    def test_single_community_zero_conductance(self) -> None:
        """A single community covering the whole graph has 0 conductance."""
        graph = _two_clique_graph(clique_size=5)
        undirected = graph.as_undirected(mode="collapse")
        membership = [0] * 10
        conductance = compute_conductance(undirected, membership)
        # Only one community => no inter-community edges => conductance = 0
        assert conductance["mean"] == pytest.approx(0.0)

    def test_conductance_keys(self) -> None:
        """Result should contain expected keys."""
        graph = _two_clique_graph(clique_size=5)
        undirected = graph.as_undirected(mode="collapse")
        membership = compute_leiden(graph, resolution=1.0, seed=42)
        conductance = compute_conductance(undirected, membership)
        assert "mean" in conductance
        assert "median" in conductance
        assert "max" in conductance
        assert "n_communities" in conductance


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_two_cliques_high_coverage(self) -> None:
        """Two well-separated cliques should have high coverage (most edges intra-community)."""
        graph = _two_clique_graph(clique_size=10)
        undirected = graph.as_undirected(mode="collapse")
        membership = compute_leiden(graph, resolution=1.0, seed=42)
        coverage = compute_coverage(undirected, membership)
        # Only 1 inter-community edge out of many
        assert coverage > 0.95

    def test_all_singleton_communities_zero_coverage(self) -> None:
        """If every node is its own community, no intra-community edges exist."""
        graph = _two_clique_graph(clique_size=5)
        undirected = graph.as_undirected(mode="collapse")
        membership = list(range(10))
        coverage = compute_coverage(undirected, membership)
        assert coverage == pytest.approx(0.0)

    def test_single_community_full_coverage(self) -> None:
        """One community covering everything has coverage = 1."""
        graph = _two_clique_graph(clique_size=5)
        undirected = graph.as_undirected(mode="collapse")
        membership = [0] * 10
        coverage = compute_coverage(undirected, membership)
        assert coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sweep resolutions tests
# ---------------------------------------------------------------------------


class TestSweepResolutions:
    def test_returns_results_for_each_resolution(self) -> None:
        """Sweep should return one result dict per resolution value."""
        graph = _two_clique_graph(clique_size=10)
        resolutions = [0.01, 0.1, 1.0]
        results = sweep_resolutions(graph, resolutions=resolutions, seed=42)
        assert len(results) == 3
        assert [r["resolution"] for r in results] == resolutions

    def test_result_keys(self) -> None:
        """Each result should contain all expected quality metric keys."""
        graph = _two_clique_graph(clique_size=10)
        results = sweep_resolutions(graph, resolutions=[1.0], seed=42)
        result = results[0]
        expected_keys = {
            "resolution",
            "n_communities",
            "membership",
            "size_stats",
            "conductance",
            "coverage",
        }
        assert expected_keys.issubset(result.keys())

    def test_higher_resolution_more_communities(self) -> None:
        """Higher resolutions should yield at least as many communities."""
        graph = _two_clique_graph(clique_size=15)
        results = sweep_resolutions(graph, resolutions=[0.01, 1.0, 10.0], seed=42)
        n_comms = [r["n_communities"] for r in results]
        for i in range(len(n_comms) - 1):
            assert n_comms[i + 1] >= n_comms[i], (
                f"Resolution {results[i + 1]['resolution']} yielded {n_comms[i + 1]} "
                f"communities, expected >= {n_comms[i]}"
            )

    def test_cpm_partition_type(self) -> None:
        """Sweep with CPM partition type should run without error."""
        graph = _two_clique_graph(clique_size=10)
        results = sweep_resolutions(
            graph,
            resolutions=[0.01, 0.1],
            partition_type="CPM",
            seed=42,
        )
        assert len(results) == 2
        for r in results:
            assert r["n_communities"] >= 1

    def test_cpm_detects_two_cliques(self) -> None:
        """CPM on two cliques should find 2 communities at appropriate resolution."""
        graph = _two_clique_graph(clique_size=10)
        # For CPM, resolution is edge density threshold.
        # Internal edges in a 10-clique: 9*10/2 = 45 for undirected, density ~ 1.0
        # So a resolution near 0.5 should separate two cliques joined by 1 bridge.
        results = sweep_resolutions(
            graph,
            resolutions=[0.5],
            partition_type="CPM",
            seed=42,
        )
        assert results[0]["n_communities"] == 2

    def test_modularity_partition_type(self) -> None:
        """Sweep with modularity (RB) partition type should be the default."""
        graph = _two_clique_graph(clique_size=10)
        results_default = sweep_resolutions(graph, resolutions=[1.0], seed=42)
        results_explicit = sweep_resolutions(
            graph, resolutions=[1.0], partition_type="modularity", seed=42
        )
        # Same partition type -> same result
        assert results_default[0]["n_communities"] == results_explicit[0]["n_communities"]

    def test_invalid_partition_type_raises(self) -> None:
        """Invalid partition type should raise ValueError."""
        graph = _two_clique_graph(clique_size=5)
        with pytest.raises(ValueError, match="partition_type"):
            sweep_resolutions(graph, resolutions=[1.0], partition_type="invalid")

    def test_empty_resolutions_returns_empty(self) -> None:
        """Empty resolution list should return empty results."""
        graph = _two_clique_graph(clique_size=5)
        results = sweep_resolutions(graph, resolutions=[], seed=42)
        assert results == []

    def test_coverage_in_expected_range(self) -> None:
        """Coverage should be between 0 and 1 for all sweep results."""
        graph = _two_clique_graph(clique_size=10)
        results = sweep_resolutions(graph, resolutions=[0.1, 1.0, 5.0], seed=42)
        for r in results:
            assert 0.0 <= r["coverage"] <= 1.0

    def test_conductance_keys_present(self) -> None:
        """Conductance summary should have the standard keys."""
        graph = _two_clique_graph(clique_size=10)
        results = sweep_resolutions(graph, resolutions=[1.0], seed=42)
        cond = results[0]["conductance"]
        assert "mean" in cond
        assert "median" in cond
        assert "n_communities" in cond

    def test_reference_labels_produce_nmi(self) -> None:
        """When reference_labels are provided, NMI should be computed."""
        graph = _two_clique_graph(clique_size=10)
        # Reference: nodes 0-9 = class 0, nodes 10-19 = class 1
        ref_labels = [0] * 10 + [1] * 10
        results = sweep_resolutions(graph, resolutions=[1.0], seed=42, reference_labels=ref_labels)
        assert "nmi" in results[0]
        # Two cliques should have high NMI with correct reference
        assert results[0]["nmi"] > 0.8

    def test_no_reference_labels_no_nmi(self) -> None:
        """Without reference_labels, NMI should not be in results."""
        graph = _two_clique_graph(clique_size=10)
        results = sweep_resolutions(graph, resolutions=[1.0], seed=42)
        assert "nmi" not in results[0]


# ---------------------------------------------------------------------------
# Compare partitions tests
# ---------------------------------------------------------------------------


class TestComparePartitions:
    def test_pairwise_nmi_matrix(self) -> None:
        """Should return an NxN matrix of NMI values for N partitions."""
        partitions = {
            "coarse": [0, 0, 0, 1, 1, 1],
            "fine": [0, 0, 1, 2, 2, 3],
        }
        result = compare_partitions(partitions)
        assert "nmi_matrix" in result
        matrix = result["nmi_matrix"]
        # 2 partitions -> 2x2 matrix
        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix.values())

    def test_diagonal_is_one(self) -> None:
        """Self-NMI should be 1.0."""
        partitions = {
            "a": [0, 0, 1, 1],
            "b": [0, 1, 0, 1],
        }
        result = compare_partitions(partitions)
        for name, row in result["nmi_matrix"].items():
            assert row[name] == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self) -> None:
        """NMI(a, b) should equal NMI(b, a)."""
        partitions = {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [0, 1, 1, 2, 2, 0],
        }
        result = compare_partitions(partitions)
        matrix = result["nmi_matrix"]
        assert matrix["a"]["b"] == pytest.approx(matrix["b"]["a"], abs=1e-10)

    def test_partition_names_in_result(self) -> None:
        """Result should list partition names."""
        partitions = {
            "coarse": [0, 0, 1, 1],
            "medium": [0, 1, 2, 3],
            "fine": [0, 1, 2, 3],
        }
        result = compare_partitions(partitions)
        assert result["partition_names"] == ["coarse", "medium", "fine"]

    def test_community_counts_in_result(self) -> None:
        """Result should include community counts per partition."""
        partitions = {
            "a": [0, 0, 0, 1],
            "b": [0, 1, 2, 3],
        }
        result = compare_partitions(partitions)
        assert result["community_counts"]["a"] == 2
        assert result["community_counts"]["b"] == 4

    def test_single_partition(self) -> None:
        """A single partition should produce a 1x1 NMI matrix of 1.0."""
        partitions = {"only": [0, 0, 1, 1]}
        result = compare_partitions(partitions)
        assert result["nmi_matrix"]["only"]["only"] == pytest.approx(1.0)

    def test_empty_partitions_raises(self) -> None:
        """Empty partitions dict should raise ValueError."""
        with pytest.raises(ValueError, match="partitions"):
            compare_partitions({})
