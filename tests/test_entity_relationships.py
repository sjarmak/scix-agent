"""Unit tests for src/scix/entity_relationships.py — hierarchy parsers.

Pure-function tests (no DB). The extractors consume the entity rows we
already store in public.entities and emit edge tuples that the population
script bulk-inserts into public.entity_relationships.
"""

from __future__ import annotations

import pytest

from scix.entity_relationships import (
    CURATED_FLAGSHIP_INSTRUMENTS,
    EdgeCandidate,
    extract_curated_flagship_edges,
    extract_gcmd_edges,
    extract_spase_region_edges,
    extract_ssodnet_class_edges,
    parse_gcmd_hierarchy,
    parse_spase_region_path,
    parse_sso_class_path,
)

# ---------------------------------------------------------------------------
# parse_gcmd_hierarchy
# ---------------------------------------------------------------------------


class TestParseGcmdHierarchy:
    def test_splits_on_delimiter(self) -> None:
        path = parse_gcmd_hierarchy(
            "Science Keywords > EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > ANISOTROPY"
        )
        assert path == [
            "Science Keywords",
            "EARTH SCIENCE",
            "ATMOSPHERE",
            "ATMOSPHERIC RADIATION",
            "ANISOTROPY",
        ]

    def test_strips_whitespace(self) -> None:
        path = parse_gcmd_hierarchy("A  >  B >C")
        assert path == ["A", "B", "C"]

    def test_drops_empty_segments(self) -> None:
        # Real GCMD payloads never produce these, but defensive parsing
        # matters for scraped input
        assert parse_gcmd_hierarchy("A >  > B") == ["A", "B"]

    def test_empty_input_returns_empty(self) -> None:
        assert parse_gcmd_hierarchy("") == []
        assert parse_gcmd_hierarchy(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_spase_region_path
# ---------------------------------------------------------------------------


class TestParseSpaseRegion:
    def test_dot_separated(self) -> None:
        assert parse_spase_region_path("Jupiter.Io") == ["Jupiter", "Io"]
        assert parse_spase_region_path("Heliosphere.NearEarth") == [
            "Heliosphere",
            "NearEarth",
        ]

    def test_single_segment(self) -> None:
        assert parse_spase_region_path("Asteroid") == ["Asteroid"]

    def test_empty(self) -> None:
        assert parse_spase_region_path("") == []


# ---------------------------------------------------------------------------
# parse_sso_class_path
# ---------------------------------------------------------------------------


class TestParseSsoClass:
    def test_angle_separated(self) -> None:
        assert parse_sso_class_path("NEA>Apollo>PHA") == ["NEA", "Apollo", "PHA"]
        assert parse_sso_class_path("MB>Inner") == ["MB", "Inner"]

    def test_single_segment_keeps_as_root(self) -> None:
        assert parse_sso_class_path("Hungaria") == ["Hungaria"]

    def test_strips_whitespace(self) -> None:
        assert parse_sso_class_path(" NEA > Apollo ") == ["NEA", "Apollo"]


# ---------------------------------------------------------------------------
# extract_gcmd_edges
# ---------------------------------------------------------------------------


class TestExtractGcmdEdges:
    def test_emits_adjacent_parent_child_edges(self) -> None:
        # Fake row: (entity_id, canonical_name, scheme, hierarchy_str)
        rows = [
            (1, "Science Keywords", "sciencekeywords", "Science Keywords"),
            (2, "EARTH SCIENCE", "sciencekeywords", "Science Keywords > EARTH SCIENCE"),
            (
                3,
                "ATMOSPHERE",
                "sciencekeywords",
                "Science Keywords > EARTH SCIENCE > ATMOSPHERE",
            ),
        ]
        # Name -> id lookup
        by_name = {
            ("Science Keywords", "sciencekeywords"): 1,
            ("EARTH SCIENCE", "sciencekeywords"): 2,
            ("ATMOSPHERE", "sciencekeywords"): 3,
        }

        edges = list(extract_gcmd_edges(rows, by_name))
        # Expect (1->2), (2->3) parent_of edges
        edge_pairs = [(e.subject_id, e.object_id, e.predicate) for e in edges]
        assert (1, 2, "parent_of") in edge_pairs
        assert (2, 3, "parent_of") in edge_pairs

    def test_missing_parent_skipped(self) -> None:
        # Only the leaf exists; parent names don't map to entity ids
        rows = [
            (
                3,
                "ATMOSPHERE",
                "sciencekeywords",
                "Science Keywords > EARTH SCIENCE > ATMOSPHERE",
            )
        ]
        by_name = {("ATMOSPHERE", "sciencekeywords"): 3}
        edges = list(extract_gcmd_edges(rows, by_name))
        assert edges == []

    def test_includes_evidence_path(self) -> None:
        rows = [
            (1, "Science Keywords", "sciencekeywords", "Science Keywords"),
            (2, "EARTH SCIENCE", "sciencekeywords", "Science Keywords > EARTH SCIENCE"),
        ]
        by_name = {
            ("Science Keywords", "sciencekeywords"): 1,
            ("EARTH SCIENCE", "sciencekeywords"): 2,
        }
        edges = list(extract_gcmd_edges(rows, by_name))
        assert edges
        assert "gcmd_hierarchy" in edges[0].evidence["method"]
        assert edges[0].evidence["scheme"] == "sciencekeywords"

    def test_self_loops_skipped(self) -> None:
        # If an entity row's final segment matches itself and parent path
        # somehow resolves to the same id, we must not emit self-loops
        rows = [(1, "A", "x", "A > A")]
        by_name = {("A", "x"): 1}
        edges = list(extract_gcmd_edges(rows, by_name))
        assert all(e.subject_id != e.object_id for e in edges)


# ---------------------------------------------------------------------------
# extract_spase_region_edges
# ---------------------------------------------------------------------------


class TestExtractSpaseRegion:
    def test_parent_from_dot_prefix(self) -> None:
        rows = [
            (10, "Jupiter"),
            (11, "Jupiter.Io"),
            (12, "Jupiter.Europa"),
        ]
        by_name = {"Jupiter": 10, "Jupiter.Io": 11, "Jupiter.Europa": 12}
        edges = list(extract_spase_region_edges(rows, by_name))
        pairs = {(e.subject_id, e.object_id) for e in edges}
        assert (10, 11) in pairs
        assert (10, 12) in pairs
        # predicate should be parent_of
        assert all(e.predicate == "parent_of" for e in edges)

    def test_no_parent_skipped(self) -> None:
        rows = [(11, "Jupiter.Io")]
        by_name = {"Jupiter.Io": 11}
        # "Jupiter" is not in by_name -> no edge
        assert list(extract_spase_region_edges(rows, by_name)) == []

    def test_top_level_regions_have_no_edge(self) -> None:
        rows = [(10, "Jupiter")]
        by_name = {"Jupiter": 10}
        assert list(extract_spase_region_edges(rows, by_name)) == []


# ---------------------------------------------------------------------------
# extract_ssodnet_class_edges
# ---------------------------------------------------------------------------


class TestExtractSsodnetClassEdges:
    def test_generates_taxon_edges(self) -> None:
        # Two asteroid rows with classes
        rows = [
            (100, "Ceres", "MB>Middle"),
            (101, "Apophis", "NEA>Apollo>PHA"),
        ]

        edges, taxa = extract_ssodnet_class_edges(rows, include_targets=False)

        # Taxa to create: MB, MB>Middle, NEA, NEA>Apollo, NEA>Apollo>PHA
        taxon_names = {t.canonical_name for t in taxa}
        assert taxon_names == {
            "MB",
            "MB>Middle",
            "NEA",
            "NEA>Apollo",
            "NEA>Apollo>PHA",
        }

        # Class->class edges: MB->MB>Middle, NEA->NEA>Apollo, NEA>Apollo->NEA>Apollo>PHA
        taxon_edges = {(e.subject_name, e.object_name) for e in edges}
        assert ("MB", "MB>Middle") in taxon_edges
        assert ("NEA", "NEA>Apollo") in taxon_edges
        assert ("NEA>Apollo", "NEA>Apollo>PHA") in taxon_edges

    def test_include_targets_adds_part_of_edges(self) -> None:
        rows = [(100, "Ceres", "MB>Middle")]
        edges, _ = extract_ssodnet_class_edges(rows, include_targets=True)
        # Class->class edge + asteroid part_of leaf-class
        predicates = {e.predicate for e in edges}
        assert "part_of" in predicates

    def test_deduplicates_identical_classes(self) -> None:
        rows = [
            (100, "Ceres", "MB>Middle"),
            (101, "Pallas", "MB>Middle"),
        ]
        edges, taxa = extract_ssodnet_class_edges(rows, include_targets=False)
        # Two asteroids, one shared class; taxa should have 2 entries
        assert len(taxa) == 2
        # Class->class edges (only MB->MB>Middle) — no duplicates
        taxon_class_edges = [e for e in edges if e.predicate == "parent_of"]
        assert len(taxon_class_edges) == 1

    def test_skips_blank_class(self) -> None:
        rows = [(100, "X", ""), (101, "Y", None)]
        edges, taxa = extract_ssodnet_class_edges(rows, include_targets=False)
        assert edges == []
        assert taxa == []


# ---------------------------------------------------------------------------
# extract_curated_flagship_edges
# ---------------------------------------------------------------------------


class TestCuratedFlagshipEdges:
    def test_known_missions_have_instrument_children(self) -> None:
        # Sanity: table has entries for well-known flagships
        assert "James Webb Space Telescope" in CURATED_FLAGSHIP_INSTRUMENTS
        assert "Hubble Space Telescope" in CURATED_FLAGSHIP_INSTRUMENTS
        jwst_instruments = CURATED_FLAGSHIP_INSTRUMENTS["James Webb Space Telescope"]
        assert "NIRSpec" in jwst_instruments
        assert "MIRI" in jwst_instruments

    def test_emits_has_instrument_edges_when_both_exist(self) -> None:
        # Minimal fixture: JWST mission exists, two instruments exist
        missions_by_name = {"James Webb Space Telescope": 500}
        instruments_by_name = {"NIRSpec": 600, "MIRI": 601}

        edges = list(extract_curated_flagship_edges(missions_by_name, instruments_by_name))
        pairs = {(e.subject_id, e.object_id, e.predicate) for e in edges}
        assert (500, 600, "has_instrument") in pairs
        assert (500, 601, "has_instrument") in pairs

    def test_skips_when_mission_missing(self) -> None:
        # Instrument exists but mission does not — no edge
        edges = list(
            extract_curated_flagship_edges(
                missions_by_name={},
                instruments_by_name={"NIRSpec": 600},
            )
        )
        assert edges == []

    def test_skips_when_instrument_missing(self) -> None:
        edges = list(
            extract_curated_flagship_edges(
                missions_by_name={"James Webb Space Telescope": 500},
                instruments_by_name={},
            )
        )
        assert edges == []


# ---------------------------------------------------------------------------
# EdgeCandidate dataclass
# ---------------------------------------------------------------------------


class TestEdgeCandidate:
    def test_frozen(self) -> None:
        e = EdgeCandidate(
            subject_id=1,
            object_id=2,
            predicate="parent_of",
            source="gcmd",
            evidence={"method": "test"},
        )
        with pytest.raises((AttributeError, Exception)):
            e.subject_id = 99  # type: ignore[misc]
