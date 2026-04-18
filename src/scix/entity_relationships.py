"""Extract structural entity relationships from source hierarchies.

The public.entities table stores hierarchy metadata inside the
``properties`` JSONB column (e.g. ``gcmd_hierarchy`` ``"A > B > C"``,
``sso_class`` ``"NEA>Apollo>PHA"``, SPASE ``ObservedRegion`` canonical
names like ``Jupiter.Io``).  This module turns those flat strings into
directed edges suitable for insertion into ``entity_relationships``.

Predicate vocabulary used here:

* ``parent_of`` — taxonomic / hierarchical containment (GCMD, SPASE
  region, SsODNet class tree).  Subject is strictly above the object.
* ``part_of`` — asteroid belongs to a leaf SsODNet class.  Subject is
  the member, object is the class.  (Opt-in only.)
* ``has_instrument`` — curated flagship mission to its flagship
  instrument (e.g. JWST -> NIRSpec).

All extractors are pure functions that emit :class:`EdgeCandidate`
tuples.  The DB writer lives in ``scripts/populate_entity_relationships.py``
so this module stays DB-free and trivially unit-testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge / taxon dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeCandidate:
    """One directed entity-to-entity edge ready for DB insertion.

    ``subject_id`` / ``object_id`` are integer FK references into
    ``entities.id``.  ``subject_name`` / ``object_name`` are present for
    SsODNet synthetic taxa where ids are not yet known at extraction
    time — the populate script resolves them to ids after taxon upsert.
    """

    predicate: str
    source: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    subject_id: int | None = None
    object_id: int | None = None
    subject_name: str | None = None
    object_name: str | None = None


@dataclass(frozen=True)
class SyntheticTaxon:
    """A SsODNet taxonomic class that needs to exist as an entity.

    The populate script upserts these into ``entities`` under
    ``source='ssodnet'`` / ``entity_type='taxon'`` before resolving
    class->class edges.
    """

    canonical_name: str
    depth: int


# ---------------------------------------------------------------------------
# GCMD
# ---------------------------------------------------------------------------


GCMD_DELIMITER = ">"


def parse_gcmd_hierarchy(path: str | None) -> list[str]:
    """Split a GCMD hierarchy string like ``"A > B > C"`` into segments.

    Whitespace around the delimiter is stripped; empty segments (e.g.
    from defensive input) are dropped.  Returns an empty list for
    ``None`` or an empty string.
    """
    if not path:
        return []
    return [seg.strip() for seg in path.split(GCMD_DELIMITER) if seg.strip()]


def extract_gcmd_edges(
    rows: Iterable[tuple[int, str, str, str | None]],
    by_name: Mapping[tuple[str, str], int],
) -> Iterator[EdgeCandidate]:
    """Emit ``parent_of`` edges from GCMD hierarchy paths.

    Parameters
    ----------
    rows
        Iterable of ``(entity_id, canonical_name, gcmd_scheme, gcmd_hierarchy)``.
    by_name
        Mapping ``(canonical_name, gcmd_scheme) -> entity_id`` for
        resolving parent segments to entity ids.  Only edges where
        both endpoints resolve are emitted.
    """
    emitted: set[tuple[int, int]] = set()
    for _leaf_id, _canonical, scheme, hierarchy in rows:
        segments = parse_gcmd_hierarchy(hierarchy)
        if len(segments) < 2:
            continue
        for parent_seg, child_seg in zip(segments, segments[1:]):
            parent_id = by_name.get((parent_seg, scheme))
            child_id = by_name.get((child_seg, scheme))
            if parent_id is None or child_id is None:
                continue
            if parent_id == child_id:
                # Defensive: never emit self-loops
                continue
            key = (parent_id, child_id)
            if key in emitted:
                continue
            emitted.add(key)
            yield EdgeCandidate(
                subject_id=parent_id,
                object_id=child_id,
                predicate="parent_of",
                source="gcmd",
                evidence={
                    "method": "gcmd_hierarchy_path",
                    "scheme": scheme,
                    "parent_segment": parent_seg,
                    "child_segment": child_seg,
                },
            )


# ---------------------------------------------------------------------------
# SPASE ObservedRegion (dot notation)
# ---------------------------------------------------------------------------


SPASE_DELIMITER = "."


def parse_spase_region_path(name: str) -> list[str]:
    """Split SPASE region canonical name on ``.``.

    Example: ``"Jupiter.Io" -> ["Jupiter", "Io"]``.
    """
    if not name:
        return []
    return [seg for seg in name.split(SPASE_DELIMITER) if seg]


def extract_spase_region_edges(
    rows: Iterable[tuple[int, str]],
    by_name: Mapping[str, int],
) -> Iterator[EdgeCandidate]:
    """Emit ``parent_of`` edges from SPASE ObservedRegion dot notation.

    Parameters
    ----------
    rows
        Iterable of ``(entity_id, canonical_name)`` for SPASE observable
        entities tagged as ObservedRegion.
    by_name
        Mapping ``canonical_name -> entity_id`` for the full SPASE
        region set (so we can look up the dot-prefixed parent).
    """
    emitted: set[tuple[int, int]] = set()
    for leaf_id, canonical in rows:
        segments = parse_spase_region_path(canonical)
        if len(segments) < 2:
            continue
        parent_name = SPASE_DELIMITER.join(segments[:-1])
        parent_id = by_name.get(parent_name)
        if parent_id is None or parent_id == leaf_id:
            continue
        key = (parent_id, leaf_id)
        if key in emitted:
            continue
        emitted.add(key)
        yield EdgeCandidate(
            subject_id=parent_id,
            object_id=leaf_id,
            predicate="parent_of",
            source="spase",
            evidence={
                "method": "spase_region_dot_notation",
                "parent_name": parent_name,
                "child_name": canonical,
            },
        )


# ---------------------------------------------------------------------------
# SsODNet sso_class (asteroid family tree)
# ---------------------------------------------------------------------------


SSO_DELIMITER = ">"


def parse_sso_class_path(path: str | None) -> list[str]:
    """Split an SsODNet ``sso_class`` value like ``"NEA>Apollo>PHA"``."""
    if not path:
        return []
    return [seg.strip() for seg in path.split(SSO_DELIMITER) if seg.strip()]


def extract_ssodnet_class_edges(
    rows: Iterable[tuple[int, str, str | None]],
    *,
    include_targets: bool,
) -> tuple[list[EdgeCandidate], list[SyntheticTaxon]]:
    """Derive SsODNet class-tree edges and the synthetic taxa they need.

    Parameters
    ----------
    rows
        Iterable of ``(entity_id, canonical_name, sso_class)`` for
        SsODNet ``entity_type='target'`` entities.
    include_targets
        When True, also emit a ``part_of`` edge from each asteroid to
        its leaf taxonomic class.  Produces O(N_asteroids) edges — keep
        True for full-coverage runs.

    Returns
    -------
    (edges, taxa)
        ``edges`` is the list of taxon->taxon parent_of edges (and
        optional asteroid->leaf part_of edges).  Taxon endpoints use
        ``subject_name``/``object_name`` since ids must be resolved
        post-upsert.  ``taxa`` is the deduplicated set of class nodes
        the caller must upsert into ``entities``.
    """
    taxa_seen: dict[str, int] = {}  # canonical -> depth
    class_edges_seen: set[tuple[str, str]] = set()
    edges: list[EdgeCandidate] = []

    for entity_id, canonical, sso_class in rows:
        segments = parse_sso_class_path(sso_class)
        if not segments:
            continue

        # Accumulate synthetic taxa for every prefix path
        # e.g. for NEA>Apollo>PHA -> "NEA", "NEA>Apollo", "NEA>Apollo>PHA"
        prefix_names: list[str] = []
        for depth, _seg in enumerate(segments, start=1):
            name = SSO_DELIMITER.join(segments[:depth])
            prefix_names.append(name)
            # Record the shallowest depth we've seen this taxon at
            if name not in taxa_seen or depth < taxa_seen[name]:
                taxa_seen[name] = depth

        # Class->class parent_of edges (one per adjacent prefix pair)
        for parent_name, child_name in zip(prefix_names, prefix_names[1:]):
            key = (parent_name, child_name)
            if key in class_edges_seen:
                continue
            class_edges_seen.add(key)
            edges.append(
                EdgeCandidate(
                    subject_name=parent_name,
                    object_name=child_name,
                    predicate="parent_of",
                    source="ssodnet",
                    evidence={
                        "method": "sso_class_path",
                        "parent_class": parent_name,
                        "child_class": child_name,
                    },
                )
            )

        # asteroid -> leaf-class part_of edge
        if include_targets:
            leaf_name = prefix_names[-1]
            edges.append(
                EdgeCandidate(
                    subject_id=entity_id,
                    object_name=leaf_name,
                    predicate="part_of",
                    source="ssodnet",
                    evidence={
                        "method": "sso_class_leaf_membership",
                        "target": canonical,
                        "leaf_class": leaf_name,
                    },
                )
            )

    taxa = [
        SyntheticTaxon(canonical_name=name, depth=depth)
        for name, depth in sorted(taxa_seen.items())
    ]
    return edges, taxa


# ---------------------------------------------------------------------------
# Curated flagship mission -> instrument
# ---------------------------------------------------------------------------


# Mission canonical_name -> tuple of instrument canonical_names.
# Sourced from NASA science web pages and each observatory's instrument
# suite; intentionally conservative — only instruments that are
# unambiguously attached to exactly one observatory are listed.  Ground
# instruments and gravitational-wave interferometers are excluded
# because they don't have instrument suites in the same sense.
CURATED_FLAGSHIP_INSTRUMENTS: Mapping[str, tuple[str, ...]] = {
    "James Webb Space Telescope": ("NIRSpec", "NIRCam", "MIRI", "NIRISS", "FGS"),
    "Hubble Space Telescope": ("WFC3", "ACS", "STIS", "COS", "NICMOS", "WFPC2", "FGS"),
    "Chandra X-ray Observatory": ("ACIS", "HRC", "LETG", "HETG"),
    "Spitzer Space Telescope": ("IRAC", "MIPS", "IRS"),
    "XMM-Newton": ("EPIC", "RGS", "OM"),
    "Gaia Space Observatory": ("RVS", "BP", "RP"),
    "Kepler Space Telescope": ("Kepler Photometer",),
    "Transiting Exoplanet Survey Satellite": ("TESS Camera",),
    "Fermi Gamma-ray Space Telescope": ("Fermi-LAT", "Fermi GBM"),
    "Neil Gehrels Swift Observatory": ("BAT", "XRT", "UVOT"),
    "Herschel Space Observatory": ("PACS", "SPIRE", "HIFI"),
    "Planck Space Observatory": ("HFI", "LFI"),
    "Wide-field Infrared Survey Explorer": ("WISE Camera",),
    "Wilkinson Microwave Anisotropy Probe": ("WMAP Radiometer",),
    "Nuclear Spectroscopic Telescope Array": ("NuSTAR FPMA", "NuSTAR FPMB"),
    "Rossi X-ray Timing Explorer": ("PCA", "HEXTE", "RXTE ASM"),
    "INTErnational Gamma-Ray Astrophysics Laboratory": (
        "IBIS",
        "SPI",
        "JEM-X",
        "OMC",
    ),
    "Compton Gamma Ray Observatory": ("EGRET", "COMPTEL", "OSSE", "BATSE"),
    "Galaxy Evolution Explorer": ("GALEX FUV", "GALEX NUV"),
    "Solar Dynamics Observatory": ("AIA", "HMI", "EVE"),
    "Solar and Heliospheric Observatory": (
        "EIT",
        "LASCO",
        "MDI",
        "SUMER",
        "CDS",
    ),
}


def extract_curated_flagship_edges(
    missions_by_name: Mapping[str, int],
    instruments_by_name: Mapping[str, int],
) -> Iterator[EdgeCandidate]:
    """Emit ``has_instrument`` edges from the curated flagship table.

    Both endpoints must already exist in ``entities`` (the curated
    flagship mission entities were seeded by
    ``scripts/curate_flagship_entities.py``; instrument entities come
    from whichever source — GCMD, AAS, SPASE — happened to register
    them).  Edges where either endpoint is missing are silently
    skipped so this function is safe to run against partial seeds.
    """
    for mission_name, instruments in CURATED_FLAGSHIP_INSTRUMENTS.items():
        mission_id = missions_by_name.get(mission_name)
        if mission_id is None:
            continue
        for instrument_name in instruments:
            instrument_id = instruments_by_name.get(instrument_name)
            if instrument_id is None:
                continue
            yield EdgeCandidate(
                subject_id=mission_id,
                object_id=instrument_id,
                predicate="has_instrument",
                source="curated_flagship_v1",
                evidence={
                    "method": "curated_flagship_lookup",
                    "mission": mission_name,
                    "instrument": instrument_name,
                },
            )
