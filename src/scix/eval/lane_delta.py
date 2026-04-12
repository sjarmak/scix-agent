"""Lane-consistency arithmetic for PRD §M4.5.

Computes per-bibcode Jaccard across three entity-resolution lanes
(citation-chain, hybrid_search[enrich_entities=True], canonical static
read) with optional adjustment for a Wikidata-backfill ``lane_delta_set``
— entities that exist in the static lane via Wikidata backfill but are
structurally unreachable from the JIT lane's candidate derivation.

At u12 the ``lane_delta_set`` is a stub returning an empty set; the
arithmetic path still subtracts it from both numerator and denominator so
that the API shape is correct when u07's Wikidata backfill lands.

The 90th-percentile-of-divergence gate uses :func:`numpy.percentile` and
passes when ``p90 ≤ 0.05``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Basic set math
# ---------------------------------------------------------------------------


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """Standard Jaccard index over two entity-id sets.

    Returns 1.0 when both sets are empty — an agreement that neither lane
    knows any entity for this bibcode, which should not count as
    divergence.
    """
    if not a and not b:
        return 1.0
    inter = a & b
    union = a | b
    if not union:
        return 1.0
    return len(inter) / len(union)


def adjusted_jaccard(
    a: frozenset[int],
    b: frozenset[int],
    lane_delta_set: frozenset[int],
) -> float:
    """Jaccard with the Wikidata-backfill delta removed from numerator and
    denominator.

    Both the intersection and the union are recomputed after subtracting
    ``lane_delta_set`` from both input sets. This is the shape the PRD
    specifies even though ``lane_delta_set`` is empty at u12.
    """
    a_adj = a - lane_delta_set
    b_adj = b - lane_delta_set
    return jaccard(a_adj, b_adj)


def divergence(jaccard_value: float) -> float:
    """Divergence = 1 - Jaccard, clamped to [0, 1]."""
    return max(0.0, min(1.0, 1.0 - jaccard_value))


# ---------------------------------------------------------------------------
# Lane delta stub — will be replaced by u07's Wikidata backfill
# ---------------------------------------------------------------------------


def compute_lane_delta_set(bibcode: str) -> frozenset[int]:
    """Return the Wikidata-backfill ``lane_delta_set`` for ``bibcode``.

    u12 in-scope: returns an empty set. The structurally-unreachable
    entity list is owned by u07 (Wikidata backfill) and the surrounding
    arithmetic is already plumbed so that substituting a real set here
    will immediately adjust the §M4.5 gate computation.

    TODO(u07-wikidata-backfill): replace with real lookup.
    """
    _ = bibcode  # documented parameter, kept for API stability
    return frozenset()


# ---------------------------------------------------------------------------
# Per-bibcode divergence bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneEntitySets:
    """Three lanes' entity-id sets for one bibcode."""

    bibcode: str
    citation_chain: frozenset[int]
    hybrid_enrich: frozenset[int]
    static_canonical: frozenset[int]


@dataclass(frozen=True)
class BibcodeDivergence:
    """Per-bibcode divergence record written into the §M4.5 report."""

    bibcode: str
    raw_jaccard_chain_hybrid: float
    raw_jaccard_chain_static: float
    raw_jaccard_hybrid_static: float
    adj_jaccard_chain_hybrid: float
    adj_jaccard_chain_static: float
    adj_jaccard_hybrid_static: float
    mean_raw_jaccard: float
    mean_adj_jaccard: float

    @property
    def mean_raw_divergence(self) -> float:
        return divergence(self.mean_raw_jaccard)

    @property
    def mean_adj_divergence(self) -> float:
        return divergence(self.mean_adj_jaccard)


def per_bibcode_divergence(
    sets: LaneEntitySets,
    lane_delta_set: frozenset[int] | None = None,
) -> BibcodeDivergence:
    """Compute raw + adjusted Jaccard for all three lane pairs."""
    delta = lane_delta_set if lane_delta_set is not None else compute_lane_delta_set(sets.bibcode)

    raw_ch = jaccard(sets.citation_chain, sets.hybrid_enrich)
    raw_cs = jaccard(sets.citation_chain, sets.static_canonical)
    raw_hs = jaccard(sets.hybrid_enrich, sets.static_canonical)

    adj_ch = adjusted_jaccard(sets.citation_chain, sets.hybrid_enrich, delta)
    adj_cs = adjusted_jaccard(sets.citation_chain, sets.static_canonical, delta)
    adj_hs = adjusted_jaccard(sets.hybrid_enrich, sets.static_canonical, delta)

    return BibcodeDivergence(
        bibcode=sets.bibcode,
        raw_jaccard_chain_hybrid=raw_ch,
        raw_jaccard_chain_static=raw_cs,
        raw_jaccard_hybrid_static=raw_hs,
        adj_jaccard_chain_hybrid=adj_ch,
        adj_jaccard_chain_static=adj_cs,
        adj_jaccard_hybrid_static=adj_hs,
        mean_raw_jaccard=(raw_ch + raw_cs + raw_hs) / 3.0,
        mean_adj_jaccard=(adj_ch + adj_cs + adj_hs) / 3.0,
    )


@dataclass(frozen=True)
class LaneConsistencyAggregate:
    """Distribution summary over a batch of :class:`BibcodeDivergence`."""

    n: int
    mean_raw_jaccard: float
    mean_adj_jaccard: float
    p50_adj_divergence: float
    p90_adj_divergence: float
    p99_adj_divergence: float
    gate_threshold: float = 0.05
    gate_passed: bool = True
    per_pair_mean_divergence: dict[str, float] = field(default_factory=dict)


GATE_THRESHOLD = 0.05


def aggregate_divergence(
    records: list[BibcodeDivergence],
    gate_threshold: float = GATE_THRESHOLD,
) -> LaneConsistencyAggregate:
    """Roll per-bibcode divergences into an aggregate with the p90 gate."""
    if not records:
        return LaneConsistencyAggregate(
            n=0,
            mean_raw_jaccard=1.0,
            mean_adj_jaccard=1.0,
            p50_adj_divergence=0.0,
            p90_adj_divergence=0.0,
            p99_adj_divergence=0.0,
            gate_threshold=gate_threshold,
            gate_passed=True,
            per_pair_mean_divergence={},
        )

    adj_divergences = [r.mean_adj_divergence for r in records]
    p50, p90, p99 = np.percentile(adj_divergences, [50, 90, 99]).tolist()

    per_pair = {
        "citation_chain_vs_hybrid": float(
            np.mean([divergence(r.adj_jaccard_chain_hybrid) for r in records])
        ),
        "citation_chain_vs_static": float(
            np.mean([divergence(r.adj_jaccard_chain_static) for r in records])
        ),
        "hybrid_vs_static": float(
            np.mean([divergence(r.adj_jaccard_hybrid_static) for r in records])
        ),
    }

    return LaneConsistencyAggregate(
        n=len(records),
        mean_raw_jaccard=float(np.mean([r.mean_raw_jaccard for r in records])),
        mean_adj_jaccard=float(np.mean([r.mean_adj_jaccard for r in records])),
        p50_adj_divergence=float(p50),
        p90_adj_divergence=float(p90),
        p99_adj_divergence=float(p99),
        gate_threshold=gate_threshold,
        gate_passed=float(p90) <= gate_threshold,
        per_pair_mean_divergence=per_pair,
    )


def gate_p90(
    records: list[BibcodeDivergence],
    gate_threshold: float = GATE_THRESHOLD,
) -> tuple[float, bool]:
    """Convenience wrapper that returns ``(p90, passed)``."""
    agg = aggregate_divergence(records, gate_threshold=gate_threshold)
    return agg.p90_adj_divergence, agg.gate_passed


__all__ = [
    "BibcodeDivergence",
    "GATE_THRESHOLD",
    "LaneConsistencyAggregate",
    "LaneEntitySets",
    "adjusted_jaccard",
    "aggregate_divergence",
    "compute_lane_delta_set",
    "divergence",
    "gate_p90",
    "jaccard",
    "per_bibcode_divergence",
]
