#!/usr/bin/env python3
"""Lane consistency eval for PRD §M4.5.

Computes per-bibcode Jaccard across three entity-resolution lanes:

1. **citation_chain** — union of resolver-static entities over the
   bibcode's citation neighbors. Stands in for the ``citation_chain``
   MCP tool's entity-derivation path.
2. **hybrid_search[enrich_entities=True]** — the resolver static lane,
   which is what the hybrid-search enrichment call reads under the
   hood.
3. **JIT resolver lane** — calls ``resolve_entities(mode='jit')`` to
   exercise the just-in-time resolution path. The point of having
   both "2" (static) and "3" (JIT) is to surface any divergence
   between the two resolver modes; if a future refactor introduces
   a discrepancy, it will show up here.

The report adjusts every Jaccard by subtracting the Wikidata-backfill
``lane_delta_set`` (empty at u12 — TODO in
``src/scix/eval/lane_delta.py``).

Gate: ``np.percentile(divergences, 90) <= 0.05``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix import resolve_entities as re_mod
from scix.eval.lane_delta import (
    BibcodeDivergence,
    GATE_THRESHOLD,
    LaneConsistencyAggregate,
    LaneEntitySets,
    aggregate_divergence,
    compute_lane_delta_set,
    per_bibcode_divergence,
)
from scix.resolve_entities import (
    EntityResolveContext,
    ResolutionFailed,
    resolve_entities,
)

logger = logging.getLogger("eval_lane_consistency")

DEFAULT_CONSISTENCY_OUTPUT = Path("build-artifacts/m45_consistency.md")
DEFAULT_LANE_DELTA_OUTPUT = Path("build-artifacts/m45_lane_delta.md")

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureBibcode:
    """One seeded bibcode for the §M4.5 fixture."""

    bibcode: str
    citation_neighbors: tuple[str, ...]
    static_entities: frozenset[int]
    jit_entities: frozenset[int]


def build_fixture() -> list[FixtureBibcode]:
    """Return 5 bibcodes with a deliberately known divergence pattern.

    Bibcodes 1-3 have fully-agreeing lanes (Jaccard = 1.0, divergence
    0.0). Bibcode 4 has a small disagreement (Jaccard ~0.67). Bibcode 5
    has full disagreement (Jaccard 0.0). On a 5-element fixture, the
    90th percentile lives at index 3.6 which ``numpy.percentile`` linearly
    interpolates; the test asserts the exact value so the gate
    computation stays honest.
    """
    return [
        FixtureBibcode(
            bibcode="2024M45..1",
            citation_neighbors=("2024N1A", "2024N1B"),
            static_entities=frozenset({10, 20}),
            jit_entities=frozenset({10, 20}),
        ),
        FixtureBibcode(
            bibcode="2024M45..2",
            citation_neighbors=("2024N2A",),
            static_entities=frozenset({30}),
            jit_entities=frozenset({30}),
        ),
        FixtureBibcode(
            bibcode="2024M45..3",
            citation_neighbors=("2024N3A", "2024N3B"),
            static_entities=frozenset({40, 50}),
            jit_entities=frozenset({40, 50}),
        ),
        FixtureBibcode(
            bibcode="2024M45..4",
            citation_neighbors=("2024N4A",),
            static_entities=frozenset({60, 61}),
            jit_entities=frozenset({60, 62}),
        ),
        FixtureBibcode(
            bibcode="2024M45..5",
            citation_neighbors=("2024N5A",),
            static_entities=frozenset({70}),
            jit_entities=frozenset({80}),
        ),
    ]


def seed_resolver_from_fixture(fixture: list[FixtureBibcode]) -> None:
    """Seed the resolver mocks so lane lookups match the fixture."""
    re_mod._reset_mocks()

    # Static lane for each bibcode + every neighbor (citation chain
    # derivation reads neighbors via mode='static').
    for item in fixture:
        re_mod._seed_static(item.bibcode, item.static_entities)
        # Neighbors carry the SAME ids as the seed bibcode so the
        # citation-chain-analog lane returns the same union (bibcodes
        # 1-3 fully agree; bibcodes 4 and 5 intentionally do not).
        for neighbor in item.citation_neighbors:
            re_mod._seed_static(neighbor, item.static_entities)

    # JIT lane — seed jit cache for each bibcode with the jit entity
    # set under the empty candidate set + v1 model.
    from scix.resolve_entities import candidate_set_hash

    jit_ctx = EntityResolveContext(candidate_set=frozenset(), mode="jit", model_version="v1")
    cset_hash = candidate_set_hash(jit_ctx)
    for item in fixture:
        re_mod._seed_jit_cache(item.bibcode, cset_hash, "v1", item.jit_entities)


# ---------------------------------------------------------------------------
# Lane lookups — all routed through scix.resolve_entities per M13.
# ---------------------------------------------------------------------------


def _entities(bibcode: str, mode: str) -> frozenset[int]:
    try:
        link_set = resolve_entities(
            bibcode,
            EntityResolveContext(candidate_set=frozenset(), mode=mode, model_version="v1"),
        )
    except ResolutionFailed:
        return frozenset()
    return link_set.entity_ids()


def citation_chain_entities(
    bibcode: str,
    neighbors: Iterable[str],
) -> frozenset[int]:
    """Citation-chain lane: union of resolver-static entities for neighbors.

    This stands in for the ``citation_chain`` MCP tool's entity
    derivation pathway. We deliberately union the neighbors' entities
    rather than the seed's so the lane is structurally distinct from
    the "hybrid search enrichment" lane.
    """
    ids: set[int] = set()
    for neighbor in neighbors:
        ids.update(_entities(neighbor, "static"))
    return frozenset(ids)


def hybrid_search_entities(bibcode: str) -> frozenset[int]:
    """Lane B — hybrid_search[enrich_entities=True] goes through static."""
    return _entities(bibcode, "static")


def static_canonical_entities(bibcode: str) -> frozenset[int]:
    """Lane C — JIT resolver read.

    Uses ``resolve_entities(mode='jit')`` to exercise the just-in-time
    resolution path. Comparing this lane against lanes A and B (both
    ``mode='static'``) surfaces any divergence between the two resolver
    modes. The M13 single-entry-point contract is preserved because
    both modes route through ``resolve_entities``.
    """
    return _entities(bibcode, "jit")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneConsistencyReportInputs:
    records: list[BibcodeDivergence]
    aggregate: LaneConsistencyAggregate
    lane_delta_rows: list[tuple[str, int, str]]


def compute_lane_consistency(
    fixture: list[FixtureBibcode],
) -> LaneConsistencyReportInputs:
    """Run the full M4.5 computation over the fixture."""
    records: list[BibcodeDivergence] = []
    lane_delta_rows: list[tuple[str, int, str]] = []

    for item in fixture:
        sets = LaneEntitySets(
            bibcode=item.bibcode,
            citation_chain=citation_chain_entities(item.bibcode, item.citation_neighbors),
            hybrid_enrich=hybrid_search_entities(item.bibcode),
            static_canonical=static_canonical_entities(item.bibcode),
        )
        delta = compute_lane_delta_set(item.bibcode)
        for entity_id in sorted(delta):
            lane_delta_rows.append((item.bibcode, entity_id, "wikidata-backfill-unreachable"))
        records.append(per_bibcode_divergence(sets, lane_delta_set=delta))

    aggregate = aggregate_divergence(records)
    return LaneConsistencyReportInputs(
        records=records,
        aggregate=aggregate,
        lane_delta_rows=lane_delta_rows,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def format_consistency_report(inputs: LaneConsistencyReportInputs) -> str:
    agg = inputs.aggregate
    lines: list[str] = []
    lines.append("# M4.5 — Lane Consistency Report")
    lines.append("")
    lines.append(f"**n**: {agg.n} bibcodes")
    lines.append(f"**Mean raw Jaccard**: {agg.mean_raw_jaccard:.4f}")
    lines.append(f"**Mean adjusted Jaccard**: {agg.mean_adj_jaccard:.4f}")
    lines.append(f"**p50 adjusted divergence**: {agg.p50_adj_divergence:.4f}")
    lines.append(f"**p90 adjusted divergence**: {agg.p90_adj_divergence:.4f}")
    lines.append(f"**p99 adjusted divergence**: {agg.p99_adj_divergence:.4f}")
    verdict = "PASS" if agg.gate_passed else "FAIL"
    lines.append(f"**Gate (p90 ≤ {agg.gate_threshold:.2f})**: {verdict}")
    lines.append("")

    lines.append("## Per-bibcode Jaccard")
    lines.append("")
    lines.append(
        "| bibcode | raw_chain_hybrid | raw_chain_static | raw_hybrid_static "
        "| adj_chain_hybrid | adj_chain_static | adj_hybrid_static "
        "| mean_adj_divergence |"
    )
    lines.append(
        "|---------|------------------|------------------|-------------------"
        "|------------------|------------------|-------------------"
        "|---------------------|"
    )
    for r in inputs.records:
        lines.append(
            f"| {r.bibcode} "
            f"| {r.raw_jaccard_chain_hybrid:.4f} "
            f"| {r.raw_jaccard_chain_static:.4f} "
            f"| {r.raw_jaccard_hybrid_static:.4f} "
            f"| {r.adj_jaccard_chain_hybrid:.4f} "
            f"| {r.adj_jaccard_chain_static:.4f} "
            f"| {r.adj_jaccard_hybrid_static:.4f} "
            f"| {r.mean_adj_divergence:.4f} |"
        )
    lines.append("")

    lines.append("## Aggregate distribution")
    lines.append("")
    lines.append("| statistic | value |")
    lines.append("|-----------|-------|")
    lines.append(f"| n | {agg.n} |")
    lines.append(f"| mean_raw_jaccard | {agg.mean_raw_jaccard:.4f} |")
    lines.append(f"| mean_adj_jaccard | {agg.mean_adj_jaccard:.4f} |")
    lines.append(f"| p50_adj_divergence | {agg.p50_adj_divergence:.4f} |")
    lines.append(f"| p90_adj_divergence | {agg.p90_adj_divergence:.4f} |")
    lines.append(f"| p99_adj_divergence | {agg.p99_adj_divergence:.4f} |")
    lines.append(f"| gate_threshold | {agg.gate_threshold:.4f} |")
    lines.append(f"| gate_passed | {agg.gate_passed} |")
    lines.append("")

    lines.append("## Per-lane-pair mean divergence")
    lines.append("")
    lines.append("| pair | mean_adjusted_divergence |")
    lines.append("|------|--------------------------|")
    for name, value in agg.per_pair_mean_divergence.items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def format_lane_delta_report(inputs: LaneConsistencyReportInputs) -> str:
    lines: list[str] = []
    lines.append("# M4.5 — Lane Delta (Wikidata-backfill unreachable entities)")
    lines.append("")
    lines.append(
        "One row per entity that exists in the static lane via Wikidata "
        "backfill but is structurally unreachable from the JIT lane's "
        "candidate derivation."
    )
    lines.append("")
    lines.append(
        "> u12 status: the lane_delta_set is a stub (empty). The real "
        "set will land with u07 Wikidata backfill; the arithmetic path "
        "already subtracts this set from the numerator and denominator "
        "of the adjusted Jaccard."
    )
    lines.append("")
    lines.append("| bibcode | entity_id | reason |")
    lines.append("|---------|-----------|--------|")
    if not inputs.lane_delta_rows:
        lines.append("| _(none)_ | _(none)_ | _(none — stub returns empty set)_ |")
    else:
        for bib, eid, reason in inputs.lane_delta_rows:
            lines.append(f"| {bib} | {eid} | {reason} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(
    consistency_output: Path = DEFAULT_CONSISTENCY_OUTPUT,
    lane_delta_output: Path = DEFAULT_LANE_DELTA_OUTPUT,
) -> LaneConsistencyReportInputs:
    fixture = build_fixture()
    seed_resolver_from_fixture(fixture)
    inputs = compute_lane_consistency(fixture)

    consistency_output.parent.mkdir(parents=True, exist_ok=True)
    consistency_output.write_text(format_consistency_report(inputs))
    logger.info("Wrote consistency report to %s", consistency_output)

    lane_delta_output.parent.mkdir(parents=True, exist_ok=True)
    lane_delta_output.write_text(format_lane_delta_report(inputs))
    logger.info("Wrote lane delta report to %s", lane_delta_output)

    agg = inputs.aggregate
    verdict = "PASS" if agg.gate_passed else "FAIL"
    print(
        f"M4.5 gate: p90 adjusted divergence = {agg.p90_adj_divergence:.4f} "
        f"(threshold {agg.gate_threshold:.2f}) — {verdict}"
    )
    return inputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="M4.5 lane consistency eval")
    parser.add_argument(
        "--consistency-output",
        type=Path,
        default=DEFAULT_CONSISTENCY_OUTPUT,
    )
    parser.add_argument(
        "--lane-delta-output",
        type=Path,
        default=DEFAULT_LANE_DELTA_OUTPUT,
    )
    args = parser.parse_args()
    run(
        consistency_output=args.consistency_output,
        lane_delta_output=args.lane_delta_output,
    )


if __name__ == "__main__":
    main()

# Re-export for test import convenience.
_ = GATE_THRESHOLD
