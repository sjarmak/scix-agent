#!/usr/bin/env python3
"""Three-way entity-enrichment eval runner for PRD §M4.

Compares three configurations:

1. ``hybrid_baseline``     — hybrid search alone, no entity enrichment.
2. ``hybrid_plus_static``  — hybrid search + static-core entity filter
   that restricts candidates to bibcodes present in
   ``document_entities_canonical``.
3. ``hybrid_plus_jit``     — hybrid search + JIT re-rank that boosts
   candidates whose entity set overlaps with the seed's entity set,
   resolved at query time against ``document_entities_canonical``.

Two modes are supported:

* **Fixture mode** (default) — 5 seeded queries + 2 graph-walk tasks,
  no DB required. Useful for CI.
* **Real-data mode** (``--real-data``) — samples seed papers from the
  populated ``scix`` database, builds citation-based ground truth, and
  runs the three lanes through :mod:`scix.search`. This is the PRD
  hinge runner.

Usage
-----

::

    python scripts/eval_three_way.py
    python scripts/eval_three_way.py --output build-artifacts/m4_inhouse_eval.md
    python scripts/eval_three_way.py --real-data --n-queries 50 --n-graph-walks 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix import resolve_entities as re_mod
from scix.eval.metrics import (
    GraphWalkTask,
    QueryFixture,
    ThreeWayConfig,
    ThreeWayResults,
    format_m4_report,
    run_three_way_eval,
)
from scix.resolve_entities import (
    EntityResolveContext,
    ResolutionFailed,
    resolve_entities,
)

logger = logging.getLogger("eval_three_way")

DEFAULT_OUTPUT = Path("build-artifacts/m4_inhouse_eval.md")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def build_fixture() -> tuple[list[QueryFixture], list[GraphWalkTask]]:
    """Build the §M4 fixture: 5 queries + 2 graph-walk tasks.

    Note: The PRD aspires to 50 queries + 20 graph-walk tasks at full
    scale. The fixture runner uses a scaled-down set so the test stays
    fast and deterministic; extend in-place when the real scix_test
    fixture is available.
    """
    queries = [
        QueryFixture(
            query_id="q1",
            query_text="exoplanet habitable zone",
            relevant_bibcodes=frozenset({"2024EXO...1", "2024EXO...2", "2024EXO...3"}),
        ),
        QueryFixture(
            query_id="q2",
            query_text="gravitational wave binary merger",
            relevant_bibcodes=frozenset({"2024GW....1", "2024GW....2"}),
        ),
        QueryFixture(
            query_id="q3",
            query_text="dark matter halo profile",
            relevant_bibcodes=frozenset({"2024DM....1", "2024DM....2", "2024DM....3"}),
        ),
        QueryFixture(
            query_id="q4",
            query_text="active galactic nuclei variability",
            relevant_bibcodes=frozenset({"2024AGN...1", "2024AGN...2"}),
        ),
        QueryFixture(
            query_id="q5",
            query_text="cosmic microwave background polarization",
            relevant_bibcodes=frozenset({"2024CMB...1", "2024CMB...2"}),
        ),
    ]
    tasks = [
        GraphWalkTask(
            task_id="gw1",
            seed_bibcode="2024EXO...1",
            expected_bibcodes=frozenset({"2024EXO...2", "2024EXO...3"}),
        ),
        GraphWalkTask(
            task_id="gw2",
            seed_bibcode="2024GW....1",
            expected_bibcodes=frozenset({"2024GW....2"}),
        ),
    ]
    return queries, tasks


# ---------------------------------------------------------------------------
# Seeded retrievers
# ---------------------------------------------------------------------------


# For each query, we pre-seed ordered retrieval lists so metric
# computation is deterministic. The "baseline" misranks a bit; the
# "static" filter boosts one correct answer; the "jit" lane lifts a
# second. This gives the report a clear signal without pretending to be
# a real retrieval system.
_BASELINE_ORDER: dict[str, list[str]] = {
    "q1": ["2024EXO...1", "2024NOISE1", "2024EXO...2", "2024NOISE2", "2024EXO...3"],
    "q2": ["2024NOISE1", "2024GW....1", "2024NOISE2", "2024GW....2"],
    "q3": ["2024DM....1", "2024NOISE1", "2024DM....2", "2024NOISE2", "2024DM....3"],
    "q4": ["2024NOISE1", "2024AGN...1", "2024NOISE2", "2024AGN...2"],
    "q5": ["2024NOISE1", "2024NOISE2", "2024CMB...1", "2024CMB...2"],
    "gw1": ["2024EXO...1", "2024NOISE1", "2024EXO...2", "2024EXO...3"],
    "gw2": ["2024GW....1", "2024NOISE1", "2024GW....2"],
}


def _seed_resolver_mocks() -> None:
    """Seed the resolver mocks used by the static and jit lanes."""
    re_mod._reset_mocks()
    # Static lane: entities associated with each relevant bibcode. The
    # eval only consults the resolver for enrichment; the retrieved
    # ordering comes from the seeded pools above.
    for bib in [
        "2024EXO...1",
        "2024EXO...2",
        "2024EXO...3",
        "2024GW....1",
        "2024GW....2",
        "2024DM....1",
        "2024DM....2",
        "2024DM....3",
        "2024AGN...1",
        "2024AGN...2",
        "2024CMB...1",
        "2024CMB...2",
    ]:
        re_mod._seed_static(bib, frozenset({hash(bib) & 0xFFFF}))

    # JIT lane: seed with the same sets + model_version='v1' so
    # resolve_entities(mode='jit') returns hits for relevant bibcodes.
    # candidate_set_hash uses candidate_set+model_version; fixture uses
    # candidate_set=frozenset() so hashes are stable.
    empty_ctx = EntityResolveContext(candidate_set=frozenset(), mode="jit", model_version="v1")
    from scix.resolve_entities import candidate_set_hash

    cset_hash = candidate_set_hash(empty_ctx)
    for bib in [
        "2024EXO...1",
        "2024EXO...2",
        "2024GW....1",
        "2024GW....2",
        "2024DM....1",
        "2024DM....2",
        "2024AGN...1",
        "2024AGN...2",
        "2024CMB...1",
        "2024CMB...2",
    ]:
        re_mod._seed_jit_cache(bib, cset_hash, "v1", frozenset({hash(bib) & 0xFFFF}))


def _resolved_entities(bibcode: str, mode: str) -> frozenset[int]:
    """Look up entities for ``bibcode`` via the M13 resolver."""
    try:
        link_set = resolve_entities(
            bibcode,
            EntityResolveContext(candidate_set=frozenset(), mode=mode, model_version="v1"),
        )
    except ResolutionFailed:
        return frozenset()
    return link_set.entity_ids()


def _boost(
    base_order: list[str],
    boosted_bibs: set[str],
) -> list[str]:
    """Stable-reorder ``base_order`` so ``boosted_bibs`` appear first.

    Simulates the effect of entity enrichment lifting relevant results.
    """
    boosted = [b for b in base_order if b in boosted_bibs]
    rest = [b for b in base_order if b not in boosted_bibs]
    return boosted + rest


def baseline_retrieve(fixture: QueryFixture) -> tuple[list[str], float]:
    return list(_BASELINE_ORDER.get(fixture.query_id, [])), 1.0


def static_retrieve(fixture: QueryFixture) -> tuple[list[str], float]:
    base = list(_BASELINE_ORDER.get(fixture.query_id, []))
    # For the static config, "relevant + has static entity" gets boosted.
    boosted = {b for b in fixture.relevant_bibcodes if _resolved_entities(b, "static")}
    return _boost(base, boosted), 1.5


def jit_retrieve(fixture: QueryFixture) -> tuple[list[str], float]:
    base = list(_BASELINE_ORDER.get(fixture.query_id, []))
    boosted = {b for b in fixture.relevant_bibcodes if _resolved_entities(b, "jit")}
    return _boost(base, boosted), 2.5


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def build_configs() -> list[ThreeWayConfig]:
    return [
        ThreeWayConfig(
            name="hybrid_baseline",
            description="Hybrid lexical + dense search with no entity enrichment.",
            retrieve=baseline_retrieve,
        ),
        ThreeWayConfig(
            name="hybrid_plus_static",
            description="Hybrid search + static-core filter via resolve_entities(mode='static').",
            retrieve=static_retrieve,
        ),
        ThreeWayConfig(
            name="hybrid_plus_jit",
            description="Hybrid search + JIT enrichment via resolve_entities(mode='jit').",
            retrieve=jit_retrieve,
        ),
    ]


def run(output_path: Path = DEFAULT_OUTPUT) -> ThreeWayResults:
    """Run the §M4 eval and write the report to ``output_path``."""
    _seed_resolver_mocks()
    queries, tasks = build_fixture()
    configs = build_configs()
    results = run_three_way_eval(queries, tasks, configs)
    report = format_m4_report(
        results=results,
        configs=configs,
        n_queries=len(queries),
        n_graph_walk=len(tasks),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Wrote M4 eval report to %s", output_path)
    return results


def run_real_data(
    output_path: Path,
    n_queries: int = 50,
    n_graph_walks: int = 20,
    min_neighbors: int = 10,
    random_seed: int = 42,
    baseline_top_n: int | None = None,
) -> ThreeWayResults:
    """Run the §M4 eval against the populated prod corpus.

    Samples ``n_queries + n_graph_walks`` seed bibcodes that have an
    INDUS embedding, at least one entity link in
    ``document_entities_canonical``, and ``min_neighbors+`` citation
    neighbors. Builds binary relevance from the seed's citation
    neighborhood and runs the three retrieval lanes defined in
    :mod:`scix.eval.real_data`.
    """
    # Deferred imports so fixture-mode tests never touch the DB module.
    from scix.db import get_connection  # noqa: WPS433 — intentional local import
    from scix.eval import real_data as rd  # noqa: WPS433

    conn = get_connection()
    ctx = rd.RealEvalContext(conn=conn)

    total_seeds = n_queries + n_graph_walks
    seeds = rd.sample_seed_papers(
        conn,
        n_seeds=total_seeds,
        min_neighbors=min_neighbors,
        random_seed=random_seed,
    )
    if len(seeds) < total_seeds:
        logger.warning(
            "Sampler returned %d seeds, requested %d — continuing with smaller n",
            len(seeds),
            total_seeds,
        )

    # Split seeds: first N for query set, remainder for graph-walk tasks.
    query_seeds = seeds[:n_queries]
    task_seeds = seeds[n_queries : n_queries + n_graph_walks]

    queries: list[QueryFixture] = []
    tasks: list[GraphWalkTask] = []
    for idx, seed in enumerate(query_seeds):
        truth = rd.ground_truth_for_seed(conn, seed.bibcode)
        if not truth:
            continue
        queries.append(
            QueryFixture(
                query_id=f"q_{idx}_{seed.bibcode}",
                query_text=seed.lexical_query or seed.title[:120],
                relevant_bibcodes=frozenset(truth),
            )
        )
    for idx, seed in enumerate(task_seeds):
        truth = rd.ground_truth_for_seed(conn, seed.bibcode)
        if not truth:
            continue
        tasks.append(
            GraphWalkTask(
                task_id=f"gw_{idx}_{seed.bibcode}",
                seed_bibcode=seed.bibcode,
                expected_bibcodes=frozenset(truth),
            )
        )

    # Map bibcode -> seed so retriever callbacks can resolve seed metadata.
    seed_map = {s.bibcode: s for s in seeds}
    qid_to_seed: dict[str, rd.SeedPaper] = {}
    for q, s in zip(queries, query_seeds):
        qid_to_seed[q.query_id] = s
    for t, s in zip(tasks, task_seeds):
        qid_to_seed[t.task_id] = s

    top_n = baseline_top_n if baseline_top_n is not None else rd.BASELINE_TOP_N

    def _lookup_seed(fixture: QueryFixture) -> rd.SeedPaper:
        # For graph-walk tasks the fixture's query_id is the task_id we mapped.
        seed = qid_to_seed.get(fixture.query_id)
        if seed is not None:
            return seed
        # Fallback: synthesise a SeedPaper from the fixture metadata.
        bibcode = fixture.query_text.replace("graph_walk:", "")
        return seed_map.get(
            bibcode,
            rd.SeedPaper(
                bibcode=bibcode,
                title=fixture.query_text,
                abstract=None,
                year=None,
                citation_count=0,
                n_neighbors=0,
                n_entities=0,
            ),
        )

    def baseline(fixture: QueryFixture) -> tuple[list[str], float]:
        return rd.baseline_retrieve(ctx, _lookup_seed(fixture), limit=top_n)

    def static(fixture: QueryFixture) -> tuple[list[str], float]:
        return rd.static_filter_retrieve(ctx, _lookup_seed(fixture), limit=top_n)

    def jit(fixture: QueryFixture) -> tuple[list[str], float]:
        return rd.jit_rerank_retrieve(ctx, _lookup_seed(fixture), limit=top_n)

    configs = [
        ThreeWayConfig(
            name="hybrid_baseline",
            description="Hybrid lexical + dense search (INDUS + BM25 via RRF), no enrichment.",
            retrieve=baseline,
        ),
        ThreeWayConfig(
            name="hybrid_plus_static",
            description=(
                "Hybrid search + static-core filter: candidates restricted to bibcodes "
                "present in document_entities_canonical."
            ),
            retrieve=static,
        ),
        ThreeWayConfig(
            name="hybrid_plus_jit",
            description=(
                "Hybrid search + JIT enrichment: baseline list re-ranked by seed-to-candidate "
                "entity-set overlap (resolved at query time)."
            ),
            retrieve=jit,
        ),
    ]

    logger.info(
        "Running real-data eval: %d queries, %d graph-walk tasks, 3 lanes",
        len(queries),
        len(tasks),
    )
    results = run_three_way_eval(queries, tasks, configs)
    report = format_m4_report(
        results=results,
        configs=configs,
        n_queries=len(queries),
        n_graph_walk=len(tasks),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Wrote real-data M4 report to %s", output_path)
    conn.close()
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="M4 three-way entity-enrichment eval")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the markdown report (default: %(default)s)",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Run against the populated prod DB instead of the in-process fixture.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=50,
        help="Number of seed queries for --real-data mode (default: 50).",
    )
    parser.add_argument(
        "--n-graph-walks",
        type=int,
        default=20,
        help="Number of graph-walk tasks for --real-data mode (default: 20).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=10,
        help="Minimum citation neighbors per seed (default: 10).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Sampling seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--baseline-top-n",
        type=int,
        default=None,
        help="Override the retrieval top-N used by each lane.",
    )
    args = parser.parse_args()
    if args.real_data:
        results = run_real_data(
            output_path=args.output,
            n_queries=args.n_queries,
            n_graph_walks=args.n_graph_walks,
            min_neighbors=args.min_neighbors,
            random_seed=args.random_seed,
            baseline_top_n=args.baseline_top_n,
        )
    else:
        results = run(output_path=args.output)
    for name, report in results.query_reports.items():
        logger.info(
            "%s — nDCG@10=%.4f Recall@20=%.4f MRR=%.4f",
            name,
            report.mean_ndcg_at_10,
            report.mean_recall_at_20,
            report.mean_mrr,
        )


if __name__ == "__main__":
    main()
