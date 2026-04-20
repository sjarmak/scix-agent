#!/usr/bin/env python3
"""Before/after benchmark for entity-aware query intent (xz4.1.24 + xz4.1.25).

Compares four `hybrid_search` variants on 10 queries drawn from the
calibration set:

    baseline           — both flags off (current production behaviour)
    alias              — enable_alias_expansion=True
    ontology           — enable_ontology_parser=True
    both               — both flags on

Each variant returns its top-K bibcodes. We report:

    * **nDCG@10** vs a "kitchen-sink" oracle. The oracle is constructed
      per query as the top-30 RRF fusion of every available signal at
      lexical_limit=200 (vector when query embeddings are available,
      lexical, body BM25, plus the union of alias and ontology signals).
      Both variants are evaluated on the SAME oracle, so improvements
      reflect actual recovery of fused-signal candidates rather than
      tautological wins.
    * **Top-10 entity coverage** — fraction of top-10 papers linked to
      any of the entities the alias expansion identified for the query.
      Independent signal: measures whether the variant surfaces papers
      that are entity-linked, regardless of oracle membership.

This is a structural benchmark, NOT a topical-relevance evaluation. A
proper topical eval requires the persona-Claude judge harness (see
docs/prd/amendments/2026-04-18-m6-m12-collapse.md).

Usage
-----
    SCIX_DSN=dbname=scix python scripts/bench_entity_query_expansion.py \
        --output results/entity_query_expansion/run_$(date +%F).json

Refuses to write data; the script only reads from `entities`,
`entity_aliases`, `papers`, `paper_embeddings`, and
`document_entities_canonical`.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.alias_expansion import build_alias_automaton, expand_query
from scix.db import get_connection
from scix.ir_metrics import ndcg_at_k
from scix.ontology_query_parser import parse_query
from scix.search import SearchFilters, hybrid_search

logger = logging.getLogger(__name__)


# 10 benchmark queries: 4 alias-heavy, 4 ontology-heavy, 2 mixed.
# Drawn directly from src/scix/eval/prompts/calibration_queries.yaml.
BENCHMARK_QUERIES: list[dict[str, str]] = [
    # alias-heavy
    {"id": "alias_01", "query": "HST observations of cool brown dwarfs"},
    {"id": "alias_03", "query": "JWST MIRI mid-infrared imaging of planetary nebulae"},
    {"id": "alias_05", "query": "CMB anisotropy measurements from WMAP"},
    {"id": "alias_06", "query": "X-ray binaries detected by Chandra"},
    # ontology-heavy
    {"id": "ontology_01", "query": "M-type asteroid metallic composition from near-infrared spectroscopy"},
    {"id": "ontology_03", "query": "C-type asteroid water and organics content"},
    {"id": "ontology_04", "query": "infrared instruments on space telescopes for exoplanet detection"},
    {"id": "ontology_07", "query": "flagship NASA missions to the outer solar system"},
    # mixed (both signals plausible)
    {"id": "mixed_01", "query": "Kepler space telescope exoplanet occurrence statistics"},
    {"id": "mixed_02", "query": "Cassini spacecraft instruments at Saturn"},
]


def _build_oracle(
    conn: psycopg.Connection,
    query: str,
    automaton,
    *,
    oracle_k: int = 30,
    lexical_limit: int = 200,
    timeout_ms: int = 60_000,
) -> dict[str, float] | None:
    """Return a relevance map: bibcode -> grade in (0, 1].

    The oracle is the top-K of a kitchen-sink hybrid_search run with both
    flags ON, body BM25 ON, and a high lexical_limit. Grades decay by
    reciprocal rank so nDCG rewards exact ordering, not just membership.
    Returns ``None`` if the oracle build itself times out — caller should
    skip nDCG computation for that query.
    """
    # Postgres SET requires a literal, not a $-param. timeout_ms is an int
    # we control, so f-string is safe (no SQL injection surface).
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
    try:
        res = hybrid_search(
            conn,
            query,
            enable_alias_expansion=True,
            enable_ontology_parser=True,
            alias_automaton=automaton,
            include_body=True,
            lexical_limit=lexical_limit,
            top_n=oracle_k,
        )
    except psycopg.errors.QueryCanceled:
        logger.warning("Oracle build timed out for query=%r — skipping nDCG", query)
        return None
    finally:
        # autocommit=True keeps the connection alive across QueryCanceled, so
        # this SET always succeeds.
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0")

    relevance: dict[str, float] = {}
    for rank, paper in enumerate(res.papers, start=1):
        bibcode = paper.get("bibcode")
        if not bibcode:
            continue
        relevance[bibcode] = 1.0 / rank
    return relevance


def _entity_linked_bibcodes(
    conn: psycopg.Connection,
    entity_ids: tuple[int, ...],
    cap: int = 500,
) -> set[str]:
    """Return bibcodes linked to any of the given entity_ids."""
    if not entity_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode FROM document_entities_canonical "
            "WHERE entity_id = ANY(%s) LIMIT %s",
            (list(entity_ids), cap),
        )
        return {row[0] for row in cur.fetchall()}


def _run_variant(
    conn: psycopg.Connection,
    query: str,
    *,
    automaton,
    enable_alias: bool,
    enable_ontology: bool,
    top_n: int = 20,
    timeout_ms: int = 30_000,
) -> tuple[list[str] | None, dict[str, float], dict[str, Any]]:
    """Run one hybrid_search variant with a hard statement timeout.

    Returns (bibcodes, timing, metadata). On timeout, bibcodes is None and
    metadata carries ``{"timeout": True, "duration_ms": <elapsed>}`` so the
    caller can record the gap without crashing the whole benchmark.
    """
    # Reset the connection's statement_timeout per-call. Postgres SET requires
    # a literal value (not a $-param); timeout_ms is an int we control.
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")

    t0 = time.perf_counter()
    try:
        res = hybrid_search(
            conn,
            query,
            enable_alias_expansion=enable_alias,
            enable_ontology_parser=enable_ontology,
            alias_automaton=automaton if enable_alias else None,
            include_body=True,
            top_n=top_n,
        )
    except psycopg.errors.QueryCanceled:
        elapsed = (time.perf_counter() - t0) * 1000
        return None, {"wallclock_ms": round(elapsed, 2)}, {"timeout": True}
    finally:
        # autocommit=True (set by run_benchmark) keeps the connection valid
        # across QueryCanceled. Restore default cap.
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0")

    elapsed = (time.perf_counter() - t0) * 1000
    bibcodes = [p.get("bibcode") for p in res.papers if p.get("bibcode")]
    timing = {**res.timing_ms, "wallclock_ms": round(elapsed, 2)}
    return bibcodes, timing, res.metadata


def run_benchmark(dsn: str | None, output: Path, oracle_k: int = 30) -> dict[str, Any]:
    conn = get_connection(dsn) if dsn else get_connection()
    # Read-only benchmark: autocommit so QueryCanceled doesn't leave the
    # connection in an aborted-transaction state that poisons later queries.
    conn.autocommit = True
    try:
        # Build a single shared automaton scoped to the entity types most
        # relevant to the benchmark queries. None = full corpus, which is
        # too large for an interactive run; we restrict to the small set.
        scoped_types = (
            "telescope",
            "instrument",
            "mission",
            "spacecraft",
            "asteroid",
            "exoplanet",
        )
        logger.info("Building alias automaton for scoped entity types: %s", scoped_types)
        automaton = build_alias_automaton(conn, entity_types=scoped_types)
        logger.info(
            "Automaton ready: %d entities, %d surface forms",
            len(automaton.canonical_by_entity),
            sum(1 for _ in automaton.automaton.keys()),
        )

        per_query: list[dict[str, Any]] = []
        for q in BENCHMARK_QUERIES:
            qid = q["id"]
            query = q["query"]
            logger.info("[%s] %s", qid, query)

            # Identify entities up-front for the entity-coverage signal.
            expansion = expand_query(conn, query, automaton=automaton)
            parsed = parse_query(query)
            alias_entity_ids = expansion.entity_ids
            entity_set = _entity_linked_bibcodes(conn, alias_entity_ids)

            oracle = _build_oracle(conn, query, automaton, oracle_k=oracle_k)
            if oracle is None:
                # Even the kitchen-sink oracle hung — record and continue.
                per_query.append(
                    {
                        "query_id": qid,
                        "query": query,
                        "oracle_timeout": True,
                    }
                )
                continue

            variants: dict[str, dict[str, Any]] = {}
            for variant_name, alias_on, onto_on in [
                ("baseline", False, False),
                ("alias", True, False),
                ("ontology", False, True),
                ("both", True, True),
            ]:
                bibcodes, timing, metadata = _run_variant(
                    conn,
                    query,
                    automaton=automaton,
                    enable_alias=alias_on,
                    enable_ontology=onto_on,
                )
                if bibcodes is None:
                    variants[variant_name] = {
                        "ndcg_at_10": None,
                        "entity_coverage_top10": None,
                        "top10": [],
                        "timing_ms": timing,
                        "search_metadata": metadata,
                        "timeout": True,
                    }
                    continue

                ndcg = ndcg_at_k(bibcodes, oracle, k=10)
                top10 = bibcodes[:10]
                if entity_set:
                    coverage = sum(1 for b in top10 if b in entity_set) / max(len(top10), 1)
                else:
                    coverage = float("nan")
                variants[variant_name] = {
                    "ndcg_at_10": round(ndcg, 4),
                    "entity_coverage_top10": (
                        round(coverage, 4) if coverage == coverage else None
                    ),
                    "top10": top10,
                    "timing_ms": timing,
                    "search_metadata": metadata,
                }

            per_query.append(
                {
                    "query_id": qid,
                    "query": query,
                    "alias_entities_found": [
                        {"id": m.entity_id, "canonical": m.canonical_name, "type": m.entity_type}
                        for m in expansion.matches
                    ],
                    "ontology_clauses": [
                        {
                            "entity_type": c.entity_type,
                            "properties_filter": c.properties_filter,
                            "surface": c.surface,
                        }
                        for c in parsed.clauses
                    ],
                    "oracle_size": len(oracle),
                    "entity_linked_pool_size": len(entity_set),
                    "variants": variants,
                }
            )

        # Aggregate across queries — skip per-query rows where the oracle
        # timed out (no `variants` key) or individual variants timed out.
        summary: dict[str, dict[str, Any]] = {}
        scored_queries = [pq for pq in per_query if "variants" in pq]
        for variant in ("baseline", "alias", "ontology", "both"):
            ndcgs = [
                pq["variants"][variant]["ndcg_at_10"]
                for pq in scored_queries
                if pq["variants"][variant]["ndcg_at_10"] is not None
            ]
            covs = [
                pq["variants"][variant]["entity_coverage_top10"]
                for pq in scored_queries
                if pq["variants"][variant]["entity_coverage_top10"] is not None
            ]
            wallclocks = [
                pq["variants"][variant]["timing_ms"].get("wallclock_ms", 0.0)
                for pq in scored_queries
            ]
            timeouts = sum(
                1 for pq in scored_queries if pq["variants"][variant].get("timeout")
            )
            summary[variant] = {
                "mean_ndcg_at_10": round(statistics.mean(ndcgs), 4) if ndcgs else None,
                "median_ndcg_at_10": round(statistics.median(ndcgs), 4) if ndcgs else None,
                "mean_entity_coverage_top10": (
                    round(statistics.mean(covs), 4) if covs else None
                ),
                "mean_wallclock_ms": (
                    round(statistics.mean(wallclocks), 2) if wallclocks else None
                ),
                "n_scored": len(ndcgs),
                "n_timeouts": timeouts,
                "n_queries_with_entity_pool": len(covs),
            }

        report = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "oracle": {
                "kind": "kitchen_sink_hybrid_top_k",
                "top_k": oracle_k,
                "lexical_limit": 200,
                "include_body": True,
                "grading": "reciprocal_rank",
                "note": (
                    "Oracle is hybrid_search with both flags ON. Variants are "
                    "evaluated against the same oracle so wins reflect signal "
                    "recovery, not entity-id filter membership tautologies."
                ),
            },
            "limitations": [
                "Structural benchmark — does not measure topical relevance.",
                "Oracle is self-referential (built with both flags ON); it "
                "rewards variants that recover the fused-signal top-K, but "
                "may under-credit ontology-only wins on queries where the "
                "oracle is dominated by alias signals.",
                "Topical evaluation requires the persona-Claude judge harness "
                "(see docs/prd/amendments/2026-04-18-m6-m12-collapse.md).",
            ],
            "queries": BENCHMARK_QUERIES,
            "summary": summary,
            "per_query": per_query,
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Wrote %s", output)

        return report
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (default: SCIX_DSN env or dbname=scix)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/entity_query_expansion/run.json"),
        help="Where to write the JSON report",
    )
    parser.add_argument(
        "--oracle-k",
        type=int,
        default=30,
        help="Top-K used to construct the oracle relevance map",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    report = run_benchmark(args.dsn, args.output, oracle_k=args.oracle_k)

    print()
    print("=" * 70)
    print("Entity-aware query intent benchmark — summary")
    print("=" * 70)
    for variant, stats in report["summary"].items():
        ndcg = stats["mean_ndcg_at_10"]
        cov = stats.get("mean_entity_coverage_top10")
        lat = stats["mean_wallclock_ms"]
        print(
            f"  {variant:<10}  "
            f"nDCG@10={'n/a' if ndcg is None else f'{ndcg:.4f}'}  "
            f"coverage@10={'n/a' if cov is None else f'{cov:.4f}'}  "
            f"latency={'n/a' if lat is None else f'{lat:>7.1f}ms'}  "
            f"timeouts={stats['n_timeouts']}/{stats['n_scored'] + stats['n_timeouts']}"
        )
    print()


if __name__ == "__main__":
    main()
