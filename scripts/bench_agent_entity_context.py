#!/usr/bin/env python3
"""Benchmark the agent_entity_context rewrite (work unit d2-mv-rewrite).

Compares the per-materialization p95 latency of the original LATERAL
``count(*)`` shape (pre-055) against the CTE-based shape (post-055) on a
synthetic bench schema of ``--batch`` entities.

The benchmark times the full inline SELECT that defines the MV body — the
same work that REFRESH MATERIALIZED VIEW performs — because that is where
the LATERAL subquery's per-entity cost actually bites. Writes a JSON report
to ``.claude/prd-build-artifacts/bench-d2-mv-rewrite.json``.

Usage:
    python scripts/bench_agent_entity_context.py --db scix_test --batch 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import psycopg

logger = logging.getLogger("bench_agent_entity_context")

# Per-entity fan-out for synthetic data. Tuned so the LATERAL subquery's
# N index scans clearly dominate runtime vs a single GROUP BY scan.
# Production scale has O(10M) document_entities for O(100K) entities —
# ~100:1. We seed 200:1 for headroom. Identifiers/aliases/relationships are
# kept at realistic per-entity fan-outs (see migrations/021_entity_graph.sql
# and matview_benchmark.py for reference densities).
DOC_ENTITIES_PER_ENTITY = 200
IDENTIFIERS_PER_ENTITY = 1.2
ALIASES_PER_ENTITY = 0.5
RELATIONSHIPS_PER_ENTITY = 0.2

# Number of timed repetitions per variant. p95 over 10+ samples is stable
# enough for the >=5x acceptance margin without making the benchmark slow.
DEFAULT_ITERATIONS = 10

# Default artifact path (relative to repo root).
REPORT_PATH = Path(".claude/prd-build-artifacts/bench-d2-mv-rewrite.json")


@dataclass(frozen=True)
class VariantResult:
    """Per-variant timing summary."""

    variant: str
    iterations: int
    samples_ms: list[float]
    p50_ms: float
    p95_ms: float
    mean_ms: float


def _p95(samples: list[float]) -> float:
    """Return the 95th percentile (nearest-rank) of a list of floats."""
    if not samples:
        return float("nan")
    sorted_samples = sorted(samples)
    k = max(0, min(len(sorted_samples) - 1, int(round(0.95 * len(sorted_samples))) - 1))
    return sorted_samples[k]


@contextmanager
def _cleanup_schema(conn: psycopg.Connection, schema: str) -> Iterator[None]:
    """Drop the bench schema on exit, even when exceptions propagate."""
    try:
        yield
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            logger.info("Dropped bench schema %s", schema)
        except psycopg.Error:  # noqa: BLE001 — cleanup must not raise
            logger.exception("Failed to drop bench schema %s", schema)


def _build_bench_schema(conn: psycopg.Connection, schema: str) -> None:
    """Create the bench schema with tables mirroring the entity graph."""
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"""
            CREATE TABLE {schema}.entities (
                id SERIAL PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                discipline TEXT,
                source TEXT NOT NULL
            )
            """)
        cur.execute(f"""
            CREATE TABLE {schema}.entity_identifiers (
                entity_id INT REFERENCES {schema}.entities(id),
                id_scheme TEXT NOT NULL,
                external_id TEXT NOT NULL,
                is_primary BOOLEAN DEFAULT false,
                PRIMARY KEY (id_scheme, external_id)
            )
            """)
        cur.execute(f"""
            CREATE TABLE {schema}.entity_aliases (
                entity_id INT REFERENCES {schema}.entities(id),
                alias TEXT NOT NULL,
                alias_source TEXT,
                PRIMARY KEY (entity_id, alias)
            )
            """)
        cur.execute(f"""
            CREATE TABLE {schema}.entity_relationships (
                id SERIAL PRIMARY KEY,
                subject_entity_id INT REFERENCES {schema}.entities(id),
                predicate TEXT NOT NULL,
                object_entity_id INT REFERENCES {schema}.entities(id),
                confidence REAL DEFAULT 1.0
            )
            """)
        cur.execute(f"""
            CREATE TABLE {schema}.document_entities (
                bibcode TEXT NOT NULL,
                entity_id INT REFERENCES {schema}.entities(id),
                link_type TEXT NOT NULL,
                confidence REAL,
                PRIMARY KEY (bibcode, entity_id, link_type)
            )
            """)


def _seed_bench_data(conn: psycopg.Connection, schema: str, n_entities: int) -> None:
    """Populate the bench tables with synthetic data sized by n_entities."""
    n_doc_entities = n_entities * DOC_ENTITIES_PER_ENTITY
    n_identifiers = int(n_entities * IDENTIFIERS_PER_ENTITY)
    n_aliases = int(n_entities * ALIASES_PER_ENTITY)
    n_relationships = int(n_entities * RELATIONSHIPS_PER_ENTITY)

    with conn.cursor() as cur:
        logger.info("Seeding entities (%d)", n_entities)
        entity_types = "ARRAY['mission','instrument','telescope','dataset','object']"
        disciplines = "ARRAY['astronomy','physics','planetary','heliophysics','earth_science']"
        sources = "ARRAY['ads','wikidata','pds','spase','ascl']"
        cur.execute(
            f"""
            INSERT INTO {schema}.entities (canonical_name, entity_type, discipline, source)
            SELECT
                'entity_' || i,
                ({entity_types})[1 + (i %% 5)],
                ({disciplines})[1 + (i %% 5)],
                ({sources})[1 + (i %% 5)]
            FROM generate_series(1, %s) AS i
            """,
            (n_entities,),
        )

        if n_identifiers > 0:
            logger.info("Seeding entity_identifiers (%d)", n_identifiers)
            cur.execute(
                f"""
                INSERT INTO {schema}.entity_identifiers
                    (entity_id, id_scheme, external_id, is_primary)
                SELECT
                    1 + (i %% %s),
                    (ARRAY['wikidata','pds','doi','ror'])[1 + (i %% 4)],
                    'EXT_' || i,
                    (i %% 5 = 0)
                FROM generate_series(1, %s) AS i
                ON CONFLICT DO NOTHING
                """,
                (n_entities, n_identifiers),
            )

        if n_aliases > 0:
            logger.info("Seeding entity_aliases (%d)", n_aliases)
            cur.execute(
                f"""
                INSERT INTO {schema}.entity_aliases (entity_id, alias, alias_source)
                SELECT
                    1 + (i %% %s),
                    'alias_' || i,
                    'synthetic'
                FROM generate_series(1, %s) AS i
                ON CONFLICT DO NOTHING
                """,
                (n_entities, n_aliases),
            )

        if n_relationships > 0:
            logger.info("Seeding entity_relationships (%d)", n_relationships)
            cur.execute(
                f"""
                INSERT INTO {schema}.entity_relationships
                    (subject_entity_id, predicate, object_entity_id, confidence)
                SELECT
                    1 + (i %% %s),
                    (ARRAY['part_of','uses','related_to'])[1 + (i %% 3)],
                    1 + ((i * 7) %% %s),
                    0.8
                FROM generate_series(1, %s) AS i
                WHERE (i %% %s) <> ((i * 7) %% %s)
                """,
                (n_entities, n_entities, n_relationships, n_entities, n_entities),
            )

        logger.info("Seeding document_entities (%d)", n_doc_entities)
        cur.execute(
            f"""
            INSERT INTO {schema}.document_entities
                (bibcode, entity_id, link_type, confidence)
            SELECT
                'BENCH' || LPAD((1 + (i %% 100000))::text, 12, '0'),
                1 + (i %% %s),
                (ARRAY['extraction','keyword','citation'])[1 + (i %% 3)],
                0.5 + random() * 0.5
            FROM generate_series(1, %s) AS i
            ON CONFLICT DO NOTHING
            """,
            (n_entities, n_doc_entities),
        )

        cur.execute(f"CREATE INDEX ON {schema}.document_entities (entity_id)")
        cur.execute(f"CREATE INDEX ON {schema}.entity_identifiers (entity_id)")
        cur.execute(f"CREATE INDEX ON {schema}.entity_relationships (subject_entity_id)")
        cur.execute(f"ANALYZE {schema}.entities")
        cur.execute(f"ANALYZE {schema}.document_entities")
        cur.execute(f"ANALYZE {schema}.entity_identifiers")
        cur.execute(f"ANALYZE {schema}.entity_aliases")
        cur.execute(f"ANALYZE {schema}.entity_relationships")


def _sql_before(schema: str) -> str:
    """Original LATERAL-based SELECT (migration 024) — full MV body."""
    return f"""
    SELECT
        e.id AS entity_id,
        e.canonical_name,
        e.entity_type,
        e.discipline,
        e.source,
        COALESCE(
            jsonb_agg(DISTINCT jsonb_build_object('scheme', ei.id_scheme, 'id', ei.external_id))
                FILTER (WHERE ei.external_id IS NOT NULL),
            '[]'::jsonb
        ) AS identifiers,
        COALESCE(
            array_agg(DISTINCT ea.alias) FILTER (WHERE ea.alias IS NOT NULL),
            ARRAY[]::text[]
        ) AS aliases,
        COALESCE(
            jsonb_agg(DISTINCT jsonb_build_object(
                'predicate', er.predicate,
                'object_id', er.object_entity_id,
                'confidence', er.confidence
            )) FILTER (WHERE er.id IS NOT NULL),
            '[]'::jsonb
        ) AS relationships,
        COALESCE(cnt.doc_count, 0) AS citing_paper_count
    FROM {schema}.entities e
    LEFT JOIN {schema}.entity_identifiers ei ON ei.entity_id = e.id
    LEFT JOIN {schema}.entity_aliases ea ON ea.entity_id = e.id
    LEFT JOIN {schema}.entity_relationships er ON er.subject_entity_id = e.id
    LEFT JOIN LATERAL (
        SELECT count(*) AS doc_count
        FROM {schema}.document_entities de
        WHERE de.entity_id = e.id
    ) cnt ON true
    GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, cnt.doc_count
    """


def _sql_after(schema: str) -> str:
    """New CTE-based SELECT (migration 055) — full MV body."""
    return f"""
    WITH de_counts AS (
        SELECT entity_id, count(*) AS doc_count
        FROM {schema}.document_entities
        GROUP BY entity_id
    )
    SELECT
        e.id AS entity_id,
        e.canonical_name,
        e.entity_type,
        e.discipline,
        e.source,
        COALESCE(
            jsonb_agg(DISTINCT jsonb_build_object('scheme', ei.id_scheme, 'id', ei.external_id))
                FILTER (WHERE ei.external_id IS NOT NULL),
            '[]'::jsonb
        ) AS identifiers,
        COALESCE(
            array_agg(DISTINCT ea.alias) FILTER (WHERE ea.alias IS NOT NULL),
            ARRAY[]::text[]
        ) AS aliases,
        COALESCE(
            jsonb_agg(DISTINCT jsonb_build_object(
                'predicate', er.predicate,
                'object_id', er.object_entity_id,
                'confidence', er.confidence
            )) FILTER (WHERE er.id IS NOT NULL),
            '[]'::jsonb
        ) AS relationships,
        COALESCE(dc.doc_count, 0) AS citing_paper_count
    FROM {schema}.entities e
    LEFT JOIN {schema}.entity_identifiers ei ON ei.entity_id = e.id
    LEFT JOIN {schema}.entity_aliases ea ON ea.entity_id = e.id
    LEFT JOIN {schema}.entity_relationships er ON er.subject_entity_id = e.id
    LEFT JOIN de_counts dc ON dc.entity_id = e.id
    GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, dc.doc_count
    """


def _time_variant(
    conn: psycopg.Connection,
    sql: str,
    iterations: int,
    variant_label: str,
) -> VariantResult:
    """Execute ``sql`` ``iterations`` times and return per-run timings.

    Each iteration runs the full MV body (no WHERE filter) — this mirrors
    what REFRESH MATERIALIZED VIEW does, and is where the LATERAL
    count(*)'s per-entity cost dominates.
    """
    samples_ms: list[float] = []
    with conn.cursor() as cur:
        # One warm-up run to prime shared buffers / plan cache.
        cur.execute(sql)
        cur.fetchall()
        for _ in range(iterations):
            start = time.perf_counter()
            cur.execute(sql)
            cur.fetchall()
            samples_ms.append((time.perf_counter() - start) * 1000.0)
    return VariantResult(
        variant=variant_label,
        iterations=len(samples_ms),
        samples_ms=samples_ms,
        p50_ms=statistics.median(samples_ms),
        p95_ms=_p95(samples_ms),
        mean_ms=statistics.fmean(samples_ms),
    )


def run_benchmark(dsn: str, batch_size: int, iterations: int) -> dict:
    """Run the full benchmark and return the report dict.

    ``batch_size`` sets the number of entities in the bench schema — the
    materialization is then sized off that, because it's the per-entity
    LATERAL subquery scan whose cost we are optimising.
    """
    schema = f"bench_d2_{os.getpid()}"
    logger.info("Connecting to %s (bench schema %s)", dsn, schema)

    conn = psycopg.connect(dsn, autocommit=True)
    try:
        with _cleanup_schema(conn, schema):
            _build_bench_schema(conn, schema)
            _seed_bench_data(conn, schema, n_entities=batch_size)

            logger.info("Timing BEFORE variant (LATERAL count(*))")
            before = _time_variant(
                conn, _sql_before(schema), iterations=iterations, variant_label="before"
            )

            logger.info("Timing AFTER variant (CTE de_counts)")
            after = _time_variant(
                conn, _sql_after(schema), iterations=iterations, variant_label="after"
            )

            speedup_x = (before.p95_ms / after.p95_ms) if after.p95_ms > 0 else float("inf")

            return {
                "work_unit": "d2-mv-rewrite",
                "dsn": dsn,
                "batch": batch_size,
                "iterations": iterations,
                "n_entities": batch_size,
                "n_doc_entities": batch_size * DOC_ENTITIES_PER_ENTITY,
                "p95_ms_before": before.p95_ms,
                "p95_ms_after": after.p95_ms,
                "p50_ms_before": before.p50_ms,
                "p50_ms_after": after.p50_ms,
                "mean_ms_before": before.mean_ms,
                "mean_ms_after": after.mean_ms,
                "speedup_x": speedup_x,
                "before": asdict(before),
                "after": asdict(after),
            }
    finally:
        conn.close()


def _write_report(report: dict, path: Path) -> None:
    """Write the JSON report to disk, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote report to %s", path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=os.environ.get("SCIX_DSN", "scix_test"),
        help="Postgres DSN or database name (default: $SCIX_DSN or scix_test).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Entity batch size — number of entities in the bench schema (default 1000).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of timed iterations per variant (default {DEFAULT_ITERATIONS}).",
    )
    parser.add_argument(
        "--out",
        default=str(REPORT_PATH),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Python logging level.",
    )
    return parser.parse_args(argv)


def _resolve_dsn(db_arg: str) -> str:
    """Accept either a bare dbname or a full DSN."""
    if "=" in db_arg or db_arg.startswith("postgres://") or db_arg.startswith("postgresql://"):
        return db_arg
    return f"dbname={db_arg}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dsn = _resolve_dsn(args.db)

    report = run_benchmark(dsn=dsn, batch_size=args.batch, iterations=args.iterations)
    _write_report(report, Path(args.out))

    speedup = report["speedup_x"]
    logger.info(
        "p95 before=%.2f ms  after=%.2f ms  speedup=%.2fx",
        report["p95_ms_before"],
        report["p95_ms_after"],
        speedup,
    )
    if speedup < 5.0:
        logger.warning("Speedup %.2fx is below the 5x acceptance threshold", speedup)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
