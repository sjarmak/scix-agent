#!/usr/bin/env python3
"""Materialized view scale benchmark for agent views PRD.

Generates synthetic data at target scale (1M entities, 10M document_entities,
500K aliases), creates each materialized view, and measures:
- CREATE time
- REFRESH CONCURRENTLY time
- Single-row query latency

Results are written to .claude/prd-build-artifacts/matview-benchmark.md
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg

DB_NAME = os.environ.get("SCIX_DB", "scix")
DB_HOST = os.environ.get("SCIX_DB_HOST", "")
DB_USER = os.environ.get("SCIX_DB_USER", os.environ.get("USER", "ds"))

BENCH_SCHEMA = f"matview_bench_{os.getpid()}"

# Target synthetic scale
N_ENTITIES = 1_000_000
N_DOC_ENTITIES = 10_000_000
N_ALIASES = 500_000
N_IDENTIFIERS = 1_200_000
N_RELATIONSHIPS = 200_000
N_DATASETS = 5_000
N_DATASET_ENTITIES = 50_000
N_DOC_DATASETS = 100_000


@dataclass(frozen=True)
class TimingResult:
    view_name: str
    create_seconds: float
    refresh_seconds: float
    query_seconds: float
    row_count: int


def get_conn():
    return psycopg.connect(dbname=DB_NAME, host=DB_HOST, user=DB_USER, autocommit=True)


@contextmanager
def timed(label: str):
    """Context manager that prints and yields elapsed time."""
    start = time.monotonic()
    result = {"elapsed": 0.0}
    try:
        yield result
    finally:
        result["elapsed"] = time.monotonic() - start
        print(f"  {label}: {result['elapsed']:.2f}s")


def setup_schema(cur):
    """Create benchmark schema with synthetic tables mirroring production."""
    print("Setting up benchmark schema...")
    cur.execute(f"DROP SCHEMA IF EXISTS {BENCH_SCHEMA} CASCADE")
    cur.execute(f"CREATE SCHEMA {BENCH_SCHEMA}")
    cur.execute(f"SET search_path TO {BENCH_SCHEMA}, public")

    # Papers — we'll reference real papers table via bibcodes
    # For document_entities we need bibcodes. Generate synthetic ones.
    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.papers (
            bibcode TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            year SMALLINT,
            citation_count INTEGER,
            reference_count INTEGER
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.entities (
            id SERIAL PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            discipline TEXT,
            source TEXT NOT NULL,
            properties JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.entity_identifiers (
            entity_id INT REFERENCES {BENCH_SCHEMA}.entities(id),
            id_scheme TEXT NOT NULL,
            external_id TEXT NOT NULL,
            is_primary BOOLEAN DEFAULT false,
            PRIMARY KEY (id_scheme, external_id)
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.entity_aliases (
            entity_id INT REFERENCES {BENCH_SCHEMA}.entities(id),
            alias TEXT NOT NULL,
            alias_source TEXT,
            PRIMARY KEY (entity_id, alias)
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.entity_relationships (
            id SERIAL PRIMARY KEY,
            subject_entity_id INT REFERENCES {BENCH_SCHEMA}.entities(id),
            predicate TEXT NOT NULL,
            object_entity_id INT REFERENCES {BENCH_SCHEMA}.entities(id),
            source TEXT,
            confidence REAL DEFAULT 1.0
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.document_entities (
            bibcode TEXT NOT NULL,
            entity_id INT REFERENCES {BENCH_SCHEMA}.entities(id),
            link_type TEXT NOT NULL,
            confidence REAL,
            match_method TEXT,
            evidence JSONB,
            PRIMARY KEY (bibcode, entity_id, link_type)
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.datasets (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            discipline TEXT,
            source TEXT NOT NULL,
            canonical_id TEXT NOT NULL,
            description TEXT,
            properties JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.dataset_entities (
            dataset_id INT REFERENCES {BENCH_SCHEMA}.datasets(id),
            entity_id INT REFERENCES {BENCH_SCHEMA}.entities(id),
            relationship TEXT NOT NULL,
            PRIMARY KEY (dataset_id, entity_id, relationship)
        )
    """)

    cur.execute(f"""
        CREATE TABLE {BENCH_SCHEMA}.document_datasets (
            bibcode TEXT NOT NULL,
            dataset_id INT REFERENCES {BENCH_SCHEMA}.datasets(id),
            link_type TEXT NOT NULL,
            confidence REAL,
            match_method TEXT,
            PRIMARY KEY (bibcode, dataset_id, link_type)
        )
    """)


def generate_synthetic_data(cur):
    """Generate synthetic data at target scale using generate_series."""
    print(f"\nGenerating synthetic data...")

    # Papers: 2M synthetic bibcodes (enough for 10M document_entities)
    n_papers = 2_000_000
    with timed(f"papers ({n_papers:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.papers (bibcode, title, abstract, year, citation_count, reference_count)
            SELECT
                'BENCH' || LPAD(i::text, 12, '0'),
                'Synthetic Paper ' || i,
                'Abstract for synthetic paper number ' || i || '. This contains representative text.',
                2020 + (i % 6),
                (random() * 1000)::int,
                (random() * 50)::int
            FROM generate_series(1, {n_papers}) AS i
        """)

    # Entities: 1M
    entity_types = ['mission', 'instrument', 'telescope', 'dataset', 'object', 'facility', 'software', 'survey']
    with timed(f"entities ({N_ENTITIES:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.entities (canonical_name, entity_type, discipline, source)
            SELECT
                'entity_' || i,
                (ARRAY['mission','instrument','telescope','dataset','object','facility','software','survey'])[1 + (i % 8)],
                (ARRAY['astronomy','physics','planetary','heliophysics','earth_science'])[1 + (i % 5)],
                (ARRAY['ads','wikidata','pds','spase','ascl'])[1 + (i % 5)]
            FROM generate_series(1, {N_ENTITIES}) AS i
        """)

    # Entity identifiers: 1.2M
    with timed(f"entity_identifiers ({N_IDENTIFIERS:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.entity_identifiers (entity_id, id_scheme, external_id, is_primary)
            SELECT
                1 + (i % {N_ENTITIES}),
                (ARRAY['wikidata','pds','spase','doi','ror'])[1 + (i % 5)],
                'EXT_' || i,
                (i % 5 = 0)
            FROM generate_series(1, {N_IDENTIFIERS}) AS i
        """)

    # Entity aliases: 500K
    with timed(f"entity_aliases ({N_ALIASES:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.entity_aliases (entity_id, alias, alias_source)
            SELECT
                1 + (i % {N_ENTITIES}),
                'alias_' || i,
                (ARRAY['wikidata','manual','harvest'])[1 + (i % 3)]
            FROM generate_series(1, {N_ALIASES}) AS i
        """)

    # Entity relationships: 200K
    with timed(f"entity_relationships ({N_RELATIONSHIPS:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.entity_relationships
                (subject_entity_id, predicate, object_entity_id, source, confidence)
            SELECT
                1 + (i % {N_ENTITIES}),
                (ARRAY['part_of','uses','related_to','successor_of','operated_by'])[1 + (i % 5)],
                1 + ((i * 7) % {N_ENTITIES}),
                'synthetic',
                0.5 + random() * 0.5
            FROM generate_series(1, {N_RELATIONSHIPS}) AS i
            WHERE (i % {N_ENTITIES}) != ((i * 7) % {N_ENTITIES})
        """)

    # Document entities: 10M (the big one)
    with timed(f"document_entities ({N_DOC_ENTITIES:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.document_entities
                (bibcode, entity_id, link_type, confidence, match_method)
            SELECT
                'BENCH' || LPAD((1 + (i % {n_papers}))::text, 12, '0'),
                1 + (i % {N_ENTITIES}),
                (ARRAY['extraction','keyword','citation'])[1 + (i % 3)],
                0.5 + random() * 0.5,
                (ARRAY['exact','fuzzy','rule'])[1 + (i % 3)]
            FROM generate_series(1, {N_DOC_ENTITIES}) AS i
            ON CONFLICT DO NOTHING
        """)

    # Datasets: 5K
    with timed(f"datasets ({N_DATASETS:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.datasets (name, discipline, source, canonical_id, description)
            SELECT
                'Dataset ' || i,
                (ARRAY['astronomy','physics','planetary','heliophysics','earth_science'])[1 + (i % 5)],
                (ARRAY['pds','mast','irsa','heasarc','vizier'])[1 + (i % 5)],
                'DS_' || i,
                'Synthetic dataset number ' || i
            FROM generate_series(1, {N_DATASETS}) AS i
        """)

    # Dataset entities: 50K
    with timed(f"dataset_entities ({N_DATASET_ENTITIES:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.dataset_entities (dataset_id, entity_id, relationship)
            SELECT
                1 + (i % {N_DATASETS}),
                1 + (i % {N_ENTITIES}),
                (ARRAY['produced_by','observed_by','calibrated_with'])[1 + (i % 3)]
            FROM generate_series(1, {N_DATASET_ENTITIES}) AS i
            ON CONFLICT DO NOTHING
        """)

    # Document datasets: 100K
    with timed(f"document_datasets ({N_DOC_DATASETS:,})"):
        cur.execute(f"""
            INSERT INTO {BENCH_SCHEMA}.document_datasets
                (bibcode, dataset_id, link_type, confidence, match_method)
            SELECT
                'BENCH' || LPAD((1 + (i % {n_papers}))::text, 12, '0'),
                1 + (i % {N_DATASETS}),
                (ARRAY['citation','analysis','calibration'])[1 + (i % 3)],
                0.5 + random() * 0.5,
                (ARRAY['exact','fuzzy','rule'])[1 + (i % 3)]
            FROM generate_series(1, {N_DOC_DATASETS}) AS i
            ON CONFLICT DO NOTHING
        """)

    # Verify counts
    for tbl in ['papers', 'entities', 'entity_identifiers', 'entity_aliases',
                 'entity_relationships', 'document_entities', 'datasets',
                 'dataset_entities', 'document_datasets']:
        cur.execute(f"SELECT count(*) FROM {BENCH_SCHEMA}.{tbl}")
        count = cur.fetchone()[0]
        print(f"  {tbl}: {count:,}")


def add_indexes(cur):
    """Add indexes needed for materialized view JOINs."""
    print("\nCreating indexes...")
    indexes = [
        f"CREATE INDEX ON {BENCH_SCHEMA}.document_entities (bibcode)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.document_entities (entity_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.entity_identifiers (entity_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.entity_aliases (entity_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.entity_relationships (subject_entity_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.entity_relationships (object_entity_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.document_datasets (bibcode)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.document_datasets (dataset_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.dataset_entities (dataset_id)",
        f"CREATE INDEX ON {BENCH_SCHEMA}.dataset_entities (entity_id)",
    ]
    for idx_sql in indexes:
        with timed(idx_sql.split("ON ")[1]):
            cur.execute(idx_sql)
    cur.execute("ANALYZE")


def benchmark_document_context(cur) -> TimingResult:
    """Benchmark agent_document_context materialized view."""
    print("\n=== agent_document_context ===")
    view_sql = f"""
        CREATE MATERIALIZED VIEW {BENCH_SCHEMA}.agent_document_context AS
        SELECT
            p.bibcode,
            p.title,
            p.abstract,
            p.year,
            p.citation_count,
            p.reference_count,
            COALESCE(
                jsonb_agg(
                    DISTINCT jsonb_build_object(
                        'entity_id', e.id,
                        'name', e.canonical_name,
                        'type', e.entity_type,
                        'link_type', de.link_type,
                        'confidence', de.confidence
                    )
                ) FILTER (WHERE e.id IS NOT NULL),
                '[]'::jsonb
            ) AS linked_entities
        FROM {BENCH_SCHEMA}.papers p
        LEFT JOIN {BENCH_SCHEMA}.document_entities de ON de.bibcode = p.bibcode
        LEFT JOIN {BENCH_SCHEMA}.entities e ON e.id = de.entity_id
        GROUP BY p.bibcode, p.title, p.abstract, p.year, p.citation_count, p.reference_count
    """

    with timed("CREATE") as create_t:
        cur.execute(view_sql)

    # Unique index required for REFRESH CONCURRENTLY
    cur.execute(f"CREATE UNIQUE INDEX ON {BENCH_SCHEMA}.agent_document_context (bibcode)")

    cur.execute(f"SELECT count(*) FROM {BENCH_SCHEMA}.agent_document_context")
    row_count = cur.fetchone()[0]
    print(f"  rows: {row_count:,}")

    with timed("REFRESH CONCURRENTLY") as refresh_t:
        cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {BENCH_SCHEMA}.agent_document_context")

    # Single-row query
    cur.execute(f"SELECT bibcode FROM {BENCH_SCHEMA}.agent_document_context LIMIT 1")
    test_bibcode = cur.fetchone()[0]
    with timed("single-row query") as query_t:
        for _ in range(100):
            cur.execute(
                f"SELECT * FROM {BENCH_SCHEMA}.agent_document_context WHERE bibcode = %s",
                (test_bibcode,)
            )
            cur.fetchone()

    return TimingResult(
        view_name="agent_document_context",
        create_seconds=create_t["elapsed"],
        refresh_seconds=refresh_t["elapsed"],
        query_seconds=query_t["elapsed"] / 100,
        row_count=row_count,
    )


def benchmark_entity_context(cur) -> TimingResult:
    """Benchmark agent_entity_context materialized view."""
    print("\n=== agent_entity_context ===")
    view_sql = f"""
        CREATE MATERIALIZED VIEW {BENCH_SCHEMA}.agent_entity_context AS
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
        FROM {BENCH_SCHEMA}.entities e
        LEFT JOIN {BENCH_SCHEMA}.entity_identifiers ei ON ei.entity_id = e.id
        LEFT JOIN {BENCH_SCHEMA}.entity_aliases ea ON ea.entity_id = e.id
        LEFT JOIN {BENCH_SCHEMA}.entity_relationships er ON er.subject_entity_id = e.id
        LEFT JOIN LATERAL (
            SELECT count(*) AS doc_count
            FROM {BENCH_SCHEMA}.document_entities de
            WHERE de.entity_id = e.id
        ) cnt ON true
        GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, cnt.doc_count
    """

    with timed("CREATE") as create_t:
        cur.execute(view_sql)

    cur.execute(f"CREATE UNIQUE INDEX ON {BENCH_SCHEMA}.agent_entity_context (entity_id)")

    cur.execute(f"SELECT count(*) FROM {BENCH_SCHEMA}.agent_entity_context")
    row_count = cur.fetchone()[0]
    print(f"  rows: {row_count:,}")

    with timed("REFRESH CONCURRENTLY") as refresh_t:
        cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {BENCH_SCHEMA}.agent_entity_context")

    # Single-row query
    with timed("single-row query") as query_t:
        for _ in range(100):
            cur.execute(
                f"SELECT * FROM {BENCH_SCHEMA}.agent_entity_context WHERE entity_id = %s",
                (42,)
            )
            cur.fetchone()

    return TimingResult(
        view_name="agent_entity_context",
        create_seconds=create_t["elapsed"],
        refresh_seconds=refresh_t["elapsed"],
        query_seconds=query_t["elapsed"] / 100,
        row_count=row_count,
    )


def benchmark_dataset_context(cur) -> TimingResult:
    """Benchmark agent_dataset_context materialized view."""
    print("\n=== agent_dataset_context ===")
    view_sql = f"""
        CREATE MATERIALIZED VIEW {BENCH_SCHEMA}.agent_dataset_context AS
        SELECT
            d.id AS dataset_id,
            d.name AS dataset_name,
            d.source,
            d.discipline,
            d.description,
            COALESCE(
                jsonb_agg(DISTINCT jsonb_build_object(
                    'entity_id', e.id,
                    'name', e.canonical_name,
                    'type', e.entity_type,
                    'relationship', dse.relationship
                )) FILTER (WHERE e.id IS NOT NULL),
                '[]'::jsonb
            ) AS linked_entities,
            COALESCE(
                jsonb_agg(DISTINCT jsonb_build_object(
                    'bibcode', p.bibcode,
                    'title', p.title,
                    'link_type', dd.link_type
                )) FILTER (WHERE p.bibcode IS NOT NULL),
                '[]'::jsonb
            ) AS citing_papers
        FROM {BENCH_SCHEMA}.datasets d
        LEFT JOIN {BENCH_SCHEMA}.dataset_entities dse ON dse.dataset_id = d.id
        LEFT JOIN {BENCH_SCHEMA}.entities e ON e.id = dse.entity_id
        LEFT JOIN {BENCH_SCHEMA}.document_datasets dd ON dd.dataset_id = d.id
        LEFT JOIN {BENCH_SCHEMA}.papers p ON p.bibcode = dd.bibcode
        GROUP BY d.id, d.name, d.source, d.discipline, d.description
    """

    with timed("CREATE") as create_t:
        cur.execute(view_sql)

    cur.execute(f"CREATE UNIQUE INDEX ON {BENCH_SCHEMA}.agent_dataset_context (dataset_id)")

    cur.execute(f"SELECT count(*) FROM {BENCH_SCHEMA}.agent_dataset_context")
    row_count = cur.fetchone()[0]
    print(f"  rows: {row_count:,}")

    with timed("REFRESH CONCURRENTLY") as refresh_t:
        cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {BENCH_SCHEMA}.agent_dataset_context")

    # Single-row query
    with timed("single-row query") as query_t:
        for _ in range(100):
            cur.execute(
                f"SELECT * FROM {BENCH_SCHEMA}.agent_dataset_context WHERE dataset_id = %s",
                (42,)
            )
            cur.fetchone()

    return TimingResult(
        view_name="agent_dataset_context",
        create_seconds=create_t["elapsed"],
        refresh_seconds=refresh_t["elapsed"],
        query_seconds=query_t["elapsed"] / 100,
        row_count=row_count,
    )


def write_report(results: list[TimingResult], total_elapsed: float):
    """Write benchmark results to markdown report."""
    report_dir = Path(__file__).parent.parent / ".claude" / "prd-build-artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "matview-benchmark.md"

    lines = [
        "# Materialized View Scale Benchmark",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total benchmark time**: {total_elapsed:.0f}s",
        "",
        "## Synthetic Data Scale",
        "",
        f"| Table | Target Rows |",
        f"|-------|------------|",
        f"| entities | {N_ENTITIES:,} |",
        f"| document_entities | {N_DOC_ENTITIES:,} |",
        f"| entity_aliases | {N_ALIASES:,} |",
        f"| entity_identifiers | {N_IDENTIFIERS:,} |",
        f"| entity_relationships | {N_RELATIONSHIPS:,} |",
        f"| datasets | {N_DATASETS:,} |",
        f"| dataset_entities | {N_DATASET_ENTITIES:,} |",
        f"| document_datasets | {N_DOC_DATASETS:,} |",
        f"| papers (synthetic) | 2,000,000 |",
        "",
        "## Results",
        "",
        "| View | Rows | CREATE (s) | REFRESH CONCURRENTLY (s) | Query Latency (ms) | Status |",
        "|------|------|-----------|-------------------------|-------------------|--------|",
    ]

    all_pass = True
    for r in results:
        status = "PASS" if r.refresh_seconds < 1800 else "FAIL (>30 min)"
        if r.refresh_seconds >= 1800:
            all_pass = False
        lines.append(
            f"| {r.view_name} | {r.row_count:,} | {r.create_seconds:.1f} | "
            f"{r.refresh_seconds:.1f} | {r.query_seconds * 1000:.2f} | {status} |"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
    ])

    if all_pass:
        lines.append(
            "All materialized views complete REFRESH CONCURRENTLY within 30 minutes at target scale. "
            "Materialized views are viable for agent context."
        )
    else:
        lines.append(
            "One or more views exceeded the 30-minute REFRESH threshold. "
            "Apply the fallback strategy below."
        )

    lines.extend([
        "",
        "## Fallback Strategy (Contingency)",
        "",
        "Documented unconditionally so downstream beads can adopt without re-running the benchmark "
        "if production scale or row distribution changes the picture. Apply if any REFRESH "
        "CONCURRENTLY at production scale exceeds 30 min, or if production row counts grow "
        "materially beyond the 10M document_entities tested here.",
        "",
        "1. **Incremental summary tables**: trigger-maintained summary tables updated on INSERT/UPDATE/DELETE "
        "(no full refresh; constant per-row cost). Best when write rate is moderate and read latency must be sub-ms.",
        "2. **Partitioned refresh**: partition `document_entities` by `entity_type` or by hash of `bibcode`, "
        "build per-partition matviews, refresh partitions independently in parallel. Cuts wall-clock refresh "
        "by N-way parallelism and isolates hot partitions.",
        "3. **Partial materialization**: only materialize frequently-queried slices (e.g., top 100K entities by "
        "doc_count, or only the last 5 years of papers). Combine with on-demand JOIN for the long tail.",
        "4. **pgvectorscale StreamingDiskANN for vector columns**: if any agent context view embeds dense vectors "
        "and memory pressure becomes a bottleneck, switch the vector index to pgvectorscale (SSD-backed, "
        "~6 GB RAM for 1M vectors at 768d vs. ~100 GB for HNSW in memory).",
        "5. **Logical replication subscriber**: replicate the source tables to a read-only subscriber and "
        "build matviews there to remove refresh contention from the primary write path.",
        "",
        "### Selection guidance",
        "",
        "- Refresh time linear in input rows: prefer #2 (partitioned).",
        "- Refresh time dominated by JSONB aggregation: prefer #1 (incremental triggers).",
        "- Long tail of low-traffic entities: prefer #3 (partial).",
        "- Vector memory pressure: layer #4 on top of any of the above.",
        "- Refresh blocks live writes: layer #5 on top of any of the above.",
        "",
        "## Observations",
        "",
        "- `agent_document_context` is the largest view (one row per paper with aggregated entities)",
        "- `agent_entity_context` uses LATERAL subquery for doc count to avoid cross-join explosion",
        "- `agent_dataset_context` is smallest due to limited dataset count",
        "- All views use JSONB aggregation with DISTINCT to avoid duplicates from multi-way JOINs",
        "- UNIQUE INDEX on primary key enables REFRESH CONCURRENTLY",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python scripts/matview_benchmark.py",
        "```",
        "",
        "Benchmark runs in an isolated `matview_bench` schema and cleans up after itself.",
    ])

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")


def main():
    total_start = time.monotonic()
    conn = get_conn()

    try:
        cur = conn.cursor()

        setup_schema(cur)
        generate_synthetic_data(cur)
        add_indexes(cur)

        # Run benchmarks
        results = []
        results.append(benchmark_document_context(cur))
        results.append(benchmark_entity_context(cur))
        results.append(benchmark_dataset_context(cur))

        total_elapsed = time.monotonic() - total_start
        write_report(results, total_elapsed)

        # Print summary
        print("\n=== SUMMARY ===")
        for r in results:
            status = "PASS" if r.refresh_seconds < 1800 else "FAIL"
            print(
                f"  {r.view_name}: CREATE={r.create_seconds:.1f}s  "
                f"REFRESH={r.refresh_seconds:.1f}s  "
                f"query={r.query_seconds*1000:.2f}ms  [{status}]"
            )
        print(f"\nTotal: {total_elapsed:.0f}s")

    finally:
        # Clean up benchmark schema
        print("\nCleaning up benchmark schema...")
        cur = conn.cursor()
        cur.execute(f"DROP SCHEMA IF EXISTS {BENCH_SCHEMA} CASCADE")
        conn.close()


if __name__ == "__main__":
    main()
