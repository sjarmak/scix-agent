"""Tests for migration 055 — agent_entity_context MV rewrite."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import psycopg
import pytest

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1] / "migrations" / "055_agent_entity_context_rewrite.sql"
)


# ---------------------------------------------------------------------------
# Unit-level: migration file structure (no DB required)
# ---------------------------------------------------------------------------


class TestMigrationFile:
    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.is_file(), f"missing migration: {MIGRATION_PATH}"

    def test_wraps_in_transaction(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert "BEGIN;" in sql
        assert "COMMIT;" in sql

    def test_drops_old_materialized_view(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert "DROP MATERIALIZED VIEW IF EXISTS agent_entity_context" in sql

    def test_creates_materialized_view_with_cte(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert "CREATE MATERIALIZED VIEW" in sql
        assert "agent_entity_context" in sql
        # The whole point of the rewrite: a pre-aggregated CTE for doc counts.
        assert "de_counts" in sql.lower() or "WITH" in sql.upper()
        assert "FROM document_entities" in sql
        assert "GROUP BY entity_id" in sql

    def test_no_lateral_count(self) -> None:
        """The rewrite MUST NOT contain the LATERAL count(*) pattern."""
        sql = MIGRATION_PATH.read_text().lower()
        # The old pattern — regression check.
        assert "left join lateral" not in sql
        assert "cnt.doc_count" not in sql

    def test_preserves_unique_index(self) -> None:
        sql = MIGRATION_PATH.read_text()
        assert "CREATE UNIQUE INDEX" in sql
        assert "idx_agent_entity_ctx_id" in sql
        assert "agent_entity_context (entity_id)" in sql

    def test_preserves_column_set(self) -> None:
        sql = MIGRATION_PATH.read_text()
        # Spot-check each column name is present.
        for col in (
            "entity_id",
            "canonical_name",
            "entity_type",
            "discipline",
            "source",
            "identifiers",
            "aliases",
            "relationships",
            "citing_paper_count",
        ):
            assert col in sql, f"missing column {col!r}"


# ---------------------------------------------------------------------------
# Integration: apply migration on a live DB and row-diff the semantics
# ---------------------------------------------------------------------------

BENCH_SCHEMA = f"test_d2_{os.getpid()}"

# Small deterministic dataset (enough to exercise aliases/identifiers/
# relationships/doc-counts variety but fast to set up).
N_ENTITIES = 60
N_DOC_ENTITIES = 600
N_IDENTIFIERS = 40
N_ALIASES = 80
N_RELATIONSHIPS = 25


def _connect_or_skip() -> psycopg.Connection:
    """Connect to scix_test or skip the integration test."""
    dsn = os.environ.get("SCIX_DSN", "dbname=scix_test")
    try:
        return psycopg.connect(dsn, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"scix_test unreachable: {exc}")


def _build_bench_schema(conn: psycopg.Connection, schema: str) -> None:
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

        # Seed entities
        cur.execute(
            f"""
            INSERT INTO {schema}.entities (canonical_name, entity_type, discipline, source)
            SELECT
                'ent_' || i,
                (ARRAY['mission','instrument','telescope'])[1 + (i %% 3)],
                'astronomy',
                (ARRAY['ads','wikidata'])[1 + (i %% 2)]
            FROM generate_series(1, %s) AS i
            """,
            (N_ENTITIES,),
        )

        # Identifiers
        cur.execute(
            f"""
            INSERT INTO {schema}.entity_identifiers (entity_id, id_scheme, external_id, is_primary)
            SELECT
                1 + (i %% %s),
                (ARRAY['wikidata','doi'])[1 + (i %% 2)],
                'EXT_' || i,
                (i %% 3 = 0)
            FROM generate_series(1, %s) AS i
            ON CONFLICT DO NOTHING
            """,
            (N_ENTITIES, N_IDENTIFIERS),
        )

        # Aliases
        cur.execute(
            f"""
            INSERT INTO {schema}.entity_aliases (entity_id, alias, alias_source)
            SELECT
                1 + (i %% %s),
                'alias_' || i,
                'test'
            FROM generate_series(1, %s) AS i
            ON CONFLICT DO NOTHING
            """,
            (N_ENTITIES, N_ALIASES),
        )

        # Relationships
        cur.execute(
            f"""
            INSERT INTO {schema}.entity_relationships
                (subject_entity_id, predicate, object_entity_id, confidence)
            SELECT
                1 + (i %% %s),
                (ARRAY['part_of','uses'])[1 + (i %% 2)],
                1 + ((i * 7) %% %s),
                0.9
            FROM generate_series(1, %s) AS i
            WHERE (i %% %s) <> ((i * 7) %% %s)
            """,
            (N_ENTITIES, N_ENTITIES, N_RELATIONSHIPS, N_ENTITIES, N_ENTITIES),
        )

        # Document entities — non-uniform distribution so some entities
        # have 0 doc links and others have many.
        cur.execute(
            f"""
            INSERT INTO {schema}.document_entities
                (bibcode, entity_id, link_type, confidence)
            SELECT
                'T' || LPAD(((i %% 200) + 1)::text, 12, '0'),
                1 + ((i * 13) %% %s),
                (ARRAY['extraction','keyword'])[1 + (i %% 2)],
                0.9
            FROM generate_series(1, %s) AS i
            ON CONFLICT DO NOTHING
            """,
            (N_ENTITIES, N_DOC_ENTITIES),
        )

        cur.execute(f"CREATE INDEX ON {schema}.document_entities (entity_id)")
        cur.execute(f"CREATE INDEX ON {schema}.entity_identifiers (entity_id)")
        cur.execute(f"CREATE INDEX ON {schema}.entity_relationships (subject_entity_id)")
        cur.execute(f"ANALYZE {schema}.entities")
        cur.execute(f"ANALYZE {schema}.document_entities")


def _sql_before(schema: str) -> str:
    """Exact semantics of the pre-055 MV body, parameterised by schema."""
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
    WHERE e.id = ANY(%s)
    GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, cnt.doc_count
    ORDER BY e.id
    """


def _sql_after(schema: str) -> str:
    """Exact semantics of the post-055 MV body, parameterised by schema."""
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
    WHERE e.id = ANY(%s)
    GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, dc.doc_count
    ORDER BY e.id
    """


def _normalize_row(row: tuple) -> tuple:
    """Sort inner JSONB arrays / alias list so two semantically-equal rows compare equal."""
    (
        entity_id,
        canonical_name,
        entity_type,
        discipline,
        source,
        identifiers,
        aliases,
        relationships,
        count,
    ) = row

    def _sort_jsonb_list(lst):
        # jsonb_agg returns a Python list of dicts from psycopg.
        if not lst:
            return []
        return sorted(lst, key=lambda d: tuple(sorted(d.items())))

    return (
        entity_id,
        canonical_name,
        entity_type,
        discipline,
        source,
        _sort_jsonb_list(identifiers),
        sorted(aliases or []),
        _sort_jsonb_list(relationships),
        count,
    )


@pytest.mark.integration
class TestRowDiff:
    def test_semantic_equality_on_sample(self) -> None:
        """The pre-055 and post-055 MV bodies must return identical rows."""
        conn = _connect_or_skip()
        try:
            _build_bench_schema(conn, BENCH_SCHEMA)
            with conn.cursor() as cur:
                cur.execute(f"SELECT id FROM {BENCH_SCHEMA}.entities ORDER BY id")
                all_ids = [r[0] for r in cur.fetchall()]

            # Sample 50 entity_ids deterministically.
            rng = random.Random(0xD1FFD2)
            sample_size = min(50, len(all_ids))
            sample_ids = sorted(rng.sample(all_ids, sample_size))

            with conn.cursor() as cur:
                cur.execute(_sql_before(BENCH_SCHEMA), (sample_ids,))
                rows_before = [_normalize_row(r) for r in cur.fetchall()]
                cur.execute(_sql_after(BENCH_SCHEMA), (sample_ids,))
                rows_after = [_normalize_row(r) for r in cur.fetchall()]

            assert (
                len(rows_before) == sample_size
            ), f"expected {sample_size} before-rows, got {len(rows_before)}"
            assert len(rows_after) == len(
                rows_before
            ), f"row count mismatch: before={len(rows_before)} after={len(rows_after)}"
            for b, a in zip(rows_before, rows_after):
                assert b == a, f"row diverged:\nbefore={b}\nafter ={a}"
        finally:
            try:
                with conn.cursor() as cur:
                    cur.execute(f"DROP SCHEMA IF EXISTS {BENCH_SCHEMA} CASCADE")
            finally:
                conn.close()


@pytest.mark.integration
class TestMigrationApplies:
    def test_migration_applies_cleanly(self) -> None:
        """psql -d scix_test -f migrations/055_*.sql exits 0 and preserves row count."""
        dsn = os.environ.get("SCIX_DSN", "dbname=scix_test")
        try:
            conn = psycopg.connect(dsn, autocommit=True)
        except psycopg.OperationalError as exc:
            pytest.skip(f"scix_test unreachable: {exc}")

        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM agent_entity_context")
                count_before = cur.fetchone()[0]

            # Apply via psql (matches the acceptance criterion exactly).
            db_name = "scix_test"
            result = subprocess.run(
                ["psql", "-d", db_name, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_PATH)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert (
                result.returncode == 0
            ), f"psql failed: stdout={result.stdout!r} stderr={result.stderr!r}"

            with conn.cursor() as cur:
                # Materialized views don't appear in information_schema.columns;
                # use pg_attribute for authoritative column metadata.
                cur.execute("""
                    SELECT a.attname
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = 'agent_entity_context'
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    ORDER BY a.attnum
                    """)
                columns = [r[0] for r in cur.fetchall()]

            expected = {
                "entity_id",
                "canonical_name",
                "entity_type",
                "discipline",
                "source",
                "identifiers",
                "aliases",
                "relationships",
                "citing_paper_count",
            }
            assert (
                set(columns) == expected
            ), f"column set mismatch: got {set(columns)} expected {expected}"

            with conn.cursor() as cur:
                cur.execute("REFRESH MATERIALIZED VIEW agent_entity_context")
                cur.execute("SELECT count(*) FROM agent_entity_context")
                count_after = cur.fetchone()[0]

            assert (
                count_after == count_before
            ), f"row count changed: before={count_before} after={count_after}"
        finally:
            conn.close()
