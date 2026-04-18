"""Integration tests for migration 049 + scripts/promote_staging_extractions.py.

Safety: these tests write to staging and public tables and therefore
require ``SCIX_TEST_DSN`` to be set and to *not* point at the production
``scix`` database.  They SKIP cleanly otherwise so running ``pytest`` in
a plain checkout never touches production data.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MIGRATION_PATH = REPO_ROOT / "migrations" / "049_staging_ner_extractions.sql"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import get_test_dsn  # noqa: E402


def _load_promote_module():
    """Import scripts/promote_staging_extractions.py as a module.

    scripts/ is not a package so we use importlib to load the file by
    path.  Falling back to this pattern (rather than mutating sys.path
    to include scripts/) keeps the script self-contained and mirrors how
    the cron-style scripts are consumed elsewhere in this repo.
    """
    mod_name = "promote_staging_extractions"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        SCRIPTS_DIR / "promote_staging_extractions.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: dataclass(frozen=True) + PEP 563
    # annotations need sys.modules[cls.__module__] to be populated.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# SCIX_TEST_DSN guard — destructive tests MUST have a dedicated test DB
# ---------------------------------------------------------------------------

TEST_DSN = get_test_dsn()

pytestmark = pytest.mark.skipif(
    TEST_DSN is None,
    reason=(
        "SCIX_TEST_DSN is not set or points at production — "
        "promote_staging_extractions tests require a dedicated test DB"
    ),
)


# Safety marker — a prefix we can use to delete only test rows.
TEST_SOURCE = "PROMOTE_STG_TEST"
TEST_BIBCODE_PREFIX = "PSTGTEST."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    assert TEST_DSN is not None  # narrow for the type checker
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migration(dsn: str) -> None:
    """Apply migration 049 to the test DB (idempotent)."""
    sql = MIGRATION_PATH.read_text()
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(sql)


@pytest.fixture
def conn(dsn: str, applied_migration: None) -> psycopg.Connection:
    c = psycopg.connect(dsn)
    c.autocommit = False
    try:
        yield c
    finally:
        c.rollback()
        c.close()


def _cleanup(conn: psycopg.Connection) -> None:
    """Idempotent removal of any leftover test rows."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM public.extraction_entity_links WHERE source = %s",
            (TEST_SOURCE,),
        )
        cur.execute(
            "DELETE FROM public.extractions WHERE bibcode LIKE %s",
            (TEST_BIBCODE_PREFIX + "%",),
        )
        cur.execute(
            "DELETE FROM staging.extraction_entity_links WHERE source = %s",
            (TEST_SOURCE,),
        )
        cur.execute(
            "DELETE FROM staging.extractions WHERE bibcode LIKE %s",
            (TEST_BIBCODE_PREFIX + "%",),
        )
        cur.execute(
            "DELETE FROM public.papers WHERE bibcode LIKE %s",
            (TEST_BIBCODE_PREFIX + "%",),
        )
    conn.commit()


def _seed_papers(conn: psycopg.Connection, bibcodes: list[str]) -> None:
    """Ensure the given bibcodes exist in public.papers (FK target)."""
    with conn.cursor() as cur:
        for bib in bibcodes:
            cur.execute(
                """
                INSERT INTO public.papers (bibcode, title, year, doctype)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (bibcode) DO NOTHING
                """,
                (bib, "Test Paper", 2024, "article"),
            )
    conn.commit()


@pytest.fixture(autouse=True)
def _isolate_test(conn: psycopg.Connection):
    _cleanup(conn)
    yield
    _cleanup(conn)


# ---------------------------------------------------------------------------
# Schema / migration tests
# ---------------------------------------------------------------------------


def test_migration_is_idempotent(dsn: str) -> None:
    """Re-applying migration 049 must not raise."""
    sql = MIGRATION_PATH.read_text()
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(sql)
            cur.execute(sql)


def test_staging_tables_are_logged(conn: psycopg.Connection) -> None:
    """Every staging.* table and public.extraction_entity_links must be LOGGED."""
    expected = [
        ("staging", "extractions"),
        ("staging", "extraction_entity_links"),
        ("staging", "extraction_entity_links_software"),
        ("staging", "extraction_entity_links_instrument"),
        ("staging", "extraction_entity_links_dataset"),
        ("staging", "extraction_entity_links_method"),
        ("staging", "extraction_entity_links_default"),
        ("public", "extraction_entity_links"),
    ]
    with conn.cursor() as cur:
        for schema, table in expected:
            cur.execute(
                """
                SELECT cl.relpersistence
                  FROM pg_class cl
                  JOIN pg_namespace n ON n.oid = cl.relnamespace
                 WHERE n.nspname = %s AND cl.relname = %s
                """,
                (schema, table),
            )
            row = cur.fetchone()
            assert row is not None, f"{schema}.{table} missing"
            assert row[0] == "p", f"{schema}.{table} is not LOGGED (got {row[0]!r})"


def test_entity_links_is_partitioned_by_list(conn: psycopg.Connection) -> None:
    """Parent table is PARTITION BY LIST (entity_type) with expected partitions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT partstrat
              FROM pg_partitioned_table pt
              JOIN pg_class c ON c.oid = pt.partrelid
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'staging' AND c.relname = 'extraction_entity_links'
            """
        )
        row = cur.fetchone()
        assert row is not None, "extraction_entity_links is not partitioned"
        assert row[0] == "l", f"expected LIST partitioning (l), got {row[0]!r}"

        cur.execute(
            """
            SELECT c.relname
              FROM pg_inherits i
              JOIN pg_class c ON c.oid = i.inhrelid
              JOIN pg_class p ON p.oid = i.inhparent
              JOIN pg_namespace n ON n.oid = p.relnamespace
             WHERE n.nspname = 'staging' AND p.relname = 'extraction_entity_links'
             ORDER BY c.relname
            """
        )
        partitions = {r[0] for r in cur.fetchall()}
    required = {
        "extraction_entity_links_software",
        "extraction_entity_links_instrument",
        "extraction_entity_links_dataset",
        "extraction_entity_links_method",
    }
    assert required.issubset(partitions), (
        f"missing required partitions: {required - partitions}"
    )


def test_provenance_columns_on_staging_extractions(conn: psycopg.Connection) -> None:
    """staging.extractions carries source + confidence_tier provenance."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
              FROM information_schema.columns
             WHERE table_schema = 'staging' AND table_name = 'extractions'
            """
        )
        cols = {name: dtype for name, dtype in cur.fetchall()}
    assert cols.get("source") == "text"
    assert cols.get("confidence_tier") == "smallint"
    assert cols.get("extraction_version") == "text"
    assert "timestamp" in (cols.get("created_at") or "")


# ---------------------------------------------------------------------------
# Promotion tests
# ---------------------------------------------------------------------------


def _insert_staging_extraction(
    cur: psycopg.Cursor,
    bibcode: str,
    extraction_type: str,
    extraction_version: str,
    payload_json: str,
) -> None:
    cur.execute(
        """
        INSERT INTO staging.extractions
            (bibcode, extraction_type, extraction_version, payload,
             source, confidence_tier)
        VALUES (%s, %s, %s, %s::jsonb, %s, %s)
        """,
        (bibcode, extraction_type, extraction_version, payload_json, TEST_SOURCE, 1),
    )


def _insert_staging_entity_link(
    cur: psycopg.Cursor,
    bibcode: str,
    entity_type: str,
    surface: str,
    extraction_version: str = "ner_v1",
) -> None:
    cur.execute(
        """
        INSERT INTO staging.extraction_entity_links
            (bibcode, entity_type, entity_surface, source,
             confidence_tier, confidence, extraction_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (bibcode, entity_type, surface, TEST_SOURCE, 1, 0.95, extraction_version),
    )


def test_promote_extractions_roundtrip(conn: psycopg.Connection) -> None:
    mod = _load_promote_module()
    bibcodes = [f"{TEST_BIBCODE_PREFIX}e{i:03d}" for i in range(3)]
    _seed_papers(conn, bibcodes)

    with conn.cursor() as cur:
        for bib in bibcodes:
            _insert_staging_extraction(cur, bib, "ner", "ner_v1", '{"entities": []}')
    conn.commit()

    counts = mod.promote(conn, batch_size=100, dry_run=False)

    assert counts.extractions == 3
    assert counts.entity_links == 0

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM public.extractions WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        assert cur.fetchone()[0] == 3


def test_promote_entity_links_roundtrip(conn: psycopg.Connection) -> None:
    mod = _load_promote_module()
    bibcodes = [f"{TEST_BIBCODE_PREFIX}l{i:03d}" for i in range(4)]
    _seed_papers(conn, bibcodes)

    with conn.cursor() as cur:
        # One row per named partition, exercising LIST routing.
        _insert_staging_entity_link(cur, bibcodes[0], "software", "astropy")
        _insert_staging_entity_link(cur, bibcodes[1], "instrument", "HST/WFC3")
        _insert_staging_entity_link(cur, bibcodes[2], "dataset", "SDSS DR17")
        _insert_staging_entity_link(cur, bibcodes[3], "method", "PCA")
    conn.commit()

    counts = mod.promote(conn, batch_size=100, dry_run=False)

    assert counts.entity_links == 4

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT entity_type, entity_surface
              FROM public.extraction_entity_links
             WHERE source = %s
             ORDER BY entity_type
            """,
            (TEST_SOURCE,),
        )
        rows = cur.fetchall()
    assert [r[0] for r in rows] == ["dataset", "instrument", "method", "software"]


def test_dry_run_rolls_back(conn: psycopg.Connection) -> None:
    mod = _load_promote_module()
    bib = f"{TEST_BIBCODE_PREFIX}dry001"
    _seed_papers(conn, [bib])

    with conn.cursor() as cur:
        _insert_staging_extraction(cur, bib, "ner", "ner_v1", "{}")
        _insert_staging_entity_link(cur, bib, "software", "numpy")
    conn.commit()

    counts = mod.promote(conn, batch_size=100, dry_run=True)

    # Counts reflect what *would* have been promoted.
    assert counts.extractions == 1
    assert counts.entity_links == 1

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.extractions WHERE bibcode = %s", (bib,))
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT count(*) FROM public.extraction_entity_links WHERE source = %s",
            (TEST_SOURCE,),
        )
        assert cur.fetchone()[0] == 0


def test_on_conflict_do_nothing(conn: psycopg.Connection) -> None:
    """Re-promoting identical staging rows does not duplicate public rows."""
    mod = _load_promote_module()
    bib = f"{TEST_BIBCODE_PREFIX}dup001"
    _seed_papers(conn, [bib])

    with conn.cursor() as cur:
        _insert_staging_extraction(cur, bib, "ner", "ner_v1", '{"k": 1}')
        _insert_staging_entity_link(cur, bib, "software", "scipy")
    conn.commit()

    # First promotion copies the rows into public.
    mod.promote(conn, batch_size=100, dry_run=False)

    # The promotion script does NOT delete from staging — re-running it
    # should re-attempt the same INSERTs and hit ON CONFLICT DO NOTHING.
    counts = mod.promote(conn, batch_size=100, dry_run=False)
    assert counts.extractions == 0
    assert counts.entity_links == 0

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.extractions WHERE bibcode = %s", (bib,))
        assert cur.fetchone()[0] == 1
        cur.execute(
            """
            SELECT count(*) FROM public.extraction_entity_links
             WHERE bibcode = %s AND source = %s
            """,
            (bib, TEST_SOURCE),
        )
        assert cur.fetchone()[0] == 1


def test_source_filter_restricts_promotion(conn: psycopg.Connection) -> None:
    mod = _load_promote_module()
    bib_a = f"{TEST_BIBCODE_PREFIX}srcA"
    bib_b = f"{TEST_BIBCODE_PREFIX}srcB"
    _seed_papers(conn, [bib_a, bib_b])

    other_source = "OTHER_PROMOTE_TEST"
    with conn.cursor() as cur:
        _insert_staging_extraction(cur, bib_a, "ner", "ner_v1", "{}")
        cur.execute(
            """
            INSERT INTO staging.extractions
                (bibcode, extraction_type, extraction_version, payload,
                 source, confidence_tier)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s)
            """,
            (bib_b, "ner", "ner_v1", "{}", other_source, 1),
        )
    conn.commit()

    counts = mod.promote(conn, batch_size=100, dry_run=False, source_filter=TEST_SOURCE)
    assert counts.extractions == 1

    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode FROM public.extractions WHERE bibcode = ANY(%s)",
            ([bib_a, bib_b],),
        )
        rows = [r[0] for r in cur.fetchall()]
    assert rows == [bib_a]

    # Clean up the OTHER_PROMOTE_TEST row from staging so the autouse
    # _isolate_test fixture (which matches on TEST_SOURCE / TEST_BIBCODE_PREFIX)
    # does not leave it behind.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM staging.extractions WHERE source = %s", (other_source,))
    conn.commit()
