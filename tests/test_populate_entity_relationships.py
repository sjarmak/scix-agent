"""Integration tests for scripts/populate_entity_relationships.py.

Runs against ``SCIX_TEST_DSN`` (defaults to ``dbname=scix_test``).
Creates fixture entities, runs each per-source extractor, and asserts
the expected edges land in public.entity_relationships with the right
predicate and evidence.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import uuid

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the script as a module (hyphens in "populate_entity_relationships"
# are fine; keeping the dash-free filename makes it directly importable).
_SCRIPT_PATH = REPO_ROOT / "scripts" / "populate_entity_relationships.py"
_spec = importlib.util.spec_from_file_location("populate_entity_relationships", _SCRIPT_PATH)
populate = importlib.util.module_from_spec(_spec)
# Register in sys.modules before exec so @dataclass can resolve its own module
sys.modules["populate_entity_relationships"] = populate
_spec.loader.exec_module(populate)  # type: ignore[union-attr]

from scix.db import is_production_dsn  # noqa: E402

TEST_DSN = os.environ.get("SCIX_TEST_DSN", "dbname=scix_test")


pytestmark = pytest.mark.skipif(
    is_production_dsn(TEST_DSN),
    reason="Refusing to run destructive tests against production DSN",
)


@pytest.fixture
def conn() -> psycopg.Connection:
    """Yield a connection and rollback on exit so tests are isolated."""
    c = psycopg.connect(TEST_DSN)
    yield c
    c.rollback()
    c.close()


@pytest.fixture
def harvest_run_id(conn: psycopg.Connection) -> int:
    """Create a harvest_runs row for FK-safe inserts."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO harvest_runs (source, status)
            VALUES (%s, 'running')
            RETURNING id
            """,
            ("test_entity_relationships",),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


# Marker string to namespace fixture rows so we never collide with
# anything another test or past run left behind.
_RUN_TAG = f"test-{uuid.uuid4().hex[:8]}"


def _seed_entity(
    conn: psycopg.Connection,
    *,
    canonical: str,
    entity_type: str,
    source: str,
    properties: dict | None = None,
) -> int:
    """Insert an entity and return its id (idempotent in tests)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, source, properties)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
                SET properties = EXCLUDED.properties
            RETURNING id
            """,
            (canonical, entity_type, source, json.dumps(properties or {})),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# GCMD
# ---------------------------------------------------------------------------


def test_populate_gcmd_inserts_parent_of_edges(
    conn: psycopg.Connection, harvest_run_id: int
) -> None:
    # Seed a three-level GCMD hierarchy
    name_root = f"ROOT_{_RUN_TAG}"
    name_mid = f"MID_{_RUN_TAG}"
    name_leaf = f"LEAF_{_RUN_TAG}"
    scheme = "sciencekeywords"

    root_id = _seed_entity(
        conn,
        canonical=name_root,
        entity_type="observable",
        source="gcmd",
        properties={"gcmd_scheme": scheme, "gcmd_hierarchy": name_root},
    )
    mid_id = _seed_entity(
        conn,
        canonical=name_mid,
        entity_type="observable",
        source="gcmd",
        properties={
            "gcmd_scheme": scheme,
            "gcmd_hierarchy": f"{name_root} > {name_mid}",
        },
    )
    leaf_id = _seed_entity(
        conn,
        canonical=name_leaf,
        entity_type="observable",
        source="gcmd",
        properties={
            "gcmd_scheme": scheme,
            "gcmd_hierarchy": f"{name_root} > {name_mid} > {name_leaf}",
        },
    )
    conn.commit()

    stats = populate.populate_gcmd(conn, harvest_run_id=harvest_run_id)
    assert stats.edges_emitted >= 2
    assert stats.edges_inserted >= 2

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT subject_entity_id, predicate, object_entity_id, source
              FROM entity_relationships
             WHERE source = 'gcmd'
               AND (subject_entity_id, object_entity_id) IN (
                   (%s, %s), (%s, %s)
               )
            """,
            (root_id, mid_id, mid_id, leaf_id),
        )
        edges = cur.fetchall()
    edge_pairs = {(r[0], r[2]) for r in edges}
    assert (root_id, mid_id) in edge_pairs
    assert (mid_id, leaf_id) in edge_pairs
    assert all(r[1] == "parent_of" for r in edges)


def test_populate_gcmd_is_idempotent(conn: psycopg.Connection, harvest_run_id: int) -> None:
    name_root = f"IDEMP_ROOT_{_RUN_TAG}"
    name_leaf = f"IDEMP_LEAF_{_RUN_TAG}"
    scheme = "sciencekeywords"

    _seed_entity(
        conn,
        canonical=name_root,
        entity_type="observable",
        source="gcmd",
        properties={"gcmd_scheme": scheme, "gcmd_hierarchy": name_root},
    )
    _seed_entity(
        conn,
        canonical=name_leaf,
        entity_type="observable",
        source="gcmd",
        properties={
            "gcmd_scheme": scheme,
            "gcmd_hierarchy": f"{name_root} > {name_leaf}",
        },
    )
    conn.commit()

    first = populate.populate_gcmd(conn, harvest_run_id=harvest_run_id)
    second = populate.populate_gcmd(conn, harvest_run_id=harvest_run_id)
    # Second run should insert nothing (ON CONFLICT)
    assert second.edges_inserted == 0
    assert first.edges_inserted >= 1


# ---------------------------------------------------------------------------
# SPASE
# ---------------------------------------------------------------------------


def test_populate_spase_region(conn: psycopg.Connection, harvest_run_id: int) -> None:
    parent_name = f"TestPlanet_{_RUN_TAG}"
    child_name = f"{parent_name}.Moon"

    parent_id = _seed_entity(
        conn,
        canonical=parent_name,
        entity_type="observable",
        source="spase",
        properties={"spase_list": "ObservedRegion"},
    )
    child_id = _seed_entity(
        conn,
        canonical=child_name,
        entity_type="observable",
        source="spase",
        properties={"spase_list": "ObservedRegion"},
    )
    conn.commit()

    stats = populate.populate_spase(conn, harvest_run_id=harvest_run_id)
    assert stats.edges_inserted >= 1

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT predicate FROM entity_relationships
             WHERE subject_entity_id = %s AND object_entity_id = %s
            """,
            (parent_id, child_id),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "parent_of"


# ---------------------------------------------------------------------------
# SsODNet
# ---------------------------------------------------------------------------


def test_populate_ssodnet_creates_taxa_and_edges(
    conn: psycopg.Connection, harvest_run_id: int
) -> None:
    # Seed two asteroid targets with class paths that share a prefix
    ast1 = f"TestAst1_{_RUN_TAG}"
    ast2 = f"TestAst2_{_RUN_TAG}"
    # Use unique class names so we don't clash with any real taxa
    class_a = f"TestClassA_{_RUN_TAG}"
    class_b = f"TestClassB_{_RUN_TAG}"

    _seed_entity(
        conn,
        canonical=ast1,
        entity_type="target",
        source="ssodnet",
        properties={"sso_class": f"{class_a}>{class_b}"},
    )
    _seed_entity(
        conn,
        canonical=ast2,
        entity_type="target",
        source="ssodnet",
        properties={"sso_class": f"{class_a}>{class_b}"},
    )
    conn.commit()

    stats = populate.populate_ssodnet(conn, harvest_run_id=harvest_run_id, include_targets=True)
    assert stats.taxa_created >= 2
    # Expect at least: 1 class->class + 2 asteroid->leaf part_of
    assert stats.edges_inserted >= 3

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT canonical_name FROM entities
             WHERE source='ssodnet' AND entity_type='taxon'
               AND canonical_name IN (%s, %s)
            """,
            (class_a, f"{class_a}>{class_b}"),
        )
        names = {r[0] for r in cur.fetchall()}
    assert class_a in names
    assert f"{class_a}>{class_b}" in names

    with conn.cursor() as cur:
        # class->class parent_of edge must exist
        cur.execute(
            """
            SELECT count(*) FROM entity_relationships er
              JOIN entities a ON a.id = er.subject_entity_id
              JOIN entities b ON b.id = er.object_entity_id
             WHERE a.canonical_name = %s
               AND b.canonical_name = %s
               AND er.predicate = 'parent_of'
               AND er.source = 'ssodnet'
            """,
            (class_a, f"{class_a}>{class_b}"),
        )
        cnt = cur.fetchone()[0]
    assert cnt == 1


def test_populate_ssodnet_skip_targets_omits_part_of(
    conn: psycopg.Connection, harvest_run_id: int
) -> None:
    ast = f"TestAstSkip_{_RUN_TAG}"
    class_x = f"TestClassSkipX_{_RUN_TAG}"
    class_y = f"TestClassSkipY_{_RUN_TAG}"
    _seed_entity(
        conn,
        canonical=ast,
        entity_type="target",
        source="ssodnet",
        properties={"sso_class": f"{class_x}>{class_y}"},
    )
    conn.commit()

    stats = populate.populate_ssodnet(conn, harvest_run_id=harvest_run_id, include_targets=False)
    # class->class only; no part_of edges
    assert stats.edges_inserted == 1
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM entity_relationships
             WHERE predicate = 'part_of'
               AND source = 'ssodnet'
               AND evidence->>'target' = %s
            """,
            (ast,),
        )
        part_of_cnt = cur.fetchone()[0]
    assert part_of_cnt == 0


# ---------------------------------------------------------------------------
# Curated flagship
# ---------------------------------------------------------------------------


def test_populate_curated_flagship(conn: psycopg.Connection, harvest_run_id: int) -> None:
    # Seed fixture JWST + NIRSpec (use test-tag so we don't collide with
    # the real curated seed)
    mission_name = "James Webb Space Telescope"
    instrument_name = "NIRSpec"

    mission_id = _seed_entity(
        conn,
        canonical=mission_name,
        entity_type="mission",
        source="curated_flagship_v1",
    )
    _seed_entity(
        conn,
        canonical=instrument_name,
        entity_type="instrument",
        source=f"gcmd_test_{_RUN_TAG}",
    )
    conn.commit()

    populate.populate_curated_flagship(conn, harvest_run_id=harvest_run_id)

    # Look up the edge by mission subject + any NIRSpec object — the
    # populate script picks the lowest-id NIRSpec across the DB which
    # may not be the one this test seeded (prior test runs leave rows).
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT er.predicate, er.source, er.evidence
              FROM entity_relationships er
              JOIN entities e ON e.id = er.object_entity_id
             WHERE er.subject_entity_id = %s
               AND e.canonical_name = %s
               AND er.predicate = 'has_instrument'
            """,
            (mission_id, instrument_name),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "has_instrument"
    assert row[1] == "curated_flagship_v1"
    assert row[2]["mission"] == mission_name


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


def test_run_populates_multiple_sources(
    conn: psycopg.Connection,
) -> None:
    """Running the full pipeline through run() commits via its own conn."""
    # Seed minimal fixtures for gcmd + spase + curated_flagship
    scheme = "sciencekeywords"
    gcmd_root = f"E2E_ROOT_{_RUN_TAG}"
    gcmd_leaf = f"E2E_LEAF_{_RUN_TAG}"
    _seed_entity(
        conn,
        canonical=gcmd_root,
        entity_type="observable",
        source="gcmd",
        properties={"gcmd_scheme": scheme, "gcmd_hierarchy": gcmd_root},
    )
    _seed_entity(
        conn,
        canonical=gcmd_leaf,
        entity_type="observable",
        source="gcmd",
        properties={
            "gcmd_scheme": scheme,
            "gcmd_hierarchy": f"{gcmd_root} > {gcmd_leaf}",
        },
    )
    conn.commit()

    # run() uses this same connection and manages its own harvest_run
    stats_list = populate.run(
        conn,
        sources=["gcmd"],
        include_ssodnet_targets=False,
        ssodnet_limit=None,
    )
    assert len(stats_list) == 1
    assert stats_list[0].source == "gcmd"
    assert stats_list[0].edges_inserted >= 1
