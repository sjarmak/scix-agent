"""Integration tests for migration 051 + scripts/compute_semantic_communities.py.

These tests write to and delete from ``paper_embeddings`` / ``paper_metrics``
and therefore require ``SCIX_TEST_DSN`` to be set to a non-production DB.
They SKIP cleanly otherwise so ``pytest`` in a plain checkout never touches
the production ``scix`` database.

Covers:
    (a) end-to-end run on a 1K-vector fixture → paper_metrics populated
    (b) determinism: same seed → identical cluster assignments (ARI=1.0)
    (c) migration idempotence: drop columns, re-apply, verify
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MIGRATION_PATH = REPO_ROOT / "migrations" / "051_community_semantic_columns.sql"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import get_test_dsn  # noqa: E402

TEST_DSN = get_test_dsn()

pytestmark = pytest.mark.skipif(
    TEST_DSN is None,
    reason=(
        "SCIX_TEST_DSN is not set or points at production — "
        "compute_semantic_communities tests require a dedicated test DB"
    ),
)


TEST_BIBCODE_PREFIX = "SEMCOMMTEST."
TEST_BIBCODE_TEMPLATE = TEST_BIBCODE_PREFIX + "{i:04d}"
N_FIXTURE_POINTS = 1_000
N_FIXTURE_CLUSTERS = 3
SEED = 42


# ---------------------------------------------------------------------------
# Dynamic module import for the script
# ---------------------------------------------------------------------------


def _load_script_module():
    mod_name = "compute_semantic_communities_script"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        SCRIPTS_DIR / "compute_semantic_communities.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    assert TEST_DSN is not None
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migration(dsn: str) -> None:
    """Apply migration 051 to the test DB (idempotent)."""
    sql = MIGRATION_PATH.read_text()
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(sql)


def _pgvector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"


def _insert_fixture(dsn: str) -> None:
    """Insert 1K synthetic papers + INDUS embeddings in 3 Gaussian clusters."""
    rng = np.random.default_rng(SEED)
    dim = 768

    # Three far-apart centroids so k-means cluster discovery is unambiguous.
    centroids = np.zeros((N_FIXTURE_CLUSTERS, dim), dtype=np.float32)
    for i in range(N_FIXTURE_CLUSTERS):
        centroids[i, i * 64 : (i + 1) * 64] = 5.0  # non-overlapping hot blocks

    per_cluster = N_FIXTURE_POINTS // N_FIXTURE_CLUSTERS
    bibcodes: list[str] = []
    vectors: list[np.ndarray] = []
    for i in range(N_FIXTURE_POINTS):
        cluster_idx = min(i // per_cluster, N_FIXTURE_CLUSTERS - 1)
        noise = rng.normal(0, 0.1, size=dim).astype(np.float32)
        vectors.append(centroids[cluster_idx] + noise)
        bibcodes.append(TEST_BIBCODE_TEMPLATE.format(i=i))

    with psycopg.connect(dsn) as c:
        c.autocommit = False
        with c.cursor() as cur:
            # papers rows (INDUS FK requires papers.bibcode)
            cur.executemany(
                "INSERT INTO papers (bibcode, title) VALUES (%s, %s) "
                "ON CONFLICT (bibcode) DO NOTHING",
                [(b, f"fixture {b}") for b in bibcodes],
            )
            # embeddings
            with cur.copy(
                "COPY paper_embeddings (bibcode, model_name, embedding, input_type) "
                "FROM STDIN"
            ) as copy:
                for bib, vec in zip(bibcodes, vectors):
                    copy.write_row((bib, "indus", _pgvector_literal(vec), "title_abstract"))
        c.commit()


def _delete_fixture(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_metrics WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM paper_embeddings WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )


@pytest.fixture
def fixture_data(dsn: str, applied_migration: None):
    """Insert 1K synthetic fixture rows, delete after test."""
    _delete_fixture(dsn)
    _insert_fixture(dsn)
    try:
        yield
    finally:
        _delete_fixture(dsn)


# ---------------------------------------------------------------------------
# Helper: fetch assignments
# ---------------------------------------------------------------------------


def _fetch_assignments(dsn: str) -> dict[str, dict[str, int | None]]:
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT bibcode, community_semantic_coarse, community_semantic_medium, "
            "community_semantic_fine "
            "FROM paper_metrics "
            "WHERE bibcode LIKE %s "
            "ORDER BY bibcode",
            (TEST_BIBCODE_PREFIX + "%",),
        )
        return {
            row[0]: {
                "coarse": row[1],
                "medium": row[2],
                "fine": row[3],
            }
            for row in cur.fetchall()
        }


def _run_script(dsn: str, seed: int, tmp_path: pathlib.Path) -> None:
    mod = _load_script_module()
    args = [
        "--dsn", dsn,
        "--k-coarse", "3",
        "--k-medium", "5",
        "--k-fine", "8",
        "--seed", str(seed),
        "--batch-size", "500",
        "--silhouette-sample", "500",
        "--results-path", str(tmp_path / "results.json"),
        "--run-meta-path", str(tmp_path / "run_meta.json"),
    ]
    rc = mod.main(args)
    assert rc == 0, f"script returned non-zero exit code {rc}"


# ---------------------------------------------------------------------------
# Test (a) — end-to-end run populates paper_metrics for all fixture bibcodes
# ---------------------------------------------------------------------------


def test_end_to_end_populates_all_bibcodes(
    dsn: str,
    fixture_data: None,
    tmp_path: pathlib.Path,
) -> None:
    _run_script(dsn, seed=SEED, tmp_path=tmp_path)

    assignments = _fetch_assignments(dsn)
    assert len(assignments) == N_FIXTURE_POINTS, (
        f"expected {N_FIXTURE_POINTS} paper_metrics rows, got {len(assignments)}"
    )

    for bib, cols in assignments.items():
        assert cols["coarse"] is not None, f"{bib} has NULL community_semantic_coarse"
        assert cols["medium"] is not None, f"{bib} has NULL community_semantic_medium"
        assert cols["fine"] is not None, f"{bib} has NULL community_semantic_fine"

    # results.json + run_meta.json were written
    results_path = tmp_path / "results.json"
    meta_path = tmp_path / "run_meta.json"
    assert results_path.exists()
    assert meta_path.exists()

    import json
    results = json.loads(results_path.read_text())
    assert "resolutions" in results
    assert set(results["resolutions"].keys()) == {"coarse", "medium", "fine"}
    assert "peak_rss_mb" in results

    meta = json.loads(meta_path.read_text())
    assert "run_id" in meta and meta["run_id"]
    assert "git_sha" in meta
    assert meta["params"]["seed"] == SEED


# ---------------------------------------------------------------------------
# Test (b) — identical seed → identical assignments (ARI = 1.0)
# ---------------------------------------------------------------------------


def _adjusted_rand_index(a: list[int], b: list[int]) -> float:
    """Small inline ARI to avoid importing sklearn at test-collect time."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def test_deterministic_with_same_seed(
    dsn: str,
    fixture_data: None,
    tmp_path: pathlib.Path,
) -> None:
    _run_script(dsn, seed=SEED, tmp_path=tmp_path / "run1")
    first = _fetch_assignments(dsn)

    _run_script(dsn, seed=SEED, tmp_path=tmp_path / "run2")
    second = _fetch_assignments(dsn)

    assert first.keys() == second.keys()

    for resolution in ("coarse", "medium", "fine"):
        labels_a = [first[b][resolution] for b in sorted(first)]
        labels_b = [second[b][resolution] for b in sorted(second)]
        ari = _adjusted_rand_index(labels_a, labels_b)
        assert ari == pytest.approx(1.0), (
            f"resolution {resolution!r}: same-seed ARI={ari}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Test (c) — migration is idempotent and works on a fresh schema
# ---------------------------------------------------------------------------


def test_migration_idempotent_fresh_db(dsn: str, applied_migration: None) -> None:
    sql = MIGRATION_PATH.read_text()

    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            # Drop all three columns (and their btree indexes via CASCADE).
            cur.execute(
                "ALTER TABLE paper_metrics "
                "DROP COLUMN IF EXISTS community_semantic_coarse CASCADE, "
                "DROP COLUMN IF EXISTS community_semantic_medium CASCADE, "
                "DROP COLUMN IF EXISTS community_semantic_fine CASCADE"
            )

            # First apply — must create columns + indexes.
            cur.execute(sql)
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='paper_metrics' "
                "  AND column_name LIKE 'community_semantic_%' "
                "ORDER BY column_name"
            )
            cols = [r[0] for r in cur.fetchall()]
            assert cols == [
                "community_semantic_coarse",
                "community_semantic_fine",
                "community_semantic_medium",
            ]

            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename='paper_metrics' "
                "  AND indexname LIKE 'idx_pm_community_semantic_%' "
                "ORDER BY indexname"
            )
            idxs = [r[0] for r in cur.fetchall()]
            assert idxs == [
                "idx_pm_community_semantic_coarse",
                "idx_pm_community_semantic_fine",
                "idx_pm_community_semantic_medium",
            ]

            # Second apply — must be a no-op (no exceptions).
            cur.execute(sql)


# ---------------------------------------------------------------------------
# Safety: non-production DSN default behaviour
# ---------------------------------------------------------------------------


def test_script_refuses_production_dsn_without_allow_prod(tmp_path: pathlib.Path) -> None:
    mod = _load_script_module()
    # A synthetic prod DSN — we never actually connect, the guard fires first.
    rc = mod.main([
        "--dsn", "dbname=scix",
        "--k-coarse", "3",
        "--k-medium", "5",
        "--k-fine", "8",
        "--seed", "1",
        "--batch-size", "100",
        "--silhouette-sample", "100",
        "--results-path", str(tmp_path / "r.json"),
        "--run-meta-path", str(tmp_path / "m.json"),
    ])
    assert rc == 2, "production DSN without --allow-prod must exit with code 2"
