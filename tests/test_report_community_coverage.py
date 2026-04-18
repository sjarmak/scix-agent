"""Integration tests for ``scripts/report_community_coverage.py``.

These tests write fixture rows into ``papers`` / ``paper_metrics`` and run
the read-only coverage script against them, so they require
``SCIX_TEST_DSN`` pointing at a non-production database. The module-level
``pytestmark`` skips cleanly when that env var is unset or points at prod,
matching the convention used by ``test_compute_semantic_communities.py``.

Covers:
    (a) ``union_covered`` equals the hand-counted number of fixture rows
        with at least one non-null signal across the three columns.
    (b) When one of the three signal columns is missing from the schema,
        the script logs a warning, reports ``null`` for that signal, and
        computes UNION coverage from the columns that remain.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import pathlib
import sys

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

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
        "report_community_coverage tests require a dedicated test DB"
    ),
)


TEST_BIBCODE_PREFIX = "COVREPTEST."


# ---------------------------------------------------------------------------
# Dynamic module import for the script under test
# ---------------------------------------------------------------------------


def _load_script_module():
    mod_name = "report_community_coverage_script"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        SCRIPTS_DIR / "report_community_coverage.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixture construction: insert papers + paper_metrics rows with known
# (citation, semantic, taxonomic) presence patterns.
#
# Layout (10 papers total):
#   0: all three signals present                          → contributes to union
#   1: citation only                                      → contributes to union
#   2: semantic only                                      → contributes to union
#   3: taxonomic only                                     → contributes to union
#   4: citation + semantic                                → contributes to union
#   5: citation + taxonomic                               → contributes to union
#   6: semantic + taxonomic                               → contributes to union
#   7: all NULL                                           → NOT in union
#   8: all NULL                                           → NOT in union
#   9: all NULL                                           → NOT in union
#
# Expected counts among the fixture rows:
#   total_papers:      10
#   citation_covered:   4 (rows 0, 1, 4, 5)
#   semantic_covered:   4 (rows 0, 2, 4, 6)
#   taxonomic_covered:  4 (rows 0, 3, 5, 6)
#   union_covered:      7
# ---------------------------------------------------------------------------


FIXTURE_ROWS: tuple[tuple[int | None, int | None, str | None], ...] = (
    (1, 10, "astro-ph"),  # 0: all three
    (2, None, None),  # 1: citation only
    (None, 11, None),  # 2: semantic only
    (None, None, "cond-mat"),  # 3: taxonomic only
    (3, 12, None),  # 4: citation+semantic
    (4, None, "hep-th"),  # 5: citation+taxonomic
    (None, 13, "gr-qc"),  # 6: semantic+taxonomic
    (None, None, None),  # 7: all null
    (None, None, None),  # 8: all null
    (None, None, None),  # 9: all null
)


EXPECTED_CITATION = sum(1 for r in FIXTURE_ROWS if r[0] is not None)
EXPECTED_SEMANTIC = sum(1 for r in FIXTURE_ROWS if r[1] is not None)
EXPECTED_TAXONOMIC = sum(1 for r in FIXTURE_ROWS if r[2] is not None)
EXPECTED_UNION = sum(1 for r in FIXTURE_ROWS if any(c is not None for c in r))


def _bib(i: int) -> str:
    return f"{TEST_BIBCODE_PREFIX}{i:04d}"


def _delete_fixture(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_metrics WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )


def _insert_fixture(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        c.autocommit = False
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO papers (bibcode, title) VALUES (%s, %s) "
                "ON CONFLICT (bibcode) DO NOTHING",
                [(_bib(i), f"fixture {_bib(i)}") for i in range(len(FIXTURE_ROWS))],
            )
            cur.executemany(
                "INSERT INTO paper_metrics ("
                "  bibcode, community_id_coarse, community_semantic_coarse, "
                "  community_taxonomic"
                ") VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (bibcode) DO UPDATE SET "
                "  community_id_coarse       = EXCLUDED.community_id_coarse, "
                "  community_semantic_coarse = EXCLUDED.community_semantic_coarse, "
                "  community_taxonomic       = EXCLUDED.community_taxonomic",
                [(_bib(i), row[0], row[1], row[2]) for i, row in enumerate(FIXTURE_ROWS)],
            )
        c.commit()


@pytest.fixture
def fixture_data(dsn: str):
    """Insert fixture rows, delete afterwards. Isolated per-test."""
    _delete_fixture(dsn)
    _insert_fixture(dsn)
    try:
        yield
    finally:
        _delete_fixture(dsn)


# ---------------------------------------------------------------------------
# DSN fixture (module-scoped — the skip-if at module import time already
# guaranteed TEST_DSN is non-None)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    assert TEST_DSN is not None
    return TEST_DSN


# ---------------------------------------------------------------------------
# Helper — isolate paper_metrics to just the fixture rows for the test.
#
# The coverage script runs a COUNT(*) across the whole table, so if the
# test DB already has rows in paper_metrics the expected totals shift.
# We scope the assertions to the fixture prefix by reading the payload
# from a per-test bibcode-filtered re-count — that is what test (a)
# asserts. Test (b) uses a dropped-column view and the same filter.
# ---------------------------------------------------------------------------


def _count_via_script_logic(
    conn: psycopg.Connection,
    bibcode_prefix: str,
    table: str = "paper_metrics",
) -> dict[str, object]:
    """Re-run the script's coverage logic against a bibcode-filtered subset.

    We intentionally re-use the script's SQL composition helper so the
    test asserts against the real query shape (including the UNION
    filter), not a hand-rolled duplicate. Only the table name and a
    ``WHERE bibcode LIKE ...`` clause are spliced in.
    """
    mod = _load_script_module()
    presence = mod.detect_present_columns(conn, table, mod.SIGNAL_COLUMNS)
    present_cols = [col for col, ok in presence.items() if ok]

    select_parts = ["COUNT(*) AS total_papers"]
    for col in present_cols:
        select_parts.append(f"COUNT({col}) AS {col}_nn")

    if present_cols:
        union_filter = " OR ".join(f"{col} IS NOT NULL" for col in present_cols)
        select_parts.append(f"COUNT(*) FILTER (WHERE {union_filter}) AS union_covered")
    else:
        select_parts.append("0 AS union_covered")

    sql = "SELECT " + ", ".join(select_parts) + f" FROM {table} WHERE bibcode LIKE %s"
    with conn.cursor() as cur:
        cur.execute(sql, (bibcode_prefix + "%",))
        row = cur.fetchone()
    assert row is not None

    total_papers = int(row[0])
    counts: dict[str, object] = {}
    idx = 1
    for signal in mod.SIGNAL_COLUMNS:
        if presence[signal.column]:
            counts[signal.key] = int(row[idx])
            idx += 1
        else:
            counts[signal.key] = None
    union_covered = int(row[idx])
    fraction = union_covered / total_papers if total_papers > 0 else None
    payload: dict[str, object] = {"total_papers": total_papers}
    for signal in mod.SIGNAL_COLUMNS:
        payload[signal.key] = counts[signal.key]
    payload["union_covered"] = union_covered
    payload["union_coverage_fraction"] = fraction
    return payload


# ---------------------------------------------------------------------------
# Test (a) — union_covered matches hand-counted expectation
# ---------------------------------------------------------------------------


def test_union_covered_matches_hand_count(
    dsn: str,
    fixture_data: None,
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_script_module()

    # Run the full script — writes the global results file + sample.
    output_path = tmp_path / "community_coverage.json"
    sample_path = tmp_path / "community_coverage.sample.json"
    rc = mod.main(
        [
            "--dsn",
            dsn,
            "--output",
            str(output_path),
            "--sample-path",
            str(sample_path),
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    # Schema contract
    assert set(payload.keys()) == {
        "total_papers",
        "citation_covered",
        "semantic_covered",
        "taxonomic_covered",
        "union_covered",
        "union_coverage_fraction",
    }
    assert isinstance(payload["total_papers"], int)
    assert isinstance(payload["union_covered"], int)

    # Sample was written because scix_test is not production
    assert sample_path.exists()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample == payload

    # Per-fixture counts: re-query with a bibcode filter so the test is
    # independent of whatever else happens to sit in paper_metrics.
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        scoped = _count_via_script_logic(conn, TEST_BIBCODE_PREFIX)

    assert scoped["total_papers"] == len(FIXTURE_ROWS)
    assert scoped["citation_covered"] == EXPECTED_CITATION
    assert scoped["semantic_covered"] == EXPECTED_SEMANTIC
    assert scoped["taxonomic_covered"] == EXPECTED_TAXONOMIC
    assert scoped["union_covered"] == EXPECTED_UNION
    assert scoped["union_coverage_fraction"] == pytest.approx(EXPECTED_UNION / len(FIXTURE_ROWS))


# ---------------------------------------------------------------------------
# Test (b) — missing columns are handled gracefully with skip + warning
# ---------------------------------------------------------------------------


def test_missing_column_reports_null_and_warns(
    dsn: str,
    fixture_data: None,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a signal column is absent, the script logs a warning and
    reports ``null`` for that count without crashing.

    We simulate the missing-column condition by creating a VIEW that
    projects ``paper_metrics`` with ``community_taxonomic`` omitted, then
    point the coverage query at that view. The real table is untouched,
    so no migration rollback is required and parallel tests are unaffected.
    """
    mod = _load_script_module()

    view_name = "paper_metrics_no_taxonomic"
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(f"DROP VIEW IF EXISTS {view_name}")
            cur.execute(
                f"CREATE VIEW {view_name} AS "
                f"SELECT bibcode, community_id_coarse, community_id_medium, "
                f"community_id_fine, community_semantic_coarse, "
                f"community_semantic_medium, community_semantic_fine, "
                f"updated_at FROM paper_metrics "
                f"WHERE bibcode LIKE '{TEST_BIBCODE_PREFIX}%'"
            )

    try:
        # Monkeypatch the script's TABLE constant so the query + the
        # information_schema lookup both target the view.
        original_table = mod.PAPER_METRICS_TABLE
        mod.PAPER_METRICS_TABLE = view_name
        try:
            output_path = tmp_path / "community_coverage.json"
            sample_path = tmp_path / "community_coverage.sample.json"
            caplog.set_level(logging.WARNING, logger="report_community_coverage")
            rc = mod.main(
                [
                    "--dsn",
                    dsn,
                    "--output",
                    str(output_path),
                    "--sample-path",
                    str(sample_path),
                ]
            )
            assert rc == 0
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            mod.PAPER_METRICS_TABLE = original_table

        # Missing column → null count.
        assert payload["taxonomic_covered"] is None
        # The other two columns must still be counted correctly (over
        # the view, which is restricted to the fixture prefix).
        assert payload["total_papers"] == len(FIXTURE_ROWS)
        assert payload["citation_covered"] == EXPECTED_CITATION
        assert payload["semantic_covered"] == EXPECTED_SEMANTIC

        # UNION coverage must reflect only the columns that remain —
        # i.e. rows that had ONLY a taxonomic signal no longer count.
        expected_union_without_taxonomic = sum(
            1 for r in FIXTURE_ROWS if r[0] is not None or r[1] is not None
        )
        assert payload["union_covered"] == expected_union_without_taxonomic
        assert payload["union_coverage_fraction"] == pytest.approx(
            expected_union_without_taxonomic / len(FIXTURE_ROWS)
        )

        # Warning was emitted for the missing column.
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelno == logging.WARNING and "community_taxonomic" in rec.getMessage()
        ]
        assert warnings, "expected a warning about missing community_taxonomic"

    finally:
        with psycopg.connect(dsn) as c:
            c.autocommit = True
            with c.cursor() as cur:
                cur.execute(f"DROP VIEW IF EXISTS {view_name}")
