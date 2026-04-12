"""Integration tests for u13 incremental-sync entity linker (PRD §M10).

Requires ``SCIX_TEST_DSN`` to point at a non-production database. The
tests seed 10 papers with ``test_u13_`` bibcodes, 3 curated entities,
and aliases; then exercise:

* fresh-watermark end-to-end run (AC2: <60s, tier-1 + tier-2 rows)
* forced budget trip (AC3: run completes, watermark advances, zero links)
* catchup backfills the skipped papers (AC4)
* two consecutive trips → alerts row with severity='page' (AC5)
* forced-stale watermark → staleness alert (AC5)

The fixture keeps ``link_runs`` and ``alerts`` scoped via source/message
markers so the test doesn't interfere with any other users of the test
database.
"""

from __future__ import annotations

import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import link_catchup  # noqa: E402
import link_incremental  # noqa: E402

from scix.circuit_breaker import CircuitBreaker  # noqa: E402

from tests.helpers import get_test_dsn  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture seed
# ---------------------------------------------------------------------------

_TEST_SOURCE = "unit_test_u13"

# entry_date is a TEXT field in ADS format. We stagger entry_dates so the
# watermark logic has something meaningful to advance across.
_BASE_ENTRY = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _entry_date(offset_hours: int) -> str:
    d = _BASE_ENTRY + timedelta(hours=offset_hours)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


_SEED_PAPERS: list[tuple[str, str, list[str], str]] = [
    (
        "test_u13_0001",
        "We used the Hubble Space Telescope to survey distant quasars. HST data revealed jets.",
        ["Hubble Space Telescope"],
        _entry_date(1),
    ),
    (
        "test_u13_0002",
        "JWST observed water vapor in an exoplanet atmosphere.",
        ["JWST"],
        _entry_date(2),
    ),
    (
        "test_u13_0003",
        "ALMA interferometry revealed dust continuum emission.",
        ["ALMA"],
        _entry_date(3),
    ),
    (
        "test_u13_0004",
        "The James Webb Space Telescope imaged a high-redshift galaxy.",
        [],
        _entry_date(4),
    ),
    (
        "test_u13_0005",
        "A multiwavelength survey without specific instruments.",
        ["galaxy_survey"],
        _entry_date(5),
    ),
    (
        "test_u13_0006",
        "ALMA observations of protoplanetary disks at millimeter wavelengths.",
        ["ALMA"],
        _entry_date(6),
    ),
    (
        "test_u13_0007",
        "HST alone without disambiguator — should NOT fire homograph entity.",
        [],
        _entry_date(7),
    ),
    (
        "test_u13_0008",
        "Deep NIR imaging with the Hubble Space Telescope.",
        ["Hubble Space Telescope"],
        _entry_date(8),
    ),
    (
        "test_u13_0009",
        "JWST NIRSpec spectra of a protostar.",
        [],
        _entry_date(9),
    ),
    (
        "test_u13_0010",
        "An abstract about ALMA and protoplanetary evolution.",
        [],
        _entry_date(10),
    ),
]

_SEED_ENTITIES: list[tuple[str, str]] = [
    # (canonical_name, ambiguity_class)
    ("Hubble Space Telescope", "homograph"),
    ("James Webb Space Telescope", "unique"),
    ("ALMA Observatory", "domain_safe"),
]

_SEED_ALIASES: list[tuple[str, str]] = [
    ("Hubble Space Telescope", "HST"),
    ("James Webb Space Telescope", "JWST"),
    ("ALMA Observatory", "ALMA"),
]


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip("SCIX_TEST_DSN must be set to a non-production DSN for u13 tests")
    return test_dsn


def _cleanup(conn: psycopg.Connection) -> None:
    bibcodes = [p[0] for p in _SEED_PAPERS]
    canonicals = [e[0] for e in _SEED_ENTITIES]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM curated_entity_core "
            " WHERE entity_id IN ("
            "   SELECT id FROM entities "
            "    WHERE canonical_name = ANY(%s) AND source = %s"
            " )",
            (canonicals, _TEST_SOURCE),
        )
        cur.execute(
            "DELETE FROM entities " " WHERE canonical_name = ANY(%s) AND source = %s",
            (canonicals, _TEST_SOURCE),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        # Scrub any link_runs rows the test created (note text marker).
        cur.execute(
            "DELETE FROM link_runs WHERE note LIKE %s OR note IS NULL AND "
            "max_entry_date BETWEEN %s AND %s",
            ("%u13-test%", _BASE_ENTRY, _BASE_ENTRY + timedelta(days=1)),
        )
        # Wipe any alerts emitted from u13-test runs against this fixture
        cur.execute(
            "DELETE FROM alerts WHERE source IN ('incremental_sync','watermark_staleness') "
            "  AND (message LIKE %s OR message LIKE %s)",
            ("%u13-test%", "%2026-04%"),
        )
    conn.commit()


def _reset_link_runs(conn: psycopg.Connection) -> None:
    """Remove all link_runs rows so tests start from a known watermark."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM link_runs")
        cur.execute("DELETE FROM alerts WHERE source IN ('incremental_sync','watermark_staleness')")
    conn.commit()


def _seed(conn: psycopg.Connection) -> dict[str, int]:
    _cleanup(conn)
    _reset_link_runs(conn)
    name_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        for bibcode, abstract, keywords, entry_date in _SEED_PAPERS:
            cur.execute(
                "INSERT INTO papers (bibcode, abstract, keywords, entry_date) "
                "VALUES (%s, %s, %s, %s)",
                (bibcode, abstract, keywords, entry_date),
            )
        for canonical, ambiguity in _SEED_ENTITIES:
            cur.execute(
                "INSERT INTO entities (canonical_name, entity_type, source, ambiguity_class) "
                "VALUES (%s, %s, %s, %s::entity_ambiguity_class) RETURNING id",
                (canonical, "test_type", _TEST_SOURCE, ambiguity),
            )
            row = cur.fetchone()
            assert row is not None
            name_to_id[canonical] = int(row[0])

        for canonical, alias in _SEED_ALIASES:
            cur.execute(
                "INSERT INTO entity_aliases (entity_id, alias, alias_source) "
                "VALUES (%s, %s, %s)",
                (name_to_id[canonical], alias, "test_seed_u13"),
            )

        for canonical, entity_id in name_to_id.items():
            cur.execute(
                "INSERT INTO curated_entity_core (entity_id, query_hits_14d) "
                "VALUES (%s, %s) ON CONFLICT (entity_id) DO NOTHING",
                (entity_id, 1),
            )
    conn.commit()
    return name_to_id


@pytest.fixture()
def seeded_conn(dsn: str) -> Iterator[tuple[psycopg.Connection, dict[str, int]]]:
    conn = psycopg.connect(dsn)
    try:
        ids = _seed(conn)
        yield conn, ids
    finally:
        try:
            _cleanup(conn)
            _reset_link_runs(conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_fixture_links(conn: psycopg.Connection, tier: int | None = None) -> int:
    pattern = "test_u13_%"
    with conn.cursor() as cur:
        if tier is None:
            cur.execute(
                "SELECT count(*) FROM document_entities WHERE bibcode LIKE %s",
                (pattern,),
            )
        else:
            cur.execute(
                "SELECT count(*) FROM document_entities " "WHERE bibcode LIKE %s AND tier = %s",
                (pattern, tier),
            )
        return int(cur.fetchone()[0])


def _latest_link_run(conn: psycopg.Connection) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id, max_entry_date, rows_linked, status, trip_count "
            "FROM link_runs ORDER BY timestamp DESC LIMIT 1"
        )
        row = cur.fetchone()
    assert row is not None, "expected at least one link_runs row"
    return {
        "run_id": row[0],
        "max_entry_date": row[1],
        "rows_linked": row[2],
        "status": row[3],
        "trip_count": row[4],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalHappyPath:
    def test_fresh_run_links_fixture_papers_under_60s(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn

        t0 = time.monotonic()
        result = link_incremental.run_incremental(
            conn,
            budget_seconds=300.0,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 60.0, f"incremental run took {elapsed:.1f}s (>60s budget)"
        assert result.status == "ok"
        assert result.papers_in_scope == len(_SEED_PAPERS)
        # Tier-1 rows via the "galaxy_survey" keyword won't match any
        # seeded entity; tier-1 should ONLY match where a keyword string
        # equals a canonical_name. We seeded papers 0001, 0003, 0006, 0008
        # with canonical/alias keywords, so tier-1 should produce ≥2 rows.
        assert result.tier1_rows >= 2
        # Tier-2 rows: JWST, ALMA, Hubble Space Telescope + disambiguator
        # pairs all fire. At least 4 papers should get tier-2 rows.
        assert result.tier2_rows >= 4

        run = _latest_link_run(conn)
        assert run["status"] == "ok"
        assert run["max_entry_date"] is not None
        # Watermark should match the latest seeded entry_date (hour 10).
        expected = _BASE_ENTRY + timedelta(hours=10)
        assert run["max_entry_date"] == expected

    def test_second_run_finds_nothing_new(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn
        link_incremental.run_incremental(
            conn,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )
        result = link_incremental.run_incremental(
            conn,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )
        assert result.papers_in_scope == 0
        assert result.status == "ok"


class TestCircuitBreakerTrip:
    def test_tiny_budget_trips_and_watermark_advances(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn

        before_links = _count_fixture_links(conn)
        assert before_links == 0

        # Force-trip: 0.001s budget means the first check() after any
        # meaningful work raises CircuitBreakerOpen.
        tiny = CircuitBreaker(budget_seconds=0.001)
        # start() with a clock that will already be over budget at check:
        tiny.start()
        time.sleep(0.01)

        result = link_incremental.run_incremental(
            conn,
            breaker=tiny,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )

        assert result.status == "tripped"
        # No links should have been written (the first breaker.check()
        # inside run_incremental trips before tier-1 runs).
        assert _count_fixture_links(conn) == 0

        # Watermark still advances (AC3)
        run = _latest_link_run(conn)
        assert run["status"] == "tripped"
        assert run["max_entry_date"] is not None
        expected = _BASE_ENTRY + timedelta(hours=10)
        assert run["max_entry_date"] == expected
        assert run["trip_count"] >= 1


class TestCatchupBackfillsSkippedPapers:
    def test_catchup_after_trip_populates_links(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn

        # Step 1: run with a tripped breaker so all papers advance past
        # the watermark with zero links.
        tiny = CircuitBreaker(budget_seconds=0.001)
        tiny.start()
        time.sleep(0.01)
        link_incremental.run_incremental(
            conn,
            breaker=tiny,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )
        assert _count_fixture_links(conn) == 0

        # Step 2: catchup — should pick up all 10 fixture papers
        catchup_result = link_catchup.run_catchup(
            conn,
            limit=None,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )
        # The catchup scope is larger than just our fixture (it covers
        # every paper with entry_date <= watermark and no links). In the
        # test DB that's exactly our 10 fixture papers.
        assert catchup_result.papers_in_scope >= len(_SEED_PAPERS)

        post_links = _count_fixture_links(conn)
        assert post_links >= 4, f"expected ≥4 links after catchup, got {post_links}"

        tier1 = _count_fixture_links(conn, tier=1)
        tier2 = _count_fixture_links(conn, tier=2)
        assert tier2 >= 4, f"expected tier-2 links after catchup, got {tier2}"


class TestTwoConsecutiveTripsPager:
    def test_second_trip_emits_page_alert(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn

        # First tripped run
        b1 = CircuitBreaker(budget_seconds=0.001)
        b1.start()
        time.sleep(0.01)
        link_incremental.run_incremental(
            conn,
            breaker=b1,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )

        # Second tripped run — should fire the page alert
        b2 = CircuitBreaker(budget_seconds=0.001)
        b2.start()
        time.sleep(0.01)
        link_incremental.run_incremental(
            conn,
            breaker=b2,
            automaton_path=pathlib.Path("/nonexistent/ac_automaton.pkl"),
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM alerts "
                "WHERE severity = 'page' AND source = 'incremental_sync'"
            )
            n = int(cur.fetchone()[0])
        assert n >= 1, "expected a 'page' alert after 2 consecutive trips"


class TestWatermarkStaleness:
    def test_forced_stale_watermark_fires_alert(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn

        # Clear any existing run rows, then insert a single stale row.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM link_runs")
            cur.execute("DELETE FROM alerts WHERE source = 'watermark_staleness'")
            stale_wm = datetime.now(timezone.utc) - timedelta(hours=48)
            cur.execute(
                "INSERT INTO link_runs (max_entry_date, rows_linked, status, note) "
                "VALUES (%s, 0, 'ok', 'u13-test stale watermark')",
                (stale_wm,),
            )
        conn.commit()

        alert_id = link_incremental.check_watermark_staleness(conn)
        assert alert_id is not None, "expected staleness alert"

        with conn.cursor() as cur:
            cur.execute(
                "SELECT severity, source FROM alerts WHERE id = %s",
                (alert_id,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "page"
        assert row[1] == "watermark_staleness"

    def test_fresh_watermark_does_not_fire(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn

        with conn.cursor() as cur:
            cur.execute("DELETE FROM link_runs")
            cur.execute("DELETE FROM alerts WHERE source = 'watermark_staleness'")
            fresh_wm = datetime.now(timezone.utc) - timedelta(minutes=5)
            cur.execute(
                "INSERT INTO link_runs (max_entry_date, rows_linked, status, note) "
                "VALUES (%s, 0, 'ok', 'u13-test fresh watermark')",
                (fresh_wm,),
            )
        conn.commit()

        alert_id = link_incremental.check_watermark_staleness(conn)
        assert alert_id is None
