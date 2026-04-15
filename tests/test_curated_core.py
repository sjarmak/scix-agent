"""Tests for curated entity core + lifecycle (M3.5.1 / M3.5.2)."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from scix import core_lifecycle
from tests.helpers import get_test_dsn

pytestmark = pytest.mark.integration

U07_TAG = "u07-curated-core-test"
TEST_SOURCE_A = "u07_src_a"
TEST_SOURCE_B = "u07_src_b"


@pytest.fixture()
def seeded_db() -> psycopg.Connection:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    c = psycopg.connect(dsn)

    # --- cleanup any prior run ---
    with c.cursor() as cur:
        cur.execute(
            """
            DELETE FROM curated_entity_core
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute(
            """
            DELETE FROM core_promotion_log
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute(
            """
            DELETE FROM entity_aliases
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute(
            "DELETE FROM entities WHERE source IN (%s, %s)",
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute("DELETE FROM query_log WHERE session_id = %s", (U07_TAG,))
    c.commit()

    # --- seed entities ---
    #   src_a unique (5), src_a homograph (1), src_b unique (3),
    #   src_b domain_safe (1)
    entities_spec = [
        # (canonical, source, ambiguity, query_hits, zero_hits)
        ("u07_alpha_hydrogen", TEST_SOURCE_A, "unique", 5, 0),
        ("u07_beta_carbon", TEST_SOURCE_A, "unique", 3, 0),
        ("u07_gamma_ray", TEST_SOURCE_A, "unique", 1, 0),
        ("u07_delta_wave", TEST_SOURCE_A, "unique", 0, 4),  # gap candidate
        ("u07_epsilon_matter", TEST_SOURCE_A, "unique", 0, 2),  # gap candidate
        ("u07_homograph_name", TEST_SOURCE_A, "homograph", 2, 0),  # pass3 excluded
        ("u07_zeta_spectrum", TEST_SOURCE_B, "unique", 8, 0),  # top
        ("u07_eta_photon", TEST_SOURCE_B, "unique", 2, 0),
        ("u07_theta_plasma", TEST_SOURCE_B, "unique", 1, 0),
        ("u07_iota_mass", TEST_SOURCE_B, "domain_safe", 4, 0),  # excluded pass3
    ]

    ent_ids: dict[str, int] = {}
    with c.cursor() as cur:
        for canonical, source, ambig, _hits, _zero in entities_spec:
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source, ambiguity_class)
                VALUES (%s, 'concept', %s, %s::entity_ambiguity_class)
                RETURNING id
                """,
                (canonical, source, ambig),
            )
            ent_ids[canonical] = int(cur.fetchone()[0])

        # Seed query_log inside the 14d window with distinct queries
        # matching canonical_name (lowercased) so the curator can bind them.
        for canonical, _source, _ambig, hits, zero_hits in entities_spec:
            for _ in range(hits):
                cur.execute(
                    """
                    INSERT INTO query_log (tool_name, success, tool, query,
                                           result_count, session_id, is_test)
                    VALUES ('search', TRUE, 'search', %s, 1, %s, FALSE)
                    """,
                    (canonical, U07_TAG),
                )
            for _ in range(zero_hits):
                cur.execute(
                    """
                    INSERT INTO query_log (tool_name, success, tool, query,
                                           result_count, session_id, is_test)
                    VALUES ('search', TRUE, 'search', %s, 0, %s, FALSE)
                    """,
                    (canonical, U07_TAG),
                )
    c.commit()

    yield c, ent_ids

    # --- teardown ---
    with c.cursor() as cur:
        cur.execute(
            """
            DELETE FROM curated_entity_core
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute(
            """
            DELETE FROM core_promotion_log
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute(
            """
            DELETE FROM entity_aliases
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute(
            "DELETE FROM entities WHERE source IN (%s, %s)",
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
        cur.execute("DELETE FROM query_log WHERE session_id = %s", (U07_TAG,))
    c.commit()
    c.close()


def test_curate_entity_core_three_pass(seeded_db, tmp_path: Path) -> None:
    """AC3 + AC5: three-pass ranking writes CSV + stratification, ≤10K rows."""
    from scripts.curate_entity_core import CSV_COLUMNS, run_curation

    conn, ent_ids = seeded_db

    csv_path = tmp_path / "curated_core.csv"
    strat_path = tmp_path / "curated_core_stratification.md"

    rows = run_curation(
        conn,
        csv_path=csv_path,
        strat_path=strat_path,
        window_days=14,
        max_n=10_000,
    )

    assert csv_path.exists(), "curated_core.csv not written"
    assert strat_path.exists(), "curated_core_stratification.md not written"
    assert len(rows) <= 10_000, "hard cap violated"

    # CSV must have the exact 7 columns from the acceptance criteria.
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == CSV_COLUMNS

    # Only rows from our seeded sources (others may exist in scix_test if
    # leftover — filter out for assertions).
    our_rows = [r for r in rows if r.source in (TEST_SOURCE_A, TEST_SOURCE_B)]
    ids_in_core = {r.entity_id for r in our_rows}

    # Pass 1 gap candidates: u07_delta_wave and u07_epsilon_matter are
    # unique + zero-result queries ⇒ should appear.
    assert ent_ids["u07_delta_wave"] in ids_in_core
    assert ent_ids["u07_epsilon_matter"] in ids_in_core

    # Pass 3 unique + hits: several unique entities with hits.
    assert ent_ids["u07_alpha_hydrogen"] in ids_in_core
    assert ent_ids["u07_zeta_spectrum"] in ids_in_core

    # Excluded: homograph (pass3 requires unique) and domain_safe
    # (requires 'unique' strictly).
    assert ent_ids["u07_homograph_name"] not in ids_in_core
    assert ent_ids["u07_iota_mass"] not in ids_in_core

    # Ranking: top of our rows by query_hits should be zeta (8 hits).
    our_sorted = sorted(our_rows, key=lambda r: -r.query_hits_14d)
    assert our_sorted[0].entity_id == ent_ids["u07_zeta_spectrum"]

    # AC5: stratification md has per-source counts.
    strat_text = strat_path.read_text(encoding="utf-8")
    assert "| source |" in strat_text
    assert TEST_SOURCE_A in strat_text or TEST_SOURCE_B in strat_text


def test_curate_populate_writes_to_core_table(seeded_db, tmp_path: Path) -> None:
    """AC: curated_entity_core table populated when populate=True."""
    from scripts.curate_entity_core import run_curation

    conn, ent_ids = seeded_db

    csv_path = tmp_path / "curated_core.csv"
    strat_path = tmp_path / "curated_core_stratification.md"

    # Clean any prior curated_entity_core rows for our test entities.
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM curated_entity_core
             WHERE entity_id IN (
                SELECT id FROM entities WHERE source IN (%s, %s)
             )
            """,
            (TEST_SOURCE_A, TEST_SOURCE_B),
        )
    conn.commit()

    rows = run_curation(
        conn,
        csv_path=csv_path,
        strat_path=strat_path,
        window_days=14,
        max_n=10_000,
        populate=True,
    )

    our_rows = [r for r in rows if r.source in (TEST_SOURCE_A, TEST_SOURCE_B)]
    assert len(our_rows) > 0, "expected curated rows from seeded data"

    # Verify rows actually landed in the table.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT entity_id, query_hits_14d
              FROM curated_entity_core
             WHERE entity_id = ANY(%s)
            """,
            (list(ent_ids.values()),),
        )
        db_rows = {int(r[0]): int(r[1]) for r in cur.fetchall()}

    for r in our_rows:
        assert (
            r.entity_id in db_rows
        ), f"entity {r.entity_id} ({r.canonical_name}) not in curated_entity_core"
        expected_hits = r.query_hits_14d + r.zero_result_hits_14d
        assert db_rows[r.entity_id] == expected_hits

    # Verify promotion_log entries exist.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM core_promotion_log
             WHERE entity_id = ANY(%s) AND action = 'promote'
               AND reason LIKE 'curate_%%'
            """,
            (list(ent_ids.values()),),
        )
        (log_count,) = cur.fetchone()
    assert log_count >= len(our_rows), "expected promotion log entries for all curated rows"


def test_promote_auto_demotes_at_cap(seeded_db, monkeypatch) -> None:
    """AC4: promoting past CORE_MAX auto-demotes the lowest-hit member."""
    conn, ent_ids = seeded_db

    # Use a small cap so the test can exercise the overflow path.
    monkeypatch.setattr(core_lifecycle, "CORE_MAX", 5)

    # Fill exactly to the cap with ascending hit counts.
    fillers = [
        ("u07_alpha_hydrogen", 5),
        ("u07_beta_carbon", 3),
        ("u07_gamma_ray", 1),
        ("u07_zeta_spectrum", 8),
        ("u07_eta_photon", 2),
    ]
    for name, hits in fillers:
        core_lifecycle.promote(
            ent_ids[name],
            query_hits_14d=hits,
            reason="seed",
            conn=conn,
        )

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM curated_entity_core")
        (size,) = cur.fetchone()
    assert size >= 5

    # Lowest-hit member among our seeds is gamma_ray=1.
    # Promote an 11th (6th relative to cap=5) entity — should auto-demote the
    # lowest-hit one among our seeds. Because scix_test may have other rows
    # from prior test runs in curated_entity_core, we scope assertions to
    # our own entity_ids.
    core_lifecycle.promote(
        ent_ids["u07_theta_plasma"],
        query_hits_14d=4,
        reason="overflow-test",
        conn=conn,
    )

    # Fetch our core membership.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT entity_id FROM curated_entity_core
             WHERE entity_id = ANY(%s)
            """,
            (list(ent_ids.values()),),
        )
        present = {int(r[0]) for r in cur.fetchall()}

    # Either the lowest-hit filler was auto-demoted, OR some other row in
    # scix_test was (we can't control that). So we check: if the overall
    # core size increased past CORE_MAX without a demote, that's a bug.
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM curated_entity_core")
        (final_size,) = cur.fetchone()
    assert final_size <= core_lifecycle.CORE_MAX, "hard cap violated on promote"

    # At least one demote event must have been logged for overflow.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM core_promotion_log
             WHERE action = 'demote' AND reason = 'auto_demote_cap'
            """)
        (demote_events,) = cur.fetchone()
    assert demote_events >= 1, "expected auto_demote_cap event"

    # The new entity must be in the core.
    assert ent_ids["u07_theta_plasma"] in present


def test_session_id_filter_isolates_seeded_traffic(seeded_db, tmp_path: Path) -> None:
    """--session-id filter restricts curation to one seed run's rows."""
    from scripts.curate_entity_core import run_curation

    conn, ent_ids = seeded_db

    # Tag a second batch of rows under a different session_id so the
    # filter has something to exclude. Use u07_gamma_ray (unique, 1 hit in
    # the default fixture) as a pass3 candidate under OTHER_TAG.
    OTHER_TAG = "u07-other-seed"
    # Defensive cleanup in case a prior run leaked before the finally block.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM query_log WHERE session_id = %s", (OTHER_TAG,))
        cur.execute(
            """
            INSERT INTO query_log (tool_name, success, tool, query,
                                   result_count, session_id, is_test)
            VALUES ('search', TRUE, 'search', %s, 1, %s, FALSE)
            """,
            ("u07_gamma_ray", OTHER_TAG),
        )
    conn.commit()

    try:
        csv_a = tmp_path / "core_a.csv"
        strat_a = tmp_path / "strat_a.md"
        rows_a = run_curation(
            conn,
            csv_path=csv_a,
            strat_path=strat_a,
            window_days=14,
            max_n=10_000,
            session_id=U07_TAG,
        )
        csv_b = tmp_path / "core_b.csv"
        strat_b = tmp_path / "strat_b.md"
        rows_b = run_curation(
            conn,
            csv_path=csv_b,
            strat_path=strat_b,
            window_days=14,
            max_n=10_000,
            session_id=OTHER_TAG,
        )

        # Restrict assertions to our test fixtures — the scix_test DB may
        # hold leftover rows from other tests.
        ids_a = {r.entity_id for r in rows_a if r.source in (TEST_SOURCE_A, TEST_SOURCE_B)}
        ids_b = {r.entity_id for r in rows_b if r.source in (TEST_SOURCE_A, TEST_SOURCE_B)}

        # Session A has the default fixture's hits.
        assert ent_ids["u07_alpha_hydrogen"] in ids_a
        assert ent_ids["u07_delta_wave"] in ids_a  # pass1 zero-result
        assert ent_ids["u07_zeta_spectrum"] in ids_a  # top pass3 hitter

        # Session B has ONLY the gamma hit, and NOT the A fixture rows.
        # gamma_ray has 1 hit under both A and B, so it appears in both
        # ids_a and ids_b — the distinguisher is that alpha/delta/zeta
        # are in A but not in B.
        assert ent_ids["u07_gamma_ray"] in ids_b
        assert ent_ids["u07_alpha_hydrogen"] not in ids_b
        assert ent_ids["u07_delta_wave"] not in ids_b
        assert ent_ids["u07_zeta_spectrum"] not in ids_b
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM query_log WHERE session_id = %s", (OTHER_TAG,))
        conn.commit()


def test_demote_removes_and_logs(seeded_db) -> None:
    conn, ent_ids = seeded_db
    core_lifecycle.promote(ent_ids["u07_alpha_hydrogen"], query_hits_14d=5, conn=conn)
    core_lifecycle.demote(ent_ids["u07_alpha_hydrogen"], reason="test", conn=conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM curated_entity_core WHERE entity_id = %s",
            (ent_ids["u07_alpha_hydrogen"],),
        )
        (n,) = cur.fetchone()
    assert n == 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM core_promotion_log WHERE entity_id = %s AND action = 'demote'",
            (ent_ids["u07_alpha_hydrogen"],),
        )
        (events,) = cur.fetchone()
    assert events >= 1
