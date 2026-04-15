"""Tests for scripts/seed_query_log.py (xz4.1.18).

Covers the seed bootstrap script that classifies unique entities via
in-process MCP tool dispatch and writes the results to query_log under a
deterministic session_id. No real MCP tool calls are issued — the
dispatcher is replaced with a stub that returns canned JSON so the tests
run against a clean scix_test DB without network or model dependencies.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

import psycopg
import pytest

from tests.helpers import get_test_dsn

pytestmark = pytest.mark.integration


SEED_TAG_PREFIX = "seed-bootstrap-v1-"
TEST_SOURCE_A = "u18_src_a"
TEST_SOURCE_B = "u18_src_b"

# Shared dispatch lookup: 3 zero-result (alpha/gamma/zeta -> pass1),
# 2 non-zero (beta/epsilon -> pass3). Reused by several tests.
DEFAULT_LOOKUP = {
    "u18_alpha_wave": 0,
    "u18_beta_field": 3,
    "u18_gamma_peak": 0,
    "u18_epsilon_line": 1,
    "u18_zeta_burst": 0,
}


@pytest.fixture()
def seeded_entities() -> tuple[psycopg.Connection, dict[str, int]]:
    """Seed a handful of unique entities across two sources.

    Returns (connection, {canonical_name_lower: entity_id}).
    """
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    c = psycopg.connect(dsn)

    def _cleanup() -> None:
        with c.cursor() as cur:
            cur.execute(
                """
                DELETE FROM query_log WHERE session_id LIKE %s
                """,
                (f"{SEED_TAG_PREFIX}%",),
            )
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
                "DELETE FROM entities WHERE source IN (%s, %s)",
                (TEST_SOURCE_A, TEST_SOURCE_B),
            )
        c.commit()

    _cleanup()

    # src_a: 3 unique + 1 homograph (homograph must NOT be picked up)
    # src_b: 2 unique
    entities_spec = [
        ("u18_alpha_wave", TEST_SOURCE_A, "unique"),
        ("u18_beta_field", TEST_SOURCE_A, "unique"),
        ("u18_gamma_peak", TEST_SOURCE_A, "unique"),
        ("u18_delta_shift", TEST_SOURCE_A, "homograph"),
        ("u18_epsilon_line", TEST_SOURCE_B, "unique"),
        ("u18_zeta_burst", TEST_SOURCE_B, "unique"),
    ]
    ids: dict[str, int] = {}
    with c.cursor() as cur:
        for canonical, source, ambig in entities_spec:
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source, ambiguity_class)
                VALUES (%s, 'concept', %s, %s::entity_ambiguity_class)
                RETURNING id
                """,
                (canonical, source, ambig),
            )
            ids[canonical] = int(cur.fetchone()[0])
    c.commit()

    yield c, ids

    _cleanup()
    c.close()


def _fake_dispatcher(lookup: dict[str, int]):
    """Build a fake _dispatch_tool that returns JSON with ``total`` = lookup[query]."""

    def _dispatch(conn: psycopg.Connection, name: str, args: dict[str, Any]) -> str:
        # Extract query text similarly to _extract_query_text.
        for key in ("query", "terms", "entity_name"):
            val = args.get(key)
            if val is not None:
                query = str(val)
                break
        else:
            query = ""
        total = lookup.get(query, 0)
        return json.dumps({"total": total, "papers": []})

    return _dispatch


# ---------------------------------------------------------------------------
# Deterministic session_id
# ---------------------------------------------------------------------------


def test_session_id_is_deterministic_from_manifest() -> None:
    from scripts.seed_query_log import compute_session_id

    manifest_a = {"version": 1, "per_source_target": 5, "tool": "keyword_search"}
    manifest_b = {"version": 1, "per_source_target": 5, "tool": "keyword_search"}
    manifest_c = {"version": 1, "per_source_target": 6, "tool": "keyword_search"}

    sid_a = compute_session_id(manifest_a)
    sid_b = compute_session_id(manifest_b)
    sid_c = compute_session_id(manifest_c)

    assert sid_a == sid_b
    assert sid_a != sid_c
    assert sid_a.startswith(SEED_TAG_PREFIX)


# ---------------------------------------------------------------------------
# Hard-fail: SCIX_TEST_DSN + prod DSN
# ---------------------------------------------------------------------------


def test_hard_fail_when_targeting_prod_with_test_dsn_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.seed_query_log import assert_safe_target_dsn

    monkeypatch.setenv("SCIX_TEST_DSN", "dbname=scix_test")
    with pytest.raises(SystemExit):
        # Pointing at prod while SCIX_TEST_DSN is set — ambiguous, hard fail.
        assert_safe_target_dsn("dbname=scix")


def test_safe_target_dsn_accepts_unambiguous_prod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.seed_query_log import assert_safe_target_dsn

    monkeypatch.delenv("SCIX_TEST_DSN", raising=False)
    # No SCIX_TEST_DSN set and target is prod — fine.
    assert_safe_target_dsn("dbname=scix") is None


def test_safe_target_dsn_accepts_test_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.seed_query_log import assert_safe_target_dsn

    monkeypatch.setenv("SCIX_TEST_DSN", "dbname=scix_test")
    # Targeting the test DSN while SCIX_TEST_DSN is set — also fine.
    assert_safe_target_dsn("dbname=scix_test") is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_query_applies_nfc_lower_strip() -> None:
    from scripts.seed_query_log import normalize_query

    raw = "  Caf\u00e9  "  # NFC precomposed é with surrounding whitespace
    assert normalize_query(raw) == "café"
    # NFD should be re-composed to NFC before lower+strip.
    nfd = unicodedata.normalize("NFD", "Café")
    assert normalize_query(nfd) == "café"


# ---------------------------------------------------------------------------
# Classifier + writer (integration against scix_test)
# ---------------------------------------------------------------------------


def test_classifier_buckets_by_result_count(seeded_entities, monkeypatch) -> None:
    """Entities with result_count=0 go to P1, result_count>0 go to P3."""
    from scripts import seed_query_log

    conn, ids = seeded_entities

    # alpha=0 (P1), beta=3 (P3), gamma=0 (P1), epsilon=1 (P3), zeta=0 (P1)
    monkeypatch.setattr(seed_query_log, "_dispatch_tool", _fake_dispatcher(DEFAULT_LOOKUP))

    manifest = {
        "version": 1,
        "per_source_target": 10,  # pool is only 5 entities total; targets won't constrain
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    result = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)

    assert result.pass1_written == 3
    assert result.pass3_written == 2

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT query, result_count, is_test, session_id
              FROM query_log
             WHERE session_id = %s
             ORDER BY query
            """,
            (result.session_id,),
        )
        rows = cur.fetchall()

    assert len(rows) == 5

    # Bucket counters must correspond to the actual result_count values
    # in query_log — catches a pass1/pass3 label swap in the seed script.
    zero_rows = [r for r in rows if r[1] == 0]
    nonzero_rows = [r for r in rows if r[1] > 0]
    assert len(zero_rows) == result.pass1_written
    assert len(nonzero_rows) == result.pass3_written

    # And the specific entities must land on the correct side.
    zero_queries = {r[0] for r in zero_rows}
    nonzero_queries = {r[0] for r in nonzero_rows}
    assert zero_queries == {"u18_alpha_wave", "u18_gamma_peak", "u18_zeta_burst"}
    assert nonzero_queries == {"u18_beta_field", "u18_epsilon_line"}
    by_query = {r[0]: r for r in rows}
    # Stored queries must be normalized (lower/strip/NFC).
    assert by_query["u18_alpha_wave"][1] == 0
    assert by_query["u18_alpha_wave"][2] is False  # is_test=False
    assert by_query["u18_alpha_wave"][3].startswith(SEED_TAG_PREFIX)
    assert by_query["u18_beta_field"][1] == 3
    assert by_query["u18_beta_field"][2] is False


def test_homograph_entities_are_skipped(seeded_entities, monkeypatch) -> None:
    """Only ambiguity_class='unique' entities should be considered."""
    from scripts import seed_query_log

    conn, ids = seeded_entities
    # Even if the homograph delta_shift got dispatched, the loader should
    # not have included it. Return high counts for everything to make a
    # bug visible.
    monkeypatch.setattr(
        seed_query_log,
        "_dispatch_tool",
        _fake_dispatcher(
            {
                "u18_alpha_wave": 5,
                "u18_beta_field": 5,
                "u18_gamma_peak": 5,
                "u18_delta_shift": 5,  # homograph — must not appear in query_log
                "u18_epsilon_line": 5,
                "u18_zeta_burst": 5,
            }
        ),
    )

    manifest = {
        "version": 1,
        "per_source_target": 10,
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    result = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT query FROM query_log WHERE session_id = %s",
            (result.session_id,),
        )
        queries = {r[0] for r in cur.fetchall()}

    assert "u18_delta_shift" not in queries
    assert len(queries) == 5


def test_dry_run_makes_no_db_writes(seeded_entities, monkeypatch) -> None:
    from scripts import seed_query_log

    conn, _ = seeded_entities
    monkeypatch.setattr(seed_query_log, "_dispatch_tool", _fake_dispatcher({"u18_alpha_wave": 0}))

    manifest = {
        "version": 1,
        "per_source_target": 10,
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    result = seed_query_log.run_seed(conn, manifest=manifest, dry_run=True)

    # Classification still happens, so counts are populated, but no rows written.
    assert result.pass1_written == 0
    assert result.pass3_written == 0

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM query_log WHERE session_id = %s",
            (result.session_id,),
        )
        assert cur.fetchone()[0] == 0


def test_not_exists_guard_skips_duplicates(seeded_entities, monkeypatch) -> None:
    """Running twice with the same manifest must not double-insert."""
    from scripts import seed_query_log

    conn, _ = seeded_entities
    monkeypatch.setattr(seed_query_log, "_dispatch_tool", _fake_dispatcher(DEFAULT_LOOKUP))

    manifest = {
        "version": 1,
        "per_source_target": 10,
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    r1 = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)
    r2 = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)

    assert r1.session_id == r2.session_id
    assert r1.pass1_written + r1.pass3_written == 5
    assert r2.pass1_written + r2.pass3_written == 0  # all skipped

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM query_log WHERE session_id = %s",
            (r1.session_id,),
        )
        assert cur.fetchone()[0] == 5


def test_rollback_deletes_only_session_id(seeded_entities, monkeypatch) -> None:
    from scripts import seed_query_log

    conn, _ = seeded_entities
    monkeypatch.setattr(seed_query_log, "_dispatch_tool", _fake_dispatcher({"u18_alpha_wave": 0}))

    manifest = {
        "version": 1,
        "per_source_target": 10,
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    SURVIVOR_TAG = "u18-survivor-session"

    # Defensive cleanup in case a prior run leaked.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM query_log WHERE session_id = %s", (SURVIVOR_TAG,))
    conn.commit()

    r = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)

    try:
        # Insert an unrelated row that must survive rollback.
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log (tool_name, success, tool, query,
                                       result_count, session_id, is_test)
                VALUES ('search', TRUE, 'search', 'survivor', 0, %s, FALSE)
                """,
                (SURVIVOR_TAG,),
            )
        conn.commit()

        deleted = seed_query_log.run_rollback(conn, session_id=r.session_id)
        assert deleted >= 1

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM query_log WHERE session_id = %s",
                (r.session_id,),
            )
            assert cur.fetchone()[0] == 0
            cur.execute(
                "SELECT count(*) FROM query_log WHERE session_id = %s",
                (SURVIVOR_TAG,),
            )
            assert cur.fetchone()[0] == 1
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM query_log WHERE session_id = %s", (SURVIVOR_TAG,))
        conn.commit()


def test_bind_rate_sanity_check(seeded_entities, monkeypatch) -> None:
    """Every written row must bind to its source entity via exact canonical name."""
    from scripts import seed_query_log

    conn, ids = seeded_entities
    monkeypatch.setattr(seed_query_log, "_dispatch_tool", _fake_dispatcher(DEFAULT_LOOKUP))

    manifest = {
        "version": 1,
        "per_source_target": 10,
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    r = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)
    bind_rate = seed_query_log.compute_bind_rate(conn, session_id=r.session_id)
    # All 5 normalized canonical names must bind back to entities.
    assert bind_rate == pytest.approx(1.0)


def test_stratification_respects_per_source_target(seeded_entities, monkeypatch) -> None:
    """per_source_target caps the number of entities classified per source."""
    from scripts import seed_query_log

    conn, _ = seeded_entities
    monkeypatch.setattr(seed_query_log, "_dispatch_tool", _fake_dispatcher(DEFAULT_LOOKUP))

    # Only 2 per source: src_a has 3 uniques, src_b has 2 uniques.
    manifest = {
        "version": 1,
        "per_source_target": 2,
        "tool": "keyword_search",
        "sources": [TEST_SOURCE_A, TEST_SOURCE_B],
    }
    r = seed_query_log.run_seed(conn, manifest=manifest, dry_run=False)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.source, count(*)
              FROM query_log q
              JOIN entities e ON lower(e.canonical_name) = lower(trim(q.query))
             WHERE q.session_id = %s
             GROUP BY e.source
            """,
            (r.session_id,),
        )
        by_source = dict(cur.fetchall())

    # Each source contributed at most per_source_target rows.
    assert by_source.get(TEST_SOURCE_A, 0) <= 2
    assert by_source.get(TEST_SOURCE_B, 0) <= 2
    # And at least 1 per source (demonstrates stratification happened).
    assert by_source.get(TEST_SOURCE_A, 0) >= 1
    assert by_source.get(TEST_SOURCE_B, 0) >= 1
