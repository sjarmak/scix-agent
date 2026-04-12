"""Tests for entity ambiguity classification (u05)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest

from scix.ambiguity import classify, is_banned_name

# Make the scripts dir importable so we can call the classifier main()
# without shelling out in the integration test.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from tests.helpers import get_test_dsn  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests — pure classify()
# ---------------------------------------------------------------------------


class TestIsBannedName:
    def test_empty_string_is_banned_as_too_short(self) -> None:
        assert is_banned_name("") is True

    def test_one_character_is_banned(self) -> None:
        assert is_banned_name("a") is True

    def test_two_characters_is_banned(self) -> None:
        assert is_banned_name("UV") is True
        assert is_banned_name("Hi") is True

    def test_three_characters_non_common_word_not_banned(self) -> None:
        # 'xyz' has zipf << 3.0
        assert is_banned_name("xyz") is False

    def test_common_english_word_is_banned_even_when_long(self) -> None:
        assert is_banned_name("the") is True
        assert is_banned_name("The") is True
        assert is_banned_name("hubble") is True  # zipf ~3.12

    def test_rare_long_name_not_banned(self) -> None:
        assert is_banned_name("GALFA-HI") is False


class TestClassify:
    def test_two_char_name_is_banned(self) -> None:
        assert (
            classify(
                canonical_name="UV",
                aliases=[],
                source_count=1,
                collision_count=0,
            )
            == "banned"
        )

    def test_common_english_word_is_banned(self) -> None:
        # 'the' has zipf 7.73 >> 3.0
        assert (
            classify(
                canonical_name="the",
                aliases=[],
                source_count=1,
                collision_count=0,
            )
            == "banned"
        )

    def test_banned_alias_propagates_to_banned(self) -> None:
        assert (
            classify(
                canonical_name="Obscure123",
                aliases=["the"],
                source_count=1,
                collision_count=0,
            )
            == "banned"
        )

    def test_homograph_when_collision(self) -> None:
        # 'HST' — 3 chars (not banned by length), low zipf, colliding.
        assert (
            classify(
                canonical_name="HST",
                aliases=[],
                source_count=2,
                collision_count=1,
            )
            == "homograph"
        )

    def test_banned_precedes_homograph(self) -> None:
        # A common word WITH a collision is still banned, not homograph.
        assert (
            classify(
                canonical_name="the",
                aliases=[],
                source_count=2,
                collision_count=5,
            )
            == "banned"
        )

    def test_domain_safe_for_long_unique_single_source(self) -> None:
        assert (
            classify(
                canonical_name="GALFA-HI",
                aliases=[],
                source_count=1,
                collision_count=0,
            )
            == "domain_safe"
        )

    def test_unique_when_short_non_banned_non_colliding(self) -> None:
        # 3 chars, non-common, no collision, one source — too short for
        # domain_safe (requires >=6), so falls through to 'unique'.
        assert (
            classify(
                canonical_name="XYZ",
                aliases=[],
                source_count=1,
                collision_count=0,
            )
            == "unique"
        )

    def test_unique_when_long_unique_but_multi_source(self) -> None:
        # Single entity but somehow shares source pool with 2 — still
        # not domain_safe because source_count > 1.
        assert (
            classify(
                canonical_name="LongRareName",
                aliases=[],
                source_count=2,
                collision_count=0,
            )
            == "unique"
        )

    def test_unique_when_short_and_multi_source(self) -> None:
        assert (
            classify(
                canonical_name="XYZ",
                aliases=[],
                source_count=3,
                collision_count=0,
            )
            == "unique"
        )


# ---------------------------------------------------------------------------
# Integration test — full pipeline against scix_test
# ---------------------------------------------------------------------------


_SEED_ROWS = [
    # (canonical_name, entity_type, source, aliases, expected_class)
    ("the", "concept", "uat", [], "banned"),
    ("UV", "instrument", "uat", [], "banned"),
    ("hubble", "facility", "uat", [], "banned"),  # zipf >= 3
    # NB: aliases here must themselves be non-banned. "Hubble Space
    # Telescope" has zipf ~3.03, so it would flip HST into banned. Use
    # an alias that collides with the other HST row instead, so we
    # still exercise the alias-collision path.
    ("HST", "facility", "ads_aas", ["HST-ACS"], "homograph"),
    ("HST", "facility", "wikidata", ["HST-ACS"], "homograph"),
    ("GALFA-HI", "survey", "uat", [], "domain_safe"),
    ("CHANDRA-XRAY", "facility", "uat", [], "domain_safe"),
    ("XYZ", "unknown", "uat", [], "unique"),
    ("QZX", "unknown", "uat", [], "unique"),
]


@pytest.fixture()
def integration_conn() -> psycopg.Connection:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set or points at production — " "integration test skipped")
    conn = psycopg.connect(dsn)
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


def _seed_entities(
    conn: psycopg.Connection, rows: list[tuple[str, str, str, list[str], str]]
) -> dict[str, int]:
    """Insert seed rows; return map keyed by '{canonical}|{source}' -> id."""
    ids: dict[str, int] = {}
    with conn.cursor() as cur:
        for canonical, etype, source, aliases, _expected in rows:
            cur.execute(
                "INSERT INTO entities (canonical_name, entity_type, source) "
                "VALUES (%s, %s, %s) RETURNING id",
                (canonical, etype, source),
            )
            row = cur.fetchone()
            assert row is not None
            entity_id = int(row[0])
            ids[f"{canonical}|{source}"] = entity_id
            for alias in aliases:
                cur.execute(
                    "INSERT INTO entity_aliases (entity_id, alias, alias_source) "
                    "VALUES (%s, %s, %s)",
                    (entity_id, alias, "test"),
                )
    return ids


@pytest.mark.integration
def test_classify_all_populates_four_buckets_on_fixture(
    integration_conn: psycopg.Connection, tmp_path: Path
) -> None:
    from classify_entity_ambiguity import classify_all, write_audit_report

    # Clean slate: scix_test is empty in baseline, but be defensive.
    with integration_conn.cursor() as cur:
        cur.execute("DELETE FROM entity_aliases")
        cur.execute("DELETE FROM entities")

    id_map = _seed_entities(integration_conn, _SEED_ROWS)
    integration_conn.commit()

    # Run the classifier.
    counts = classify_all(integration_conn, batch_size=100)

    # Each seeded entity should have its expected class.
    with integration_conn.cursor() as cur:
        for canonical, _etype, source, _aliases, expected in _SEED_ROWS:
            entity_id = id_map[f"{canonical}|{source}"]
            cur.execute(
                "SELECT ambiguity_class::text FROM entities WHERE id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == expected, (
                f"{canonical}@{source} (id={entity_id}) got {row[0]!r}, " f"expected {expected!r}"
            )

    # AC4: 4 buckets with non-null counts.
    with integration_conn.cursor() as cur:
        cur.execute(
            "SELECT ambiguity_class::text, count(*) FROM entities "
            "WHERE ambiguity_class IS NOT NULL "
            "GROUP BY ambiguity_class"
        )
        bucket_counts = {row[0]: int(row[1]) for row in cur.fetchall()}
    assert set(bucket_counts.keys()) == {
        "banned",
        "homograph",
        "domain_safe",
        "unique",
    }, f"missing buckets: got {bucket_counts}"
    for cls, count in bucket_counts.items():
        assert count > 0, f"class {cls} has zero rows"

    # classify_all's counts should match the DB.
    assert counts == bucket_counts

    # AC5: audit report generation. Write to the tmp dir to avoid
    # touching the checked-in build-artifacts path from the test run.
    audit_path = tmp_path / "ambiguity_audit.md"
    write_audit_report(integration_conn, audit_path, sample_size=50)
    assert audit_path.exists()
    content = audit_path.read_text()
    for cls in ("banned", "homograph", "domain_safe", "unique"):
        assert f"## {cls}" in content

    # Also write to the real build-artifacts/ location for the PRD artifact.
    real_audit = _REPO_ROOT / "build-artifacts" / "ambiguity_audit.md"
    write_audit_report(integration_conn, real_audit, sample_size=50)
    assert real_audit.exists()
