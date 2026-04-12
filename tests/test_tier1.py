"""Tests for u06 tier-1 keyword-match linker + audit.

These tests REQUIRE ``SCIX_TEST_DSN`` to point at a non-production database.
They seed their own fixture rows (≥10 papers, ≥20 entities) and then assert
that ``run_tier1_link`` produces ≥5 tier-1 rows and that ``run_audit``
generates a valid markdown audit file.

The Wilson CI helper is also tested against the spec-given anchor input
(95 / 100 → [0.887, 0.978]).
"""

from __future__ import annotations

import pathlib
import sys
from typing import Iterator

import psycopg
import pytest

from tests.helpers import get_test_dsn

# Make scripts/ importable (they are not a package).
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_tier1  # noqa: E402
import link_tier1  # noqa: E402

# ---------------------------------------------------------------------------
# Wilson CI unit test (runs without a DB)
# ---------------------------------------------------------------------------


class TestWilson95CI:
    def test_known_input_95_of_100(self) -> None:
        lo, hi = audit_tier1.wilson_95_ci(95, 100)
        # Spec anchor: [0.887, 0.978] — tolerance ±0.005 accounts for z choice.
        assert lo == pytest.approx(0.887, abs=0.005)
        assert hi == pytest.approx(0.978, abs=0.005)
        assert 0.0 <= lo <= hi <= 1.0

    def test_zero_total_returns_full_interval(self) -> None:
        assert audit_tier1.wilson_95_ci(0, 0) == (0.0, 1.0)

    def test_all_successes(self) -> None:
        lo, hi = audit_tier1.wilson_95_ci(10, 10)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert lo < 1.0

    def test_zero_successes(self) -> None:
        lo, hi = audit_tier1.wilson_95_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi > 0.0

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            audit_tier1.wilson_95_ci(-1, 10)
        with pytest.raises(ValueError):
            audit_tier1.wilson_95_ci(11, 10)


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip("SCIX_TEST_DSN must be set to a non-production DSN for tier-1 tests")
    return test_dsn


# Seed data designed so that:
#   - 12 papers (AC2 asks ≥10)
#   - 25 entities (AC2 asks ≥20), across 2 sources + 2 arxiv_class buckets
#   - expected tier-1 matches via canonical_name AND entity_aliases
#   - expected distinct (bibcode, entity_id) matches ≥ 5

_SEED_PAPERS: list[tuple[str, list[str], list[str]]] = [
    ("test_u06_0001", ["astro-ph.GA"], ["Hubble Space Telescope", "redshift"]),
    ("test_u06_0002", ["astro-ph.GA"], ["JWST", "galaxy"]),
    ("test_u06_0003", ["astro-ph.HE"], ["Chandra", "X-ray"]),
    ("test_u06_0004", ["astro-ph.HE"], ["NuSTAR", "accretion"]),
    ("test_u06_0005", ["astro-ph.SR"], ["Kepler", "exoplanet"]),
    ("test_u06_0006", ["astro-ph.SR"], ["TESS", "transit"]),
    ("test_u06_0007", ["astro-ph.CO"], ["Planck", "CMB"]),
    ("test_u06_0008", ["astro-ph.CO"], ["ACT", "lensing"]),
    ("test_u06_0009", ["astro-ph.IM"], ["ALMA", "interferometry"]),
    ("test_u06_0010", ["astro-ph.IM"], ["VLA", "radio"]),
    ("test_u06_0011", ["astro-ph.GA"], ["HST"]),  # alias-match only
    ("test_u06_0012", ["astro-ph.HE"], ["nothing_matches_here"]),
]

# (canonical_name, source)
_SEED_ENTITIES: list[tuple[str, str]] = [
    ("Hubble Space Telescope", "unit_test_a"),
    ("JWST", "unit_test_a"),
    ("Chandra", "unit_test_a"),
    ("NuSTAR", "unit_test_a"),
    ("Kepler", "unit_test_a"),
    ("TESS", "unit_test_a"),
    ("Planck", "unit_test_a"),
    ("ACT", "unit_test_a"),
    ("ALMA", "unit_test_a"),
    ("VLA", "unit_test_a"),
    ("redshift", "unit_test_b"),
    ("galaxy", "unit_test_b"),
    ("X-ray", "unit_test_b"),
    ("accretion", "unit_test_b"),
    ("exoplanet", "unit_test_b"),
    ("transit", "unit_test_b"),
    ("CMB", "unit_test_b"),
    ("lensing", "unit_test_b"),
    ("interferometry", "unit_test_b"),
    ("radio", "unit_test_b"),
    ("unrelated_alpha", "unit_test_b"),
    ("unrelated_beta", "unit_test_b"),
    ("unrelated_gamma", "unit_test_b"),
    ("unrelated_delta", "unit_test_b"),
    ("unrelated_epsilon", "unit_test_b"),
]

# (entity_canonical_name, alias)
_SEED_ALIASES: list[tuple[str, str]] = [
    ("Hubble Space Telescope", "HST"),
]


def _cleanup(conn: psycopg.Connection) -> None:
    """Remove only the fixture rows, leaving any other data intact."""
    bibcodes = [p[0] for p in _SEED_PAPERS]
    canonical_names = [e[0] for e in _SEED_ENTITIES]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        # entity_aliases cascade on entities delete
        cur.execute(
            "DELETE FROM entities WHERE canonical_name = ANY(%s) "
            "AND source IN ('unit_test_a','unit_test_b')",
            (canonical_names,),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
    conn.commit()


def _seed(conn: psycopg.Connection) -> None:
    _cleanup(conn)
    with conn.cursor() as cur:
        # Papers
        cur.executemany(
            "INSERT INTO papers (bibcode, arxiv_class, keywords) " "VALUES (%s, %s, %s)",
            _SEED_PAPERS,
        )
        # Entities (return id for alias wiring)
        name_to_id: dict[str, int] = {}
        for name, source in _SEED_ENTITIES:
            cur.execute(
                "INSERT INTO entities (canonical_name, entity_type, source) "
                "VALUES (%s, %s, %s) RETURNING id",
                (name, "test_type", source),
            )
            row = cur.fetchone()
            assert row is not None
            name_to_id[name] = row[0]
        # Aliases
        for canonical, alias in _SEED_ALIASES:
            cur.execute(
                "INSERT INTO entity_aliases (entity_id, alias, alias_source) "
                "VALUES (%s, %s, %s)",
                (name_to_id[canonical], alias, "test_seed"),
            )
    conn.commit()


@pytest.fixture()
def seeded_conn(dsn: str) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(dsn)
    try:
        _seed(conn)
        yield conn
    finally:
        try:
            _cleanup(conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


class TestLinkTier1EndToEnd:
    def test_inserts_rows_with_expected_shape(self, seeded_conn: psycopg.Connection) -> None:
        inserted = link_tier1.run_tier1_link(seeded_conn, dry_run=False)
        assert inserted >= 5, f"expected ≥5 tier-1 rows, got {inserted}"

        with seeded_conn.cursor() as cur:
            cur.execute(
                "SELECT tier, link_type, confidence, match_method "
                "FROM document_entities "
                "WHERE bibcode LIKE 'test_u06_%' "
                "  AND link_type = 'keyword_match' "
                "  AND tier = 1"
            )
            rows = cur.fetchall()
        assert len(rows) >= 5
        for tier, link_type, confidence, match_method in rows:
            assert tier == 1
            assert link_type == "keyword_match"
            assert confidence == pytest.approx(1.0)
            assert match_method == "keyword_exact_lower"

    def test_idempotent_on_rerun(self, seeded_conn: psycopg.Connection) -> None:
        first = link_tier1.run_tier1_link(seeded_conn, dry_run=False)
        assert first >= 5
        second = link_tier1.run_tier1_link(seeded_conn, dry_run=False)
        assert second == 0, "re-run must be idempotent via ON CONFLICT DO NOTHING"

    def test_alias_match_is_picked_up(self, seeded_conn: psycopg.Connection) -> None:
        link_tier1.run_tier1_link(seeded_conn, dry_run=False)
        with seeded_conn.cursor() as cur:
            cur.execute(
                "SELECT evidence FROM document_entities "
                "WHERE bibcode = 'test_u06_0011' "
                "  AND tier = 1 AND link_type = 'keyword_match'"
            )
            rows = cur.fetchall()
        assert rows, "paper 0011 should link to HST via alias"
        # at least one evidence row should reference alias match
        evidence_sources = {r[0].get("match_source") for r in rows}
        assert "alias" in evidence_sources

    def test_dry_run_does_not_persist(self, seeded_conn: psycopg.Connection) -> None:
        count = link_tier1.run_tier1_link(seeded_conn, dry_run=True)
        assert count >= 5
        with seeded_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_entities "
                "WHERE bibcode LIKE 'test_u06_%' AND tier = 1"
            )
            persisted = cur.fetchone()[0]
        assert persisted == 0


class TestAuditTier1:
    def test_generates_markdown_with_expected_columns(
        self, seeded_conn: psycopg.Connection, tmp_path: pathlib.Path
    ) -> None:
        link_tier1.run_tier1_link(seeded_conn, dry_run=False)

        out_path = tmp_path / "tier1_audit.md"
        audit_tier1.run_audit(seeded_conn, sample_size=200, output_path=out_path)
        assert out_path.exists()

        content = out_path.read_text(encoding="utf-8")
        assert "# Tier-1 keyword-match audit" in content
        assert "bibcode" in content
        assert "entity_id" in content
        assert "canonical_name" in content
        assert "source" in content
        assert "arxiv_class" in content
        assert "label_placeholder" in content
        assert "unlabeled" in content
        # Wilson CI for worked example
        assert "wilson_95_ci(95, 100)" in content
        # Check the CI is present (tolerate z-value rounding: 0.887 or 0.888)
        assert ("[0.887," in content) or ("[0.888," in content)

        # Sampled rows ≤ 200
        data_lines = [line for line in content.splitlines() if line.startswith("| test_u06_")]
        assert 1 <= len(data_lines) <= 200

    def test_stratified_sample_respects_bounds(self) -> None:
        rows = [
            audit_tier1.Tier1Row(
                bibcode=f"b{i}",
                entity_id=i,
                canonical_name=f"name_{i}",
                source=("unit_test_a" if i % 2 == 0 else "unit_test_b"),
                arxiv_class=("astro-ph.GA" if i % 3 == 0 else "astro-ph.HE"),
            )
            for i in range(500)
        ]
        sample = audit_tier1.stratified_sample(rows, sample_size=200)
        assert len(sample) == 200
        assert len({r.entity_id for r in sample}) == 200  # no duplicates

    def test_stratified_sample_small_population(self) -> None:
        rows = [
            audit_tier1.Tier1Row(
                bibcode=f"b{i}",
                entity_id=i,
                canonical_name=f"n{i}",
                source="unit_test_a",
                arxiv_class="astro-ph.GA",
            )
            for i in range(3)
        ]
        sample = audit_tier1.stratified_sample(rows, sample_size=200)
        assert len(sample) == 3
