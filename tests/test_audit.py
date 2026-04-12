"""Unit + integration tests for the M9 audit sampler and report writer."""

from __future__ import annotations

import pathlib

import pytest

from scix.eval.audit import (
    AuditCandidate,
    sample_stratified,
    write_audit_report,
)
from scix.eval.wilson import wilson_95_ci

from tests.helpers import get_test_dsn

# ---------------------------------------------------------------------------
# Wilson CI — unit tests (no DB)
# ---------------------------------------------------------------------------


class TestWilson95CI:
    def test_known_anchor_95_of_100(self) -> None:
        lo, hi = wilson_95_ci(95, 100)
        # Spec: [0.887, 0.978] ±0.005 per u06 convention.
        assert lo == pytest.approx(0.887, abs=0.005)
        assert hi == pytest.approx(0.978, abs=0.005)
        assert 0.0 <= lo <= hi <= 1.0

    def test_zero_total_returns_full_interval(self) -> None:
        assert wilson_95_ci(0, 0) == (0.0, 1.0)

    def test_all_successes_upper_bound_is_one(self) -> None:
        lo, hi = wilson_95_ci(10, 10)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert lo < 1.0

    def test_zero_successes_lower_bound_is_zero(self) -> None:
        lo, hi = wilson_95_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi > 0.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            wilson_95_ci(-1, 10)
        with pytest.raises(ValueError):
            wilson_95_ci(11, 10)


# ---------------------------------------------------------------------------
# write_audit_report — unit test (no DB)
# ---------------------------------------------------------------------------


class TestWriteAuditReport:
    def test_report_contains_tier_rows_and_wilson_ci(self, tmp_path: pathlib.Path) -> None:
        candidates = [
            AuditCandidate(tier=1, bibcode="2099TEST..001A", entity_id=10),
            AuditCandidate(tier=1, bibcode="2099TEST..002A", entity_id=11),
            AuditCandidate(tier=2, bibcode="2099TEST..003A", entity_id=12),
            AuditCandidate(tier=2, bibcode="2099TEST..004A", entity_id=13),
        ]
        labels = {
            ("2099TEST..001A", 10): "correct",
            ("2099TEST..002A", 11): "correct",
            ("2099TEST..003A", 12): "incorrect",
            ("2099TEST..004A", 13): "correct",
        }

        out = tmp_path / "eval_report.md"
        write_audit_report(out, candidates, labels, title="unit test report")

        text = out.read_text(encoding="utf-8")
        assert "unit test report" in text
        assert "| tier |" in text
        # Tier 1: 2/2 correct → Wilson lower >0, upper == 1
        assert "| 1 | 2 | 2 |" in text
        # Tier 2: 1/2 correct
        assert "| 2 | 1 | 2 |" in text
        # Worked example present
        assert "wilson_95_ci(95, 100)" in text


# ---------------------------------------------------------------------------
# sample_stratified — integration test (needs SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


FIXTURE_BIBCODES = [f"2099AUDT..{i:03d}A" for i in range(6)]
FIXTURE_TIERS = (1, 2, 4, 5)
FIXTURE_PER_TIER = 3  # 4 tiers × 3 = 12 rows


@pytest.fixture
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip("SCIX_TEST_DSN must be set to a non-production DSN")
    return test_dsn


@pytest.fixture
def seeded_conn(dsn: str):
    import psycopg

    conn = psycopg.connect(dsn)
    _seed(conn)
    try:
        yield conn
    finally:
        try:
            _cleanup(conn)
        finally:
            conn.close()


def _seed(conn) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO papers (bibcode, title) VALUES (%s, %s) "
            "ON CONFLICT (bibcode) DO NOTHING",
            [(b, f"audit fixture paper {b}") for b in FIXTURE_BIBCODES],
        )
        cur.execute("""
            INSERT INTO entities (canonical_name, entity_type, source)
            SELECT 'm9_audit_ent_' || g::text, 'concept', 'm9_audit_fixture'
            FROM generate_series(1, 4) AS g
            ON CONFLICT DO NOTHING
            """)
        cur.execute("SELECT id FROM entities WHERE source = 'm9_audit_fixture' ORDER BY id LIMIT 4")
        eids = [int(r[0]) for r in cur.fetchall()]
        assert len(eids) == 4

        rows = []
        for ti, tier in enumerate(FIXTURE_TIERS):
            for i in range(FIXTURE_PER_TIER):
                bib = FIXTURE_BIBCODES[(ti * FIXTURE_PER_TIER + i) % len(FIXTURE_BIBCODES)]
                eid = eids[i % len(eids)]
                rows.append((bib, eid, f"m9_audit_tier{tier}", tier, 0.5))
        cur.executemany(
            "INSERT INTO document_entities (bibcode, entity_id, link_type, tier, confidence) "  # noqa: resolver-lint
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()


def _cleanup(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE link_type LIKE 'm9_audit_tier%'"  # noqa: resolver-lint
        )
        cur.execute("DELETE FROM entities WHERE source = 'm9_audit_fixture'")
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (FIXTURE_BIBCODES,),
        )
    conn.commit()


@pytest.mark.integration
class TestSampleStratified:
    def test_samples_contain_fixture_tiers(self, seeded_conn) -> None:
        # Sample up to 10 per tier — we only seeded 3 per tier, so every
        # fixture row should come back (but base document_entities may have
        # other rows in scix_test; filter to fixture bibcodes).
        candidates = sample_stratified(seeded_conn, n_per_tier=10, seed=0.0)
        fixture = [c for c in candidates if c.bibcode in set(FIXTURE_BIBCODES)]

        tiers_seen = {c.tier for c in fixture}
        for tier in FIXTURE_TIERS:
            assert tier in tiers_seen, f"missing tier {tier} in sample"

        # Per-tier cap: no tier should exceed 10 fixture rows (it's 3 total).
        for tier in FIXTURE_TIERS:
            got = [c for c in fixture if c.tier == tier]
            assert len(got) <= FIXTURE_PER_TIER

    def test_n_per_tier_zero_returns_empty(self, seeded_conn) -> None:
        assert sample_stratified(seeded_conn, n_per_tier=0) == []
