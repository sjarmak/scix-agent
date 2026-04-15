"""Tests for scripts/backfill_papers_ads_body.py — populate papers_ads_body from papers.body.

Integration tests run against scix_test. They:
  1. Insert test papers with body text into papers.
  2. Run the backfill function.
  3. Verify papers_ads_body has the correct rows, body_length, and tsv.
  4. Verify ON CONFLICT DO NOTHING behavior (idempotency).

Unit tests cover argument parsing, production guard, and stats reporting
without any database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Determine test DSN — skip integration tests if not set.
TEST_DSN = os.environ.get("SCIX_TEST_DSN", "")
SKIP_INTEGRATION = not TEST_DSN
INTEGRATION_REASON = "SCIX_TEST_DSN not set"


# ---------------------------------------------------------------------------
# Unit tests (no database)
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Test argument parsing for the backfill CLI."""

    def test_defaults(self) -> None:
        from backfill_papers_ads_body import _parse_args

        args = _parse_args([])
        assert args.dsn is None
        assert args.batch_size == 50_000
        assert args.yes_production is False
        assert args.verbose is False

    def test_custom_args(self) -> None:
        from backfill_papers_ads_body import _parse_args

        args = _parse_args(
            [
                "--dsn",
                "dbname=scix_test",
                "--batch-size",
                "1000",
                "--yes-production",
                "-v",
            ]
        )
        assert args.dsn == "dbname=scix_test"
        assert args.batch_size == 1000
        assert args.yes_production is True
        assert args.verbose is True

    def test_dry_run_flag(self) -> None:
        from backfill_papers_ads_body import _parse_args

        args = _parse_args(["--dry-run"])
        assert args.dry_run is True


class TestProductionGuard:
    """Backfill must refuse production DSN without --yes-production."""

    def test_refuses_production_dsn(self) -> None:
        from backfill_papers_ads_body import main

        # main() should return exit code 2 when production guard fires.
        code = main(["--dsn", "dbname=scix"])
        assert code == 2

    def test_none_dsn_resolves_to_default(self) -> None:
        from backfill_papers_ads_body import main

        # With no --dsn, effective DSN is DEFAULT_DSN (dbname=scix).
        # Should refuse unless --yes-production is passed.
        code = main([])
        assert code == 2


class TestBackfillConfig:
    """BackfillConfig is frozen and has sensible defaults."""

    def test_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from backfill_papers_ads_body import BackfillConfig

        cfg = BackfillConfig(dsn="dbname=scix_test")
        with pytest.raises(FrozenInstanceError):
            cfg.dsn = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        from backfill_papers_ads_body import BackfillConfig

        cfg = BackfillConfig(dsn="dbname=scix_test")
        assert cfg.batch_size == 50_000
        assert cfg.dry_run is False
        assert cfg.yes_production is False


class TestBackfillStats:
    """BackfillStats is frozen."""

    def test_stats_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from backfill_papers_ads_body import BackfillStats

        stats = BackfillStats(
            rows_inserted=100,
            rows_already_present=5,
            total_body_rows=105,
            elapsed_seconds=1.0,
            dry_run=False,
        )
        with pytest.raises(FrozenInstanceError):
            stats.rows_inserted = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration tests (require scix_test)
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_conn():
    """Yield a psycopg connection to scix_test, rolling back after each test."""
    import psycopg

    conn = psycopg.connect(TEST_DSN)
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture()
def seed_papers(test_conn):
    """Insert test papers into papers + clean up papers_ads_body for those bibcodes."""
    bibcodes = [
        "2024TEST..001A",
        "2024TEST..002B",
        "2024TEST..003C",
        "2024TEST..004D",  # will have NULL body
    ]
    bodies = [
        "This is a test paper about dark matter and galaxy formation.",
        "A study of stellar evolution and nucleosynthesis in massive stars.",
        "Observations of gravitational waves from binary neutron star mergers.",
        None,  # no body
    ]

    with test_conn.cursor() as cur:
        # Clean up from previous test runs
        cur.execute(
            "DELETE FROM papers_ads_body WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )

        # Insert test papers
        for bib, body in zip(bibcodes, bodies):
            cur.execute(
                "INSERT INTO papers (bibcode, title, body) VALUES (%s, %s, %s)",
                (bib, f"Test paper {bib}", body),
            )

    test_conn.commit()

    yield bibcodes, bodies

    # Cleanup
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM papers_ads_body WHERE bibcode = ANY(%s)", (bibcodes,))
        cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (bibcodes,))
    test_conn.commit()


@pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
class TestBackfillIntegration:
    """Integration tests against scix_test."""

    def test_backfill_populates_rows(self, test_conn, seed_papers) -> None:
        from backfill_papers_ads_body import BackfillConfig, run_backfill

        bibcodes, bodies = seed_papers
        cfg = BackfillConfig(dsn=TEST_DSN, batch_size=2)
        stats = run_backfill(cfg)

        # Should have inserted 3 rows (4th has NULL body)
        assert stats.rows_inserted >= 3
        assert stats.dry_run is False

        # Verify rows exist with correct body_length
        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, body_length FROM papers_ads_body "
                "WHERE bibcode = ANY(%s) ORDER BY bibcode",
                (bibcodes[:3],),
            )
            rows = cur.fetchall()
            assert len(rows) == 3
            for bib, length in rows:
                idx = bibcodes.index(bib)
                assert length == len(bodies[idx])

    def test_backfill_idempotent(self, test_conn, seed_papers) -> None:
        from backfill_papers_ads_body import BackfillConfig, run_backfill

        cfg = BackfillConfig(dsn=TEST_DSN, batch_size=2)

        # First run
        stats1 = run_backfill(cfg)
        assert stats1.rows_inserted >= 3

        # Second run — should be idempotent (ON CONFLICT DO NOTHING)
        stats2 = run_backfill(cfg)
        assert stats2.rows_already_present >= 3
        assert stats2.rows_inserted == 0

    def test_backfill_null_body_skipped(self, test_conn, seed_papers) -> None:
        from backfill_papers_ads_body import BackfillConfig, run_backfill

        bibcodes, _ = seed_papers
        cfg = BackfillConfig(dsn=TEST_DSN, batch_size=10)
        run_backfill(cfg)

        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM papers_ads_body WHERE bibcode = %s",
                (bibcodes[3],),  # the NULL body paper
            )
            assert cur.fetchone() is None

    def test_tsquery_works(self, test_conn, seed_papers) -> None:
        from backfill_papers_ads_body import BackfillConfig, run_backfill

        cfg = BackfillConfig(dsn=TEST_DSN, batch_size=10)
        run_backfill(cfg)

        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode FROM papers_ads_body "
                "WHERE tsv @@ to_tsquery('english', 'dark & matter')"
            )
            rows = cur.fetchall()
            assert len(rows) >= 1
            assert rows[0][0] == "2024TEST..001A"

    def test_dry_run_does_not_insert(self, test_conn, seed_papers) -> None:
        from backfill_papers_ads_body import BackfillConfig, run_backfill

        bibcodes, _ = seed_papers
        cfg = BackfillConfig(dsn=TEST_DSN, batch_size=10, dry_run=True)
        stats = run_backfill(cfg)

        assert stats.dry_run is True
        assert stats.total_body_rows >= 3

        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM papers_ads_body WHERE bibcode = ANY(%s)",
                (bibcodes,),
            )
            assert cur.fetchone()[0] == 0
