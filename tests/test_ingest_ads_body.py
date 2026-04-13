"""Tests for scripts/ingest_ads_body.py + src/scix/ads_body.py.

Verifies:
- Loader COPY-loads JSONL into papers_ads_body.
- papers_external_ids.has_ads_body flips to true for loaded bibcodes.
- tsv generated column is populated and searchable via GIN.
- Unknown bibcodes (no matching papers row) are filtered, not errored.
- Resumable via ingest_log.
- Refuses to target production DSN without --yes-production.

All destructive tests gate on SCIX_TEST_DSN.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import is_production_dsn  # noqa: E402

# Make src/ importable so tests can import scix.ads_body directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

pytestmark = pytest.mark.skipif(
    TEST_DSN is None or (TEST_DSN is not None and is_production_dsn(TEST_DSN)),
    reason="Destructive loader tests require non-production SCIX_TEST_DSN.",
)


# Fixture bibcodes and bodies (small enough for a smoke test, realistic enough
# to exercise the code paths).
_FIXTURE_RECORDS: list[dict[str, object]] = [
    {
        "bibcode": "2099TEST.body.001",
        "body": (
            "We present a novel analysis of gravitational wave signals detected by LIGO. "
            "The matched filtering pipeline achieves 99.5 percent recall on the test set."
        ),
        "entry_date": "2099-01-15T00:00:00Z",
    },
    {
        "bibcode": "2099TEST.body.002",
        "body": (
            "This paper reports the discovery of an exoplanet orbiting a G-type star "
            "using transit photometry from the Kepler space telescope."
        ),
        "entry_date": "2099-02-20T00:00:00Z",
    },
    {
        "bibcode": "2099TEST.body.003",
        "body": "A short body for coverage testing.",
        "entry_date": "2099-03-01T00:00:00Z",
    },
    {
        "bibcode": "2099TEST.body.004",
        "body": (
            "X-ray observations of the galactic center reveal a previously unknown source "
            "with a luminosity of 1e36 erg/s."
        ),
        "entry_date": "2099-04-10T00:00:00Z",
    },
    {
        "bibcode": "2099TEST.body.005",
        "body": "Dark matter halo simulations at z=2.",
        "entry_date": "2099-05-05T00:00:00Z",
    },
]

# Records whose bibcodes are intentionally NOT in papers — loader must skip them.
_UNKNOWN_RECORDS: list[dict[str, object]] = [
    {
        "bibcode": "2099TEST.unknown.001",
        "body": "This bibcode has no matching papers row and must be filtered.",
        "entry_date": "2099-06-01T00:00:00Z",
    },
]

# Records with empty body — loader must skip.
_EMPTY_BODY_RECORDS: list[dict[str, object]] = [
    {
        "bibcode": "2099TEST.body.006",
        "body": "",
        "entry_date": "2099-07-01T00:00:00Z",
    },
    {
        "bibcode": "2099TEST.body.007",
        "body": None,
        "entry_date": "2099-08-01T00:00:00Z",
    },
]


@pytest.fixture()
def conn() -> Iterator[psycopg.Connection]:
    assert TEST_DSN is not None
    if is_production_dsn(TEST_DSN):
        pytest.skip("Refuses to run loader tests against production DSN.")
    c = psycopg.connect(TEST_DSN)
    c.autocommit = False
    try:
        yield c
    finally:
        c.rollback()
        c.close()


@pytest.fixture()
def seeded_papers(conn: psycopg.Connection) -> Iterator[list[str]]:
    """Insert fixture bibcodes into papers so the FK is satisfied."""
    bibcodes = [str(rec["bibcode"]) for rec in _FIXTURE_RECORDS + _EMPTY_BODY_RECORDS]
    with conn.cursor() as cur:
        for bibcode in bibcodes:
            cur.execute(
                "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
                (bibcode,),
            )
        # Pre-insert external_ids rows so the loader can UPDATE has_ads_body.
        for bibcode in bibcodes:
            cur.execute(
                "INSERT INTO papers_external_ids (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
                (bibcode,),
            )
    conn.commit()
    try:
        yield bibcodes
    finally:
        # Clean up in reverse FK order.
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM papers_ads_body WHERE bibcode = ANY(%s)",
                (bibcodes,),
            )
            cur.execute(
                "DELETE FROM papers_external_ids WHERE bibcode = ANY(%s)",
                (bibcodes,),
            )
            cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (bibcodes,))
            cur.execute(
                "DELETE FROM ingest_log WHERE filename LIKE 'test_ads_body_%'",
            )
        conn.commit()


def _write_fixture_jsonl(
    tmp_path: Path, records: list[dict[str, object]], name: str = "test_ads_body_fixture.jsonl"
) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    return path


# ---------------------------------------------------------------------------
# Loader module tests (direct API)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAdsBodyLoader:
    def test_loads_all_valid_records(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        cfg = LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=3)
        loader = AdsBodyLoader(cfg)
        stats = loader.run()

        assert stats.records_loaded == len(_FIXTURE_RECORDS)
        assert stats.records_skipped == 0

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, body_length FROM papers_ads_body WHERE bibcode = ANY(%s) "
                "ORDER BY bibcode",
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            rows = cur.fetchall()
        assert len(rows) == len(_FIXTURE_RECORDS)
        expected_by_bibcode = {str(r["bibcode"]): str(r["body"]) for r in _FIXTURE_RECORDS}
        for bibcode, length in rows:
            assert bibcode in expected_by_bibcode
            assert length == len(expected_by_bibcode[bibcode])

    def test_tsv_populated_and_searchable(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        AdsBodyLoader(LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=10)).run()

        with conn.cursor() as cur:
            # tsv non-null
            cur.execute(
                "SELECT COUNT(*) FROM papers_ads_body WHERE bibcode = ANY(%s) AND tsv IS NOT NULL",
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            assert cur.fetchone()[0] == len(_FIXTURE_RECORDS)

            # Text search via GIN: find the exoplanet paper
            cur.execute(
                """
                SELECT bibcode FROM papers_ads_body
                WHERE bibcode = ANY(%s) AND tsv @@ plainto_tsquery('english', 'exoplanet')
                """,
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            hits = [r[0] for r in cur.fetchall()]
            assert "2099TEST.body.002" in hits

    def test_flips_has_ads_body_flag(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        # Pre-condition: all flags false.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bool_or(has_ads_body) FROM papers_external_ids WHERE bibcode = ANY(%s)",
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            assert cur.fetchone()[0] is False

        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        AdsBodyLoader(LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=10)).run()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bool_and(has_ads_body) FROM papers_external_ids WHERE bibcode = ANY(%s)",
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            assert cur.fetchone()[0] is True

    def test_skips_unknown_bibcodes(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        """Bibcodes absent from papers must be skipped, not fail the load."""
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        mixed = _FIXTURE_RECORDS[:2] + _UNKNOWN_RECORDS
        jsonl = _write_fixture_jsonl(tmp_path, mixed)
        stats = AdsBodyLoader(LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=10)).run()

        assert stats.records_loaded == 2  # only the valid ones
        assert stats.records_skipped == 1  # unknown bibcode

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode FROM papers_ads_body WHERE bibcode LIKE '2099TEST.unknown.%'"
            )
            assert cur.fetchall() == []

    def test_skips_empty_body(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        """Records with empty/null body must be skipped."""
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        jsonl = _write_fixture_jsonl(tmp_path, _EMPTY_BODY_RECORDS)
        stats = AdsBodyLoader(LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=10)).run()

        assert stats.records_loaded == 0
        assert stats.records_skipped == len(_EMPTY_BODY_RECORDS)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM papers_ads_body WHERE bibcode = ANY(%s)",
                ([str(r["bibcode"]) for r in _EMPTY_BODY_RECORDS],),
            )
            assert cur.fetchone()[0] == 0

    def test_idempotent_reload(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        """Running the loader twice on the same fixture must not duplicate rows
        and must not fail (ingest_log resume + ON CONFLICT DO UPDATE)."""
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        cfg = LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=10)
        AdsBodyLoader(cfg).run()
        stats2 = AdsBodyLoader(cfg).run()

        # Second run is a no-op per ingest_log.is_complete.
        assert stats2.records_loaded == 0
        assert stats2.already_complete is True

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM papers_ads_body WHERE bibcode = ANY(%s)",
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            assert cur.fetchone()[0] == len(_FIXTURE_RECORDS)

    def test_refuses_production_dsn(self, tmp_path: Path) -> None:
        """Loader must refuse to run against a production DSN without --yes-production."""
        from scix.ads_body import AdsBodyLoader, LoaderConfig, ProductionGuardError

        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        cfg = LoaderConfig(dsn="dbname=scix", jsonl_path=jsonl, batch_size=10)
        loader = AdsBodyLoader(cfg)
        with pytest.raises(ProductionGuardError):
            loader.run()

    def test_dry_run_does_not_write(
        self,
        tmp_path: Path,
        conn: psycopg.Connection,
        seeded_papers: list[str],
    ) -> None:
        from scix.ads_body import AdsBodyLoader, LoaderConfig

        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        cfg = LoaderConfig(dsn=TEST_DSN, jsonl_path=jsonl, batch_size=10, dry_run=True)
        stats = AdsBodyLoader(cfg).run()

        assert stats.records_loaded == len(_FIXTURE_RECORDS)
        assert stats.dry_run is True

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM papers_ads_body WHERE bibcode = ANY(%s)",
                ([str(r["bibcode"]) for r in _FIXTURE_RECORDS],),
            )
            assert cur.fetchone()[0] == 0


# LoaderConfig frozen check lives in tests/test_ads_body_unit.py (no DB needed).


# ---------------------------------------------------------------------------
# CLI entry point smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCli:
    def test_cli_help(self) -> None:
        script = REPO_ROOT / "scripts" / "ingest_ads_body.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--jsonl" in result.stdout
        assert "--dsn" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--yes-production" in result.stdout

    def test_cli_refuses_production_without_flag(self, tmp_path: Path) -> None:
        """CLI must exit non-zero when DSN points at production without --yes-production."""
        script = REPO_ROOT / "scripts" / "ingest_ads_body.py"
        jsonl = _write_fixture_jsonl(tmp_path, _FIXTURE_RECORDS)
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dsn",
                "dbname=scix",
                "--jsonl",
                str(jsonl),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "production" in combined
