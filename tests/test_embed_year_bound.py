"""The nightly embed run must bound its unembedded scan by publication year.

Unbounded, the anti-join that finds papers with no dense vector is a parallel
seq scan over every paper (width 351, so every abstract is read out of TOAST)
hashed against a full scan of the 35M-row indus_qdrant_synced watermark:
planned ~10.1M, measured 530 s before the first row on a cold cache. That was
98% of a 540 s nightly drain whose actual GPU work was ~10 s.

Bounding on ``papers.year`` uses the existing ``idx_papers_year`` and drops the
plan to ~2.06M. The cost is that ``year`` is publication year, not ingest date,
so a newly ingested old paper is invisible to a bounded run unless supplied by
an explicit bibcode source.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.embed import (  # noqa: E402
    NIGHTLY_YEAR_LOOKBACK,
    default_year_floor,
    unembedded_predicate,
)
from scripts.embed import load_bibcodes  # noqa: E402
from tests.helpers import is_production_dsn  # noqa: E402


class TestUnembeddedPredicate:
    def test_unbounded_form_has_no_year_clause_and_no_params(self) -> None:
        sql, params = unembedded_predicate(None)
        assert "p.year" not in sql
        assert params == []

    def test_bounded_form_adds_an_indexable_year_floor(self) -> None:
        sql, params = unembedded_predicate(2025)
        assert "p.year >= %s" in sql
        assert params == [2025]

    def test_bounded_form_keeps_the_watermark_anti_join(self) -> None:
        """The year bound narrows the scan; it must not change what 'unembedded' means."""
        sql, _ = unembedded_predicate(2025)
        assert "indus_qdrant_synced" in sql
        assert "s.bibcode IS NULL" in sql
        assert "p.title IS NOT NULL" in sql

    def test_placeholder_count_matches_param_count(self) -> None:
        """Guards the ordering contract: params are consumed before any LIMIT."""
        for floor in (None, 1995, 2026):
            sql, params = unembedded_predicate(floor)
            assert sql.count("%s") == len(params)

    def test_explicit_bibcodes_are_unioned_with_the_year_bound(self) -> None:
        sql, params = unembedded_predicate(2025, ("1995ApJ...1A", "2026ApJ...2B"))

        assert "(p.year >= %s OR p.bibcode = ANY(%s))" in sql
        assert params == [2025, ["1995ApJ...1A", "2026ApJ...2B"]]

    def test_explicit_bibcodes_do_not_narrow_a_full_scan(self) -> None:
        sql, params = unembedded_predicate(None, ("1995ApJ...1A",))

        assert sql == unembedded_predicate(None)[0]
        assert params == []


class TestDefaultYearFloor:
    def test_looks_back_from_the_given_year(self) -> None:
        assert default_year_floor(date(2026, 7, 28)) == 2026 - NIGHTLY_YEAR_LOOKBACK

    def test_covers_late_arriving_previous_year_records(self) -> None:
        """A January run must still see papers published the previous year."""
        assert default_year_floor(date(2027, 1, 2)) <= 2026

    def test_lookback_is_at_least_one_year(self) -> None:
        assert NIGHTLY_YEAR_LOOKBACK >= 1


class TestCLIWiring:
    """The CLI decides scope; the pipeline only executes it."""

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "embed.py"), *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    def test_full_and_year_floor_are_mutually_exclusive(self) -> None:
        result = self._run("--full", "--year-floor", "2020")
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_help_documents_both_flags(self) -> None:
        result = self._run("--help")
        assert result.returncode == 0
        assert "--full" in result.stdout
        assert "--year-floor" in result.stdout
        assert "--bibcodes-from-jsonl" in result.stdout

    def test_help_warns_that_full_is_not_for_the_nightly_run(self) -> None:
        """The flag's cost must be visible at the point of use, not only in a doc."""
        result = self._run("--help")
        assert "nightly" in result.stdout.lower()

    def test_bibcode_source_option_reaches_cli_validation(self, tmp_path: Path) -> None:
        """Supplying the repeatable option must not fail inside argparse."""
        source = tmp_path / "backfill.jsonl"
        source.write_text('{"bibcode": "1995A"}\n', encoding="utf-8")

        result = self._run(
            "--bibcodes-from-jsonl",
            str(source),
            "--model",
            "unsupported-model",
        )

        assert result.returncode != 0
        assert "supports only model_name='indus'" in result.stderr
        assert "has no attribute 'append'" not in result.stderr

    def test_load_bibcodes_reads_compressed_jsonl_and_deduplicates(self, tmp_path: Path) -> None:
        import gzip
        import json

        first = tmp_path / "first.jsonl.gz"
        second = tmp_path / "second.jsonl"
        with gzip.open(first, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"bibcode": "1995A"}) + "\n")
            stream.write(json.dumps({"bibcode": "2026B"}) + "\n")
        second.write_text(
            json.dumps({"bibcode": "1995A"}) + "\n" + json.dumps({"bibcode": "1987C"}) + "\n",
            encoding="utf-8",
        )

        assert load_bibcodes((first, second)) == ("1995A", "2026B", "1987C")

    def test_load_bibcodes_rejects_a_record_without_a_bibcode(self, tmp_path: Path) -> None:
        import json

        source = tmp_path / "bad.jsonl"
        source.write_text(json.dumps({"title": "missing key"}) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"bad\.jsonl line 1.*bibcode"):
            load_bibcodes((source,))


@pytest.mark.parametrize("floor", [None, 2025])
def test_predicate_is_a_prefix_of_the_same_from_clause(floor: int | None) -> None:
    """Bounded and unbounded forms must share one FROM/JOIN definition.

    Two hand-maintained copies would drift, and a drifted definition of
    'unembedded' silently under- or over-embeds.
    """
    sql, _ = unembedded_predicate(floor)
    assert sql.startswith(unembedded_predicate(None)[0])


def test_explicit_old_bibcode_is_selected_in_bounded_database_query() -> None:
    """The production-shaped predicate includes only named pre-floor papers."""
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn or is_production_dsn(dsn):
        pytest.skip("requires a non-production SCIX_TEST_DSN")

    with psycopg.connect(dsn) as conn, conn.cursor() as cursor:
        cursor.execute("CREATE TEMP TABLE papers (bibcode TEXT PRIMARY KEY, title TEXT, year INT)")
        cursor.execute("CREATE TEMP TABLE indus_qdrant_synced (bibcode TEXT PRIMARY KEY)")
        cursor.executemany(
            "INSERT INTO papers (bibcode, title, year) VALUES (%s, %s, %s)",
            (
                ("1995-target", "target", 1995),
                ("1994-other", "other", 1994),
                ("2026-recent", "recent", 2026),
            ),
        )
        sql, params = unembedded_predicate(2025, ("1995-target",))
        cursor.execute("SELECT p.bibcode " + sql + " ORDER BY p.bibcode", params)

        assert [row[0] for row in cursor.fetchall()] == ["1995-target", "2026-recent"]
