"""The nightly embed run must bound its unembedded scan by publication year.

Unbounded, the anti-join that finds papers with no dense vector is a parallel
seq scan over every paper (width 351, so every abstract is read out of TOAST)
hashed against a full scan of the 35M-row indus_qdrant_synced watermark:
planned ~10.1M, measured 530 s before the first row on a cold cache. That was
98% of a 540 s nightly drain whose actual GPU work was ~10 s.

Bounding on ``papers.year`` uses the existing ``idx_papers_year`` and drops the
plan to ~2.06M. The cost is that ``year`` is publication year, not ingest date,
so a newly ingested old paper is invisible to a bounded run — which is what
``--full`` is for.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.embed import (  # noqa: E402
    NIGHTLY_YEAR_LOOKBACK,
    default_year_floor,
    unembedded_predicate,
)


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

    def test_help_warns_that_full_is_not_for_the_nightly_run(self) -> None:
        """The flag's cost must be visible at the point of use, not only in a doc."""
        result = self._run("--help")
        assert "nightly" in result.stdout.lower()


@pytest.mark.parametrize("floor", [None, 2025])
def test_predicate_is_a_prefix_of_the_same_from_clause(floor: int | None) -> None:
    """Bounded and unbounded forms must share one FROM/JOIN definition.

    Two hand-maintained copies would drift, and a drifted definition of
    'unembedded' silently under- or over-embeds.
    """
    sql, _ = unembedded_predicate(floor)
    assert sql.startswith(unembedded_predicate(None)[0])
