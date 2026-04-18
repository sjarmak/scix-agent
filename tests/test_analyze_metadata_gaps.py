"""Tests for scripts/analyze_metadata_gaps.py — ADS metadata gap analysis."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make both the script module and the scix package importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from analyze_metadata_gaps import (  # noqa: E402
    ASTRONOMY_COHORT_SQL,
    DEFAULT_COHORT_BIBCODE_CAP,
    ENTITY_FIELDS,
    FIELD_TO_ENTITY_TYPE,
    FieldCoverage,
    GapEntityRow,
    GapReport,
    build_report,
    compute_field_coverage,
    compute_gap_ranking,
    fetch_cohort_rows,
    is_row_in_gap_cohort,
    safety_guard,
    write_report,
)


# ---------------------------------------------------------------------------
# Cohort SQL predicate
# ---------------------------------------------------------------------------


class TestAstronomyCohortSQL:
    def test_contains_arxiv_astro_ph_filter(self) -> None:
        assert "astro-ph" in ASTRONOMY_COHORT_SQL.lower()
        assert "ilike" in ASTRONOMY_COHORT_SQL.lower()
        assert "unnest(arxiv_class)" in ASTRONOMY_COHORT_SQL.lower()

    def test_contains_collection_database_filter(self) -> None:
        assert "'astronomy' = ANY(database)" in ASTRONOMY_COHORT_SQL

    def test_combines_predicates_with_or(self) -> None:
        normalized = " ".join(ASTRONOMY_COHORT_SQL.upper().split())
        assert " OR " in normalized


# ---------------------------------------------------------------------------
# is_row_in_gap_cohort
# ---------------------------------------------------------------------------


class TestGapCohortMembership:
    def test_all_fields_populated_not_in_cohort(self) -> None:
        row = {
            "bibcode": "2024A",
            "facility": ["HST"],
            "data": ["SDSS"],
            "keyword_norm": ["photometry"],
        }
        assert is_row_in_gap_cohort(row) is False

    def test_one_field_none_is_in_cohort(self) -> None:
        row = {
            "bibcode": "2024A",
            "facility": ["HST"],
            "data": None,
            "keyword_norm": ["photometry"],
        }
        assert is_row_in_gap_cohort(row) is True

    def test_one_field_empty_list_is_in_cohort(self) -> None:
        row = {
            "bibcode": "2024A",
            "facility": [],
            "data": ["SDSS"],
            "keyword_norm": ["photometry"],
        }
        assert is_row_in_gap_cohort(row) is True

    def test_list_of_whitespace_only_is_in_cohort(self) -> None:
        row = {
            "bibcode": "2024A",
            "facility": ["HST"],
            "data": ["SDSS"],
            "keyword_norm": ["", "  "],
        }
        assert is_row_in_gap_cohort(row) is True

    def test_all_fields_missing_is_in_cohort(self) -> None:
        row = {"bibcode": "2024A", "facility": None, "data": None, "keyword_norm": None}
        assert is_row_in_gap_cohort(row) is True

    def test_non_list_value_is_in_cohort(self) -> None:
        # Defensive — db row should always be list/None, but string would count as empty.
        row = {
            "bibcode": "2024A",
            "facility": "HST",
            "data": ["SDSS"],
            "keyword_norm": ["photometry"],
        }
        assert is_row_in_gap_cohort(row) is True


# ---------------------------------------------------------------------------
# compute_field_coverage
# ---------------------------------------------------------------------------


class TestComputeFieldCoverage:
    def test_basic_mix(self) -> None:
        rows = [
            {"facility": ["HST"], "data": ["SDSS"], "keyword_norm": ["m"]},
            {"facility": None, "data": ["2MASS"], "keyword_norm": None},
            {"facility": [], "data": [], "keyword_norm": ["m"]},
            {"facility": ["VLT"], "data": None, "keyword_norm": ["m"]},
        ]
        cov = compute_field_coverage(rows)

        assert cov["facility"].populated == 2
        assert cov["facility"].total == 4
        assert cov["facility"].coverage_pct == 50.0

        assert cov["data"].populated == 2
        assert cov["data"].coverage_pct == 50.0

        assert cov["keyword_norm"].populated == 3
        assert cov["keyword_norm"].coverage_pct == 75.0

    def test_empty_input(self) -> None:
        cov = compute_field_coverage([])
        for field in ENTITY_FIELDS:
            assert cov[field].populated == 0
            assert cov[field].total == 0
            assert cov[field].coverage_pct == 0.0

    def test_all_populated(self) -> None:
        rows = [{"facility": ["a"], "data": ["b"], "keyword_norm": ["c"]}]
        cov = compute_field_coverage(rows)
        for field in ENTITY_FIELDS:
            assert cov[field].coverage_pct == 100.0


# ---------------------------------------------------------------------------
# compute_gap_ranking
# ---------------------------------------------------------------------------


class TestComputeGapRanking:
    def test_orders_by_gap_count_descending(self) -> None:
        # facility: 3 missing, data: 1 missing, keyword_norm: 2 missing
        rows = [
            {"facility": None, "data": ["x"], "keyword_norm": ["x"]},
            {"facility": None, "data": ["x"], "keyword_norm": None},
            {"facility": None, "data": None, "keyword_norm": None},
            {"facility": ["a"], "data": ["x"], "keyword_norm": ["x"]},
        ]
        ranking = compute_gap_ranking(rows)

        assert ranking[0].field == "facility"
        assert ranking[0].gap_count == 3
        assert ranking[1].field == "keyword_norm"
        assert ranking[1].gap_count == 2
        assert ranking[2].field == "data"
        assert ranking[2].gap_count == 1

    def test_entity_types_match_mapping(self) -> None:
        rows = [{"facility": None, "data": None, "keyword_norm": None}]
        ranking = compute_gap_ranking(rows)
        seen = {row.field: row.entity_type for row in ranking}
        assert seen == FIELD_TO_ENTITY_TYPE

    def test_ratio_is_count_over_total(self) -> None:
        rows = [
            {"facility": None, "data": ["x"], "keyword_norm": ["x"]},
            {"facility": None, "data": ["x"], "keyword_norm": ["x"]},
            {"facility": ["a"], "data": ["x"], "keyword_norm": ["x"]},
            {"facility": ["a"], "data": ["x"], "keyword_norm": ["x"]},
        ]
        ranking = compute_gap_ranking(rows)
        facility_row = next(r for r in ranking if r.field == "facility")
        assert facility_row.gap_ratio == 0.5

    def test_top_n_limit(self) -> None:
        rows = [{"facility": None, "data": None, "keyword_norm": None}]
        assert len(compute_gap_ranking(rows, top_n=1)) == 1

    def test_empty_input_returns_zeroed_rows(self) -> None:
        ranking = compute_gap_ranking([])
        assert len(ranking) == len(ENTITY_FIELDS)
        for r in ranking:
            assert r.gap_count == 0
            assert r.gap_ratio == 0.0


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def _rows(self) -> list[dict]:
        # 10 papers: 3 with gaps, 7 complete
        rows: list[dict] = []
        for i in range(7):
            rows.append(
                {
                    "bibcode": f"complete-{i}",
                    "facility": ["HST"],
                    "data": ["SDSS"],
                    "keyword_norm": ["method"],
                }
            )
        for i in range(3):
            rows.append(
                {
                    "bibcode": f"gap-{i}",
                    "facility": ["HST"],
                    "data": None,
                    "keyword_norm": ["method"],
                }
            )
        return rows

    def test_required_report_keys(self) -> None:
        report = build_report(self._rows(), sample_size=10, dsn_redacted="dbname=scix_test")
        d = report.to_dict()
        required = {
            "total_astronomy_papers",
            "gap_cohort_count",
            "gap_cohort_ratio",
            "gap_cohort_bibcodes_sample",
            "gap_cohort_bibcodes_sidecar_path",
            "top_gap_entity_types",
            "per_field_coverage",
            "meta",
        }
        assert required.issubset(d.keys())

        for meta_key in ("sample_size", "dsn_redacted", "generated_at", "gap_ratio_under_40pct"):
            assert meta_key in d["meta"]

    def test_gap_ratio_computed(self) -> None:
        report = build_report(self._rows(), sample_size=10, dsn_redacted="x")
        assert report.total_astronomy_papers == 10
        assert report.gap_cohort_count == 3
        assert report.gap_cohort_ratio == 0.3

    def test_under_40pct_flag_true(self) -> None:
        report = build_report(self._rows(), sample_size=10, dsn_redacted="x")
        assert report.meta["gap_ratio_under_40pct"] is True

    def test_under_40pct_flag_false(self) -> None:
        # 5 of 6 have gaps -> ratio 5/6 >= 0.4
        rows = [
            {"bibcode": f"g{i}", "facility": None, "data": None, "keyword_norm": None}
            for i in range(5)
        ] + [{"bibcode": "ok", "facility": ["x"], "data": ["x"], "keyword_norm": ["x"]}]
        report = build_report(rows, sample_size=None, dsn_redacted="x")
        assert report.meta["gap_ratio_under_40pct"] is False

    def test_report_is_json_serializable(self) -> None:
        report = build_report(self._rows(), sample_size=10, dsn_redacted="x")
        json.dumps(report.to_dict())

    def test_small_cohort_is_inlined(self) -> None:
        report = build_report(self._rows(), sample_size=10, dsn_redacted="x")
        assert report.gap_cohort_bibcodes_sidecar_path is None
        assert len(report.gap_cohort_bibcodes_sample) == 3

    def test_large_cohort_spills_to_sidecar(self) -> None:
        rows = [
            {"bibcode": f"g{i}", "facility": None, "data": None, "keyword_norm": None}
            for i in range(5)
        ]
        report = build_report(
            rows,
            sample_size=5,
            dsn_redacted="x",
            cohort_bibcode_cap=2,
        )
        assert report.gap_cohort_bibcodes_sidecar_path is not None
        assert len(report.gap_cohort_bibcodes_sample) == 2
        assert report.meta["gap_cohort_full_size"] == 5

    def test_empty_rows_do_not_divide_by_zero(self) -> None:
        report = build_report([], sample_size=0, dsn_redacted="x")
        assert report.total_astronomy_papers == 0
        assert report.gap_cohort_count == 0
        assert report.gap_cohort_ratio == 0.0
        # 0 < 0.4, so flag is True
        assert report.meta["gap_ratio_under_40pct"] is True


# ---------------------------------------------------------------------------
# fetch_cohort_rows (mocked DB)
# ---------------------------------------------------------------------------


def _mock_conn_with_cursor(
    fetchall_rows: list[tuple],
    *,
    iterable_rows: list[tuple] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (conn, default_cursor, named_cursor).  Both cursor paths are wired."""
    mock_conn = MagicMock()

    default_cursor = MagicMock()
    default_cursor.fetchall.return_value = fetchall_rows
    default_ctx = MagicMock()
    default_ctx.__enter__ = MagicMock(return_value=default_cursor)
    default_ctx.__exit__ = MagicMock(return_value=False)

    named_cursor = MagicMock()
    named_cursor.__iter__ = lambda self: iter(iterable_rows or [])
    named_ctx = MagicMock()
    named_ctx.__enter__ = MagicMock(return_value=named_cursor)
    named_ctx.__exit__ = MagicMock(return_value=False)

    def _cursor_router(*args, **kwargs):
        if "name" in kwargs or args:
            return named_ctx
        return default_ctx

    mock_conn.cursor.side_effect = _cursor_router
    return mock_conn, default_cursor, named_cursor


class TestFetchCohortRows:
    def test_with_sample_size_uses_limit_query(self) -> None:
        rows = [("2024A", ["HST"], ["SDSS"], ["m"])]
        conn, default_cursor, named_cursor = _mock_conn_with_cursor(rows)

        fetched = fetch_cohort_rows(conn, sample_size=10)

        assert len(fetched) == 1
        assert fetched[0]["bibcode"] == "2024A"
        assert fetched[0]["facility"] == ["HST"]
        # Should have called the default (non-named) cursor with a LIMIT query
        default_cursor.execute.assert_called_once()
        executed_sql, executed_params = default_cursor.execute.call_args[0]
        assert "LIMIT" in executed_sql
        assert executed_params == (10,)
        # Streaming path should NOT have been used.
        named_cursor.execute.assert_not_called()

    def test_without_sample_size_uses_streaming_cursor(self) -> None:
        rows = [
            ("2024A", ["HST"], None, ["m"]),
            ("2024B", None, ["SDSS"], None),
        ]
        conn, default_cursor, named_cursor = _mock_conn_with_cursor([], iterable_rows=rows)

        fetched = fetch_cohort_rows(conn, sample_size=None)

        assert len(fetched) == 2
        assert {r["bibcode"] for r in fetched} == {"2024A", "2024B"}
        named_cursor.execute.assert_called_once()
        executed_sql = named_cursor.execute.call_args[0][0]
        assert "LIMIT" not in executed_sql


# ---------------------------------------------------------------------------
# safety_guard
# ---------------------------------------------------------------------------


class TestSafetyGuard:
    def test_blocks_prod_full_scan(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            safety_guard("dbname=scix", sample_size=None, env={})
        assert excinfo.value.code == 2

    def test_sample_size_bypasses_guard_on_prod(self) -> None:
        # Should not raise.
        safety_guard("dbname=scix", sample_size=1000, env={})

    def test_env_override_bypasses_guard_on_prod(self) -> None:
        safety_guard("dbname=scix", sample_size=None, env={"SCIX_ALLOW_FULL_SCAN": "1"})

    def test_non_prod_dsn_full_scan_allowed(self) -> None:
        safety_guard("dbname=scix_test", sample_size=None, env={})

    def test_env_override_must_be_exactly_one(self) -> None:
        with pytest.raises(SystemExit):
            safety_guard("dbname=scix", sample_size=None, env={"SCIX_ALLOW_FULL_SCAN": "yes"})

    def test_empty_dsn_does_not_trigger_guard(self) -> None:
        safety_guard("", sample_size=None, env={})
        safety_guard(None, sample_size=None, env={})


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


class TestWriteReport:
    def _report(
        self,
        *,
        sidecar_path: str | None,
        sample: list[str],
    ) -> GapReport:
        return GapReport(
            total_astronomy_papers=10,
            gap_cohort_count=len(sample) if sidecar_path is None else 5,
            gap_cohort_ratio=0.3,
            gap_cohort_bibcodes_sample=sample,
            gap_cohort_bibcodes_sidecar_path=sidecar_path,
            top_gap_entity_types=[
                GapEntityRow(
                    entity_type="instruments",
                    field="facility",
                    gap_count=3,
                    gap_ratio=0.3,
                )
            ],
            per_field_coverage={
                "facility": FieldCoverage(
                    field="facility", populated=7, total=10, coverage_pct=70.0
                ),
                "data": FieldCoverage(
                    field="data", populated=7, total=10, coverage_pct=70.0
                ),
                "keyword_norm": FieldCoverage(
                    field="keyword_norm", populated=7, total=10, coverage_pct=70.0
                ),
            },
            meta={"sample_size": 10, "dsn_redacted": "x", "generated_at": "z"},
        )

    def test_small_cohort_inlines_all(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        report = self._report(sidecar_path=None, sample=["a", "b", "c"])
        write_report(report, out, gap_bibcodes_full=["a", "b", "c"])

        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["gap_cohort_bibcodes_sample"] == ["a", "b", "c"]
        assert parsed["gap_cohort_bibcodes_sidecar_path"] is None
        # No sidecar written.
        assert not (tmp_path / "cohort.txt").exists()

    def test_large_cohort_writes_sidecar(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        sidecar = tmp_path / "cohort.txt"
        report = GapReport(
            total_astronomy_papers=10,
            gap_cohort_count=5,
            gap_cohort_ratio=0.5,
            gap_cohort_bibcodes_sample=["b0", "b1"],
            gap_cohort_bibcodes_sidecar_path=str(sidecar),
            top_gap_entity_types=[],
            per_field_coverage={},
            meta={},
        )
        write_report(
            report,
            out,
            gap_bibcodes_full=["b0", "b1", "b2", "b3", "b4"],
            cohort_bibcode_cap=2,
        )

        assert out.exists()
        assert sidecar.exists()
        lines = sidecar.read_text().splitlines()
        assert lines == ["b0", "b1", "b2", "b3", "b4"]

    def test_uses_default_cohort_cap_constant(self) -> None:
        # Sanity: ensure the default cap stays a sensible int.
        assert isinstance(DEFAULT_COHORT_BIBCODE_CAP, int)
        assert DEFAULT_COHORT_BIBCODE_CAP > 0


# ---------------------------------------------------------------------------
# psycopg3 placeholder-parser regression
# ---------------------------------------------------------------------------


def _psycopg_placeholder_safe(query: str) -> bool:
    """Return True if no raw ``%`` appears outside an allowed placeholder.

    psycopg3 treats ``%`` as the start of a placeholder when parameters are
    supplied.  Only ``%s``, ``%b``, ``%t``, and the literal-percent escape
    ``%%`` are legal.  Any other ``%X`` triggers ``ProgrammingError``.
    """
    # Replace all legal sequences with a neutral token, then look for stray %.
    scrubbed = re.sub(r"%[sbt%]", "", query)
    return "%" not in scrubbed


class TestPsycopgPlaceholderEscape:
    """Regression guard: bare ``%`` in SQL constant crashes parameterised path."""

    def test_cohort_sql_has_no_bare_percent(self) -> None:
        # The original bug: ILIKE 'astro-ph%' contained a bare % that the
        # psycopg3 placeholder parser interpreted as a malformed placeholder.
        # Escaping to %% is the fix.
        assert _psycopg_placeholder_safe(ASTRONOMY_COHORT_SQL), (
            f"Unescaped %% in ASTRONOMY_COHORT_SQL would crash psycopg3: "
            f"{ASTRONOMY_COHORT_SQL!r}"
        )

    def test_cohort_sql_escapes_astro_ph_wildcard(self) -> None:
        # Belt-and-braces: ensure the LIKE wildcard is explicitly %%.
        assert "astro-ph%%" in ASTRONOMY_COHORT_SQL
        assert "astro-ph%'" not in ASTRONOMY_COHORT_SQL.replace("astro-ph%%", "")

    def test_limit_query_string_is_placeholder_safe(self) -> None:
        # Mirror the exact query construction used in _fetch_rows_with_limit.
        query = (
            "SELECT bibcode, facility, data, keyword_norm\n"
            "FROM papers\n"
            f"WHERE {ASTRONOMY_COHORT_SQL}\n"
            "LIMIT %s"
        )
        assert _psycopg_placeholder_safe(query), (
            f"LIMIT query would crash psycopg3 placeholder parser: {query!r}"
        )

    def test_streaming_query_string_is_placeholder_safe(self) -> None:
        # Mirror the exact query construction used in _fetch_rows_streaming.
        query = (
            "SELECT bibcode, facility, data, keyword_norm\n"
            "FROM papers\n"
            f"WHERE {ASTRONOMY_COHORT_SQL}"
        )
        assert _psycopg_placeholder_safe(query), (
            f"Streaming query would crash psycopg3 placeholder parser: "
            f"{query!r}"
        )


# ---------------------------------------------------------------------------
# Real-database integration test (opt-in via SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealDatabaseSampleSize:
    """Exercise --sample-size against a real psycopg3 cursor.

    Skipped unless ``SCIX_TEST_DSN`` is set so we never touch production.
    This is the regression test for the placeholder-parser crash: mocked
    cursors never call into the real psycopg3 parameter parser, so a unit
    test cannot catch the class of bug.
    """

    def test_sample_size_executes_without_programming_error(self) -> None:
        test_dsn = os.environ.get("SCIX_TEST_DSN")
        if not test_dsn:
            pytest.skip("SCIX_TEST_DSN not set; skipping real-DB regression test")

        import psycopg  # imported lazily so unit-only runs don't need the dep

        with psycopg.connect(test_dsn) as conn:
            # If the % escape is wrong, the next call raises
            # psycopg.ProgrammingError before any rows come back.
            rows = fetch_cohort_rows(conn, sample_size=10)

        assert isinstance(rows, list)
        # scix_test may be empty; the assertion is about no exception.
        for r in rows:
            assert "bibcode" in r
            assert "facility" in r
            assert "data" in r
            assert "keyword_norm" in r
