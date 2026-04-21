"""Tests for ``scripts/report_fulltext_coverage.py``.

Unit tests run unconditionally and exercise formatting, bucketing, tier
derivation, read-only enforcement, and the write-safety guard via fake
psycopg connections.

Integration tests require ``SCIX_TEST_DSN`` pointing at a non-production
database (by convention ``dbname=scix_test``). They seed deterministic
fixture rows into ``papers``, ``papers_fulltext`` and
``papers_fulltext_failures``, run the script, and assert on the parsed
output. Seeded rows are removed on teardown regardless of test outcome.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "report_fulltext_coverage.py"

_SRC_DIR = REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import get_test_dsn  # noqa: E402

# ---------------------------------------------------------------------------
# Dynamic module import — scripts/ is not a package.
# ---------------------------------------------------------------------------


def _load_script_module():
    mod_name = "report_fulltext_coverage_script"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


script = _load_script_module()


# ---------------------------------------------------------------------------
# Fake connection — records queries; returns canned rows by SQL shape.
# ---------------------------------------------------------------------------


@dataclass
class FakeSchema:
    """Inputs + canned outputs for FakeConnection queries."""

    readonly: str = "on"  # value returned by SHOW transaction_read_only
    by_source: list[tuple[str, int]] = field(default_factory=list)
    histogram: list[tuple[int, int]] = field(default_factory=list)  # (ord, count)
    section_empty: int = 0
    latex_median: float | None = None
    fulltext_rows: int = 0
    failure_rows: int = 0


class _FakeCursor:
    def __init__(self, schema: FakeSchema, sql_log: list[str]):
        self._schema = schema
        self._log = sql_log
        self._result: list[tuple[Any, ...]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query: Any, params: Sequence[Any] | None = None) -> None:
        text = query.as_string(None) if hasattr(query, "as_string") else str(query)
        self._log.append(text)
        norm = " ".join(text.split()).lower()

        if "show transaction_read_only" in norm:
            self._result = [(self._schema.readonly,)]
            return
        if norm.startswith("set session characteristics"):
            self._result = []
            return
        if "group by source" in norm:
            self._result = [(s, n) for (s, n) in self._schema.by_source]
            return
        if "jsonb_array_length(sections) = 0" in norm and "select count(*)" in norm:
            self._result = [(self._schema.section_empty,)]
            return
        if "percentile_cont" in norm:
            self._result = [(self._schema.latex_median,)]
            return
        if "group by ord" in norm:
            self._result = [(int(ord_), int(count)) for (ord_, count) in self._schema.histogram]
            return
        if "papers_fulltext_failures" in norm and "count(*)" in norm:
            self._result = [(self._schema.failure_rows,)]
            return
        if "papers_fulltext" in norm and "count(*)" in norm:
            self._result = [(self._schema.fulltext_rows,)]
            return
        raise AssertionError(f"Unrecognised query in fake cursor: {text!r}")

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._result[0] if self._result else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._result)


class FakeConnection:
    def __init__(self, schema: FakeSchema):
        self._schema = schema
        self.autocommit = False
        self.sql_log: list[str] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._schema, self.sql_log)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Unit tests (no DB)
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_lists_dsn_and_format():
    """Criterion 2: --help exits 0 listing both --dsn and --format."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dsn" in result.stdout
    assert "--format" in result.stdout
    assert "text" in result.stdout and "json" in result.stdout


def test_tier_derivation_maps_known_and_unknown_sources():
    assert script.tier_for_source("ar5iv") == "latex"
    assert script.tier_for_source("arxiv_local") == "latex"
    assert script.tier_for_source("s2orc") == "xml"
    assert script.tier_for_source("docling") == "xml"
    assert script.tier_for_source("ads_body") == "text"
    assert script.tier_for_source("abstract") == "abstract"
    assert script.tier_for_source("brand_new_source") == "other"


def test_bucket_labels_cover_expected_ranges():
    labels = dict(script.BUCKET_LABELS)
    assert labels[0] == "0"
    assert labels[1] == "1"
    assert labels[2] == "2-4"
    assert labels[3] == "5-9"
    assert labels[4] == "10-19"
    assert labels[5] == "20+"


def test_failure_rate_pct_zero_denominator():
    assert script.failure_rate_pct(0, 0) == 0.0


def test_failure_rate_pct_computed():
    # 1 failure / (3 fulltext + 1 failure) = 25%
    assert script.failure_rate_pct(3, 1) == pytest.approx(25.0)


def test_assert_readonly_passes_when_on():
    schema = FakeSchema(readonly="on")
    conn = FakeConnection(schema)
    script.assert_readonly(conn)  # should not raise


def test_assert_readonly_raises_when_off():
    schema = FakeSchema(readonly="off")
    conn = FakeConnection(schema)
    with pytest.raises(RuntimeError, match="transaction_read_only"):
        script.assert_readonly(conn)


def test_collect_coverage_emits_no_writes(monkeypatch):
    """Criterion 8: the script does no writes.

    Run collect_coverage against a fake conn and assert that every SQL
    statement emitted is a SELECT / SET (read-only). No DDL or DML.
    """
    schema = FakeSchema(
        readonly="on",
        by_source=[("ar5iv", 4), ("ads_body", 1)],
        histogram=[(1, 3), (2, 2)],
        section_empty=0,
        latex_median=2.0,
        fulltext_rows=5,
        failure_rows=1,
    )
    conn = FakeConnection(schema)
    script.collect_coverage(conn, dsn_redacted="x")

    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "TRUNCATE", "ALTER")
    for stmt in conn.sql_log:
        upper = stmt.upper()
        for bad in forbidden:
            assert bad not in upper, f"unexpected {bad} in: {stmt!r}"


def test_collect_coverage_report_shape():
    schema = FakeSchema(
        readonly="on",
        by_source=[("ar5iv", 4), ("ads_body", 2), ("abstract", 1)],
        histogram=[(0, 1), (2, 3), (3, 3)],
        section_empty=1,
        latex_median=3.0,
        fulltext_rows=7,
        failure_rows=3,
    )
    conn = FakeConnection(schema)
    report = script.collect_coverage(conn, dsn_redacted="dbname=fake")

    assert report.total_rows == 7
    assert report.by_source == {"ar5iv": 4, "ads_body": 2, "abstract": 1}
    # by_tier: ar5iv -> latex, ads_body -> text, abstract -> abstract
    assert report.by_tier == {"latex": 4, "text": 2, "abstract": 1}
    # histogram is ordered and fills missing buckets with 0. Buckets
    # 0, 2 (2-4), 3 (5-9) are populated; 1, 4 (10-19), 5 (20+) should be 0.
    assert report.histogram == [
        ("0", 1),
        ("1", 0),
        ("2-4", 3),
        ("5-9", 3),
        ("10-19", 0),
        ("20+", 0),
    ]
    assert report.section_empty == 1
    assert report.latex_median_sections == 3.0
    # 3 / (7 + 3) = 30%
    assert report.failure_rate_pct == pytest.approx(30.0)


def test_format_text_contains_source_rows_lines():
    schema = FakeSchema(
        readonly="on",
        by_source=[("ar5iv", 4), ("ads_body", 2)],
        histogram=[(1, 2), (2, 4)],
        section_empty=0,
        latex_median=1.0,
        fulltext_rows=6,
        failure_rows=0,
    )
    conn = FakeConnection(schema)
    report = script.collect_coverage(conn, dsn_redacted="dbname=fake")
    text = script.format_text(report)
    # Criterion 3: per-source row-count table in text mode.
    assert "source=ar5iv rows=4" in text
    assert "source=ads_body rows=2" in text
    assert "histogram" in text
    assert "section_empty: 0" in text
    assert "failure_rate_pct" in text


def test_format_json_structure():
    schema = FakeSchema(
        readonly="on",
        by_source=[("ar5iv", 4)],
        histogram=[(1, 2)],
        section_empty=0,
        latex_median=1.0,
        fulltext_rows=4,
        failure_rows=0,
    )
    conn = FakeConnection(schema)
    report = script.collect_coverage(conn, dsn_redacted="dbname=fake")
    payload = json.loads(script.format_json(report))
    # Criterion 3: structured JSON keyed by source.
    assert payload["by_source"] == {"ar5iv": 4}
    assert isinstance(payload["histogram"], list)
    assert payload["section_empty"] == 0
    assert payload["failure_rate_pct"] == 0.0
    assert "latex_median_sections" in payload


def test_latex_median_is_none_when_no_rows():
    schema = FakeSchema(
        readonly="on",
        by_source=[],
        histogram=[],
        section_empty=0,
        latex_median=None,
        fulltext_rows=0,
        failure_rows=0,
    )
    conn = FakeConnection(schema)
    report = script.collect_coverage(conn, dsn_redacted="x")
    assert report.latex_median_sections is None
    assert report.failure_rate_pct == 0.0  # divide-by-zero guard
    text = script.format_text(report)
    assert "n/a" in text


def test_main_with_fake_connection(monkeypatch, capsys):
    """End-to-end main() against a fake psycopg.connect."""
    schema = FakeSchema(
        readonly="on",
        by_source=[("ar5iv", 2)],
        histogram=[(2, 2)],
        section_empty=0,
        latex_median=3.0,
        fulltext_rows=2,
        failure_rows=1,
    )

    class _ConnCtx:
        def __enter__(self_inner):
            return FakeConnection(schema)

        def __exit__(self_inner, *exc):
            return False

    monkeypatch.setattr(script.psycopg, "connect", lambda dsn: _ConnCtx())
    rc = script.main(["--dsn", "dbname=fake", "--format", "json", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["by_source"] == {"ar5iv": 2}
    assert payload["failure_rate_pct"] == pytest.approx(100.0 / 3.0)


# ---------------------------------------------------------------------------
# Integration tests — require SCIX_TEST_DSN.
# ---------------------------------------------------------------------------


TEST_DSN = get_test_dsn()
TEST_BIBCODE_PREFIX = "FTCOVTEST."


pytestmark_integration = pytest.mark.skipif(
    TEST_DSN is None,
    reason=(
        "SCIX_TEST_DSN is not set or points at production — "
        "report_fulltext_coverage integration tests require a dedicated test DB"
    ),
)


def _bib(i: int) -> str:
    return f"{TEST_BIBCODE_PREFIX}{i:04d}"


def _jsonb_sections(n: int) -> str:
    """Return a JSON-encoded list of N minimal section objects."""
    return json.dumps([{"heading": f"s{k}", "level": 1, "text": "", "offset": 0} for k in range(n)])


# Rows to seed into papers_fulltext. Each tuple is (index, source, n_sections).
#
# Layout exercises every histogram bucket + provides a known LaTeX median:
#
#   0 sections:       1 row   (ar5iv)                          bucket "0"
#   1 section :       2 rows  (arxiv_local x 1, s2orc x 1)     bucket "1"
#   2-4 :             2 rows  (ar5iv=3, ads_body=2)            bucket "2-4"
#   5-9 :             2 rows  (arxiv_local=7, abstract=5)      bucket "5-9"
#   10-19 :           1 row   (ar5iv=12)                       bucket "10-19"
#   20+   :           1 row   (docling=25)                     bucket "20+"
#
# LaTeX-derived rows (ar5iv, arxiv_local) section counts: [0, 1, 3, 7, 12]
#   → median = 3
#
# By source totals:
#   ar5iv=3, arxiv_local=2, s2orc=1, ads_body=1, abstract=1, docling=1  → 9 total
#
_FIXTURE_FULLTEXT: tuple[tuple[int, str, int], ...] = (
    (1, "ar5iv", 0),
    (2, "arxiv_local", 1),
    (3, "s2orc", 1),
    (4, "ar5iv", 3),
    (5, "ads_body", 2),
    (6, "arxiv_local", 7),
    (7, "abstract", 5),
    (8, "ar5iv", 12),
    (9, "docling", 25),
)

# Three failure rows → failure_rate = 3 / (9 + 3) = 25.0
_FIXTURE_FAILURES: tuple[int, ...] = (101, 102, 103)


def _seed(dsn: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO papers (bibcode, title) VALUES (%s, %s) "
                "ON CONFLICT (bibcode) DO NOTHING",
                [
                    (_bib(i), f"fixture {_bib(i)}")
                    for i in [row[0] for row in _FIXTURE_FULLTEXT] + list(_FIXTURE_FAILURES)
                ],
            )
            cur.executemany(
                """
                INSERT INTO papers_fulltext (
                    bibcode, source, sections, inline_cites,
                    figures, tables, equations, parser_version
                )
                VALUES (%s, %s, %s::jsonb, '[]'::jsonb,
                        '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, 'test/v1')
                ON CONFLICT (bibcode) DO UPDATE SET
                    source = EXCLUDED.source,
                    sections = EXCLUDED.sections,
                    parser_version = EXCLUDED.parser_version
                """,
                [(_bib(i), source, _jsonb_sections(n)) for (i, source, n) in _FIXTURE_FULLTEXT],
            )
            cur.executemany(
                """
                INSERT INTO papers_fulltext_failures (
                    bibcode, parser_version, failure_reason, attempts,
                    retry_after
                )
                VALUES (%s, 'test/v1', 'seeded', 1, now() + interval '1 day')
                ON CONFLICT (bibcode) DO NOTHING
                """,
                [(_bib(i),) for i in _FIXTURE_FAILURES],
            )
        conn.commit()


def _cleanup(dsn: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM papers_fulltext WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers_fulltext_failures WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )


@pytest.fixture
def integration_dsn():
    if TEST_DSN is None:
        pytest.skip("SCIX_TEST_DSN not set")
    _cleanup(TEST_DSN)
    _seed(TEST_DSN)
    try:
        yield TEST_DSN
    finally:
        _cleanup(TEST_DSN)


def _run_script(dsn: str, fmt: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dsn", dsn, "--format", fmt, "--quiet"],
        capture_output=True,
        text=True,
        check=False,
    )


@pytestmark_integration
def test_integration_by_source_text_and_json(integration_dsn: str):
    """Criterion 3: per-source text table + structured JSON keyed by source."""
    text_result = _run_script(integration_dsn, "text")
    assert text_result.returncode == 0, text_result.stderr
    # Our seeded rows appear under their source names.
    assert "source=ar5iv rows=3" in text_result.stdout
    assert "source=arxiv_local rows=2" in text_result.stdout
    assert "source=s2orc rows=1" in text_result.stdout
    assert "source=ads_body rows=1" in text_result.stdout
    assert "source=abstract rows=1" in text_result.stdout
    assert "source=docling rows=1" in text_result.stdout

    json_result = _run_script(integration_dsn, "json")
    assert json_result.returncode == 0, json_result.stderr
    payload = json.loads(json_result.stdout)
    for source in ("ar5iv", "arxiv_local", "s2orc", "ads_body", "abstract", "docling"):
        assert payload["by_source"].get(source, 0) >= 1


@pytestmark_integration
def test_integration_histogram_buckets(integration_dsn: str):
    """Criterion 4: histogram bucket counts match seeded distribution."""
    result = _run_script(integration_dsn, "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    hist = {label: count for (label, count) in payload["histogram"]}
    # Seeded distribution (accounting only for fixture rows — there may be
    # existing unrelated rows in scix_test, so use >= where other rows could
    # land. Bucket "20+" covers our docling=25 row uniquely for this fixture.
    assert hist["0"] >= 1  # one seeded 0-section row
    assert hist["1"] >= 2  # two seeded 1-section rows
    assert hist["2-4"] >= 2
    assert hist["5-9"] >= 2
    assert hist["10-19"] >= 1
    assert hist["20+"] >= 1


@pytestmark_integration
def test_integration_section_empty_count(integration_dsn: str):
    """Criterion 5: section_empty counts zero-sections rows."""
    result = _run_script(integration_dsn, "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["section_empty"] >= 1  # our seeded ar5iv 0-section row


@pytestmark_integration
def test_integration_latex_median(integration_dsn: str):
    """Criterion 6: median sections-per-paper for LaTeX-derived sources.

    This test cleans out ALL non-fixture LaTeX-derived rows for the duration
    of the assertion so the median exactly reflects our fixture.
    """
    import psycopg

    # Move existing non-fixture LaTeX rows aside to a temp table so the
    # median reflects only our fixture. We restore them on teardown via the
    # fixture.
    with psycopg.connect(integration_dsn) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE _pft_backup AS "
                "SELECT * FROM papers_fulltext "
                "WHERE source IN ('ar5iv','arxiv_local') "
                "AND bibcode NOT LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers_fulltext "
                "WHERE source IN ('ar5iv','arxiv_local') "
                "AND bibcode NOT LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            conn.commit()
            try:
                result = _run_script(integration_dsn, "json")
                assert result.returncode == 0, result.stderr
                payload = json.loads(result.stdout)
                # LaTeX-derived section counts in fixture: [0, 1, 3, 7, 12]
                # PERCENTILE_CONT(0.5) over 5 values yields 3.0.
                assert payload["latex_median_sections"] == pytest.approx(3.0)
            finally:
                cur.execute("INSERT INTO papers_fulltext SELECT * FROM _pft_backup")
                cur.execute("DROP TABLE _pft_backup")
                conn.commit()


@pytestmark_integration
def test_integration_failure_rate(integration_dsn: str):
    """Criterion 7: failure_rate_pct = failures / (fulltext + failures).

    Seeded: 9 fulltext fixture rows + 3 failure fixture rows. Because the
    test DB may contain other rows from earlier runs or other tests, we
    filter to our fixture by temporarily removing non-fixture rows from
    both tables during the assertion.
    """
    import psycopg

    with psycopg.connect(integration_dsn) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE _pft_backup2 AS "
                "SELECT * FROM papers_fulltext WHERE bibcode NOT LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "CREATE TEMP TABLE _pftf_backup2 AS "
                "SELECT * FROM papers_fulltext_failures "
                "WHERE bibcode NOT LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers_fulltext WHERE bibcode NOT LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers_fulltext_failures WHERE bibcode NOT LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            conn.commit()
            try:
                result = _run_script(integration_dsn, "json")
                assert result.returncode == 0, result.stderr
                payload = json.loads(result.stdout)
                # 3 / (9 + 3) = 25.0
                assert payload["failure_rate_pct"] == pytest.approx(25.0)
                assert payload["fulltext_rows"] == 9
                assert payload["failure_rows"] == 3
            finally:
                cur.execute("INSERT INTO papers_fulltext SELECT * FROM _pft_backup2")
                cur.execute("INSERT INTO papers_fulltext_failures " "SELECT * FROM _pftf_backup2")
                cur.execute("DROP TABLE _pft_backup2")
                cur.execute("DROP TABLE _pftf_backup2")
                conn.commit()
