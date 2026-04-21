"""Unit + integration tests for scripts/populate_papers_fulltext.py.

Integration tests require ``SCIX_TEST_DSN`` and refuse to run against a
production DSN. Unit tests run unconditionally and do not touch the DB.

Every acceptance criterion in the driver-core work unit corresponds to at
least one test below; the ``_criterion_*`` marker-comments map them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import uuid
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load the script as a module so we can unit-test its internals. We register
# the module under both the bare name and ``scripts.populate_papers_fulltext``
# so coverage.py can target either form (the repo's ``scripts/`` directory is
# not a proper Python package — no ``__init__.py`` — so we synthesise one).
_SCRIPT_PATH = REPO_ROOT / "scripts" / "populate_papers_fulltext.py"
if "scripts" not in sys.modules:
    _pkg_spec = importlib.util.spec_from_loader("scripts", loader=None)
    _pkg = importlib.util.module_from_spec(_pkg_spec)
    _pkg.__path__ = [str(REPO_ROOT / "scripts")]  # type: ignore[attr-defined]
    sys.modules["scripts"] = _pkg
_spec = importlib.util.spec_from_file_location("scripts.populate_papers_fulltext", _SCRIPT_PATH)
driver = importlib.util.module_from_spec(_spec)
sys.modules["scripts.populate_papers_fulltext"] = driver
sys.modules["populate_papers_fulltext"] = driver
_spec.loader.exec_module(driver)  # type: ignore[union-attr]

from scix.db import is_production_dsn  # noqa: E402
from scix.sources import ads_body_parser  # noqa: E402
from scix.sources.ar5iv import Section  # noqa: E402

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

_integration_skip_reason: str | None = None
if TEST_DSN is None:
    _integration_skip_reason = "SCIX_TEST_DSN not set"
elif is_production_dsn(TEST_DSN):
    _integration_skip_reason = "SCIX_TEST_DSN points at production"

integration = pytest.mark.skipif(
    _integration_skip_reason is not None,
    reason=_integration_skip_reason or "no DSN",
)


# ---------------------------------------------------------------------------
# File-shape checks (acceptance criterion 1)
# ---------------------------------------------------------------------------


def test_script_file_exists_and_executable():
    """_criterion_1: script exists + executable + has __main__ entry."""
    assert _SCRIPT_PATH.exists(), "driver script must exist"
    mode = _SCRIPT_PATH.stat().st_mode
    # owner execute bit set
    assert mode & 0o100, f"driver script must be executable (mode={oct(mode)})"
    text = _SCRIPT_PATH.read_text()
    assert 'if __name__ == "__main__"' in text


# ---------------------------------------------------------------------------
# CLI surface (acceptance criterion 2)
# ---------------------------------------------------------------------------


def test_cli_help_lists_all_required_flags(capsys):
    """_criterion_2: --help lists all required flags."""
    with pytest.raises(SystemExit) as exc:
        driver._build_arg_parser().parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    help_text = captured.out + captured.err
    for flag in ("--dsn", "--allow-prod", "--resume-from", "--chunk-size", "--limit"):
        assert flag in help_text, f"help text missing {flag}: {help_text!r}"


def test_cli_help_exits_zero_via_main():
    """_criterion_2: `python script --help` exits 0."""
    with pytest.raises(SystemExit) as exc:
        driver.main(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Production guard (acceptance criterion 3)
# ---------------------------------------------------------------------------


def test_prod_guard_rejects_prod_dsn_without_allow_prod(monkeypatch):
    """_criterion_3: prod DSN without --allow-prod → SystemExit non-zero."""
    monkeypatch.setattr(driver, "is_production_dsn", lambda dsn: True)
    with pytest.raises(SystemExit) as exc:
        driver.resolve_prod_guard(dsn="dbname=scix", allow_prod=False, env={})
    assert exc.value.code != 0


def test_prod_guard_requires_systemd_scope_with_allow_prod(monkeypatch):
    """_criterion_3: --allow-prod without SYSTEMD_SCOPE → SystemExit non-zero."""
    monkeypatch.setattr(driver, "is_production_dsn", lambda dsn: True)
    with pytest.raises(SystemExit) as exc:
        driver.resolve_prod_guard(dsn="dbname=scix", allow_prod=True, env={})  # no SYSTEMD_SCOPE
    assert exc.value.code != 0


def test_prod_guard_allows_prod_with_scope(monkeypatch):
    """_criterion_3 (positive): --allow-prod + SYSTEMD_SCOPE passes."""
    monkeypatch.setattr(driver, "is_production_dsn", lambda dsn: True)
    # No exception.
    driver.resolve_prod_guard(
        dsn="dbname=scix",
        allow_prod=True,
        env={"SYSTEMD_SCOPE": "scix-batch.scope"},
    )


def test_prod_guard_allows_test_dsn(monkeypatch):
    """_criterion_3 (negative): non-prod DSN passes without --allow-prod."""
    monkeypatch.setattr(driver, "is_production_dsn", lambda dsn: False)
    driver.resolve_prod_guard(dsn="dbname=scix_test", allow_prod=False, env={})


def test_main_exits_nonzero_on_prod_without_allow_prod(monkeypatch):
    """_criterion_3: main() returns non-zero when prod guard fires."""
    monkeypatch.setattr(driver, "is_production_dsn", lambda dsn: True)
    rc = driver.main(["--dsn", "dbname=scix"])
    assert rc != 0


# ---------------------------------------------------------------------------
# Routing + dispatch (acceptance criterion 4)
# ---------------------------------------------------------------------------


def _row(
    bibcode: str = "2024ApJ...900A...1X",
    *,
    body: str = "",
    bibstem: list[str] | None = None,
    doi: list[str] | None = None,
    doctype: str | None = None,
    openalex_has_pdf_url: bool = False,
) -> dict:
    return {
        "bibcode": bibcode,
        "body": body,
        "bibstem": bibstem if bibstem is not None else ["ApJ"],
        "doi": doi if doi is not None else [],
        "doctype": doctype,
        "openalex_has_pdf_url": openalex_has_pdf_url,
    }


def test_build_route_input_flattens_arrays():
    """_criterion_4 (shape): doi/bibstem arrays flattened to first element."""
    inp = driver.build_route_input(
        _row(doi=["10.1000/foo", "10.1000/bar"], bibstem=["MNRAS"], body="x"),
        has_fulltext_row=False,
        sibling_row_source=None,
    )
    assert inp.doi == "10.1000/foo"
    assert inp.has_ads_body is True


def test_build_route_input_empty_arrays_none():
    inp = driver.build_route_input(
        _row(doi=[], bibstem=[], body=""),
        has_fulltext_row=False,
        sibling_row_source=None,
    )
    assert inp.doi is None
    assert inp.has_ads_body is False


class _FakeConn:
    """No-op connection stand-in for _dispatch_one unit tests."""

    def cursor(self, *_, **__):
        return _FakeCursor()


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *_, **__):
        return None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    @property
    def description(self):
        return []

    def __iter__(self):
        return iter([])


class _RecordingConn:
    """Captures execute() calls for failure-record assertions."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.committed: int = 0

    def cursor(self, *_, **__):
        outer = self

        class _C:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def execute(self_inner, sql, params=()):
                outer.calls.append((sql, tuple(params)))

        return _C()

    def commit(self):
        self.committed += 1


def test_dispatch_tier1_ads_body_builds_parsed_row():
    """_criterion_4: tier1_ads_body branch exercised."""
    row = _row(body="1. Intro\nhello world\n", bibstem=["ApJ"])
    stats = driver.DriverStats()

    def fake_parse(body, bibstem):
        return [Section(heading="Intro", level=1, text="hello world", offset=0)], {}

    parsed = driver._dispatch_one(
        _FakeConn(),
        row,
        parse_fn=fake_parse,
        sibling_fetch=lambda conn, bib: None,
        stats=stats,
    )
    assert parsed is not None
    assert parsed.source == "ads_body"
    assert parsed.parser_version == ads_body_parser.PARSER_VERSION
    assert "Intro" in parsed.sections_json
    assert stats.tier_counts.get("tier1_ads_body") == 1


def test_dispatch_serve_sibling_clones_row():
    """_criterion_4: serve_sibling branch exercised."""
    row = _row(body="", bibstem=[], doctype="eprint")
    stats = driver.DriverStats()
    sibling = {
        "bibcode": "2024arXiv999Y",
        "source": "ar5iv",
        "sections": [{"heading": "X", "level": 1, "text": "y", "offset": 0}],
        "inline_cites": [],
        "figures": [],
        "tables": [],
        "equations": [],
        "parser_version": "ar5iv_html@v1",
    }
    parsed = driver._dispatch_one(
        _FakeConn(),
        row,
        parse_fn=lambda b, bs: ([], {}),
        sibling_fetch=lambda conn, bib: sibling,
        stats=stats,
    )
    assert parsed is not None
    assert parsed.source == "ar5iv"
    assert "X" in parsed.sections_json
    assert stats.tier_counts.get("serve_sibling") == 1
    assert stats.served_sibling == 1


def test_dispatch_tier3_records_failure():
    """_criterion_4: tier3_docling branch records failure."""
    row = _row(
        body="",
        doctype="article",
        doi=["10.1000/x"],
        openalex_has_pdf_url=True,
    )
    stats = driver.DriverStats()
    rec_conn = _RecordingConn()
    parsed = driver._dispatch_one(
        rec_conn,
        row,
        parse_fn=lambda b, bs: ([], {}),
        sibling_fetch=lambda conn, bib: None,
        stats=stats,
    )
    assert parsed is None
    assert stats.tier3_skipped == 1
    assert stats.tier_counts.get("tier3_docling") == 1
    # failure row upsert happened
    assert any("papers_fulltext_failures" in sql for sql, _ in rec_conn.calls)
    assert any("tier3_not_yet_wired" in str(params) for _, params in rec_conn.calls)


def test_dispatch_abstract_only_records_failure():
    """_criterion_4: abstract_only branch records failure (no row written)."""
    row = _row(body="", doctype=None, doi=[], openalex_has_pdf_url=False)
    stats = driver.DriverStats()
    rec_conn = _RecordingConn()
    parsed = driver._dispatch_one(
        rec_conn,
        row,
        parse_fn=lambda b, bs: ([], {}),
        sibling_fetch=lambda conn, bib: None,
        stats=stats,
    )
    assert parsed is None
    assert stats.abstract_only == 1
    assert stats.tier_counts.get("abstract_only") == 1
    assert any("abstract_only" in str(params) for _, params in rec_conn.calls)


def test_dispatch_tier1_parse_error_records_failure():
    """A parser exception is caught and recorded as a failure row."""
    row = _row(body="body", bibstem=["ApJ"])
    stats = driver.DriverStats()
    rec_conn = _RecordingConn()

    def raising_parse(body, bibstem):
        raise ValueError("boom")

    parsed = driver._dispatch_one(
        rec_conn,
        row,
        parse_fn=raising_parse,
        sibling_fetch=lambda conn, bib: None,
        stats=stats,
    )
    assert parsed is None
    assert stats.failures == 1
    assert any("tier1_parse_error" in str(params) for _, params in rec_conn.calls)


def test_dispatch_serve_existing_skips(monkeypatch):
    """_criterion_4: serve_existing branch exercised.

    Defense-in-depth: the outer SQL NOT-EXISTS join normally filters rows
    that already have a papers_fulltext entry, so the ``serve_existing``
    tier is unreachable in practice. We still exercise the internal
    dispatch branch directly by monkeypatching ``route_fulltext_request``
    to force the decision — asserts no row is produced and the
    ``skipped_existing`` counter is bumped.
    """
    row = _row(body="", bibstem=["ApJ"])
    stats = driver.DriverStats()

    forced = driver.RouteDecision(
        tier="serve_existing",
        reason="forced_for_test",
        source_hint=None,
    )
    monkeypatch.setattr(driver, "route_fulltext_request", lambda _inp: forced)

    parsed = driver._dispatch_one(
        _FakeConn(),
        row,
        parse_fn=lambda b, bs: ([], {}),
        sibling_fetch=lambda conn, bib: None,
        stats=stats,
    )
    assert parsed is None
    assert stats.skipped_existing == 1
    assert stats.tier_counts.get("serve_existing") == 1
    # No failure recorded for this branch.
    assert stats.failures == 0


# ---------------------------------------------------------------------------
# Pure helpers (criterion 6 shape)
# ---------------------------------------------------------------------------


def test_compute_retry_after_backoff_ladder():
    """_criterion_6: 24h / 3d / 7d / 30d ladder."""
    now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    assert driver.compute_retry_after(1, now) - now == timedelta(hours=24)
    assert driver.compute_retry_after(2, now) - now == timedelta(days=3)
    assert driver.compute_retry_after(3, now) - now == timedelta(days=7)
    assert driver.compute_retry_after(4, now) - now == timedelta(days=30)
    assert driver.compute_retry_after(99, now) - now == timedelta(days=30)


# ---------------------------------------------------------------------------
# Integration tests (criteria 5, 6, 7, 8, 9)
# ---------------------------------------------------------------------------

# Each integration run uses a uuid-tagged prefix so rows never collide across
# runs and cleanup is scoped to the current test.
_TAG = f"DRVT-{uuid.uuid4().hex[:8]}"


def _mk_bibcode(i: int) -> str:
    # Ensure ASCII ordering matches creation order (zero-padded).
    return f"{_TAG}-{i:06d}"


@pytest.fixture
def integ_conn():
    """Yield a connection for integration fixtures; clean up on teardown."""
    conn = psycopg.connect(TEST_DSN)  # type: ignore[arg-type]
    conn.autocommit = True
    yield conn
    # Cleanup any rows this test left behind (best-effort).
    with conn.cursor() as cur:
        cur.execute("DELETE FROM papers_fulltext WHERE bibcode LIKE %s", (f"{_TAG}%",))
        cur.execute(
            "DELETE FROM papers_fulltext_failures WHERE bibcode LIKE %s",
            (f"{_TAG}%",),
        )
        cur.execute(
            "DELETE FROM papers_external_ids WHERE bibcode LIKE %s",
            (f"{_TAG}%",),
        )
        cur.execute("DELETE FROM papers WHERE bibcode LIKE %s", (f"{_TAG}%",))
    conn.close()


def _seed_papers(
    conn: psycopg.Connection,
    n: int,
    *,
    body: str = "1. Intro\nHello world here.\n2. Methods\nMore text.\n",
    bibstem: str = "ApJ",
    doctype: str = "article",
) -> list[str]:
    """Insert ``n`` papers with bodies; return the bibcodes."""
    bibcodes = [_mk_bibcode(i) for i in range(n)]
    with conn.cursor() as cur:
        for bib in bibcodes:
            cur.execute(
                """
                INSERT INTO papers (bibcode, title, doctype, bibstem, body)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (bibcode) DO NOTHING
                """,
                (bib, f"test {bib}", doctype, [bibstem], body),
            )
    return bibcodes


@integration
def test_driver_writes_bulk_rows_via_copy(integ_conn):
    """_criterion_5: ≥1000 seeded bibcodes → ≥1000 rows in papers_fulltext.

    We seed exactly 1000 so the batch boundary is actually crossed (not just
    "fewer than chunk_size").
    """
    bibcodes = _seed_papers(integ_conn, 1000)
    config = driver.DriverConfig(
        dsn=TEST_DSN,  # type: ignore[arg-type]
        chunk_size=driver.DEFAULT_CHUNK_SIZE,
        resume_from=None,
        limit=len(bibcodes),
        allow_prod=False,
    )
    stats = driver.run(config)
    assert stats.wrote >= 1000
    with integ_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM papers_fulltext WHERE bibcode LIKE %s",
            (f"{_TAG}%",),
        )
        assert cur.fetchone()[0] >= 1000


@integration
def test_driver_records_failure_with_correct_retry_after(integ_conn):
    """_criterion_6: forced parser error writes a failure row with ~24h retry."""
    _seed_papers(integ_conn, 3)
    config = driver.DriverConfig(
        dsn=TEST_DSN,  # type: ignore[arg-type]
        chunk_size=10,
        resume_from=None,
        limit=3,
        allow_prod=False,
    )

    def raising_parse(body, bibstem):
        raise RuntimeError("induced failure")

    stats = driver.run(config, parse_fn=raising_parse)
    assert stats.failures >= 3
    with integ_conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, parser_version, attempts, retry_after
              FROM papers_fulltext_failures
             WHERE bibcode LIKE %s
            """,
            (f"{_TAG}%",),
        )
        rows = cur.fetchall()
    assert len(rows) >= 3
    now = datetime.now(timezone.utc)
    for bib, pver, attempts, retry_after in rows:
        assert pver == ads_body_parser.PARSER_VERSION
        assert attempts == 1
        # Expect retry_after ~24h out; allow a generous window for clock skew.
        delta = retry_after - now
        assert (
            timedelta(hours=23) <= delta <= timedelta(hours=25)
        ), f"expected ~24h backoff for first attempt, got {delta} for {bib}"


@integration
def test_driver_resumable_no_duplicate_work(integ_conn):
    """_criterion_7: two consecutive runs → second run writes 0 new rows."""
    _seed_papers(integ_conn, 5)
    config = driver.DriverConfig(
        dsn=TEST_DSN,  # type: ignore[arg-type]
        chunk_size=10,
        resume_from=None,
        limit=5,
        allow_prod=False,
    )
    first = driver.run(config)
    assert first.wrote == 5
    second = driver.run(config)
    assert second.wrote == 0
    # And --resume-from a later bibcode trims further (no bibcodes after it).
    later_cfg = driver.DriverConfig(
        dsn=TEST_DSN,  # type: ignore[arg-type]
        chunk_size=10,
        resume_from=_mk_bibcode(999999),
        limit=5,
        allow_prod=False,
    )
    third = driver.run(later_cfg)
    assert third.seen == 0


@integration
def test_driver_idempotent_does_not_overwrite(integ_conn):
    """_criterion_8: existing row is not overwritten on re-run."""
    bibcodes = _seed_papers(integ_conn, 1)
    bib = bibcodes[0]
    # Pre-seed a papers_fulltext row with a distinctive source.
    with integ_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO papers_fulltext
                (bibcode, source, sections, inline_cites, figures, tables,
                 equations, parser_version)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s)
            """,
            (
                bib,
                "preexisting_marker",
                json.dumps([{"heading": "KEEP_ME", "level": 1, "text": "x", "offset": 0}]),
                "[]",
                "[]",
                "[]",
                "[]",
                "marker@v0",
            ),
        )
    config = driver.DriverConfig(
        dsn=TEST_DSN,  # type: ignore[arg-type]
        chunk_size=10,
        resume_from=None,
        limit=1,
        allow_prod=False,
    )
    stats = driver.run(config)
    assert stats.wrote == 0
    with integ_conn.cursor() as cur:
        cur.execute(
            "SELECT source, parser_version FROM papers_fulltext WHERE bibcode = %s",
            (bib,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "preexisting_marker"
    assert row[1] == "marker@v0"


@integration
def test_driver_pins_parser_version(integ_conn):
    """_criterion_9: tier1 rows record ads_body_parser.PARSER_VERSION."""
    bibcodes = _seed_papers(integ_conn, 5)
    config = driver.DriverConfig(
        dsn=TEST_DSN,  # type: ignore[arg-type]
        chunk_size=10,
        resume_from=None,
        limit=5,
        allow_prod=False,
    )
    stats = driver.run(config)
    assert stats.wrote == 5
    with integ_conn.cursor() as cur:
        cur.execute(
            "SELECT parser_version FROM papers_fulltext WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        versions = {r[0] for r in cur.fetchall()}
    assert versions == {ads_body_parser.PARSER_VERSION}
