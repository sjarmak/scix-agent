"""Tests for scripts/verify_fulltext_populated.py.

Unit tests drive the verification logic against a fake psycopg connection,
exercising the PASS/FAIL/exit-code paths without hitting a live DB. An
integration test (marked ``integration``) seeds rows into ``SCIX_TEST_DSN``
and runs the real script as a subprocess, skipping if ``SCIX_TEST_DSN`` is
unset or points at production.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_fulltext_populated.py"

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import is_production_dsn  # noqa: E402

# ---------------------------------------------------------------------------
# Module loader — the script lives in ``scripts/`` which isn't a package.
# ---------------------------------------------------------------------------


def _load_module():
    name = "verify_fulltext_populated"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


verify_module = _load_module()


# ---------------------------------------------------------------------------
# Fake connection/cursor — just enough to satisfy the single query shape.
# ---------------------------------------------------------------------------


@dataclass
class FakeSchema:
    """Maps tier source -> row count returned for that source."""

    counts: dict[str, int] = field(default_factory=dict)


class _FakeCursor:
    def __init__(self, schema: FakeSchema):
        self._schema = schema
        self._result: list[tuple[Any, ...]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query: Any, params: Sequence[Any] | None = None) -> None:
        text = str(query)
        text_norm = " ".join(text.split()).lower()
        params = tuple(params or ())

        if "from papers_fulltext where source = %s" in text_norm:
            (tier,) = params
            self._result = [(self._schema.counts.get(tier, 0),)]
            return
        raise AssertionError(f"Unrecognised query in fake cursor: {text!r}")

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._result[0] if self._result else None


class _FakeConnection:
    def __init__(self, schema: FakeSchema):
        self._schema = schema
        self.autocommit = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._schema)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Unit tests — pure logic
# ---------------------------------------------------------------------------


def test_parse_tiers_splits_and_strips():
    assert verify_module._parse_tiers("ar5iv,ads_body") == ["ar5iv", "ads_body"]
    assert verify_module._parse_tiers(" ar5iv , , ads_body ") == ["ar5iv", "ads_body"]
    assert verify_module._parse_tiers("") == []
    assert verify_module._parse_tiers(",,") == []


def test_verify_all_tiers_pass():
    schema = FakeSchema(counts={"ar5iv": 5, "ads_body": 2})
    conn = _FakeConnection(schema)
    exit_code, messages = verify_module.verify(conn, ["ar5iv", "ads_body"])
    assert exit_code == 0
    assert messages == [
        "PASS: tier ar5iv has 5 rows",
        "PASS: tier ads_body has 2 rows",
    ]


def test_verify_reports_failure_when_tier_empty():
    schema = FakeSchema(counts={"ar5iv": 3})
    conn = _FakeConnection(schema)
    exit_code, messages = verify_module.verify(conn, ["ar5iv", "ads_body"])
    assert exit_code == 1
    assert "PASS: tier ar5iv has 3 rows" in messages
    assert "FAIL: tier ads_body has 0 rows" in messages


def test_verify_all_tiers_empty():
    schema = FakeSchema(counts={})
    conn = _FakeConnection(schema)
    exit_code, messages = verify_module.verify(conn, ["ar5iv", "ads_body"])
    assert exit_code == 1
    assert messages == [
        "FAIL: tier ar5iv has 0 rows",
        "FAIL: tier ads_body has 0 rows",
    ]


def test_count_tier_returns_zero_when_no_rows():
    schema = FakeSchema(counts={})
    conn = _FakeConnection(schema)
    assert verify_module._count_tier(conn, "ar5iv") == 0


# ---------------------------------------------------------------------------
# main() — CLI surface via monkeypatched psycopg.connect
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_all_tiers_pass(monkeypatch, capsys):
    schema = FakeSchema(counts={"ar5iv": 1, "ads_body": 1})

    class _ConnCtx:
        def __enter__(self_inner):
            return _FakeConnection(schema)

        def __exit__(self_inner, *exc):
            return False

    monkeypatch.setattr(verify_module.psycopg, "connect", lambda dsn: _ConnCtx())

    exit_code = verify_module.main(
        ["--dsn", "dbname=scix_test", "--require-tiers", "ar5iv,ads_body", "--quiet"]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS: tier ar5iv has 1 rows" in out
    assert "PASS: tier ads_body has 1 rows" in out


def test_main_returns_one_when_tier_empty(monkeypatch, capsys):
    schema = FakeSchema(counts={"ar5iv": 1})

    class _ConnCtx:
        def __enter__(self_inner):
            return _FakeConnection(schema)

        def __exit__(self_inner, *exc):
            return False

    monkeypatch.setattr(verify_module.psycopg, "connect", lambda dsn: _ConnCtx())

    exit_code = verify_module.main(
        ["--dsn", "dbname=scix_test", "--require-tiers", "ar5iv,ads_body", "--quiet"]
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL: tier ads_body has 0 rows" in out


def test_main_rejects_empty_require_tiers(monkeypatch, capsys):
    # psycopg.connect must not be called — monkeypatch it to blow up if invoked.
    def _boom(dsn):
        raise AssertionError("psycopg.connect should not be called for empty tier list")

    monkeypatch.setattr(verify_module.psycopg, "connect", _boom)

    exit_code = verify_module.main(["--dsn", "dbname=scix_test", "--require-tiers", ",", "--quiet"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_main_default_tiers_match_spec(monkeypatch, capsys):
    """Default --require-tiers is 'ar5iv,ads_body' per PRD D3 spec."""
    seen_tiers: list[str] = []
    schema = FakeSchema(counts={"ar5iv": 1, "ads_body": 1})

    class _TrackingConn(_FakeConnection):
        def cursor(self):
            parent_cursor = super().cursor()
            original_execute = parent_cursor.execute

            def tracking_execute(query, params=None):
                if params:
                    seen_tiers.append(params[0])
                original_execute(query, params)

            parent_cursor.execute = tracking_execute  # type: ignore[assignment]
            return parent_cursor

    class _ConnCtx:
        def __enter__(self_inner):
            return _TrackingConn(schema)

        def __exit__(self_inner, *exc):
            return False

    monkeypatch.setattr(verify_module.psycopg, "connect", lambda dsn: _ConnCtx())

    exit_code = verify_module.main(["--dsn", "dbname=scix_test", "--quiet"])
    assert exit_code == 0
    assert seen_tiers == ["ar5iv", "ads_body"]


def test_help_exits_zero():
    """``--help`` must exit 0 (acceptance criterion 2)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--require-tiers" in result.stdout
    assert "--dsn" in result.stdout


# ---------------------------------------------------------------------------
# Integration — seed rows into scix_test and run the real script
# ---------------------------------------------------------------------------


def _require_test_dsn() -> str:
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        pytest.skip("SCIX_TEST_DSN not set — integration test requires test DB")
    if is_production_dsn(dsn):
        pytest.skip("SCIX_TEST_DSN points at production — refusing to seed rows")
    return dsn


def _ensure_papers_fulltext_exists(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
             WHERE table_schema='public' AND table_name='papers_fulltext'
            """)
        if cur.fetchone() is None:
            pytest.skip(
                "papers_fulltext table missing in scix_test — "
                "apply migrations/041_papers_fulltext.sql first"
            )


_TEST_BIBCODES = ("ZZVERIFTST1..................A", "ZZVERIFTST2..................A")


def _seed_rows(conn: psycopg.Connection, rows: list[tuple[str, str]]) -> None:
    """Seed (bibcode, source) rows. Skips if bibcodes are not in papers."""
    with conn.cursor() as cur:
        # Temporarily drop FK enforcement by inserting parent rows if possible;
        # otherwise skip. scix_test may have the FK but no parent rows.
        for bibcode, _source in rows:
            cur.execute(
                "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
                (bibcode,),
            )
        for bibcode, source in rows:
            cur.execute(
                """
                INSERT INTO papers_fulltext
                    (bibcode, source, sections, inline_cites, parser_version)
                VALUES (%s, %s, '[]'::jsonb, '[]'::jsonb, 'test-0')
                ON CONFLICT (bibcode) DO UPDATE SET source = EXCLUDED.source
                """,
                (bibcode, source),
            )
    conn.commit()


def _cleanup(conn: psycopg.Connection, bibcodes: Sequence[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM papers_fulltext WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
    conn.commit()


@pytest.mark.integration
def test_integration_pass_when_all_tiers_have_rows():
    dsn = _require_test_dsn()
    with psycopg.connect(dsn) as conn:
        _ensure_papers_fulltext_exists(conn)
        _cleanup(conn, _TEST_BIBCODES)
        try:
            _seed_rows(
                conn,
                [
                    (_TEST_BIBCODES[0], "ar5iv"),
                    (_TEST_BIBCODES[1], "ads_body"),
                ],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dsn",
                    dsn,
                    "--require-tiers",
                    "ar5iv,ads_body",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr + result.stdout
            assert "PASS: tier ar5iv" in result.stdout
            assert "PASS: tier ads_body" in result.stdout
            assert "FAIL" not in result.stdout
        finally:
            _cleanup(conn, _TEST_BIBCODES)


@pytest.mark.integration
def test_integration_fail_when_tier_has_zero_rows():
    dsn = _require_test_dsn()
    with psycopg.connect(dsn) as conn:
        _ensure_papers_fulltext_exists(conn)
        _cleanup(conn, _TEST_BIBCODES)
        try:
            # Only seed ar5iv; ads_body should remain empty (assuming a clean
            # scix_test — if not, the test is still valid as long as the
            # unique tier name below has 0 rows).
            _seed_rows(conn, [(_TEST_BIBCODES[0], "ar5iv")])
            unique_missing_tier = "tier-that-does-not-exist-xyz"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dsn",
                    dsn,
                    "--require-tiers",
                    f"ar5iv,{unique_missing_tier}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1, result.stderr + result.stdout
            assert "PASS: tier ar5iv" in result.stdout
            assert f"FAIL: tier {unique_missing_tier} has 0 rows" in result.stdout
        finally:
            _cleanup(conn, _TEST_BIBCODES)


# JSON-output round-trip is not part of this script's surface (verify_fulltext
# emits line-oriented PASS/FAIL text, per D3 spec). We keep one sanity check
# that the subprocess actually runs when SCIX_TEST_DSN is available.


@pytest.mark.integration
def test_integration_subprocess_emits_text_lines():
    dsn = _require_test_dsn()
    with psycopg.connect(dsn) as conn:
        _ensure_papers_fulltext_exists(conn)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dsn",
            dsn,
            "--require-tiers",
            "some-tier-that-does-not-exist",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # No JSON — the script emits text. Just assert it ran and produced
    # a well-formed FAIL line for the missing tier.
    assert result.returncode == 1, result.stderr
    assert "FAIL: tier some-tier-that-does-not-exist has 0 rows" in result.stdout
    # json.loads should fail on this output — confirming text format.
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)
