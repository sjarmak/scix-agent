"""Tests for scripts/extract_claims_batch.py.

Two layers:

1. Pure-logic tests — exercise CLI argument plumbing and helpers without
   touching a database. Always run.

2. Integration tests — apply migration 062 to scix_test, seed a few papers
   plus papers_fulltext rows, then invoke the script as a subprocess and
   assert the trailing-line JSON summary matches expectations. Skipped if
   no scix_test DB is reachable.

The DSN-resolution / skip pattern mirrors tests/test_paper_claims_schema.py
and tests/test_claims_extract.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_claims_batch.py"
MIGRATION_PATH = REPO_ROOT / "migrations" / "062_paper_claims.sql"
PYTHON_BIN = sys.executable


# ---------------------------------------------------------------------------
# Module loader for pure-logic tests
# ---------------------------------------------------------------------------


def _load_script_module() -> Any:
    """Import the script as a module so its helpers are importable.

    Mirrors tests/test_backfill_citation_intent.py's pattern.
    """
    spec = importlib.util.spec_from_file_location(
        "extract_claims_batch_script", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


batch_script = _load_script_module()


# ===========================================================================
# Pure-logic tests — no DB
# ===========================================================================


class TestHelpFlag:
    """`--help` exits 0 and prints usage including all documented flags."""

    def test_help_lists_all_flags(self) -> None:
        result = subprocess.run(
            [PYTHON_BIN, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        out = result.stdout
        for flag in (
            "--batch-size",
            "--limit",
            "--arxiv-class",
            "--bibcode-glob",
            "--section-roles",
            "--prompt-version",
            "--model-name",
            "--dsn",
            "--llm",
            "--llm-stub-claims-json",
        ):
            assert flag in out, f"--help missing flag: {flag}"


class TestParseSectionRoles:
    """The comma-separated section-roles parser handles empty / whitespace cases."""

    def test_none_returns_none(self) -> None:
        assert batch_script._parse_section_roles(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert batch_script._parse_section_roles("") is None

    def test_single_role(self) -> None:
        assert batch_script._parse_section_roles("results") == ["results"]

    def test_multiple_roles_with_whitespace(self) -> None:
        assert batch_script._parse_section_roles("results, discussion, conclusion") == [
            "results",
            "discussion",
            "conclusion",
        ]

    def test_drops_empty_segments(self) -> None:
        assert batch_script._parse_section_roles("results,,discussion") == [
            "results",
            "discussion",
        ]


class TestResolveDsn:
    """DSN resolution prefers --dsn > SCIX_TEST_DSN > SCIX_DSN > default."""

    def test_explicit_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCIX_TEST_DSN", "dbname=fromenv_test")
        assert batch_script._resolve_dsn("dbname=explicit") == "dbname=explicit"

    def test_test_env_wins_over_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCIX_TEST_DSN", "dbname=fromenv_test")
        assert batch_script._resolve_dsn(None) == "dbname=fromenv_test"

    def test_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCIX_TEST_DSN", raising=False)
        # DEFAULT_DSN is set at import time from SCIX_DSN env var, so we just
        # check the resolver returns SOMETHING when no overrides are present.
        result = batch_script._resolve_dsn(None)
        assert isinstance(result, str) and result


class TestBuildLLMClient:
    """LLMClient factory honours --llm and --llm-stub-claims-json."""

    def test_stub_default_returns_empty_list(self) -> None:
        client = batch_script.build_llm_client(kind="stub", stub_claims_json=None)
        assert client.extract("p", "para") == []

    def test_stub_with_canned_response(self) -> None:
        canned = json.dumps(
            [
                {
                    "claim_text": "x",
                    "claim_type": "factual",
                    "char_span_start": 0,
                    "char_span_end": 1,
                }
            ]
        )
        client = batch_script.build_llm_client(kind="stub", stub_claims_json=canned)
        result = client.extract("p", "para")
        assert len(result) == 1
        assert result[0]["claim_type"] == "factual"
        # Stub returns the SAME default for every call.
        assert client.extract("p2", "para2") == result

    def test_stub_invalid_json_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            batch_script.build_llm_client(kind="stub", stub_claims_json="not json {{{")

    def test_stub_non_list_json_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            batch_script.build_llm_client(kind="stub", stub_claims_json='{"a":1}')

    def test_unknown_kind_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            batch_script.build_llm_client(kind="bogus", stub_claims_json=None)

    def test_claude_cli_constructs(self) -> None:
        # Constructing the ClaudeCliLLMClient does NOT invoke the binary.
        client = batch_script.build_llm_client(kind="claude-cli", stub_claims_json=None)
        # Just confirm the attribute exists.
        assert hasattr(client, "_cli_path")


# ===========================================================================
# Integration layer — real schema, subprocess invocation
# ===========================================================================


def _resolve_test_dsn() -> str | None:
    """Resolve the scix_test DSN.

    Order:
    1. SCIX_TEST_DSN env var.
    2. Fallback "dbname=scix_test" if reachable.
    Returns None if no scix_test DB is reachable.
    """
    dsn = os.environ.get("SCIX_TEST_DSN")
    if dsn:
        if "scix_test" not in dsn:
            pytest.fail(
                "SCIX_TEST_DSN must reference scix_test — got: " + dsn,
                pytrace=False,
            )
        return dsn

    fallback = "dbname=scix_test"
    try:
        import psycopg
    except ImportError:
        return None

    try:
        with psycopg.connect(fallback, connect_timeout=2):
            pass
    except Exception:
        return None
    return fallback


TEST_DSN = _resolve_test_dsn()
INTEGRATION_REASON = "scix_test database not reachable (set SCIX_TEST_DSN)"


@pytest.fixture(scope="module")
def dsn() -> str:
    if TEST_DSN is None:
        pytest.skip(INTEGRATION_REASON, allow_module_level=False)
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migration(dsn: str) -> str:
    """Apply migration 062 (idempotently) before integration tests run."""
    result = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"migration 062 failed to apply: stderr=\n{result.stderr}\nstdout=\n{result.stdout}"
    )
    return dsn


# Paragraph used in fixtures — stable offsets for the canned claim.
PARAGRAPH = (
    "We measure a rotation period of 4.21 days for TOI-1452 b. "
    "Our pipeline outperforms the RAPID baseline by 13 F1 points."
)
ANCHOR_START, ANCHOR_END = 0, 56  # "We measure a rotation period of 4.21 days for TOI-1452 b"


def _seed_papers(dsn: str, n: int) -> list[str]:
    """Insert ``n`` synthetic (papers, papers_fulltext) rows. Returns bibcodes.

    The bibcodes share a common prefix so a glob test can target them.
    """
    import psycopg

    prefix = f"9999batch_{uuid.uuid4().hex[:6]}"
    bibcodes = [f"{prefix}_p{i:02d}" for i in range(n)]

    sections_payload = json.dumps(
        [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]
    )

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for bib in bibcodes:
            cur.execute(
                "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT (bibcode) DO NOTHING",
                (bib,),
            )
            cur.execute(
                """
                INSERT INTO papers_fulltext (
                    bibcode, source, sections, inline_cites, parser_version
                ) VALUES (
                    %s, 'test', %s::jsonb, '[]'::jsonb, 'test-v0'
                )
                ON CONFLICT (bibcode) DO UPDATE SET
                    sections = EXCLUDED.sections,
                    parser_version = EXCLUDED.parser_version
                """,
                (bib, sections_payload),
            )
        conn.commit()
    return bibcodes


def _cleanup_papers(dsn: str, bibcodes: list[str]) -> None:
    import psycopg

    if not bibcodes:
        return
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM paper_claims WHERE bibcode = ANY(%s)", (bibcodes,))
        cur.execute("DELETE FROM papers_fulltext WHERE bibcode = ANY(%s)", (bibcodes,))
        cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (bibcodes,))
        conn.commit()


def _seed_paper_no_sections(dsn: str) -> str:
    """Insert one paper whose papers_fulltext.sections is an empty list."""
    import psycopg

    bib = f"9999batch_empty_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT (bibcode) DO NOTHING",
            (bib,),
        )
        cur.execute(
            """
            INSERT INTO papers_fulltext (
                bibcode, source, sections, inline_cites, parser_version
            ) VALUES (
                %s, 'test', '[]'::jsonb, '[]'::jsonb, 'test-v0'
            )
            ON CONFLICT (bibcode) DO UPDATE SET sections = '[]'::jsonb
            """,
            (bib,),
        )
        conn.commit()
    return bib


def _canned_claim_json() -> str:
    """A single canned claim that fits PARAGRAPH's ANCHOR offsets.

    The same JSON is returned for every paragraph the stub sees — combined
    with the unique-index, that means N papers x 1 paragraph each = N claims.
    """
    return json.dumps(
        [
            {
                "claim_text": "TOI-1452 b has rotation period 4.21 days.",
                "claim_type": "factual",
                "subject": "TOI-1452 b",
                "predicate": "has rotation period",
                "object": "4.21 days",
                "char_span_start": ANCHOR_START,
                "char_span_end": ANCHOR_END,
                "confidence": 0.95,
            }
        ]
    )


def _run_script(dsn: str, *extra_args: str) -> tuple[int, str, str]:
    """Run the script as a subprocess; return (returncode, stdout, stderr)."""
    cmd = [
        PYTHON_BIN,
        str(SCRIPT_PATH),
        "--dsn",
        dsn,
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


def _parse_trailing_json(stdout: str) -> dict[str, int]:
    """The script's contract: stdout's last non-empty line is JSON."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    assert lines, f"no stdout produced; raw={stdout!r}"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Acceptance criterion 9.d / 9.e — happy-path subprocess run + idempotency
# ---------------------------------------------------------------------------


class TestSubprocessHappyPath:
    """Seed 3 papers, run the script, assert stats; re-run, assert idempotency."""

    def test_three_papers_processed_then_idempotent(
        self, applied_migration: str
    ) -> None:
        import psycopg

        bibcodes = _seed_papers(applied_migration, 3)
        try:
            # Run #1: stub returns 1 canned claim per paragraph.
            rc, stdout, stderr = _run_script(
                applied_migration,
                "--limit",
                "3",
                "--batch-size",
                "10",
                "--llm",
                "stub",
                "--llm-stub-claims-json",
                _canned_claim_json(),
                "--bibcode-glob",
                bibcodes[0].rsplit("_", 1)[0] + "%",
                "--prompt-version",
                "v1",
                "--model-name",
                "stub-test-model",
            )
            assert rc == 0, f"stderr=\n{stderr}\nstdout=\n{stdout}"
            stats = _parse_trailing_json(stdout)
            assert stats["processed"] == 3, stats
            assert stats["claims_written"] >= 1, stats
            assert stats["skipped"] == 0, stats
            assert stats["failed"] == 0, stats
            first_run_written = stats["claims_written"]

            # Verify rows actually landed in paper_claims.
            with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM paper_claims WHERE bibcode = ANY(%s)",
                    (bibcodes,),
                )
                row_count_after_first = cur.fetchone()[0]
            assert row_count_after_first == first_run_written

            # Run #2: identical inputs → idempotent. claims_written drops to 0.
            rc2, stdout2, stderr2 = _run_script(
                applied_migration,
                "--limit",
                "3",
                "--batch-size",
                "10",
                "--llm",
                "stub",
                "--llm-stub-claims-json",
                _canned_claim_json(),
                "--bibcode-glob",
                bibcodes[0].rsplit("_", 1)[0] + "%",
                "--prompt-version",
                "v1",
                "--model-name",
                "stub-test-model",
            )
            assert rc2 == 0, f"stderr=\n{stderr2}\nstdout=\n{stdout2}"
            stats2 = _parse_trailing_json(stdout2)
            assert stats2["processed"] == 3, stats2
            assert stats2["claims_written"] == 0, stats2
            assert stats2["failed"] == 0, stats2

            # Row count must NOT have grown.
            with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM paper_claims WHERE bibcode = ANY(%s)",
                    (bibcodes,),
                )
                row_count_after_second = cur.fetchone()[0]
            assert row_count_after_second == row_count_after_first, (
                "duplicate rows detected after re-run"
            )
        finally:
            _cleanup_papers(applied_migration, bibcodes)


class TestSubprocessSkipsEmptySections:
    """A paper whose papers_fulltext.sections is [] increments `skipped`,
    not `processed-only`. (processed counts attempts; skipped is a subset.)"""

    def test_empty_sections_paper_is_skipped(self, applied_migration: str) -> None:
        empty_bib = _seed_paper_no_sections(applied_migration)
        try:
            rc, stdout, stderr = _run_script(
                applied_migration,
                "--limit",
                "1",
                "--llm",
                "stub",
                "--bibcode-glob",
                empty_bib + "%",
            )
            assert rc == 0, f"stderr=\n{stderr}\nstdout=\n{stdout}"
            stats = _parse_trailing_json(stdout)
            assert stats["processed"] == 1, stats
            assert stats["skipped"] == 1, stats
            assert stats["claims_written"] == 0, stats
            assert stats["failed"] == 0, stats
        finally:
            _cleanup_papers(applied_migration, [empty_bib])


class TestSubprocessStubDefaultProducesNoClaims:
    """`--llm stub` without --llm-stub-claims-json yields zero claims."""

    def test_stub_default_no_claims(self, applied_migration: str) -> None:
        bibcodes = _seed_papers(applied_migration, 2)
        try:
            rc, stdout, stderr = _run_script(
                applied_migration,
                "--limit",
                "2",
                "--llm",
                "stub",
                "--bibcode-glob",
                bibcodes[0].rsplit("_", 1)[0] + "%",
            )
            assert rc == 0, f"stderr=\n{stderr}\nstdout=\n{stdout}"
            stats = _parse_trailing_json(stdout)
            assert stats["processed"] == 2, stats
            assert stats["claims_written"] == 0, stats
            assert stats["skipped"] == 0, stats
            assert stats["failed"] == 0, stats
        finally:
            _cleanup_papers(applied_migration, bibcodes)
