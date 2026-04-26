"""Tests for src/scix/claims/extract.py — the claim extraction pipeline.

Two layers:

1. Pure-logic tests — exercise the section-role classifier, paragraph
   splitter, claim-validation, and StubLLMClient via in-memory fakes.
   Always run.

2. Integration tests — apply migration 062 to scix_test, then run
   ``extract_claims_for_paper`` end-to-end against the real schema with a
   StubLLMClient (NEVER calls the real ``claude`` CLI). Skipped if no
   scix_test DB is reachable.

The DSN-resolution / skip pattern mirrors tests/test_paper_claims_schema.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from scix.claims import (
    ClaimDict,
    ClaudeCliLLMClient,
    StubLLMClient,
    classify_section_role,
    extract_claims_for_paper,
    split_paragraphs,
)
from scix.claims.extract import _validate_claim

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_PATH = REPO_ROOT / "migrations" / "062_paper_claims.sql"

PROMPT_VERSION = "v1"
MODEL_NAME = "stub-test-model"


# ===========================================================================
# Pure-logic tests — no DB
# ===========================================================================


class TestClassifySectionRole:
    """The heading -> role classifier handles common variants."""

    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("Abstract", "abstract"),
            ("1. Introduction", "introduction"),
            ("II. Intro", "introduction"),
            ("Background and Motivation", "introduction"),
            ("Related Work", "related_work"),
            ("Methods", "methods"),
            ("3.1 Methodology", "methods"),
            ("Experimental Setup", "methods"),
            ("Results", "results"),
            ("Results and Analysis", "results"),
            ("Discussion", "discussion"),
            ("Conclusions", "conclusion"),
            ("Summary", "conclusion"),
            ("Acknowledgments", "acknowledgments"),
            ("References", "references"),
            ("Appendix A", "appendix"),
            ("Random Header", "other"),
            ("", "other"),
            (None, "other"),
        ],
    )
    def test_role_classification(self, heading: str | None, expected: str) -> None:
        assert classify_section_role(heading) == expected


class TestSplitParagraphs:
    """Paragraph splitting preserves indices and drops empties."""

    def test_single_paragraph(self) -> None:
        text = "One short paragraph with no breaks."
        result = split_paragraphs(text)
        assert len(result) == 1
        idx, body, offset = result[0]
        assert idx == 0
        assert body == text
        assert offset == 0

    def test_two_paragraphs_double_newline(self) -> None:
        text = "First paragraph.\n\nSecond paragraph here."
        result = split_paragraphs(text)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[0][1] == "First paragraph."
        assert result[1][0] == 1
        assert result[1][1] == "Second paragraph here."

    def test_empty_input(self) -> None:
        assert split_paragraphs("") == []

    def test_whitespace_only_paragraph_dropped(self) -> None:
        text = "Real text.\n\n   \n\nMore text."
        result = split_paragraphs(text)
        assert [r[1] for r in result] == ["Real text.", "More text."]


class TestValidateClaim:
    """_validate_claim is the per-claim gate; it must fail-soft."""

    def _good_claim(self, paragraph: str) -> dict[str, object]:
        return {
            "claim_text": "We measured X.",
            "claim_type": "factual",
            "subject": "we",
            "predicate": "measured",
            "object": "X",
            "char_span_start": 0,
            "char_span_end": min(10, len(paragraph)),
            "confidence": 0.9,
        }

    def test_valid_claim_round_trips(self) -> None:
        para = "We measured X using a calibrated detector."
        out = _validate_claim(self._good_claim(para), para)
        assert out is not None
        assert out["claim_text"] == "We measured X."

    def test_missing_required_field_returns_none(self) -> None:
        para = "Some paragraph."
        bad = self._good_claim(para)
        del bad["claim_text"]
        assert _validate_claim(bad, para) is None

    def test_unknown_claim_type_returns_none(self) -> None:
        para = "Some paragraph."
        bad = self._good_claim(para)
        bad["claim_type"] = "made_up"
        assert _validate_claim(bad, para) is None

    def test_oob_char_span_returns_none(self) -> None:
        para = "Short."
        bad = self._good_claim(para)
        bad["char_span_end"] = 9999
        assert _validate_claim(bad, para) is None

    def test_inverted_char_span_returns_none(self) -> None:
        para = "Some paragraph here."
        bad = self._good_claim(para)
        bad["char_span_start"] = 5
        bad["char_span_end"] = 5
        assert _validate_claim(bad, para) is None

    def test_non_integer_char_span_returns_none(self) -> None:
        para = "Some paragraph here."
        bad = self._good_claim(para)
        bad["char_span_start"] = "zero"
        assert _validate_claim(bad, para) is None


class TestStubLLMClient:
    """The stub itself behaves predictably — sanity check for the test fixture."""

    def test_default_response(self) -> None:
        stub = StubLLMClient(default=[{"x": 1}])  # type: ignore[list-item]
        assert stub.extract("p", "para") == [{"x": 1}]
        assert stub.calls == [("p", "para")]

    def test_queue_consumed_fifo(self) -> None:
        stub = StubLLMClient(responses=[[{"a": 1}], [{"b": 2}]])  # type: ignore[list-item]
        assert stub.extract("p1", "para1") == [{"a": 1}]
        assert stub.extract("p2", "para2") == [{"b": 2}]
        # Queue exhausted -> default empty list.
        assert stub.extract("p3", "para3") == []

    def test_raise_propagates(self) -> None:
        stub = StubLLMClient(raise_exc=json.JSONDecodeError("boom", "", 0))
        with pytest.raises(json.JSONDecodeError):
            stub.extract("p", "para")


class TestClaudeCliClientNoSdkImport:
    """ClaudeCliLLMClient resolves the binary path correctly without
    importing any paid-API SDK."""

    def test_default_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCIX_CLAUDE_CLI", raising=False)
        client = ClaudeCliLLMClient()
        assert client._cli_path == "claude"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCIX_CLAUDE_CLI", "/opt/bin/claude")
        client = ClaudeCliLLMClient()
        assert client._cli_path == "/opt/bin/claude"

    def test_explicit_arg_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCIX_CLAUDE_CLI", "/opt/bin/claude")
        client = ClaudeCliLLMClient(cli_path="/usr/local/bin/claude-canary")
        assert client._cli_path == "/usr/local/bin/claude-canary"

    def test_missing_binary_raises_jsondecodeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point at a binary that doesn't exist; .extract() should convert the
        # FileNotFoundError into a JSONDecodeError so the pipeline can skip.
        client = ClaudeCliLLMClient(cli_path="/nonexistent/claude-binary-xyz")
        with pytest.raises(json.JSONDecodeError):
            client.extract("prompt body", "paragraph")


# ===========================================================================
# Integration layer — real schema, StubLLMClient (no Claude calls)
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


def _ensure_test_bibcode(dsn: str) -> str:
    """Insert a synthetic bibcode for this test run; returns it."""
    import psycopg

    bibcode = f"9999test_claims_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT (bibcode) DO NOTHING",
            (bibcode,),
        )
        conn.commit()
    return bibcode


def _cleanup_paper(dsn: str, bibcode: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM paper_claims WHERE bibcode = %s", (bibcode,))
        cur.execute("DELETE FROM papers WHERE bibcode = %s", (bibcode,))
        conn.commit()


@pytest.fixture
def test_paper(applied_migration: str):
    """Yield (dsn, bibcode) and clean up after the test."""
    bibcode = _ensure_test_bibcode(applied_migration)
    yield applied_migration, bibcode
    _cleanup_paper(applied_migration, bibcode)


# ---------------------------------------------------------------------------
# Test fixtures: paragraph and section payloads
# ---------------------------------------------------------------------------

PARAGRAPH = (
    "We measure a rotation period of 4.21 days for TOI-1452 b. "
    "Our pipeline outperforms the RAPID baseline by 13 F1 points."
)
# Provenance test: the substring at these offsets must round-trip verbatim.
ANCHOR_1 = (0, 56)  # "We measure a rotation period of 4.21 days for TOI-1452 b"
ANCHOR_2 = (58, 118)  # "Our pipeline outperforms the RAPID baseline by 13 F1 points."


def _two_claims_for(paragraph: str) -> list[ClaimDict]:
    return [
        {
            "claim_text": "TOI-1452 b has rotation period 4.21 days.",
            "claim_type": "factual",
            "subject": "TOI-1452 b",
            "predicate": "has rotation period",
            "object": "4.21 days",
            "char_span_start": ANCHOR_1[0],
            "char_span_end": ANCHOR_1[1],
            "confidence": 0.95,
        },
        {
            "claim_text": "Our pipeline outperforms RAPID by 13 F1 points.",
            "claim_type": "comparative",
            "subject": "our pipeline",
            "predicate": "outperforms",
            "object": "RAPID by 13 F1 points",
            "char_span_start": ANCHOR_2[0],
            "char_span_end": ANCHOR_2[1],
            "confidence": 0.9,
        },
    ]


# ---------------------------------------------------------------------------
# Acceptance criteria 9.a — happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """1 paper, 1 section, 1 paragraph, stub returns 2 claims -> 2 rows."""

    def test_two_claims_persisted_with_correct_provenance(
        self, test_paper: tuple[str, str]
    ) -> None:
        import psycopg

        dsn, bibcode = test_paper
        sections = [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]
        stub = StubLLMClient(responses=[_two_claims_for(PARAGRAPH)])

        with psycopg.connect(dsn) as conn:
            n = extract_claims_for_paper(
                conn,
                bibcode,
                sections,
                stub,
                prompt_version=PROMPT_VERSION,
                model_name=MODEL_NAME,
            )

        assert n == 2

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT section_index, paragraph_index, char_span_start, char_span_end,
                       claim_text, claim_type, subject, predicate, object,
                       extraction_model, extraction_prompt_version
                FROM paper_claims
                WHERE bibcode = %s
                ORDER BY char_span_start
                """,
                (bibcode,),
            )
            rows = cur.fetchall()

        assert len(rows) == 2
        # Provenance check: paragraph[start:end] equals the documented anchor.
        for row, expected_anchor in zip(rows, [ANCHOR_1, ANCHOR_2]):
            section_index, paragraph_index, span_start, span_end = row[:4]
            assert section_index == 0
            assert paragraph_index == 0
            assert span_start == expected_anchor[0]
            assert span_end == expected_anchor[1]
            assert PARAGRAPH[span_start:span_end] == PARAGRAPH[
                expected_anchor[0] : expected_anchor[1]
            ]
        # Provenance for first row: "We measure ... TOI-1452 b" must be a verbatim slice.
        assert PARAGRAPH[rows[0][2] : rows[0][3]].startswith("We measure a rotation period")
        # Model + prompt-version columns recorded.
        for row in rows:
            assert row[9] == MODEL_NAME
            assert row[10] == PROMPT_VERSION


# ---------------------------------------------------------------------------
# Acceptance criteria 9.b — section role filtering
# ---------------------------------------------------------------------------


class TestSectionRoleFiltering:
    """Six sections; only those whose heading classifies into the whitelist
    get processed."""

    def test_only_abstract_and_results_processed(self, test_paper: tuple[str, str]) -> None:
        import psycopg

        dsn, bibcode = test_paper

        # Six sections with distinguishable, easy-to-anchor paragraph text.
        # Each paragraph is the same so the same anchor is valid in any of them.
        sections = [
            {"heading": "Abstract", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "1. Introduction", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "2. Methods", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "3. Results", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "4. Discussion", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "5. References", "level": 1, "text": PARAGRAPH, "offset": 0},
        ]

        # Only "abstract" and "results" should be processed -> 2 stub.extract calls.
        # Pre-load 6 responses so any over-call would be detectable; we'll assert exactly 2 below.
        responses = [_two_claims_for(PARAGRAPH) for _ in range(6)]
        stub = StubLLMClient(responses=responses)

        with psycopg.connect(dsn) as conn:
            n = extract_claims_for_paper(
                conn,
                bibcode,
                sections,
                stub,
                prompt_version=PROMPT_VERSION,
                model_name=MODEL_NAME,
                section_roles=["abstract", "results"],
            )

        # 2 sections * 2 claims each = 4.
        assert n == 4
        # Stub should have been called exactly 2 times (one per processed paragraph).
        assert len(stub.calls) == 2

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT section_index FROM paper_claims WHERE bibcode = %s "
                "ORDER BY section_index",
                (bibcode,),
            )
            section_indices = [row[0] for row in cur.fetchall()]

        # Sections 0 (Abstract) and 3 (Results) only.
        assert section_indices == [0, 3]


# ---------------------------------------------------------------------------
# Acceptance criteria 9.c — idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Re-invoking on the same paper inserts no duplicates."""

    def test_second_invocation_returns_zero_and_count_unchanged(
        self, test_paper: tuple[str, str]
    ) -> None:
        import psycopg

        dsn, bibcode = test_paper
        sections = [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]

        # First invocation: stub returns 2 claims.
        first_stub = StubLLMClient(responses=[_two_claims_for(PARAGRAPH)])
        with psycopg.connect(dsn) as conn:
            first = extract_claims_for_paper(
                conn,
                bibcode,
                sections,
                first_stub,
                prompt_version=PROMPT_VERSION,
                model_name=MODEL_NAME,
            )
        assert first == 2

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM paper_claims WHERE bibcode = %s", (bibcode,))
            count_after_first = cur.fetchone()[0]
        assert count_after_first == 2

        # Second invocation: stub returns the SAME 2 claims, same provenance.
        second_stub = StubLLMClient(responses=[_two_claims_for(PARAGRAPH)])
        with psycopg.connect(dsn) as conn:
            second = extract_claims_for_paper(
                conn,
                bibcode,
                sections,
                second_stub,
                prompt_version=PROMPT_VERSION,
                model_name=MODEL_NAME,
            )
        assert second == 0

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM paper_claims WHERE bibcode = %s", (bibcode,))
            count_after_second = cur.fetchone()[0]
        assert count_after_second == 2


# ---------------------------------------------------------------------------
# Acceptance criteria 9.d — invalid JSON tolerance
# ---------------------------------------------------------------------------


class TestInvalidJsonTolerance:
    """Stub raises JSONDecodeError -> pipeline records nothing, does not raise."""

    def test_jsondecodeerror_is_swallowed(self, test_paper: tuple[str, str]) -> None:
        import psycopg

        dsn, bibcode = test_paper
        sections = [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]
        stub = StubLLMClient(raise_exc=json.JSONDecodeError("not json", "", 0))

        with psycopg.connect(dsn) as conn:
            n = extract_claims_for_paper(
                conn,
                bibcode,
                sections,
                stub,
                prompt_version=PROMPT_VERSION,
                model_name=MODEL_NAME,
            )

        assert n == 0
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM paper_claims WHERE bibcode = %s", (bibcode,))
            assert cur.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Acceptance criteria 9.e — out-of-bounds char_span tolerance
# ---------------------------------------------------------------------------


class TestOutOfBoundsCharSpanTolerance:
    """A claim with end > len(paragraph) is dropped silently."""

    def test_oob_claim_skipped_other_claims_persist(self, test_paper: tuple[str, str]) -> None:
        import psycopg

        dsn, bibcode = test_paper
        sections = [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]

        good = _two_claims_for(PARAGRAPH)[0]
        oob: ClaimDict = {
            "claim_text": "Out of bounds claim.",
            "claim_type": "factual",
            "subject": "x",
            "predicate": "y",
            "object": "z",
            "char_span_start": 0,
            "char_span_end": 9999,
            "confidence": 0.5,
        }
        stub = StubLLMClient(responses=[[good, oob]])

        with psycopg.connect(dsn) as conn:
            n = extract_claims_for_paper(
                conn,
                bibcode,
                sections,
                stub,
                prompt_version=PROMPT_VERSION,
                model_name=MODEL_NAME,
            )

        assert n == 1  # only the good one persisted
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT claim_text FROM paper_claims WHERE bibcode = %s",
                (bibcode,),
            )
            texts = [row[0] for row in cur.fetchall()]
        assert texts == [good["claim_text"]]


# ---------------------------------------------------------------------------
# Acceptance criteria 10 — pure-logic also runs without DB (mocked conn)
# ---------------------------------------------------------------------------


class TestPipelineWithMockedConn:
    """Pure-logic test: extract_claims_for_paper is exercised via a mock
    connection so the test runs even when scix_test is unreachable."""

    class _Cursor:
        def __init__(self, statements: list[tuple[str, tuple]]) -> None:
            self._statements = statements
            self._last_sql: str = ""

        def __enter__(self) -> "TestPipelineWithMockedConn._Cursor":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def execute(self, sql: str, params: tuple = ()) -> None:
            self._last_sql = sql
            self._statements.append((sql, tuple(params)))

        def fetchone(self) -> tuple | None:
            # For the INSERT ... RETURNING claim_id pattern, pretend we always
            # successfully inserted a new row.
            if "RETURNING" in self._last_sql:
                return (uuid.uuid4(),)
            return None

    class _Conn:
        def __init__(self) -> None:
            self.statements: list[tuple[str, tuple]] = []
            self.committed = False

        def cursor(self) -> "TestPipelineWithMockedConn._Cursor":
            return TestPipelineWithMockedConn._Cursor(self.statements)

        def commit(self) -> None:
            self.committed = True

        def rollback(self) -> None:
            pass

    def test_mocked_pipeline_calls_stub_and_inserts(self) -> None:
        sections = [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]
        stub = StubLLMClient(responses=[_two_claims_for(PARAGRAPH)])

        conn = self._Conn()
        n = extract_claims_for_paper(
            conn,
            "9999mock_paper",
            sections,
            stub,
            prompt_version=PROMPT_VERSION,
            model_name=MODEL_NAME,
        )

        assert n == 2
        assert conn.committed is True
        # Stub got called exactly once (one paragraph in this section).
        assert len(stub.calls) == 1
        # We should see at least the index-create plus 2 INSERT statements.
        sqls = [s[0] for s in conn.statements]
        assert any("CREATE UNIQUE INDEX" in s for s in sqls)
        assert sum(1 for s in sqls if s.lstrip().upper().startswith("INSERT INTO PAPER_CLAIMS")) == 2

    def test_mocked_pipeline_section_role_filter(self) -> None:
        sections = [
            {"heading": "Abstract", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "Methods", "level": 1, "text": PARAGRAPH, "offset": 0},
            {"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0},
        ]
        stub = StubLLMClient(default=[])

        conn = self._Conn()
        extract_claims_for_paper(
            conn,
            "9999mock_paper",
            sections,
            stub,
            prompt_version=PROMPT_VERSION,
            model_name=MODEL_NAME,
            section_roles=["abstract", "results"],
        )

        # Methods skipped -> only 2 stub calls.
        assert len(stub.calls) == 2

    def test_mocked_pipeline_jsondecodeerror_does_not_raise(self) -> None:
        sections = [{"heading": "Results", "level": 1, "text": PARAGRAPH, "offset": 0}]
        stub = StubLLMClient(raise_exc=json.JSONDecodeError("oops", "", 0))

        conn = self._Conn()
        n = extract_claims_for_paper(
            conn,
            "9999mock_paper",
            sections,
            stub,
            prompt_version=PROMPT_VERSION,
            model_name=MODEL_NAME,
        )
        assert n == 0
        # Index-create still ran, but no INSERTs.
        sqls = [s[0] for s in conn.statements]
        assert any("CREATE UNIQUE INDEX" in s for s in sqls)
        assert not any(s.lstrip().upper().startswith("INSERT INTO PAPER_CLAIMS") for s in sqls)
