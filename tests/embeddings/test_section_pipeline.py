"""Unit tests for ``scix.embeddings.section_pipeline``.

These tests must run on CPU-only CI: no GPU, no live DB, no real model. The
section_pipeline module is designed for that — model loading is lazy and DB
access is parameterized on a duck-typed connection. Here we substitute a tiny
fake connection / cursor / model and exercise the pure helpers and the
batch-orchestration glue.

Integration tests (marked ``@pytest.mark.integration``) at the bottom of this
file exercise the real pipeline end-to-end against a non-production Postgres
DSN provided via ``SCIX_TEST_DSN``. They are skipped when the env var is
unset or points at a production database, and they use a fake encoder to keep
the test fast without burning model-load time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import psycopg
import pytest

from scix.embeddings.section_pipeline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    NOMIC_DOC_PREFIX,
    _build_parser,
    _process_batch,
    compute_section_sha,
    encode_batch,
    existing_shas,
    filter_unchanged,
    format_halfvec,
    iter_sections,
    main,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal cursor double: records ``execute`` calls and yields canned rows.

    ``rows_for_query`` is a function (sql, params) -> iterable of result rows.
    """

    def __init__(self, rows_for_query):
        self._rows_for_query = rows_for_query
        self.calls: list[tuple[str, Any]] = []
        self._rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params=None):
        self.calls.append((sql, params))
        self._rows = list(self._rows_for_query(sql, params))

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    def __init__(self, rows_for_query):
        self._rows_for_query = rows_for_query
        self.commits = 0
        self.last_cursor: _FakeCursor | None = None

    def cursor(self):
        cur = _FakeCursor(self._rows_for_query)
        self.last_cursor = cur
        return cur

    def commit(self):
        self.commits += 1

    def close(self):
        pass


class _FakeModel:
    """Deterministic stub model. ``encode`` returns a vector whose first
    component is a stable hash of the input — lets tests assert order-pairing.
    """

    def __init__(self, dim: int = DEFAULT_DIMENSIONS):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=True, truncate_dim=None):
        self.calls.append(list(texts))
        out = []
        for i, t in enumerate(texts):
            # First slot encodes input ordinal so the test can spot mismatches.
            v = [float(i)] + [float(len(t)) / 1000.0] * (self.dim - 1)
            out.append(v)
        return out


# ---------------------------------------------------------------------------
# 1. sha256 determinism
# ---------------------------------------------------------------------------


def test_sha256_determinism():
    sha_a = compute_section_sha("Methods", "We did X.")
    sha_b = compute_section_sha("Methods", "We did X.")
    assert sha_a == sha_b
    assert len(sha_a) == 64  # hex digest

    # Different heading -> different sha
    sha_c = compute_section_sha("Results", "We did X.")
    assert sha_a != sha_c

    # Different text -> different sha
    sha_d = compute_section_sha("Methods", "We did Y.")
    assert sha_a != sha_d

    # None inputs are stable too
    sha_none = compute_section_sha(None, None)
    assert sha_none == compute_section_sha("", "")


# ---------------------------------------------------------------------------
# 2. halfvec text formatting
# ---------------------------------------------------------------------------


def test_halfvec_text_formatting():
    s = format_halfvec([0.1, 0.2, 0.3])
    assert s.startswith("[") and s.endswith("]")
    # Roundtrip via Python float parsing — must yield the input.
    parts = s.strip("[]").split(",")
    parsed = [float(x) for x in parts]
    assert parsed == [0.1, 0.2, 0.3]

    # Empty vector formats to "[]"
    assert format_halfvec([]) == "[]"

    # Tolerates non-float-typed inputs (numpy scalars, ints)
    s2 = format_halfvec([1, 2, 3])
    assert [float(x) for x in s2.strip("[]").split(",")] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 3. multi-section chunking via iter_sections
# ---------------------------------------------------------------------------


def test_multi_section_chunking():
    sections_for_paper = [
        {"heading": "Intro", "level": 1, "text": "intro body"},
        {"heading": "Methods", "level": 1, "text": "method body"},
        {"heading": "Results", "level": 1, "text": "result body"},
    ]

    def rows_for_query(sql, params):
        assert "papers_fulltext" in sql
        # Single paper, three sections.
        return [("2020ApJ...900..001S", sections_for_paper)]

    conn = _FakeConn(rows_for_query)
    yielded = list(iter_sections(conn))

    assert len(yielded) == 3
    indices = [row[1] for row in yielded]
    assert indices == [0, 1, 2]
    headings = [row[2] for row in yielded]
    assert headings == ["Intro", "Methods", "Results"]
    bibcodes = {row[0] for row in yielded}
    assert bibcodes == {"2020ApJ...900..001S"}


def test_iter_sections_skips_empty_text_and_handles_json_string():
    sections = [
        {"heading": "Empty", "level": 1, "text": ""},
        {"heading": "Whitespace", "level": 1, "text": "   "},
        {"heading": "Real", "level": 1, "text": "real content"},
    ]

    def rows_for_query(sql, params):
        # Return sections as a JSON string to test the str-fallback path.
        return [("2021ApJ...920..002Z", json.dumps(sections))]

    conn = _FakeConn(rows_for_query)
    yielded = list(iter_sections(conn))
    assert len(yielded) == 1
    assert yielded[0][1] == 2  # original section_index preserved
    assert yielded[0][2] == "Real"


def test_iter_sections_passes_range_params():
    captured = {}

    def rows_for_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return []

    conn = _FakeConn(rows_for_query)
    list(iter_sections(conn, start_bibcode="A", end_bibcode="Z"))

    assert captured["params"]["start"] == "A"
    assert captured["params"]["end"] == "Z"
    assert "ORDER BY bibcode" in captured["sql"]


# ---------------------------------------------------------------------------
# 4. skip-existing logic
# ---------------------------------------------------------------------------


def test_existing_shas_returns_empty_for_no_bibcodes():
    """No bibcodes -> no DB round-trip and empty dict."""
    calls: list = []

    def rows_for_query(sql, params):
        calls.append((sql, params))
        return []

    conn = _FakeConn(rows_for_query)
    out = existing_shas(conn, [])
    assert out == {}
    # No execute call should have happened.
    assert calls == []


def test_existing_shas_keys_by_bibcode_and_index():
    def rows_for_query(sql, params):
        assert "section_embeddings" in sql
        assert params["bibcodes"] == ["b1", "b2"]
        return [
            ("b1", 0, "sha-b1-0"),
            ("b1", 1, "sha-b1-1"),
            ("b2", 0, "sha-b2-0"),
        ]

    conn = _FakeConn(rows_for_query)
    out = existing_shas(conn, ["b1", "b2"])
    assert out == {
        ("b1", 0): "sha-b1-0",
        ("b1", 1): "sha-b1-1",
        ("b2", 0): "sha-b2-0",
    }


def test_skip_existing_logic_via_filter_unchanged():
    rows = [
        ("b1", 0, "Intro", "text-a", "sha-a"),
        ("b1", 1, "Methods", "text-b", "sha-b"),
        ("b2", 0, "Intro", "text-c", "sha-c"),
    ]
    # Stored has matching sha for (b1,0) and (b2,0) — those should be skipped.
    stored = {
        ("b1", 0): "sha-a",
        ("b1", 1): "sha-OLD",  # sha differs -> re-encode
        ("b2", 0): "sha-c",
    }
    survivors = filter_unchanged(rows, stored)
    assert [(r[0], r[1]) for r in survivors] == [("b1", 1)]


def test_process_batch_skips_unchanged_and_writes_only_changed():
    """End-to-end batch step: stored sha for one row matches; the other gets
    encoded and written. Verify the model only sees the changed input and the
    COPY only carries that row."""

    rows_for_paper = [
        ("b1", 0, "Intro", "text-a"),
        ("b1", 1, "Methods", "text-b"),
    ]

    sha_a = compute_section_sha("Intro", "text-a")

    copy_writes: list = []
    inserts: list = []

    class _CopyCtx:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_row(self, row):
            copy_writes.append(row)

    class _BatchCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            self.last_sql = sql
            self.last_params = params
            if "INSERT INTO section_embeddings" in sql:
                inserts.append(sql)

        def copy(self, sql):
            inserts.append(("COPY", sql))
            return _CopyCtx()

        def __iter__(self):
            # Existing-sha lookup returns the matching sha for (b1, 0) only.
            if "section_embeddings" in getattr(self, "last_sql", ""):
                return iter([("b1", 0, sha_a)])
            return iter([])

        @property
        def rowcount(self):
            return 1

    class _BatchConn:
        def cursor(self):
            return _BatchCursor()

        def commit(self):
            pass

    model = _FakeModel(dim=DEFAULT_DIMENSIONS)
    written = _process_batch(_BatchConn(), model, rows_for_paper, dimensions=DEFAULT_DIMENSIONS)

    assert written == 1
    # Model should have been called exactly once with one input — the
    # unchanged (b1, 0) row should be filtered out before encode.
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 1
    assert model.calls[0][0].startswith(NOMIC_DOC_PREFIX)
    assert "text-b" in model.calls[0][0]

    # Exactly one row written via COPY, for (b1, 1).
    assert len(copy_writes) == 1
    bibcode, idx, heading, sha, vec_literal = copy_writes[0]
    assert (bibcode, idx) == ("b1", 1)
    assert heading == "Methods"
    assert sha == compute_section_sha("Methods", "text-b")
    assert vec_literal.startswith("[") and vec_literal.endswith("]")


# ---------------------------------------------------------------------------
# 5. CLI argument parsing
# ---------------------------------------------------------------------------


def test_cli_argument_parsing_defaults():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.model == DEFAULT_MODEL
    assert args.dimensions == DEFAULT_DIMENSIONS
    assert args.batch_size == DEFAULT_BATCH_SIZE
    assert args.start_bibcode is None
    assert args.end_bibcode is None
    assert args.dry_run is False
    assert args.dsn is None


def test_cli_argument_parsing_overrides():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--model",
            "custom/model",
            "--batch-size",
            "16",
            "--start-bibcode",
            "2020A",
            "--end-bibcode",
            "2021Z",
            "--dry-run",
            "--dimensions",
            "768",
            "--dsn",
            "dbname=test",
        ]
    )
    assert args.model == "custom/model"
    assert args.batch_size == 16
    assert args.start_bibcode == "2020A"
    assert args.end_bibcode == "2021Z"
    assert args.dry_run is True
    assert args.dimensions == 768
    assert args.dsn == "dbname=test"


def test_cli_help_exits_zero():
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_main_dry_run_returns_zero_without_db_or_model():
    """--dry-run must short-circuit before touching the DB or loading a model.
    If it tried, this test would fail in CI (no live DB, no model files)."""
    rc = main(["--dry-run"])
    assert rc == 0


# ---------------------------------------------------------------------------
# encode_batch unit (uses the fake model — no torch/sentence_transformers)
# ---------------------------------------------------------------------------


def test_encode_batch_returns_python_floats():
    model = _FakeModel(dim=8)
    out = encode_batch(model, ["a", "bb", "ccc"], dimensions=8)
    assert len(out) == 3
    for vec in out:
        assert len(vec) == 8
        for x in vec:
            assert isinstance(x, float)


def test_encode_batch_truncates_to_requested_dimensions():
    """If the model emits more dims than requested, we trim."""

    class _BiggerModel:
        def encode(self, texts, normalize_embeddings=True, truncate_dim=None):
            # Always emit 16-d, regardless of truncate_dim.
            return [[float(i)] * 16 for i, _ in enumerate(texts)]

    out = encode_batch(_BiggerModel(), ["a", "b"], dimensions=4)
    assert all(len(v) == 4 for v in out)


def test_encode_batch_empty_input():
    assert encode_batch(_FakeModel(), [], dimensions=8) == []


# ---------------------------------------------------------------------------
# Policy guard: the source file must not import paid-API SDKs
# ---------------------------------------------------------------------------


def test_no_paid_api_imports():
    """Per project policy ``feedback_no_paid_apis``, the section pipeline
    must not import any paid-API SDK. Verify by reading the source file."""
    from scix.embeddings import section_pipeline as sp

    src = Path(sp.__file__).read_text(encoding="utf-8")
    forbidden = (
        "import openai",
        "from openai",
        "import cohere",
        "from cohere",
        "from anthropic",
        "import anthropic",
        "import voyageai",
        "from voyageai",
        "import together",
        "from together",
    )
    for fragment in forbidden:
        assert fragment not in src, f"Forbidden import found: {fragment!r}"


# ---------------------------------------------------------------------------
# Integration tests against a real Postgres (scix_test). Skipped if
# SCIX_TEST_DSN is unset or points at the production database.
# ---------------------------------------------------------------------------

_PRODUCTION_DB_NAMES = {"scix"}


def _is_production_dsn(dsn: str) -> bool:
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


_TEST_DSN = os.environ.get("SCIX_TEST_DSN")
_skip_no_test_db = pytest.mark.skipif(
    _TEST_DSN is None or _is_production_dsn(_TEST_DSN or ""),
    reason=(
        "Integration test requires SCIX_TEST_DSN pointing to a non-production "
        "database with section_embeddings + papers_fulltext schema."
    ),
)


@pytest.fixture()
def _scix_test_conn():
    """Yield a psycopg connection to scix_test and clean up after the test."""
    assert _TEST_DSN is not None  # guarded by _skip_no_test_db
    conn = psycopg.connect(_TEST_DSN)
    yield conn
    conn.close()


@pytest.mark.integration
@_skip_no_test_db
def test_section_pipeline_writes_to_scix_test(_scix_test_conn) -> None:
    """End-to-end smoke: insert sample papers + papers_fulltext rows, run
    ``_process_batch`` with the deterministic ``_FakeModel``, and verify
    section_embeddings rows land with the expected sha + halfvec(1024)
    payload. Cleans up its own rows on success and on failure.
    """
    from scix.embeddings.section_pipeline import (
        DEFAULT_DIMENSIONS,
        _process_batch,
        compute_section_sha,
        iter_sections,
    )

    bib_a = "9999TEST..001..001A"
    bib_b = "9999TEST..001..002B"

    sections_a = [
        {"heading": "Intro", "level": 1, "text": "Introduction body alpha."},
        {"heading": "Methods", "level": 1, "text": "We measured X via Y."},
    ]
    sections_b = [
        {"heading": "Results", "level": 1, "text": "Result text beta."},
    ]

    cur = _scix_test_conn.cursor()
    try:
        # Insert minimal papers + papers_fulltext rows.
        cur.execute(
            "INSERT INTO papers (bibcode, title) VALUES (%s, %s), (%s, %s) "
            "ON CONFLICT (bibcode) DO NOTHING",
            (bib_a, "Test paper A", bib_b, "Test paper B"),
        )
        cur.execute(
            "INSERT INTO papers_fulltext (bibcode, source, sections, inline_cites, "
            "                              parser_version) "
            "VALUES (%s, %s, %s::jsonb, '[]'::jsonb, %s), "
            "       (%s, %s, %s::jsonb, '[]'::jsonb, %s) "
            "ON CONFLICT (bibcode) DO UPDATE SET "
            "  sections = EXCLUDED.sections, "
            "  source   = EXCLUDED.source",
            (
                bib_a, "test", json.dumps(sections_a), "test-v1",
                bib_b, "test", json.dumps(sections_b), "test-v1",
            ),
        )
        _scix_test_conn.commit()

        # Read back via iter_sections and run a single _process_batch.
        rows = list(
            iter_sections(
                _scix_test_conn,
                start_bibcode=bib_a,
                end_bibcode=bib_b,
            )
        )
        assert len(rows) == 3  # 2 sections in a + 1 in b

        model = _FakeModel(dim=DEFAULT_DIMENSIONS)
        written = _process_batch(
            _scix_test_conn,
            model,
            rows,
            dimensions=DEFAULT_DIMENSIONS,
        )
        assert written == 3

        # Verify the rows landed with the right shape.
        cur.execute(
            "SELECT bibcode, section_index, section_heading, section_text_sha256 "
            "FROM section_embeddings "
            "WHERE bibcode IN (%s, %s) "
            "ORDER BY bibcode, section_index",
            (bib_a, bib_b),
        )
        out = list(cur)
        assert out == [
            (bib_a, 0, "Intro", compute_section_sha("Intro", "Introduction body alpha.")),
            (bib_a, 1, "Methods", compute_section_sha("Methods", "We measured X via Y.")),
            (bib_b, 0, "Results", compute_section_sha("Results", "Result text beta.")),
        ]

        # Verify halfvec dimensionality is 1024.
        cur.execute(
            "SELECT vector_dims(embedding::vector) FROM section_embeddings "
            "WHERE bibcode = %s LIMIT 1",
            (bib_a,),
        )
        (dims,) = cur.fetchone()
        assert dims == DEFAULT_DIMENSIONS

        # Idempotency: re-running _process_batch on the same input must not
        # write anything (sha matches stored sha).
        written_again = _process_batch(
            _scix_test_conn,
            model,
            rows,
            dimensions=DEFAULT_DIMENSIONS,
        )
        assert written_again == 0

    finally:
        # Clean up regardless of test outcome.
        cur.execute(
            "DELETE FROM section_embeddings WHERE bibcode IN (%s, %s)",
            (bib_a, bib_b),
        )
        cur.execute(
            "DELETE FROM papers_fulltext WHERE bibcode IN (%s, %s)",
            (bib_a, bib_b),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode IN (%s, %s)",
            (bib_a, bib_b),
        )
        _scix_test_conn.commit()
        cur.close()
