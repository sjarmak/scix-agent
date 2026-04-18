"""Tests for scripts/verify_communities_populated.py.

Unit tests stub the psycopg connection with a fake in-memory schema + data
model so the script's verification logic is exercised without a live DB.
An integration test (marked ``integration``) runs the real script against
``SCIX_TEST_DSN`` when that env var is set.
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

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_communities_populated.py"


# ---------------------------------------------------------------------------
# Module loader — the script lives in ``scripts/`` which isn't a package.
# ---------------------------------------------------------------------------


def _load_module():
    name = "verify_communities_populated"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass() can resolve cls.__module__ during
    # its forward-ref handling (Python 3.12 looks the module up in sys.modules).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


verify_module = _load_module()


# ---------------------------------------------------------------------------
# Fake connection/cursor — enough to satisfy the script's query surface.
# ---------------------------------------------------------------------------


@dataclass
class FakeSchema:
    """Configurable fake of the scix DB schema + row counts."""

    tables: set[str] = field(default_factory=set)
    # ``columns`` maps table -> set of column names.
    columns: dict[str, set[str]] = field(default_factory=dict)
    # ``non_null_counts`` maps (table, col) -> int. Missing key = 0.
    non_null_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    # Total row count per table.
    table_totals: dict[str, int] = field(default_factory=dict)
    # Communities breakdown rows: list of (signal, resolution, count).
    # ``signal`` may be None if the signal column is absent.
    communities_breakdown: list[tuple[str | None, str, int]] = field(
        default_factory=list
    )


class _FakeCursor:
    def __init__(self, schema: FakeSchema):
        self._schema = schema
        self._result: list[tuple[Any, ...]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query: Any, params: Sequence[Any] | None = None) -> None:
        text = query.as_string(None) if hasattr(query, "as_string") else str(query)
        text_norm = " ".join(text.split()).lower()
        params = tuple(params or ())

        if "information_schema.columns" in text_norm:
            table, candidate_cols = params
            have = self._schema.columns.get(table, set())
            self._result = [(c,) for c in candidate_cols if c in have]
            return
        if "information_schema.tables" in text_norm:
            (candidate_tables,) = params
            self._result = [
                (t,) for t in candidate_tables if t in self._schema.tables
            ]
            return
        if "is not null" in text_norm:
            # Parse ``SELECT count(*) FROM "tbl" WHERE "col" IS NOT NULL``.
            parts = text.split('"')
            table = parts[1]
            column = parts[3]
            self._result = [
                (self._schema.non_null_counts.get((table, column), 0),)
            ]
            return
        if "select count(*) from" in text_norm:
            parts = text.split('"')
            table = parts[1]
            self._result = [(self._schema.table_totals.get(table, 0),)]
            return
        if "from communities group by" in text_norm:
            self._result = [
                (signal, resolution, count)
                for (signal, resolution, count) in self._schema.communities_breakdown
            ]
            return
        raise AssertionError(f"Unrecognised query in fake cursor: {text!r}")

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._result[0] if self._result else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._result)


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
# Unit tests
# ---------------------------------------------------------------------------


def _baseline_schema() -> FakeSchema:
    """Schema resembling production on 2026-04-18 — migration 051/052 missing."""
    schema = FakeSchema(
        tables={"paper_metrics", "communities"},
        columns={
            "paper_metrics": {
                "community_id_coarse",
                "community_id_medium",
                "community_id_fine",
                "community_taxonomic",
            },
            "communities": {
                "community_id",
                "resolution",
                "label",
                "paper_count",
                "top_keywords",
            },
        },
        non_null_counts={
            ("paper_metrics", "community_id_coarse"): 12_409_080,
            ("paper_metrics", "community_id_medium"): 12_409_080,
            ("paper_metrics", "community_id_fine"): 12_409_080,
            ("paper_metrics", "community_taxonomic"): 2_662_341,
        },
        table_totals={
            "paper_metrics": 32_390_237,
            "communities": 0,
        },
        communities_breakdown=[],
    )
    return schema


def test_verify_flags_missing_semantic_migration_and_empty_communities():
    schema = _baseline_schema()
    conn = _FakeConnection(schema)
    report = verify_module.verify(conn, dsn_redacted="dbname=scix")

    problems = set(report.problems)

    # 051 not applied — three semantic columns flagged.
    assert "migration 051 not applied: paper_metrics.community_semantic_coarse missing" in problems
    assert "migration 051 not applied: paper_metrics.community_semantic_medium missing" in problems
    assert "migration 051 not applied: paper_metrics.community_semantic_fine missing" in problems

    # 052 not applied — signal column absent.
    assert "migration 052 not applied: communities.signal missing" in problems

    # Communities table empty — M4 labels never written.
    assert "communities table empty: 0 rows (M4 labels not generated)" in problems

    # Citation counts are non-zero so no citation-empty problem raised.
    assert not any(
        p.startswith("citation community empty") for p in problems
    )


def test_verify_healthy_when_full_state_present():
    schema = _baseline_schema()
    schema.columns["paper_metrics"].update(
        {
            "community_semantic_coarse",
            "community_semantic_medium",
            "community_semantic_fine",
        }
    )
    schema.columns["communities"].add("signal")
    schema.non_null_counts.update(
        {
            ("paper_metrics", "community_semantic_coarse"): 32_000_000,
            ("paper_metrics", "community_semantic_medium"): 32_000_000,
            ("paper_metrics", "community_semantic_fine"): 32_000_000,
        }
    )
    schema.table_totals["communities"] = 4220
    schema.communities_breakdown = [
        ("citation", "coarse", 20),
        ("citation", "medium", 200),
        ("citation", "fine", 2000),
        ("semantic", "coarse", 20),
        ("semantic", "medium", 200),
        ("semantic", "fine", 2000),
    ]

    conn = _FakeConnection(schema)
    report = verify_module.verify(conn, dsn_redacted="dbname=scix")

    assert report.problems == []
    payload = report.to_dict()
    assert payload["healthy"] is True
    assert payload["communities_total"] == 4220


def test_verify_flags_missing_citation_data():
    schema = _baseline_schema()
    # Zero out citation assignment coverage to simulate a fresh DB.
    for col in ("community_id_coarse", "community_id_medium", "community_id_fine"):
        schema.non_null_counts[("paper_metrics", col)] = 0

    conn = _FakeConnection(schema)
    report = verify_module.verify(conn, dsn_redacted="dbname=scix_test")

    citation_problems = [
        p for p in report.problems if p.startswith("citation community empty")
    ]
    assert len(citation_problems) == 3


def test_verify_ignores_missing_paper_communities_table():
    """``paper_communities`` was never declared by the PRD; absence != problem."""
    schema = _baseline_schema()
    # Give it full semantic + signal coverage so only the paper_communities
    # presence is at issue.
    schema.columns["paper_metrics"].update(
        {
            "community_semantic_coarse",
            "community_semantic_medium",
            "community_semantic_fine",
        }
    )
    schema.columns["communities"].add("signal")
    schema.non_null_counts.update(
        {
            ("paper_metrics", "community_semantic_coarse"): 1,
            ("paper_metrics", "community_semantic_medium"): 1,
            ("paper_metrics", "community_semantic_fine"): 1,
        }
    )
    schema.table_totals["communities"] = 1

    conn = _FakeConnection(schema)
    report = verify_module.verify(conn, dsn_redacted="x")

    # paper_communities reported absent in tables[], but no problem raised.
    paper_communities_rows = [
        t for t in report.tables if t.table == "paper_communities"
    ]
    assert paper_communities_rows == [
        verify_module.TableCheck(table="paper_communities", present=False)
    ]
    assert not any("paper_communities" in p for p in report.problems)


def test_main_returns_nonzero_on_problems(tmp_path, monkeypatch, capsys):
    """End-to-end: main() exits 1 when problems exist, writes the output file."""

    # Monkeypatch psycopg.connect to return a fake connection with the
    # baseline (unhealthy) schema.
    schema = _baseline_schema()

    class _ConnCtx:
        def __enter__(self_inner):
            conn = _FakeConnection(schema)
            return conn

        def __exit__(self_inner, *exc):
            return False

    monkeypatch.setattr(
        verify_module.psycopg, "connect", lambda dsn: _ConnCtx()
    )

    out_path = tmp_path / "report.json"
    exit_code = verify_module.main(
        ["--dsn", "dbname=scix", "--output", str(out_path), "--quiet"]
    )
    assert exit_code == 1
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["healthy"] is False
    # stdout also has the JSON.
    captured = capsys.readouterr().out
    assert "healthy" in captured


def test_main_returns_zero_when_healthy(tmp_path, monkeypatch):
    schema = _baseline_schema()
    schema.columns["paper_metrics"].update(
        {
            "community_semantic_coarse",
            "community_semantic_medium",
            "community_semantic_fine",
        }
    )
    schema.columns["communities"].add("signal")
    schema.non_null_counts.update(
        {
            ("paper_metrics", "community_semantic_coarse"): 32_000_000,
            ("paper_metrics", "community_semantic_medium"): 32_000_000,
            ("paper_metrics", "community_semantic_fine"): 32_000_000,
        }
    )
    schema.table_totals["communities"] = 1

    class _ConnCtx:
        def __enter__(self_inner):
            return _FakeConnection(schema)

        def __exit__(self_inner, *exc):
            return False

    monkeypatch.setattr(
        verify_module.psycopg, "connect", lambda dsn: _ConnCtx()
    )

    exit_code = verify_module.main(
        ["--dsn", "dbname=scix", "--quiet"]
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Integration — run the real script against SCIX_TEST_DSN.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_runs_against_scix_test():
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        pytest.skip("SCIX_TEST_DSN not set")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dsn", dsn, "--quiet"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit code may be 0 or 1 depending on whether scix_test has migrations
    # applied. We only assert the JSON shape is valid + no crash.
    assert result.returncode in (0, 1), result.stderr
    payload = json.loads(result.stdout)
    assert "dsn_redacted" in payload
    assert "problems" in payload
    assert "paper_metrics_total" in payload
