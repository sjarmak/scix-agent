"""Tests for scripts/ast_lint_resolver.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "ast_lint_resolver.py"


def _load_lint_module():
    spec = importlib.util.spec_from_file_location("ast_lint_resolver", LINT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lint_module():
    return _load_lint_module()


def test_current_src_has_no_violations(lint_module):
    """Running the lint against ``src/`` must pass (exit code 0)."""
    violations = lint_module.run_lint(REPO_ROOT / "src")
    assert violations == [], [v.format() for v in violations]


def test_main_exits_zero_on_clean_src(lint_module):
    rc = lint_module.main([str(REPO_ROOT / "src")])
    assert rc == 0


def test_planted_violation_is_detected(lint_module, tmp_path):
    """A deliberately planted insert into document_entities must fail."""
    bad_dir = tmp_path / "scix"
    bad_dir.mkdir()
    (bad_dir / "bad.py").write_text(
        "import psycopg\n"
        "def go(conn):\n"
        "    with conn.cursor() as cur:\n"
        '        cur.execute("INSERT INTO document_entities (bibcode) VALUES (%s)", ("x",))\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert len(violations) >= 1
    assert any(v.rule == "insert_document_entities" for v in violations)


def test_planted_update_is_detected(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'SQL = "UPDATE document_entities SET confidence = 0.5"\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert any(v.rule == "update_document_entities" for v in violations)


def test_planted_delete_is_detected(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'SQL = "DELETE FROM document_entities WHERE tier = 0"\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert any(v.rule == "delete_document_entities" for v in violations)


def test_planted_jit_cache_write_is_detected(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'SQL = "INSERT INTO document_entities_jit_cache (bibcode) VALUES (%s)"\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert any(v.rule == "insert_document_entities_jit_cache" for v in violations)


def test_planted_canonical_read_is_detected(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'SQL = "SELECT bibcode, entity_id FROM document_entities_canonical"\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert any(v.rule == "select_document_entities_canonical" for v in violations)


def test_noqa_comment_exempts_violation(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'SQL = "INSERT INTO document_entities (bibcode) VALUES (%s)"  # noqa: resolver-lint\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert violations == []


def test_noqa_comment_exempts_multiline_sql(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        "def go(cur):\n"
        "    cur.execute(\n"
        '        """\n'
        "        INSERT INTO document_entities (bibcode)\n"
        "        VALUES (%s)\n"
        '        """,  # noqa: resolver-lint\n'
        '        ("x",),\n'
        "    )\n",
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert violations == []


def test_resolve_entities_itself_is_exempt(lint_module, tmp_path):
    """Even if resolve_entities.py has a forbidden pattern, it's allowed."""
    allowed_dir = tmp_path / "src" / "scix"
    allowed_dir.mkdir(parents=True)
    (allowed_dir / "resolve_entities.py").write_text(
        'SQL = "INSERT INTO document_entities (bibcode) VALUES (%s)"\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    assert violations == []


def test_main_exits_nonzero_on_violation(lint_module, tmp_path):
    (tmp_path / "bad.py").write_text(
        'SQL = "INSERT INTO document_entities (bibcode) VALUES (%s)"\n',
        encoding="utf-8",
    )
    rc = lint_module.main([str(tmp_path)])
    assert rc == 1


def test_canonical_suffix_word_boundary(lint_module, tmp_path):
    """``document_entities_canonical`` must not trip the base-table rule."""
    (tmp_path / "ok.py").write_text(
        'SQL = "REFRESH MATERIALIZED VIEW document_entities_canonical"\n',
        encoding="utf-8",
    )
    violations = lint_module.run_lint(tmp_path)
    # REFRESH isn't one of our banned verbs and the name has the _canonical
    # suffix so the base-table regex must not fire.
    assert violations == []
