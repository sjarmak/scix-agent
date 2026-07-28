"""Guard: no test module may reach production when SCIX_TEST_DSN is exported.

CLAUDE.md documents `export SCIX_TEST_DSN="dbname=scix_test"` as the way to
point the suite away from the production `scix` database. That contract was
only honoured by some modules: `tests/helpers.py` and four test modules read
`SCIX_DSN` first, so with the guard exported they still connected to
production, and the destructive modules that gate on `is_production_dsn(DSN)`
skipped silently instead of running.

This module fails if that pattern comes back.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from helpers import DSN

TESTS_DIR = Path(__file__).resolve().parent

# Modules allowed to read SCIX_DSN without the SCIX_TEST_DSN fallback, with the
# reason. Each is a deliberate read-only probe of the live corpus, named and
# documented as such at its definition site.
ALLOWED_PROD_READERS = {
    "test_mcp_entity_context_smoke.py": "_PROD_DSN_FOR_READONLY: documented read-only prod smoke test",
}


def _dsn_expressions(tree: ast.AST) -> list[ast.expr]:
    """Return the value of every assignment whose target names a DSN."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # Case-insensitive: local assignments spell it `dsn`, and one of
            # those (test_agent_entity_context_rewrite) carried the same bug.
            if isinstance(target, ast.Name) and "dsn" in target.id.lower():
                found.append(node.value)
    return found


def _reads_env(expr: ast.expr, name: str) -> bool:
    """Return True if the expression reads os.environ for ``name``."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and node.value == name:
            return True
    return False


def _test_dsn_aliases(tree: ast.AST) -> set[str]:
    """Return module-level names bound to a SCIX_TEST_DSN lookup.

    ``TEST_DSN = os.environ.get("SCIX_TEST_DSN")`` followed by
    ``DSN = TEST_DSN or ...`` honours the guard just as much as the inline
    form, so referencing such a name counts.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _reads_env(node.value, "SCIX_TEST_DSN"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def _references(expr: ast.expr, names: set[str]) -> bool:
    """Return True if the expression reads any of ``names``."""
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(expr))


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("*.py") if p.name != Path(__file__).name)


@pytest.mark.parametrize("path", _test_modules(), ids=lambda p: p.name)
def test_scix_dsn_is_never_read_without_the_test_dsn_fallback(path: Path) -> None:
    """A module reading SCIX_DSN must consult SCIX_TEST_DSN first."""
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = _test_dsn_aliases(tree)
    for expr in _dsn_expressions(tree):
        if not _reads_env(expr, "SCIX_DSN"):
            continue
        if _reads_env(expr, "SCIX_TEST_DSN") or _references(expr, aliases):
            continue
        allowed = ALLOWED_PROD_READERS.get(path.name)
        assert allowed, (
            f"{path.name} resolves a DSN from SCIX_DSN without an SCIX_TEST_DSN "
            f"fallback, so it reaches production even when the documented guard "
            f"is exported. Use: "
            f'os.environ.get("SCIX_TEST_DSN") or os.environ.get("SCIX_DSN", "dbname=scix")'
        )


def test_helpers_dsn_honours_the_exported_guard() -> None:
    """helpers.DSN is what the destructive modules gate on; it must follow SCIX_TEST_DSN."""
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if not test_dsn:
        pytest.skip("SCIX_TEST_DSN is not exported")
    assert DSN == test_dsn, (
        f"helpers.DSN resolved to {DSN!r} while SCIX_TEST_DSN={test_dsn!r}. "
        "Destructive test modules skip on is_production_dsn(DSN), so a stale "
        "helpers.DSN makes them silently skip."
    )
