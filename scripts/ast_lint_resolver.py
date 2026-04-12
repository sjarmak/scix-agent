#!/usr/bin/env python3
"""AST lint that enforces the M13 single-entry-point rule.

Walks ``src/`` with libcst and fails if any Python file outside
``src/scix/resolve_entities.py`` contains a SQL string literal that writes
to ``document_entities`` / ``document_entities_jit_cache`` or reads from
``document_entities_canonical``. The lint is pattern-based: we match against
string literals in the module so both inline SQL (``cur.execute("INSERT
INTO document_entities ...")``) and assigned SQL
(``SQL = "INSERT INTO document_entities ..."; cur.execute(SQL)``) are
caught.

Escape hatch: any line that ends with ``# noqa: resolver-lint`` is skipped.
Use this for legitimate migration tooling that must bypass M13 on purpose.

Exit codes:
    0 - clean
    1 - violations found

This script is imported by ``tests/test_ast_lint_resolver.py`` so that a
deliberately planted violation can be verified without shelling out.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst

# ---------------------------------------------------------------------------
# Forbidden SQL patterns
# ---------------------------------------------------------------------------
#
# We carefully distinguish ``document_entities`` (the base table we ban) from
# ``document_entities_canonical`` (the MV — writes are auto-banned because
# you can't write to an MV; reads we ban separately) and
# ``document_entities_jit_cache`` (partitioned table — we ban writes only).
#
# The negative lookahead ``(?!_)`` keeps the first pattern from matching the
# other two table names.
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "insert_document_entities",
        re.compile(r"\bINSERT\s+INTO\s+document_entities\b(?!_)", re.IGNORECASE),
    ),
    (
        "update_document_entities",
        re.compile(r"\bUPDATE\s+document_entities\b(?!_)", re.IGNORECASE),
    ),
    (
        "delete_document_entities",
        re.compile(r"\bDELETE\s+FROM\s+document_entities\b(?!_)", re.IGNORECASE),
    ),
    (
        "insert_document_entities_jit_cache",
        re.compile(r"\bINSERT\s+INTO\s+document_entities_jit_cache\b", re.IGNORECASE),
    ),
    (
        "update_document_entities_jit_cache",
        re.compile(r"\bUPDATE\s+document_entities_jit_cache\b", re.IGNORECASE),
    ),
    (
        "delete_document_entities_jit_cache",
        re.compile(r"\bDELETE\s+FROM\s+document_entities_jit_cache\b", re.IGNORECASE),
    ),
    (
        "select_document_entities_canonical",
        re.compile(r"\bFROM\s+document_entities_canonical\b", re.IGNORECASE),
    ),
)

NOQA_MARKER = "# noqa: resolver-lint"

# Path of the one and only file allowed to contain these patterns, expressed
# as a POSIX suffix for cross-platform match.
ALLOWED_SUFFIX = "src/scix/resolve_entities.py"


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    snippet: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.snippet}"


# ---------------------------------------------------------------------------
# libcst visitor
# ---------------------------------------------------------------------------


class _StringLiteralCollector(cst.CSTVisitor):
    """Collect every string literal in a module with its approximate line.

    We use :meth:`libcst.Module.code_for_node` to locate literals via the
    module's metadata wrapper rather than hand-parsing, because SQL strings
    may be SimpleString, ConcatenatedString, or FormattedString.
    """

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self) -> None:
        super().__init__()
        # (start_line, end_line, value)
        self.literals: list[tuple[int, int, str]] = []

    def _record(self, node: cst.CSTNode, value: str) -> None:
        pos = self.get_metadata(cst.metadata.PositionProvider, node)
        self.literals.append((pos.start.line, pos.end.line, value))

    def visit_SimpleString(self, node: cst.SimpleString) -> None:
        try:
            value = node.evaluated_value
        except Exception:
            value = node.value
        if isinstance(value, bytes):  # pragma: no cover - defensive
            value = value.decode("utf-8", errors="ignore")
        self._record(node, value)

    def visit_ConcatenatedString(self, node: cst.ConcatenatedString) -> None:
        try:
            value = node.evaluated_value or ""
        except Exception:
            value = ""
        if isinstance(value, bytes):  # pragma: no cover - defensive
            value = value.decode("utf-8", errors="ignore")
        self._record(node, value)

    def visit_FormattedString(self, node: cst.FormattedString) -> None:
        # f-strings: reconstruct literal portions so we can scan them.
        # Expression parts become a placeholder so we don't accidentally
        # match on variable names like ``{document_entities}``.
        parts: list[str] = []
        for part in node.parts:
            if isinstance(part, cst.FormattedStringText):
                parts.append(part.value)
            else:
                parts.append(" __expr__ ")
        self._record(node, "".join(parts))


def _scan_source(source: str, path: Path) -> list[Violation]:
    """Return every M13 rule violation in ``source``."""
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        # Skip files we can't parse — they'd fail the real build anyway.
        return []

    wrapper = cst.metadata.MetadataWrapper(module)
    collector = _StringLiteralCollector()
    wrapper.visit(collector)

    # Pre-index source lines for noqa lookups.
    source_lines = source.splitlines()

    violations: list[Violation] = []
    for start_line, end_line, literal in collector.literals:
        # A literal is exempt if ``# noqa: resolver-lint`` appears anywhere
        # in the line range the string spans, or on the line immediately
        # before/after.
        if _has_noqa(source_lines, start_line, end_line):
            continue
        for rule, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(literal):
                snippet = literal.strip().splitlines()[0][:120]
                violations.append(Violation(path=path, line=start_line, rule=rule, snippet=snippet))
                break
    return violations


def _has_noqa(lines: list[str], start_line: int, end_line: int) -> bool:
    """Check if any line in the string's span carries the noqa marker."""
    lo = max(1, start_line - 1)
    hi = min(len(lines), end_line + 1)
    for idx_1based in range(lo, hi + 1):
        if NOQA_MARKER in lines[idx_1based - 1]:
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _is_allowed_file(path: Path) -> bool:
    # POSIX suffix match so Windows-friendly.
    return path.as_posix().endswith(ALLOWED_SUFFIX)


def run_lint(root: Path) -> list[Violation]:
    """Walk ``root`` and return every violation found."""
    violations: list[Violation] = []
    for py_path in _iter_py_files(root):
        if _is_allowed_file(py_path):
            continue
        try:
            source = py_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        violations.extend(_scan_source(source, py_path))
    return violations


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(argv[0]) if argv else Path("src")
    if not root.exists():
        print(f"ast_lint_resolver: {root} does not exist", file=sys.stderr)
        return 2
    violations = run_lint(root)
    if not violations:
        return 0
    print(
        f"ast_lint_resolver: {len(violations)} M13 rule violation(s) found",
        file=sys.stderr,
    )
    for v in violations:
        print("  " + v.format(), file=sys.stderr)
    print(
        "These files must NOT write document_entities / document_entities_jit_cache "
        "or read document_entities_canonical. All such access must go through "
        "src/scix/resolve_entities.py. Use '# noqa: resolver-lint' to intentionally "
        "bypass (e.g. migrations).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
