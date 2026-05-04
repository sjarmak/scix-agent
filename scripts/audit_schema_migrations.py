"""Audit drift between migrations/ files and schema_migrations table.

For each file under migrations/, the audit:
1. Parses the version number and filename.
2. Determines a probe target — either auto-extracted from the first CREATE
   statement, or supplied by MANUAL_PROBES for migrations whose primary
   effect is a column add, marker INSERT, or other non-CREATE change.
3. Queries the database to check whether the effect is present.
4. Joins against schema_migrations to detect: MISSING_ROW (effects present
   but no row), MISSING_EFFECTS (row recorded but effects missing),
   FILENAME_MISMATCH (row exists at this version with a different filename),
   UNKNOWN (no probe available).

Run as a script for a tabular report; import and call ``run_audit`` for
programmatic use.

Bead: scix_experiments-l0ub.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import psycopg

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"

VERSION_PREFIX_RE = re.compile(r"^(\d+)_")
CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)
CREATE_VIEW_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)
CREATE_FUNCTION_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Probe:
    """A check the audit can run against the DB.

    kind:
      - 'table'    qualified_name = 'schema.table' or 'table' (defaults to public)
      - 'view'     qualified_name = 'schema.view'
      - 'function' qualified_name = 'schema.function'
      - 'column'   qualified_name = 'schema.table.column'
      - 'marker'   qualified_name = SQL fragment for a one-row probe
                   (e.g. "ingest_log WHERE filename='x'")
    """

    kind: str
    qualified_name: str


@dataclass(frozen=True)
class ParsedMigration:
    version: int
    filename: str
    path: Path

    def auto_probe(self) -> Probe | None:
        """Extract the first CREATE TABLE / VIEW / FUNCTION as a probe.

        Strips both ``-- line`` and ``/* block */`` comments first so a
        comment like ``-- Idempotent: CREATE TABLE IF NOT EXISTS,`` doesn't
        get matched as a real DDL statement (the trailing ``,`` defeats the
        IF-NOT-EXISTS optional and would otherwise capture ``IF`` as the
        table name).
        """
        sql = _strip_sql_comments(self.path.read_text())
        if m := CREATE_TABLE_RE.search(sql):
            return Probe("table", _qualify(m.group(1)))
        if m := CREATE_VIEW_RE.search(sql):
            return Probe("view", _qualify(m.group(1)))
        if m := CREATE_FUNCTION_RE.search(sql):
            return Probe("function", _qualify(m.group(1)))
        return None


@dataclass(frozen=True)
class AuditRow:
    version: int
    filename: str
    in_db: bool
    in_db_filename: str | None
    effects_present: bool | None
    probe: Probe | None

    @property
    def status(self) -> str:
        if self.in_db and self.in_db_filename != self.filename:
            return "FILENAME_MISMATCH"
        if self.probe is None:
            return "UNKNOWN"
        # effects_present == None means the probe raised — distinguish
        # this from genuinely-absent effects so a malformed probe or
        # transient DB error doesn't masquerade as MISSING_EFFECTS.
        if self.effects_present is None:
            return "PROBE_ERROR"
        if self.in_db and self.effects_present:
            return "OK"
        if self.in_db and not self.effects_present:
            return "MISSING_EFFECTS"
        if not self.in_db and self.effects_present:
            return "MISSING_ROW"
        return "MISSING_BOTH"


# Manual probes for migrations whose primary effect is a column add, marker
# INSERT, or other change auto_probe cannot detect.
#
# Versions 65/66/67 are the post-rename numbers for what were originally
# 055_paper_umap_2d, 056_intent_populate, and 063_section_entities.
# See l0ub for the rename rationale (collision with in-DB v=63 zpm4 BM25).
MANUAL_PROBES: dict[int, Probe] = {
    25: Probe("table", "public.entity_merge_log"),
    58: Probe("column", "public.papers.correction_events"),
    60: Probe("column", "staging.extractions.section_name"),
    66: Probe(
        "marker",
        "ingest_log WHERE filename='intent_backfill:citation_contexts'",
    ),
}


def _qualify(name: str) -> str:
    return name if "." in name else f"public.{name}"


_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_sql_comments(sql: str) -> str:
    """Strip both ``-- line`` and ``/* block */`` SQL comments."""
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", sql))


# Backwards-compat alias for the original name (unit tests + external callers).
_strip_sql_line_comments = _strip_sql_comments


def parse_migration(path: Path) -> ParsedMigration | None:
    m = VERSION_PREFIX_RE.match(path.name)
    if not m or path.suffix != ".sql":
        return None
    return ParsedMigration(
        version=int(m.group(1)),
        filename=path.name,
        path=path,
    )


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[ParsedMigration]:
    """List every parseable migration file in ``directory``."""
    out: list[ParsedMigration] = []
    for path in sorted(directory.iterdir()):
        parsed = parse_migration(path)
        if parsed is not None:
            out.append(parsed)
    return out


def probe_target(conn: psycopg.Connection, probe: Probe) -> bool:
    """Return True if the probe target is present in the DB.

    Uses ``to_regclass`` for tables/views and ``to_regproc`` for functions —
    NOT ``to_regprocedure``, which requires a full ``name(arg_types)``
    signature and silently returns NULL for bare names.
    """
    with conn.cursor() as cur:
        if probe.kind in ("table", "view"):
            cur.execute(
                "SELECT to_regclass(%s) IS NOT NULL", (probe.qualified_name,)
            )
            row = cur.fetchone()
            return row is not None and bool(row[0])
        if probe.kind == "function":
            cur.execute(
                "SELECT to_regproc(%s) IS NOT NULL", (probe.qualified_name,)
            )
            row = cur.fetchone()
            return row is not None and bool(row[0])
        if probe.kind == "column":
            try:
                schema, table, column = probe.qualified_name.split(".")
            except ValueError as e:
                raise ValueError(
                    f"column probe must be schema.table.column, got "
                    f"{probe.qualified_name!r}"
                ) from e
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s "
                "AND column_name = %s",
                (schema, table, column),
            )
            return cur.fetchone() is not None
        if probe.kind == "marker":
            # qualified_name is sourced exclusively from MANUAL_PROBES
            # (hardcoded in this file). It is not parameterizable because
            # the marker shape is "<table> WHERE <expression>" — splitting
            # it would require a richer Probe type. Validate the shape
            # before interpolation as a belt-and-suspenders against future
            # contributors who add markers.
            if not _MARKER_SHAPE_RE.match(probe.qualified_name):
                raise ValueError(
                    f"marker probe rejected unexpected shape: "
                    f"{probe.qualified_name!r}"
                )
            cur.execute(  # nosec B608 - shape-validated, source-controlled input
                f"SELECT 1 FROM {probe.qualified_name} LIMIT 1"
            )
            return cur.fetchone() is not None
    raise ValueError(f"unknown probe kind: {probe.kind!r}")


# A marker probe must look like  "ident(.ident)? [WHERE ident(.ident)?='value']"
_MARKER_SHAPE_RE = re.compile(
    r"^[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?"
    r"(?: WHERE [A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?='[^']*')?$"
)


def fetch_recorded_versions(
    conn: psycopg.Connection,
) -> dict[int, str]:
    """Return {version: filename} from schema_migrations."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT version, filename FROM schema_migrations ORDER BY version"
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def run_audit(
    conn: psycopg.Connection,
    migrations: Iterable[ParsedMigration] | None = None,
) -> list[AuditRow]:
    """Build an AuditRow per migration file, joined against schema_migrations."""
    files = list(migrations) if migrations is not None else discover_migrations()
    recorded = fetch_recorded_versions(conn)

    rows: list[AuditRow] = []
    for parsed in files:
        probe = MANUAL_PROBES.get(parsed.version) or parsed.auto_probe()
        effects_present: bool | None
        if probe is None:
            effects_present = None
        else:
            try:
                effects_present = probe_target(conn, probe)
            except Exception as exc:
                # Don't abort the whole audit on a single bad probe — but
                # log loudly so PROBE_ERROR rows are explainable. Reset the
                # connection's transaction state since the failed query
                # implicitly opened one and subsequent probes would error.
                logger.warning(
                    "probe failed for v=%d (%s): %s",
                    parsed.version,
                    probe,
                    exc,
                )
                conn.rollback()
                effects_present = None
        in_db = parsed.version in recorded
        rows.append(
            AuditRow(
                version=parsed.version,
                filename=parsed.filename,
                in_db=in_db,
                in_db_filename=recorded.get(parsed.version),
                effects_present=effects_present,
                probe=probe,
            )
        )
    return rows


def format_report(rows: Iterable[AuditRow]) -> str:
    out = [
        f"{'V':>3}  {'FILE':<48}  {'STATUS':<18}  {'PROBE'}",
        "-" * 100,
    ]
    for r in sorted(rows, key=lambda r: r.version):
        probe_str = (
            f"{r.probe.kind}:{r.probe.qualified_name}" if r.probe else "(no probe)"
        )
        suffix = (
            f"  [recorded as {r.in_db_filename}]"
            if r.status == "FILENAME_MISMATCH"
            else ""
        )
        out.append(
            f"{r.version:>3}  {r.filename:<48}  {r.status:<18}  "
            f"{probe_str}{suffix}"
        )
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN") or "dbname=scix",
        help="PostgreSQL DSN (default: $SCIX_DSN or dbname=scix)",
    )
    p.add_argument(
        "--migrations",
        type=Path,
        default=MIGRATIONS_DIR,
        help=f"Migrations directory (default: {MIGRATIONS_DIR})",
    )
    p.add_argument(
        "--exit-on-drift",
        action="store_true",
        help=(
            "Exit nonzero if any actionable drift is found (statuses: "
            "MISSING_BOTH, MISSING_EFFECTS, MISSING_ROW, FILENAME_MISMATCH, "
            "PROBE_ERROR). UNKNOWN is excluded — many migrations have no "
            "auto-detectable probe and that's not drift, just incomplete "
            "tooling coverage."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    with psycopg.connect(args.dsn) as conn:
        rows = run_audit(conn, discover_migrations(args.migrations))

    print(format_report(rows))

    if args.exit_on_drift:
        drifted = [r for r in rows if r.status in DRIFT_STATUSES]
        if drifted:
            print(
                f"\nDRIFT: {len(drifted)} migration(s) need attention",
                file=sys.stderr,
            )
            return 1
    return 0


DRIFT_STATUSES = frozenset(
    {
        "MISSING_BOTH",
        "MISSING_EFFECTS",
        "MISSING_ROW",
        "FILENAME_MISMATCH",
        "PROBE_ERROR",
    }
)


if __name__ == "__main__":
    raise SystemExit(main())
