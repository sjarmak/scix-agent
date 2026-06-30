"""Unit tests for scripts/audit_schema_migrations.py.

The audit script reads files in migrations/, parses each one's primary DDL
target, probes the database to see whether the effects are present, and
joins that against schema_migrations to produce a drift report.

Bead: scix_experiments-l0ub.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_schema_migrations as audit  # noqa: E402


class TestParseMigration:
    def test_extracts_version_and_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "061_section_embeddings.sql"
        f.write_text("CREATE TABLE IF NOT EXISTS section_embeddings (bibcode TEXT);")
        parsed = audit.parse_migration(f)
        assert parsed is not None
        assert parsed.version == 61
        assert parsed.filename == "061_section_embeddings.sql"

    def test_skips_files_without_version_prefix(self, tmp_path: Path) -> None:
        f = tmp_path / "README.md"
        f.write_text("not a migration")
        assert audit.parse_migration(f) is None

    def test_auto_probe_finds_first_create_table(self, tmp_path: Path) -> None:
        f = tmp_path / "099_widgets.sql"
        f.write_text(
            textwrap.dedent(
                """\
                BEGIN;
                CREATE TABLE IF NOT EXISTS widgets (id INT);
                CREATE INDEX idx_widgets ON widgets(id);
                COMMIT;
                """
            )
        )
        parsed = audit.parse_migration(f)
        assert parsed is not None
        probe = parsed.auto_probe()
        assert probe is not None
        assert probe.kind == "table"
        assert probe.qualified_name == "public.widgets"

    def test_auto_probe_handles_qualified_table_names(self, tmp_path: Path) -> None:
        f = tmp_path / "099_extractions_columns.sql"
        f.write_text("ALTER TABLE staging.extractions ADD COLUMN x TEXT;")
        parsed = audit.parse_migration(f)
        assert parsed is not None
        probe = parsed.auto_probe()
        # Pure ALTER TABLE has no CREATE — auto_probe returns None and the
        # caller falls back to a manual probe.
        assert probe is None

    def test_auto_probe_strips_block_comments(self, tmp_path: Path) -> None:
        """A C-style /* CREATE TABLE foo */ inside a comment block must
        not fool the regex into capturing 'foo'."""
        f = tmp_path / "099_real.sql"
        f.write_text(
            textwrap.dedent(
                """\
                /*
                 * Idempotent: CREATE TABLE IF NOT EXISTS fake (id INT);
                 */
                CREATE TABLE IF NOT EXISTS real_table (id INT);
                """
            )
        )
        parsed = audit.parse_migration(f)
        assert parsed is not None
        assert parsed.auto_probe() == audit.Probe("table", "public.real_table")

    def test_auto_probe_ignores_create_in_line_comments(self, tmp_path: Path) -> None:
        """A comment like ``-- CREATE TABLE IF NOT EXISTS,`` (trailing comma)
        previously caused the regex to capture ``IF`` as the table name."""
        f = tmp_path / "099_real.sql"
        f.write_text(
            textwrap.dedent(
                """\
                -- Idempotent: CREATE TABLE IF NOT EXISTS,
                --   plus CREATE INDEX IF NOT EXISTS.
                CREATE TABLE IF NOT EXISTS real_table (id INT);
                """
            )
        )
        parsed = audit.parse_migration(f)
        assert parsed is not None
        probe = parsed.auto_probe()
        assert probe == audit.Probe("table", "public.real_table")

    def test_auto_probe_finds_create_view(self, tmp_path: Path) -> None:
        f = tmp_path / "099_v.sql"
        f.write_text("CREATE OR REPLACE VIEW public.v_widgets AS SELECT 1;")
        parsed = audit.parse_migration(f)
        assert parsed is not None
        probe = parsed.auto_probe()
        assert probe is not None
        assert probe.kind == "view"
        assert probe.qualified_name == "public.v_widgets"


class TestManualProbes:
    def test_manual_probes_cover_known_gaps(self) -> None:
        """Migrations whose primary effect is column-add or marker-insert
        must have a manual probe entry. Auto-parse cannot detect these."""
        # 060 adds columns to staging.extractions — no CREATE TABLE.
        assert 60 in audit.MANUAL_PROBES
        probe = audit.MANUAL_PROBES[60]
        assert probe.kind == "column"
        assert probe.qualified_name == "staging.extractions.section_name"

        # 058 adds papers.correction_events column — no CREATE TABLE.
        assert 58 in audit.MANUAL_PROBES
        assert audit.MANUAL_PROBES[58].kind == "column"

        # 066 (was 056_intent_populate) is a marker INSERT into ingest_log.
        assert 66 in audit.MANUAL_PROBES
        assert audit.MANUAL_PROBES[66].kind == "marker"


class TestAuditRowStatus:
    def test_status_ok_when_in_db_and_effects_present(self) -> None:
        row = audit.AuditRow(
            version=50,
            filename="050_x.sql",
            in_db=True,
            in_db_filename="050_x.sql",
            effects_present=True,
            probe=audit.Probe("table", "public.x"),
        )
        assert row.status == "OK"

    def test_status_missing_row_when_effects_present_no_db_row(self) -> None:
        row = audit.AuditRow(
            version=62,
            filename="062_paper_claims.sql",
            in_db=False,
            in_db_filename=None,
            effects_present=True,
            probe=audit.Probe("table", "public.paper_claims"),
        )
        assert row.status == "MISSING_ROW"

    def test_status_missing_effects_when_db_row_no_effects(self) -> None:
        row = audit.AuditRow(
            version=99,
            filename="099_x.sql",
            in_db=True,
            in_db_filename="099_x.sql",
            effects_present=False,
            probe=audit.Probe("table", "public.x"),
        )
        assert row.status == "MISSING_EFFECTS"

    def test_status_filename_mismatch(self) -> None:
        row = audit.AuditRow(
            version=63,
            filename="063_section_entities.sql",
            in_db=True,
            in_db_filename="063_section_bm25.sql",
            effects_present=False,
            probe=audit.Probe("table", "public.section_entities"),
        )
        assert row.status == "FILENAME_MISMATCH"

    def test_status_unknown_when_no_probe(self) -> None:
        row = audit.AuditRow(
            version=99,
            filename="099_x.sql",
            in_db=False,
            in_db_filename=None,
            effects_present=None,
            probe=None,
        )
        assert row.status == "UNKNOWN"

    def test_status_probe_error_when_probe_set_but_effects_none(self) -> None:
        """A swallowed exception in run_audit sets effects_present=None.
        Without a dedicated PROBE_ERROR status this masquerades as
        MISSING_EFFECTS (when in_db=True) or MISSING_BOTH (when not),
        producing false drift alerts."""
        row = audit.AuditRow(
            version=99,
            filename="099_x.sql",
            in_db=True,
            in_db_filename="099_x.sql",
            effects_present=None,
            probe=audit.Probe("table", "public.x"),
        )
        assert row.status == "PROBE_ERROR"


class TestMarkerProbeShape:
    """Defense-in-depth: marker probe SQL is built via f-string. Validate
    that the regex rejects shapes that would be unsafe to interpolate."""

    def test_accepts_canonical_shape(self) -> None:
        assert audit._MARKER_SHAPE_RE.match(
            "ingest_log WHERE filename='intent_backfill:citation_contexts'"
        )

    def test_accepts_bare_table_name(self) -> None:
        assert audit._MARKER_SHAPE_RE.match("ingest_log")

    def test_rejects_semicolon(self) -> None:
        assert not audit._MARKER_SHAPE_RE.match("ingest_log; DROP TABLE papers; --")

    def test_rejects_drop_inside_where(self) -> None:
        assert not audit._MARKER_SHAPE_RE.match("ingest_log WHERE 1=1; DROP TABLE papers; --")


class TestProbeDB:
    """Live DB probes — skipped without SCIX_TEST_DSN.

    Production DSN is never used here; the production-DSN guard in db.py
    rejects writes, but probes are read-only so they are technically safe.
    We still gate on SCIX_TEST_DSN to follow CLAUDE.md §Testing.
    """

    @pytest.fixture(autouse=True)
    def require_test_dsn(self) -> None:
        if not os.environ.get("SCIX_TEST_DSN"):
            pytest.skip("SCIX_TEST_DSN not set")

    def test_probe_table_detects_information_schema(self) -> None:
        import psycopg

        with psycopg.connect(os.environ["SCIX_TEST_DSN"]) as conn:
            assert (
                audit.probe_target(
                    conn,
                    audit.Probe("table", "information_schema.tables"),
                )
                is True
            )

    def test_probe_table_returns_false_for_missing(self) -> None:
        import psycopg

        with psycopg.connect(os.environ["SCIX_TEST_DSN"]) as conn:
            assert (
                audit.probe_target(
                    conn,
                    audit.Probe("table", "public.this_table_does_not_exist_xyz"),
                )
                is False
            )

    def test_probe_column_returns_true_for_known_column(self) -> None:
        import psycopg

        with psycopg.connect(os.environ["SCIX_TEST_DSN"]) as conn:
            # information_schema.tables.table_name always exists in any pg.
            assert (
                audit.probe_target(
                    conn,
                    audit.Probe("column", "information_schema.tables.table_name"),
                )
                is True
            )

    def test_probe_column_returns_false_for_missing(self) -> None:
        import psycopg

        with psycopg.connect(os.environ["SCIX_TEST_DSN"]) as conn:
            assert (
                audit.probe_target(
                    conn,
                    audit.Probe(
                        "column",
                        "information_schema.tables.no_such_column_xyz",
                    ),
                )
                is False
            )
