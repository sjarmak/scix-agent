"""Unit tests for ``scripts/backfill_part_of_inheritance.py``.

The tests use an in-memory fake ``psycopg`` connection that understands the
small set of statements the backfill issues (fetch part_of edges, fetch
surfaces, materialize a tsquery from a SELECT, INSERT into document_entities
with a ``RETURNING``/``ON CONFLICT`` shape). The real database is never
touched; no ``SCIX_TEST_DSN`` is required.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any, Optional

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_part_of_inheritance.py"
_spec = importlib.util.spec_from_file_location(
    "backfill_part_of_inheritance", _SCRIPT_PATH
)
backfill_mod = importlib.util.module_from_spec(_spec)
sys.modules["backfill_part_of_inheritance"] = backfill_mod
assert _spec.loader is not None
_spec.loader.exec_module(backfill_mod)


# ---------------------------------------------------------------------------
# Surface-form classifier
# ---------------------------------------------------------------------------


class TestIsSpecificSurface:
    """Disambiguator policy — bare acronyms reject, long forms accept."""

    @pytest.mark.parametrize(
        "surface",
        [
            "Chandra ACIS",  # multi-word
            "Chandra/ACIS",  # slash-prefixed
            "Advanced CCD Imaging Spectrometer",  # long form
            "NIRSpec",  # CamelCase >=6 chars
            "NIRCam",
            "LSSTCam",
            "Hubble Space Telescope",  # long mission name
            "ALMA Band 3",  # multi-word with digit
            "HST/ACS",  # slash
        ],
    )
    def test_specific(self, surface: str) -> None:
        assert backfill_mod.is_specific_surface(surface)

    @pytest.mark.parametrize(
        "surface",
        [
            "ACIS",  # bare uppercase, 4 chars
            "ACS",  # bare uppercase, 3 chars
            "OM",  # 2 chars
            "IRS",
            "FGS",
            "COS",
            "HRC",
            "RGS",
            "EPIC",  # bare uppercase 4 chars — skipped
            "MIPS",
            "IRAC",
            "BOSS",
            # 5-6 char all-uppercase acronyms are also rejected: the english
            # stemmer mangles many ('BATSE' -> 'bats') and the per-token risk
            # isn't worth the recall. Long-form aliases ('Burst And Transient
            # Source Experiment') carry the precision-safe match.
            "NIRISS",
            "NICMOS",
            "BATSE",
            "LASCO",
            "SUMER",
            "APOGEE",
        ],
    )
    def test_not_specific(self, surface: str) -> None:
        assert not backfill_mod.is_specific_surface(surface)

    def test_empty_string_rejected(self) -> None:
        assert not backfill_mod.is_specific_surface("")
        assert not backfill_mod.is_specific_surface("   ")


# ---------------------------------------------------------------------------
# Quote helper
# ---------------------------------------------------------------------------


class TestQuote:
    def test_simple(self) -> None:
        assert backfill_mod._quote("foo") == "'foo'"

    def test_doubles_single_quote(self) -> None:
        assert backfill_mod._quote("O'Brien") == "'O''Brien'"

    def test_does_not_escape_other_chars(self) -> None:
        # Backslashes and slashes pass through; only single-quote needs doubling.
        assert backfill_mod._quote("HST/ACS") == "'HST/ACS'"


# ---------------------------------------------------------------------------
# tsquery builder
# ---------------------------------------------------------------------------


class TestBuildTsqueryOr:
    def test_returns_none_for_empty(self) -> None:
        assert backfill_mod.build_tsquery_or([]) is None

    def test_single_surface(self) -> None:
        out = backfill_mod.build_tsquery_or(["NIRSpec"])
        assert out == "phraseto_tsquery('english', 'NIRSpec')"

    def test_multiple_surfaces_or_joined(self) -> None:
        out = backfill_mod.build_tsquery_or(["Chandra ACIS", "Chandra/ACIS"])
        assert out == (
            "phraseto_tsquery('english', 'Chandra ACIS') || "
            "phraseto_tsquery('english', 'Chandra/ACIS')"
        )

    def test_quote_in_surface_doubled(self) -> None:
        out = backfill_mod.build_tsquery_or(["Hubble's"])
        assert out == "phraseto_tsquery('english', 'Hubble''s')"


# ---------------------------------------------------------------------------
# Fake connection / cursor (just enough for the backfill script)
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, conn: "FakeConnection") -> None:
        self._conn = conn
        self._last_rows: list[tuple[Any, ...]] = []
        self.rowcount: int = 0

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Optional[dict[str, Any]] = None) -> None:
        params = params or {}
        normalized = " ".join(sql.split())
        self._conn.executed.append((normalized, dict(params)))

        if "FROM entity_relationships" in normalized:
            self._last_rows = self._conn.fetch_part_of_edges(params)
            self.rowcount = len(self._last_rows)
            return

        if "FROM entities WHERE id" in normalized and "UNION" in normalized:
            self._last_rows = self._conn.fetch_surfaces(params)
            self.rowcount = len(self._last_rows)
            return

        if normalized.startswith("SELECT (phraseto_tsquery"):
            # The backfill materializes the OR-joined tsquery via
            # SELECT (... )::text and binds the result. The fake just
            # echoes the inner expression.
            inner = sql.strip()[len("SELECT (") : -len(")::text")]
            self._last_rows = [(inner.strip(),)]
            self.rowcount = 1
            return

        if (
            "INSERT INTO document_entities" in normalized
            and "matched AS" in normalized
        ):
            inserted = self._conn.insert_instrument_doc_entities(params)
            self.rowcount = inserted
            self._last_rows = []
            return

        if (
            "INSERT INTO document_entities" in normalized
            and "FROM document_entities de" in normalized
        ):
            inserted = self._conn.insert_parent_doc_entities(params)
            self.rowcount = inserted
            self._last_rows = []
            return

        raise AssertionError(
            f"Unexpected SQL in test: {normalized[:140]} (params={params!r})"
        )

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        if not self._last_rows:
            return None
        return self._last_rows[0]

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._last_rows
        self._last_rows = []
        return rows


class FakeConnection:
    """Models the rows the backfill reads/writes."""

    def __init__(
        self,
        *,
        edges: Optional[list[dict[str, Any]]] = None,
        surfaces_by_entity: Optional[dict[int, list[str]]] = None,
        papers_matching_tsquery: Optional[dict[str, list[str]]] = None,
        existing_doc_entities: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.edges = edges or []
        self.surfaces_by_entity = surfaces_by_entity or {}
        # Map a tsquery string -> list of bibcodes that match it. The
        # tests register expected bibcode sets per (entity_id) by passing
        # a tsquery string that the fake will match exactly.
        self.papers_matching_tsquery = papers_matching_tsquery or {}
        self.doc_entities: list[dict[str, Any]] = list(existing_doc_entities or [])

        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self, *args: Any, **kwargs: Any) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1
        # Emulate full transactional rollback for the test surface.
        self.doc_entities = [d for d in self.doc_entities if d.get("_committed", False)]

    def close(self) -> None:
        self.closed = True

    # --- emulated SQL ----------------------------------------------------

    def fetch_part_of_edges(
        self, params: dict[str, Any]
    ) -> list[tuple[Any, ...]]:
        sources = params.get("sources") or []
        rows: list[tuple[Any, ...]] = []
        for e in self.edges:
            if not sources or e.get("source") in sources:
                rows.append(
                    (
                        e["instrument_id"],
                        e["instrument_name"],
                        e["mission_id"],
                        e["mission_name"],
                    )
                )
        return rows

    def fetch_surfaces(self, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        eid = params["entity_id"]
        surfaces = self.surfaces_by_entity.get(eid, [])
        return [(s,) for s in surfaces]

    def insert_instrument_doc_entities(self, params: dict[str, Any]) -> int:
        bibcodes = self.papers_matching_tsquery.get(params["tsquery"], [])
        inserted = 0
        for bc in bibcodes:
            key = (
                bc,
                params["entity_id"],
                params["link_type"],
                params["tier"],
            )
            if any(
                (d["bibcode"], d["entity_id"], d["link_type"], d["tier"]) == key
                for d in self.doc_entities
            ):
                continue
            self.doc_entities.append(
                {
                    "bibcode": bc,
                    "entity_id": params["entity_id"],
                    "link_type": params["link_type"],
                    "tier": params["tier"],
                    "tier_version": params["tier_version"],
                    "confidence": params["confidence"],
                    "match_method": params["match_method"],
                    "evidence": json.loads(params["evidence"]),
                }
            )
            inserted += 1
        return inserted

    def insert_parent_doc_entities(self, params: dict[str, Any]) -> int:
        instrument_rows = [
            d
            for d in self.doc_entities
            if d["entity_id"] == params["instrument_id"]
            and d["link_type"] == params["instrument_link_type"]
            and d["tier"] == params["tier"]
        ]
        inserted = 0
        for src in instrument_rows:
            key = (
                src["bibcode"],
                params["parent_id"],
                params["link_type"],
                params["tier"],
            )
            if any(
                (d["bibcode"], d["entity_id"], d["link_type"], d["tier"]) == key
                for d in self.doc_entities
            ):
                continue
            self.doc_entities.append(
                {
                    "bibcode": src["bibcode"],
                    "entity_id": params["parent_id"],
                    "link_type": params["link_type"],
                    "tier": params["tier"],
                    "tier_version": params["tier_version"],
                    "confidence": params["confidence"],
                    "match_method": params["match_method"],
                    "evidence": json.loads(params["evidence"]),
                }
            )
            inserted += 1
        return inserted


# ---------------------------------------------------------------------------
# End-to-end backfill tests
# ---------------------------------------------------------------------------


def _tsquery_for(surfaces: list[str]) -> str:
    """Render the tsquery the script will materialize for ``surfaces``."""
    return backfill_mod.build_tsquery_or(surfaces) or ""


class TestRunBackfill:
    def test_inserts_instrument_and_parent_rows(self) -> None:
        # NIRSpec has a CamelCase >=6 char canonical (specific) plus a
        # multi-word alias "JWST NIRSpec" (specific).
        nirspec_id = 100
        jwst_id = 200
        nirspec_surfaces = ["NIRSpec", "JWST NIRSpec"]
        nirspec_papers = ["2023A&A...1...1N", "2023ApJ...2....N"]
        conn = FakeConnection(
            edges=[
                {
                    "instrument_id": nirspec_id,
                    "instrument_name": "NIRSpec",
                    "mission_id": jwst_id,
                    "mission_name": "James Webb Space Telescope",
                    "source": "flagship_seed",
                }
            ],
            surfaces_by_entity={nirspec_id: nirspec_surfaces},
            papers_matching_tsquery={
                _tsquery_for(nirspec_surfaces): nirspec_papers,
            },
        )

        stats = backfill_mod.run_backfill(conn, dry_run=False)

        assert stats.edges_processed == 1
        assert stats.instruments_with_surfaces == 1
        assert stats.instruments_skipped_no_surfaces == 0
        assert stats.instrument_rows_inserted == 2
        assert stats.parent_rows_inserted == 2

        # Both instrument rows landed.
        instr_rows = [d for d in conn.doc_entities if d["entity_id"] == nirspec_id]
        assert len(instr_rows) == 2
        assert {d["bibcode"] for d in instr_rows} == set(nirspec_papers)
        for d in instr_rows:
            assert d["link_type"] == "abstract_match"
            assert d["match_method"] == "part_of_backfill_tsv"
            assert d["evidence"]["matched_surfaces"] == nirspec_surfaces

        # Mission rows mirror via link_type='inherited'.
        parent_rows = [d for d in conn.doc_entities if d["entity_id"] == jwst_id]
        assert len(parent_rows) == 2
        assert {d["bibcode"] for d in parent_rows} == set(nirspec_papers)
        for d in parent_rows:
            assert d["link_type"] == "inherited"
            assert d["match_method"] == "part_of_inheritance"
            assert d["evidence"]["via_instrument_id"] == nirspec_id
            assert d["evidence"]["via_instrument_name"] == "NIRSpec"

        assert conn.commits == 1
        assert conn.rollbacks == 0

    def test_skips_instrument_with_only_bare_acronym_aliases(self) -> None:
        # Hypothetical "ZZ" instrument with only short bare aliases.
        conn = FakeConnection(
            edges=[
                {
                    "instrument_id": 300,
                    "instrument_name": "ZZ",
                    "mission_id": 400,
                    "mission_name": "Some Mission",
                    "source": "flagship_seed",
                }
            ],
            surfaces_by_entity={300: ["ZZ", "AB", "CD"]},
            papers_matching_tsquery={},
        )
        stats = backfill_mod.run_backfill(conn, dry_run=False)
        assert stats.instruments_skipped_no_surfaces == 1
        assert stats.instruments_with_surfaces == 0
        assert stats.instrument_rows_inserted == 0
        assert stats.parent_rows_inserted == 0
        # No doc_entities rows were created.
        assert conn.doc_entities == []

    def test_idempotent_second_run(self) -> None:
        # Run twice; the second run should be a no-op.
        nirspec_id = 100
        jwst_id = 200
        surfaces = ["NIRSpec"]
        papers = ["2023A&A...1...1N"]
        conn = FakeConnection(
            edges=[
                {
                    "instrument_id": nirspec_id,
                    "instrument_name": "NIRSpec",
                    "mission_id": jwst_id,
                    "mission_name": "James Webb Space Telescope",
                    "source": "flagship_seed",
                }
            ],
            surfaces_by_entity={nirspec_id: surfaces},
            papers_matching_tsquery={_tsquery_for(surfaces): papers},
        )

        first = backfill_mod.run_backfill(conn, dry_run=False)
        assert first.instrument_rows_inserted == 1
        assert first.parent_rows_inserted == 1

        # Mark previously-inserted rows as committed for the rollback model.
        for d in conn.doc_entities:
            d["_committed"] = True

        second = backfill_mod.run_backfill(conn, dry_run=False)
        assert second.instrument_rows_inserted == 0
        assert second.parent_rows_inserted == 0
        assert second.edges_processed == 1
        assert second.instruments_with_surfaces == 1

    def test_dry_run_rolls_back(self) -> None:
        nirspec_id = 100
        jwst_id = 200
        surfaces = ["NIRSpec"]
        papers = ["2023A&A...1...1N"]
        conn = FakeConnection(
            edges=[
                {
                    "instrument_id": nirspec_id,
                    "instrument_name": "NIRSpec",
                    "mission_id": jwst_id,
                    "mission_name": "James Webb Space Telescope",
                    "source": "flagship_seed",
                }
            ],
            surfaces_by_entity={nirspec_id: surfaces},
            papers_matching_tsquery={_tsquery_for(surfaces): papers},
        )

        stats = backfill_mod.run_backfill(conn, dry_run=True)

        # Stats reflect the would-have-inserted counts.
        assert stats.instrument_rows_inserted == 1
        assert stats.parent_rows_inserted == 1
        # ... but the fake's state was rolled back to empty.
        assert conn.doc_entities == []
        assert conn.rollbacks == 1
        assert conn.commits == 0

    def test_handles_zero_matched_papers(self) -> None:
        nirspec_id = 100
        jwst_id = 200
        surfaces = ["NIRSpec"]
        conn = FakeConnection(
            edges=[
                {
                    "instrument_id": nirspec_id,
                    "instrument_name": "NIRSpec",
                    "mission_id": jwst_id,
                    "mission_name": "James Webb Space Telescope",
                    "source": "flagship_seed",
                }
            ],
            surfaces_by_entity={nirspec_id: surfaces},
            papers_matching_tsquery={_tsquery_for(surfaces): []},
        )
        stats = backfill_mod.run_backfill(conn, dry_run=False)
        # Surfaces existed, query ran, but no papers matched.
        assert stats.instruments_with_surfaces == 1
        assert stats.instrument_rows_inserted == 0
        assert stats.parent_rows_inserted == 0
        assert conn.doc_entities == []

    def test_filters_by_source(self) -> None:
        # An edge from an unrelated source should be filtered out.
        conn = FakeConnection(
            edges=[
                {
                    "instrument_id": 1,
                    "instrument_name": "Foo",
                    "mission_id": 2,
                    "mission_name": "Bar",
                    "source": "ssodnet",
                },
            ],
            surfaces_by_entity={1: ["FooLong Instrument"]},
            papers_matching_tsquery={
                _tsquery_for(["FooLong Instrument"]): ["b1"],
            },
        )
        # Default sources tuple doesn't include 'ssodnet'.
        stats = backfill_mod.run_backfill(conn, dry_run=False)
        assert stats.edges_processed == 0
        assert conn.doc_entities == []

        # Explicitly opting in picks it up.
        stats2 = backfill_mod.run_backfill(conn, dry_run=False, sources=("ssodnet",))
        assert stats2.edges_processed == 1
        assert stats2.instrument_rows_inserted == 1


# ---------------------------------------------------------------------------
# Production guard
# ---------------------------------------------------------------------------


class TestProdGuard:
    def test_refuses_prod_dsn_without_allow_prod(self) -> None:
        with pytest.raises(SystemExit) as exc:
            backfill_mod.enforce_prod_guard(
                dsn="dbname=scix",
                allow_prod=False,
                env={"INVOCATION_ID": "abc"},
            )
        assert exc.value.code == 2

    def test_refuses_allow_prod_without_systemd_scope(self) -> None:
        with pytest.raises(SystemExit) as exc:
            backfill_mod.enforce_prod_guard(
                dsn="dbname=scix",
                allow_prod=True,
                env={},  # no INVOCATION_ID
            )
        assert exc.value.code == 2

    def test_allows_prod_dsn_with_allow_prod_inside_systemd(self) -> None:
        # Should not raise.
        backfill_mod.enforce_prod_guard(
            dsn="dbname=scix",
            allow_prod=True,
            env={"INVOCATION_ID": "abc"},
        )

    def test_allows_test_dsn_without_allow_prod(self) -> None:
        backfill_mod.enforce_prod_guard(
            dsn="dbname=scix_test",
            allow_prod=False,
            env={},
        )


# ---------------------------------------------------------------------------
# DSN resolver
# ---------------------------------------------------------------------------


class TestResolveDsn:
    def test_bare_dbname(self) -> None:
        assert backfill_mod._resolve_dsn("scix_test") == "dbname=scix_test"

    def test_full_keyvalue(self) -> None:
        assert (
            backfill_mod._resolve_dsn("dbname=foo host=bar")
            == "dbname=foo host=bar"
        )

    def test_uri(self) -> None:
        assert (
            backfill_mod._resolve_dsn("postgresql://u:p@h/db")
            == "postgresql://u:p@h/db"
        )
