"""Unit tests for ``scripts/seed_flagship_instruments.py``.

These tests use a lightweight in-memory fake ``psycopg`` connection that
understands just enough SQL (INSERT / SELECT / UPDATE against the small set
of tables touched by the seeder) to simulate the ``ON CONFLICT DO NOTHING
RETURNING id`` semantics the script relies on for idempotency. The real
database is never touched; no ``SCIX_TEST_DSN`` is required.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the script module by path (pytest pythonpath already includes
# scripts/, but importing via spec keeps the test self-contained and
# robust to env differences).
_SCRIPT_PATH = REPO_ROOT / "scripts" / "seed_flagship_instruments.py"
_spec = importlib.util.spec_from_file_location("seed_flagship_instruments", _SCRIPT_PATH)
seed_mod = importlib.util.module_from_spec(_spec)
sys.modules["seed_flagship_instruments"] = seed_mod
assert _spec.loader is not None
_spec.loader.exec_module(seed_mod)


# ---------------------------------------------------------------------------
# Fake connection / cursor
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal SQL executor — just enough for the seed script's statements."""

    def __init__(self, conn: "FakeConnection") -> None:
        self._conn = conn
        self._last_rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Optional[dict[str, Any]] = None) -> None:
        self._conn.executed.append((sql, dict(params or {})))
        normalized = " ".join(sql.split())
        params = params or {}

        if normalized.startswith("INSERT INTO entities"):
            self._last_rows = self._conn.insert_entity(params)
        elif normalized.startswith("SELECT id FROM entities WHERE canonical_name"):
            self._last_rows = self._conn.lookup_entity_by_key(params)
        elif normalized.startswith(
            "SELECT id, source FROM entities WHERE canonical_name"
        ):
            self._last_rows = self._conn.lookup_entity_any_source(params)
        elif normalized.startswith("INSERT INTO entity_aliases"):
            self._last_rows = self._conn.insert_alias(params)
        elif normalized.startswith("INSERT INTO entity_relationships"):
            self._last_rows = self._conn.insert_relationship(params)
        else:
            raise AssertionError(f"Unexpected SQL in test: {normalized[:120]}")

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        if not self._last_rows:
            return None
        row = self._last_rows[0]
        self._last_rows = self._last_rows[1:]
        return row


class FakeConnection:
    """In-memory fake of the tables the seeder touches."""

    def __init__(self, *, preseeded_entities: Optional[list[dict[str, Any]]] = None) -> None:
        # entities: list of dicts with id, canonical_name, entity_type,
        # source, discipline.
        self.entities: list[dict[str, Any]] = []
        self.aliases: list[dict[str, Any]] = []
        self.relationships: list[dict[str, Any]] = []
        self._next_entity_id = 1
        self._next_rel_id = 1
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

        for row in preseeded_entities or []:
            self.entities.append(
                {
                    "id": self._next_entity_id,
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                    "source": row["source"],
                    "discipline": row.get("discipline"),
                }
            )
            self._next_entity_id += 1

    # --- cursor -----------------------------------------------------------

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        # Full transactional rollback: discard all uncommitted state.
        # The seeder only commits once (at the end of seed()), so rolling
        # back here models the dry-run path precisely.
        self.rollbacks += 1
        self.entities = [
            e for e in self.entities if e.get("_committed", False)
        ]
        self.aliases = [
            a for a in self.aliases if a.get("_committed", False)
        ]
        self.relationships = [
            r for r in self.relationships if r.get("_committed", False)
        ]

    def close(self) -> None:
        self.closed = True

    # --- emulated SQL -----------------------------------------------------

    def _find_entity(
        self, canonical_name: str, entity_type: str, source: str
    ) -> Optional[dict[str, Any]]:
        for e in self.entities:
            if (
                e["canonical_name"] == canonical_name
                and e["entity_type"] == entity_type
                and e["source"] == source
            ):
                return e
        return None

    def insert_entity(self, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        existing = self._find_entity(
            params["canonical_name"], params["entity_type"], params["source"]
        )
        if existing is not None:
            # ON CONFLICT DO NOTHING RETURNING id → empty rowset.
            return []
        row = {
            "id": self._next_entity_id,
            "canonical_name": params["canonical_name"],
            "entity_type": params["entity_type"],
            "source": params["source"],
            "discipline": params.get("discipline"),
        }
        self._next_entity_id += 1
        self.entities.append(row)
        return [(row["id"],)]

    def lookup_entity_by_key(self, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        existing = self._find_entity(
            params["canonical_name"], params["entity_type"], params["source"]
        )
        return [(existing["id"],)] if existing else []

    def lookup_entity_any_source(
        self, params: dict[str, Any]
    ) -> list[tuple[Any, ...]]:
        matches = [
            e
            for e in self.entities
            if e["canonical_name"] == params["canonical_name"]
            and e["entity_type"] == params["entity_type"]
        ]
        if not matches:
            return []
        preferred = params.get("preferred")
        matches.sort(key=lambda e: (0 if e["source"] == preferred else 1, e["id"]))
        top = matches[0]
        return [(top["id"], top["source"])]

    def insert_alias(self, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        for a in self.aliases:
            if (
                a["entity_id"] == params["entity_id"]
                and a["alias"] == params["alias"]
            ):
                return []
        self.aliases.append(
            {
                "entity_id": params["entity_id"],
                "alias": params["alias"],
                "alias_source": params["alias_source"],
            }
        )
        return [(params["entity_id"],)]

    def insert_relationship(self, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        for r in self.relationships:
            if (
                r["subject_entity_id"] == params["subject"]
                and r["predicate"] == params["predicate"]
                and r["object_entity_id"] == params["object"]
            ):
                return []
        rel_id = self._next_rel_id
        self._next_rel_id += 1
        self.relationships.append(
            {
                "id": rel_id,
                "subject_entity_id": params["subject"],
                "predicate": params["predicate"],
                "object_entity_id": params["object"],
                "source": params.get("source"),
            }
        )
        return [(rel_id,)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSeed:
    def test_basic_seed_creates_entities_and_aliases(self) -> None:
        conn = FakeConnection()
        stats = seed_mod.seed(conn, dry_run=False)

        assert stats.entities_created > 0
        assert stats.aliases_created > 0
        assert conn.commits == 1
        assert conn.rollbacks == 0

        # At least 25 sub-instruments/missions from CURATED_INSTRUMENTS —
        # parents are additional entities beyond that.
        assert len(seed_mod.CURATED_INSTRUMENTS) >= 25

    def test_idempotent_second_run_is_noop(self) -> None:
        conn = FakeConnection()
        seed_mod.seed(conn, dry_run=False)

        entity_count = len(conn.entities)
        alias_count = len(conn.aliases)
        rel_count = len(conn.relationships)

        # Re-run on the same state — no new rows.
        stats2 = seed_mod.seed(conn, dry_run=False)

        assert len(conn.entities) == entity_count
        assert len(conn.aliases) == alias_count
        assert len(conn.relationships) == rel_count
        assert stats2.entities_created == 0
        assert stats2.aliases_created == 0
        assert stats2.relationships_created == 0
        # Every parent + instrument should now count as "existing".
        assert stats2.entities_existing > 0

    def test_dry_run_rolls_back(self) -> None:
        conn = FakeConnection()
        stats = seed_mod.seed(conn, dry_run=True)

        # Stats should reflect would-have-inserted rows.
        assert stats.entities_created > 0
        # But the fake connection's state was rolled back to empty.
        assert conn.entities == []
        assert conn.aliases == []
        assert conn.relationships == []
        assert conn.rollbacks == 1
        assert conn.commits == 0

    def test_alias_registered_for_jwst_nirspec(self) -> None:
        conn = FakeConnection()
        seed_mod.seed(conn, dry_run=False)

        # Locate the NIRSpec instrument entity by canonical name.
        nirspec_rows = [
            e
            for e in conn.entities
            if e["canonical_name"] == "NIRSpec" and e["entity_type"] == "instrument"
        ]
        assert len(nirspec_rows) == 1
        nirspec_id = nirspec_rows[0]["id"]

        # The "JWST NIRSpec" alias (lowercased match) must be present.
        aliases_for_nirspec = {
            a["alias"].lower() for a in conn.aliases if a["entity_id"] == nirspec_id
        }
        assert "jwst nirspec" in aliases_for_nirspec
        assert "nirspec" in aliases_for_nirspec

    def test_parent_relationship_nirspec_part_of_jwst(self) -> None:
        conn = FakeConnection()
        seed_mod.seed(conn, dry_run=False)

        jwst_rows = [
            e
            for e in conn.entities
            if e["canonical_name"] == "James Webb Space Telescope"
            and e["entity_type"] == "mission"
        ]
        assert len(jwst_rows) == 1
        jwst_id = jwst_rows[0]["id"]

        nirspec_id = next(
            e["id"]
            for e in conn.entities
            if e["canonical_name"] == "NIRSpec" and e["entity_type"] == "instrument"
        )

        matching = [
            r
            for r in conn.relationships
            if r["subject_entity_id"] == nirspec_id
            and r["object_entity_id"] == jwst_id
            and r["predicate"] == "part_of"
        ]
        assert len(matching) == 1

    def test_reuses_existing_curated_parent(self) -> None:
        # Simulate a real DB where JWST already exists under the
        # curated_flagship_v1 source. The seeder should NOT create a
        # duplicate under source='flagship_seed' — it should reuse.
        conn = FakeConnection(
            preseeded_entities=[
                {
                    "canonical_name": "James Webb Space Telescope",
                    "entity_type": "mission",
                    "source": "curated_flagship_v1",
                }
            ]
        )
        seed_mod.seed(conn, dry_run=False)

        jwst_rows = [
            e
            for e in conn.entities
            if e["canonical_name"] == "James Webb Space Telescope"
            and e["entity_type"] == "mission"
        ]
        sources = {e["source"] for e in jwst_rows}
        assert sources == {"curated_flagship_v1"}

        # And part_of edges should point at that reused row.
        jwst_id = jwst_rows[0]["id"]
        parent_edges = [r for r in conn.relationships if r["object_entity_id"] == jwst_id]
        assert len(parent_edges) >= 5  # NIRSpec, NIRCam, MIRI, NIRISS, FGS

    def test_many_instruments_with_parents(self) -> None:
        # Sanity: the curated list should include diverse missions.
        with_parents = [i for i in seed_mod.CURATED_INSTRUMENTS if i.parent is not None]
        assert len(with_parents) >= 15

        # Cover major missions: JWST, HST, Chandra, ALMA.
        parent_names = {i.parent.canonical_name for i in with_parents}
        assert "James Webb Space Telescope" in parent_names
        assert "Hubble Space Telescope" in parent_names
        assert "Chandra X-ray Observatory" in parent_names
        assert "Atacama Large Millimeter Array" in parent_names

    def test_all_entities_flagged_with_expected_source(self) -> None:
        conn = FakeConnection()
        seed_mod.seed(conn, dry_run=False)
        for e in conn.entities:
            # Everything created by this seeder is under flagship_seed.
            assert e["source"] == "flagship_seed"
        # All aliases also use the flagship_seed alias_source.
        for a in conn.aliases:
            assert a["alias_source"] == "flagship_seed"


# ---------------------------------------------------------------------------
# CLI entrypoint tests (resolve_dsn, argparse wiring)
# ---------------------------------------------------------------------------


class TestCli:
    def test_resolve_dsn_bare_dbname(self) -> None:
        assert seed_mod._resolve_dsn("scix_test") == "dbname=scix_test"

    def test_resolve_dsn_full_keyvalue(self) -> None:
        assert (
            seed_mod._resolve_dsn("dbname=foo host=bar")
            == "dbname=foo host=bar"
        )

    def test_resolve_dsn_uri(self) -> None:
        assert (
            seed_mod._resolve_dsn("postgresql://u:p@h/db")
            == "postgresql://u:p@h/db"
        )
