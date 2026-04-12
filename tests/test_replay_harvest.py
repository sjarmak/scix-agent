"""Integration tests for scripts.replay_harvest.

Covers round-tripping a snapshot file through the staging tables and back out
to JSON-Lines with zero diff.

Require SCIX_TEST_DSN pointing at a non-production database.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from helpers import is_production_dsn  # noqa: E402


def _load_replay_module():
    """Load scripts/replay_harvest.py as a module (it lives outside src/)."""
    path = REPO_ROOT / "scripts" / "replay_harvest.py"
    spec = importlib.util.spec_from_file_location("replay_harvest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay_harvest = _load_replay_module()


TEST_DSN = os.environ.get("SCIX_TEST_DSN")
TEST_SOURCE = "REPLAY_TEST_SRC"

pytestmark = pytest.mark.skipif(
    TEST_DSN is None or (TEST_DSN is not None and is_production_dsn(TEST_DSN)),
    reason="replay_harvest tests require SCIX_TEST_DSN pointing at a non-production DB",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    assert TEST_DSN is not None
    c = psycopg.connect(TEST_DSN)
    c.autocommit = False
    try:
        yield c
    finally:
        c.close()


def _cleanup(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM entities_staging WHERE source = %s", (TEST_SOURCE,))
        cur.execute(
            "DELETE FROM entity_aliases_staging WHERE source = %s",
            (TEST_SOURCE,),
        )
        cur.execute(
            "DELETE FROM entity_identifiers_staging WHERE source = %s",
            (TEST_SOURCE,),
        )
        cur.execute("DELETE FROM harvest_runs WHERE source = %s", (TEST_SOURCE,))
    conn.commit()


@pytest.fixture(autouse=True)
def _clean(conn):
    _cleanup(conn)
    yield
    _cleanup(conn)


@pytest.fixture
def sample_records() -> list[dict]:
    return [
        {
            "entity": {
                "canonical_name": "Alpha Catalog",
                "entity_type": "dataset",
                "discipline": "astronomy",
                "source": TEST_SOURCE,
                "source_version": "v1",
                "ambiguity_class": None,
                "link_policy": None,
                "properties": {"k": "v", "n": 1},
            },
            "aliases": [
                {"alias": "alpha-cat", "alias_source": "manual"},
                {"alias": "AC-1", "alias_source": None},
            ],
            "identifiers": [
                {"id_scheme": "doi", "external_id": "10.1/alpha", "is_primary": True},
            ],
        },
        {
            "entity": {
                "canonical_name": "Beta Instrument",
                "entity_type": "instrument",
                "discipline": None,
                "source": TEST_SOURCE,
                "source_version": None,
                "ambiguity_class": None,
                "link_policy": None,
                "properties": {},
            },
            "aliases": [],
            "identifiers": [],
        },
        {
            "entity": {
                "canonical_name": "Gamma Concept",
                "entity_type": "concept",
                "discipline": "physics",
                "source": TEST_SOURCE,
                "source_version": "2026-04",
                "ambiguity_class": None,
                "link_policy": None,
                "properties": {"note": "triple"},
            },
            "aliases": [{"alias": "gamma", "alias_source": "wiki"}],
            "identifiers": [
                {"id_scheme": "qid", "external_id": "Q42", "is_primary": False},
                {"id_scheme": "url", "external_id": "https://ex.org/g", "is_primary": False},
            ],
        },
    ]


def _write_snapshot(tmp_path: Path, source: str, date: str, records: list[dict]) -> Path:
    dir_ = tmp_path / source
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / f"{date}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplayLoad:
    def test_loads_records_into_staging(self, conn, tmp_path, sample_records):
        _write_snapshot(tmp_path, TEST_SOURCE, "2026-04-10", sample_records)

        run_id = replay_harvest.replay_snapshot(
            TEST_SOURCE,
            "2026-04-10",
            dsn=TEST_DSN,
            snapshots_root=tmp_path,
        )
        assert isinstance(run_id, int)
        assert run_id > 0

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM entities_staging WHERE staging_run_id = %s",
                (run_id,),
            )
            assert cur.fetchone()[0] == 3

            cur.execute(
                "SELECT COUNT(*) FROM entity_aliases_staging WHERE staging_run_id = %s",
                (run_id,),
            )
            # 2 aliases for Alpha + 0 for Beta + 1 for Gamma = 3
            assert cur.fetchone()[0] == 3

            cur.execute(
                "SELECT COUNT(*) FROM entity_identifiers_staging WHERE staging_run_id = %s",
                (run_id,),
            )
            # 1 for Alpha + 0 for Beta + 2 for Gamma = 3
            assert cur.fetchone()[0] == 3

            cur.execute("SELECT source, status FROM harvest_runs WHERE id = %s", (run_id,))
            src, status = cur.fetchone()
            assert src == TEST_SOURCE
            assert status == "replayed"


class TestReplayRoundTrip:
    def test_round_trip_produces_zero_diff(self, conn, tmp_path, sample_records):
        # 1. Write the original snapshot.
        original_path = _write_snapshot(tmp_path, TEST_SOURCE, "2026-04-11", sample_records)

        # 2. Replay into staging.
        run_id = replay_harvest.replay_snapshot(
            TEST_SOURCE,
            "2026-04-11",
            dsn=TEST_DSN,
            snapshots_root=tmp_path,
        )

        # 3. Dump back out to a new snapshot file.
        dump_path = tmp_path / "roundtrip.jsonl.gz"
        written = replay_harvest.dump_staging_to_snapshot(run_id, dump_path, dsn=TEST_DSN)
        assert written == len(sample_records)

        # 4. Parse both files and compare semantic content.
        def _load(path: Path) -> list[dict]:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]

        original = _load(original_path)
        dumped = _load(dump_path)

        # Normalize ordering (entities by canonical_name, aliases/identifiers sorted).
        def _normalize(records: list[dict]) -> list[dict]:
            def _norm_rec(r: dict) -> dict:
                ent = dict(r.get("entity") or {})
                # Ensure dict fields are comparable.
                ent["properties"] = ent.get("properties") or {}
                aliases = sorted(
                    (r.get("aliases") or []),
                    key=lambda a: (a.get("alias") or "", a.get("alias_source") or ""),
                )
                idents = sorted(
                    (r.get("identifiers") or []),
                    key=lambda i: (
                        i.get("id_scheme") or "",
                        i.get("external_id") or "",
                    ),
                )
                return {"entity": ent, "aliases": aliases, "identifiers": idents}

            return sorted(
                (_norm_rec(r) for r in records),
                key=lambda r: r["entity"].get("canonical_name") or "",
            )

        norm_original = _normalize(original)
        norm_dumped = _normalize(dumped)

        assert len(norm_original) == len(norm_dumped)
        for o, d in zip(norm_original, norm_dumped):
            # Entity fields
            o_ent = o["entity"]
            d_ent = d["entity"]
            for key in (
                "canonical_name",
                "entity_type",
                "discipline",
                "source",
                "source_version",
                "ambiguity_class",
                "link_policy",
            ):
                assert o_ent.get(key) == d_ent.get(
                    key
                ), f"{key}: {o_ent.get(key)!r} != {d_ent.get(key)!r}"
            assert o_ent["properties"] == d_ent["properties"]
            assert o["aliases"] == d["aliases"]
            assert o["identifiers"] == d["identifiers"]


class TestSnapshotPath:
    def test_snapshot_path_layout(self, tmp_path):
        path = replay_harvest.snapshot_path("VizieR", "2026-04-01", root=tmp_path)
        assert path == tmp_path / "VizieR" / "2026-04-01.jsonl.gz"

    def test_missing_snapshot_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(replay_harvest.iter_snapshot(tmp_path / "missing.jsonl.gz"))
