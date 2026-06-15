"""Tests for scripts/qdrant_outbox_sync.py (bead scix_experiments-8m0a).

Two layers:

* Unit tests (no DB, no Qdrant) cover the pure logic: point-id contract,
  latest-op-wins batch resolution, the upsert-then-delete Qdrant call shape,
  and the bounded retry.
* Integration tests (require ``SCIX_TEST_DSN`` + migration 070 applied to that
  database) seed ``papers``/``paper_embeddings``, let the trigger enqueue, and
  drive a real drain against a fake Qdrant client — verifying the trigger,
  drain/delete, DELETE-op handling, and the ``--backfill-since`` reconcile.

The Qdrant client is injected everywhere, so nothing here talks to a live
Qdrant. Integration tests skip cleanly when the test DSN or the migration is
absent.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import qdrant_outbox_sync as obx  # noqa: E402

from tests.helpers import get_test_dsn  # noqa: E402

_PREFIX = "test_obx_"


# ---------------------------------------------------------------------------
# Fake Qdrant client — records upsert/delete calls
# ---------------------------------------------------------------------------


class FakeQdrant:
    def __init__(self) -> None:
        self.upserted: dict[str, dict[str, Any]] = {}  # collection → {id: payload}
        self.vectors: dict[str, dict[str, Any]] = {}  # collection → {id: vector}
        self.deleted: dict[str, list[str]] = {}
        self.upsert_calls = 0
        self.delete_calls = 0

    def upsert(self, *, collection_name: str, points: list[Any], wait: bool = True) -> None:
        self.upsert_calls += 1
        bucket = self.upserted.setdefault(collection_name, {})
        vecs = self.vectors.setdefault(collection_name, {})
        for p in points:
            bucket[p.id] = p.payload
            vecs[p.id] = p.vector

    def delete(self, *, collection_name: str, points_selector: Any, wait: bool = True) -> None:
        self.delete_calls += 1
        self.deleted.setdefault(collection_name, []).extend(points_selector.points)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_point_id_matches_bulk_load_contract() -> None:
    # Must equal scripts/qdrant_full_load.py's uuid5(NAMESPACE_URL, bibcode)
    # or forward writes would create duplicate points instead of updating.
    bib = "2026ApJ...999..001X"
    assert obx.point_id(bib) == str(uuid.uuid5(uuid.NAMESPACE_URL, bib))
    assert obx.point_id(bib) == obx.point_id(bib)  # deterministic


def _row(id_: int, bibcode: str, op: str, t: int) -> obx.ClaimedRow:
    return obx.ClaimedRow(id=id_, bibcode=bibcode, op=op, enqueued_at=t)


def test_resolve_latest_op_delete_wins() -> None:
    rows = [
        _row(1, "A", "INSERT", 1),
        _row(2, "A", "UPDATE", 2),
        _row(3, "A", "DELETE", 3),  # newest → A is a delete
        _row(4, "B", "INSERT", 1),
    ]
    r = obx._resolve_batch(rows)
    assert r.delete_bibcodes == ["A"]
    assert r.upsert_bibcodes == ["B"]
    assert sorted(r.claimed_ids) == [1, 2, 3, 4]


def test_resolve_latest_op_update_after_delete_resurrects() -> None:
    rows = [
        _row(1, "A", "DELETE", 1),
        _row(2, "A", "INSERT", 2),  # re-created after delete → upsert wins
    ]
    r = obx._resolve_batch(rows)
    assert r.upsert_bibcodes == ["A"]
    assert r.delete_bibcodes == []


def test_apply_to_qdrant_upserts_then_deletes() -> None:
    from qdrant_client import models as qm

    client = FakeQdrant()
    pts = [qm.PointStruct(id=obx.point_id("A"), vector=[0.1, 0.2], payload={"bibcode": "A"})]
    obx._apply_to_qdrant(client, "coll", points=pts, delete_ids=[obx.point_id("Z")])

    assert client.upsert_calls == 1
    assert client.delete_calls == 1
    assert client.upserted["coll"][obx.point_id("A")] == {"bibcode": "A"}
    assert client.deleted["coll"] == [obx.point_id("Z")]


def test_apply_to_qdrant_skips_empty_calls() -> None:
    client = FakeQdrant()
    obx._apply_to_qdrant(client, "coll", points=[], delete_ids=[])
    assert client.upsert_calls == 0
    assert client.delete_calls == 0


def test_with_retry_succeeds_after_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(obx.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return "ok"

    assert obx._with_retry(flaky, what="upsert") == "ok"
    assert calls["n"] == 2


def test_with_retry_raises_after_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(obx.time, "sleep", lambda *_: None)

    def always_fails() -> None:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="failed after"):
        obx._with_retry(always_fails, what="delete")


def test_routing_has_indus() -> None:
    assert obx.ROUTING["indus"] == "scix_indus_v2_papers_s1"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def _outbox_present(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.embedding_outbox')")
        return cur.fetchone()[0] is not None


@pytest.fixture
def conn() -> Any:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN must be set to a non-production DSN")
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(dsn)
    register_vector(connection)
    if not _outbox_present(connection):
        connection.close()
        pytest.skip("migration 070 (embedding_outbox) not applied to test DB")
    _cleanup(connection)
    try:
        yield connection
    finally:
        _cleanup(connection)
        connection.close()


def _cleanup(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        # Delete embeddings first (fires DELETE-trigger enqueues), then purge
        # all outbox rows for the test bibcodes, then the papers.
        cur.execute("DELETE FROM paper_embeddings WHERE bibcode LIKE %s", (_PREFIX + "%",))
        cur.execute("DELETE FROM embedding_outbox WHERE bibcode LIKE %s", (_PREFIX + "%",))
        cur.execute("DELETE FROM papers WHERE bibcode LIKE %s", (_PREFIX + "%",))
    conn.commit()


def _seed_paper(conn: psycopg.Connection, bibcode: str, entry_date: str = "2026-06-11") -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO papers (bibcode, title, entry_date) VALUES (%s, %s, %s) "
            "ON CONFLICT (bibcode) DO NOTHING",
            (bibcode, "test paper " + bibcode, entry_date),
        )
    conn.commit()


def _vec(fill: float) -> list[float]:
    # paper_embeddings.embedding is vector(768) in the test DB.
    return [fill] * 768


def _seed_embedding(conn: psycopg.Connection, bibcode: str, vec: list[float]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO paper_embeddings (bibcode, model_name, embedding) "
            "VALUES (%s, 'indus', %s) "
            "ON CONFLICT (bibcode, model_name) DO UPDATE SET embedding = EXCLUDED.embedding",
            (bibcode, vec),
        )
    conn.commit()


def _outbox_count(conn: psycopg.Connection, bibcode_like: str = _PREFIX + "%") -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM embedding_outbox WHERE bibcode LIKE %s", (bibcode_like,))
        return cur.fetchone()[0]


@pytest.mark.integration
def test_trigger_enqueues_and_drain_upserts(conn: psycopg.Connection) -> None:
    bibs = [f"{_PREFIX}0001", f"{_PREFIX}0002"]
    for b in bibs:
        _seed_paper(conn, b)
        _seed_embedding(conn, b, _vec(0.1))

    # Trigger fired on each INSERT.
    assert _outbox_count(conn) == 2

    client = FakeQdrant()
    stats = obx.drain_model(conn, client, model_name="indus", collection="scix_indus_v2_papers_s1")

    assert stats.upserted == 2
    assert stats.deleted == 0
    # Both points upserted with the bulk-load id + bibcode payload.
    upserted = client.upserted["scix_indus_v2_papers_s1"]
    for b in bibs:
        assert upserted[obx.point_id(b)] == {"bibcode": b}
    # Queue drained.
    assert _outbox_count(conn) == 0


@pytest.mark.integration
def test_update_trigger_drains_new_vector(conn: psycopg.Connection) -> None:
    # The UPDATE branch of the trigger + the op='UPDATE' drain path: after an
    # embedding is re-written, exactly one UPDATE row must enqueue and the
    # drain must push the NEW vector (not the stale one) to Qdrant.
    b = f"{_PREFIX}upd1"
    _seed_paper(conn, b)
    _seed_embedding(conn, b, _vec(0.1))
    obx.drain_model(conn, FakeQdrant(), model_name="indus", collection="scix_indus_v2_papers_s1")
    assert _outbox_count(conn) == 0

    _seed_embedding(conn, b, _vec(0.9))  # ON CONFLICT DO UPDATE → UPDATE trigger
    assert _outbox_count(conn) == 1

    client = FakeQdrant()
    stats = obx.drain_model(conn, client, model_name="indus", collection="scix_indus_v2_papers_s1")
    assert stats.upserted == 1
    # vector(768) is float32, so compare approximately; the point is that the
    # NEW vector (0.9) reached Qdrant, not the stale 0.1.
    pushed = client.vectors["scix_indus_v2_papers_s1"][obx.point_id(b)]
    assert pushed == pytest.approx(_vec(0.9), abs=1e-5)
    assert _outbox_count(conn) == 0


@pytest.mark.integration
def test_delete_op_removes_point(conn: psycopg.Connection) -> None:
    b = f"{_PREFIX}del1"
    _seed_paper(conn, b)
    _seed_embedding(conn, b, _vec(0.5))

    # Drain the INSERT enqueue first.
    obx.drain_model(conn, FakeQdrant(), model_name="indus", collection="scix_indus_v2_papers_s1")
    assert _outbox_count(conn) == 0

    # Now delete the embedding row → DELETE trigger enqueues.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM paper_embeddings WHERE bibcode = %s", (b,))
    conn.commit()
    assert _outbox_count(conn) == 1

    client = FakeQdrant()
    stats = obx.drain_model(conn, client, model_name="indus", collection="scix_indus_v2_papers_s1")
    assert stats.deleted == 1
    assert client.deleted["scix_indus_v2_papers_s1"] == [obx.point_id(b)]
    assert _outbox_count(conn) == 0


@pytest.mark.integration
def test_backfill_since_reconciles_pretrigger_rows(conn: psycopg.Connection) -> None:
    # Simulate rows embedded *before* the trigger existed by disabling it
    # during the insert, so no INSERT enqueue is produced.
    inside = f"{_PREFIX}bf_in"
    outside = f"{_PREFIX}bf_out"
    _seed_paper(conn, inside, entry_date="2026-06-11")
    _seed_paper(conn, outside, entry_date="2026-06-01")

    with conn.cursor() as cur:
        cur.execute("ALTER TABLE paper_embeddings DISABLE TRIGGER trg_embedding_outbox")
    conn.commit()
    try:
        _seed_embedding(conn, inside, _vec(0.11))
        _seed_embedding(conn, outside, _vec(0.22))
    finally:
        # rollback first: if a seed raised, the txn is aborted and the
        # ENABLE would silently fail, leaving the trigger disabled for the
        # whole test DB (a real landmine the first run hit).
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE paper_embeddings ENABLE TRIGGER trg_embedding_outbox")
        conn.commit()

    # No trigger rows were produced for the pre-trigger inserts.
    assert _outbox_count(conn) == 0

    n = obx.enqueue_backfill(conn, "indus", "2026-06-11")
    assert n == 1  # only the entry_date >= cutoff row
    assert _outbox_count(conn) == 1

    client = FakeQdrant()
    obx.drain_model(conn, client, model_name="indus", collection="scix_indus_v2_papers_s1")
    assert obx.point_id(inside) in client.upserted["scix_indus_v2_papers_s1"]
    assert obx.point_id(outside) not in client.upserted.get("scix_indus_v2_papers_s1", {})


@pytest.mark.integration
def test_report_lag_counts_rows(conn: psycopg.Connection) -> None:
    b = f"{_PREFIX}lag1"
    _seed_paper(conn, b)
    _seed_embedding(conn, b, _vec(0.3))

    lag = {m: (rows, age) for m, rows, age in obx.report_lag(conn)}
    assert "indus" in lag
    rows, age = lag["indus"]
    assert rows >= 1
    assert age >= 0
