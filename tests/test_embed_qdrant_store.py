"""Unit test for the INDUS Qdrant store writer (bead s7cy).

Fake client + conn: no DB, no Qdrant. Asserts the at-least-once ordering —
Qdrant upsert (wait=True) happens before the watermark row insert and commit —
so a crash between them re-embeds rather than marking an un-upserted paper done.
"""

from scix import embed


class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def executemany(self, sql, rows):
        self.log.append(("executemany", [r[0] for r in rows]))


class _FakeConn:
    def __init__(self, log):
        self.log = log

    def cursor(self):
        return _FakeCursor(self.log)

    def commit(self):
        self.log.append(("commit",))


class _FakeClient:
    def __init__(self, log):
        self.log = log

    def upsert(self, *, collection_name, points, wait):
        self.log.append(("upsert", collection_name, wait, {p.payload["bibcode"] for p in points}))


def _inp(bibcode):
    return embed.EmbeddingInput(bibcode=bibcode, text="t", input_type="title_only", source_hash="h")


def test_store_upserts_to_qdrant_then_marks_synced_then_commits():
    log: list = []
    conn, client = _FakeConn(log), _FakeClient(log)

    n = embed.store_embeddings_qdrant(conn, client, [_inp("b1"), _inp("b2")], [[0.1], [0.2]])

    assert n == 2
    kinds = [e[0] for e in log]
    assert kinds.index("upsert") < kinds.index("executemany") < kinds.index("commit")

    upsert = next(e for e in log if e[0] == "upsert")
    assert upsert[1] == embed.INDUS_COLLECTION
    assert upsert[2] is True  # wait=True — durability before the watermark write
    assert upsert[3] == {"b1", "b2"}

    marked = next(e for e in log if e[0] == "executemany")
    assert set(marked[1]) == {"b1", "b2"}


def test_store_empty_is_noop():
    log: list = []
    assert embed.store_embeddings_qdrant(_FakeConn(log), _FakeClient(log), [], []) == 0
    assert log == []
