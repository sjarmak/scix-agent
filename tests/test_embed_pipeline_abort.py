"""Regression: a writer-thread failure must abort the pipeline, not deadlock it.

Before the abort-Event fix, a raise inside the writer (e.g. Qdrant unreachable)
left the main GPU thread blocked forever on a full write_queue, hanging the
unattended daily job. This drives run_embedding_pipeline with fakes (no DB, no
Qdrant, no model) and asserts it re-raises promptly instead of hanging.
"""

import threading

import pytest

from scix import embed


class _CountCursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return (3,)  # total_to_embed

    def executemany(self, sql, rows):
        pass


class _NamedCursor:
    itersize = 0

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        pass

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self, name=None):
        return _NamedCursor(self._rows) if name else _CountCursor()

    def commit(self):
        pass

    def close(self):
        pass


class _BoomClient:
    def upsert(self, **kwargs):
        raise RuntimeError("qdrant down")

    def close(self):
        pass


def test_writer_failure_aborts_without_deadlock(monkeypatch):
    rows = [(f"b{i}", f"title {i}", f"abstract {i}") for i in range(3)]

    monkeypatch.setattr(embed, "get_connection", lambda dsn=None: _FakeConn(rows))
    monkeypatch.setattr(embed, "dense_client", lambda *a, **k: _BoomClient())
    monkeypatch.setattr(embed, "load_model", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(
        embed, "embed_batch", lambda model, tok, texts, **k: [[0.0, 0.0] for _ in texts]
    )

    captured: dict = {}

    def run():
        try:
            embed.run_embedding_pipeline(model_name="indus", batch_size=32)
        except Exception as exc:  # noqa: BLE001 — capturing for the assertion
            captured["exc"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=30)

    assert not t.is_alive(), "run_embedding_pipeline hung after a writer failure"
    assert isinstance(captured.get("exc"), RuntimeError)
    assert "qdrant down" in str(captured["exc"])


def test_non_indus_model_rejected():
    with pytest.raises(ValueError, match="indus"):
        embed.run_embedding_pipeline(model_name="specter2")
