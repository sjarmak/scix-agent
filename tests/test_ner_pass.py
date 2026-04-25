"""Tests for the dbl.3 GLiNER NER backfill pipeline.

Unit tests cover canonicalization, per-doc dedup, label mapping, and the
DB writers (against ``SCIX_TEST_DSN``). Integration tests inject a stub
model so the suite never needs the real ~430 MB GLiNER weights.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from helpers import get_test_dsn

from scix.extract.ner_pass import (
    LABEL_TO_ENTITY_TYPE,
    NER_LABELS,
    NER_TIER,
    GlinerExtractor,
    Mention,
    PaperInput,
    _dedup_within_doc,
    canonicalize,
    process_batch,
    run,
)

# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------


class _StubModel:
    """Mimics the GLiNER public surface used by ``GlinerExtractor.predict``.

    Tests configure ``predictions``: a list of per-text mention dicts that
    the stub returns when ``batch_predict_entities`` is called. Order is
    preserved to align by index with the input.
    """

    def __init__(self, predictions: list[list[dict[str, Any]]]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[list[str], list[str], float, int]] = []

    def to(self, _device: str) -> "_StubModel":
        return self

    def eval(self) -> "_StubModel":  # noqa: A003 — mimic torch
        return self

    def batch_predict_entities(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float,
        batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        self.calls.append((list(texts), list(labels), threshold, batch_size))
        # Return predictions in the same order as the active texts; the
        # extractor handles re-threading by index for skipped (empty) inputs.
        return self.predictions[: len(texts)]


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_lowercase_and_strip(self) -> None:
        assert canonicalize("  PyTorch  ") == "pytorch"

    def test_drops_trailing_paren(self) -> None:
        assert canonicalize("PyTorch (Paszke et al., 2019)") == "pytorch"

    def test_collapses_internal_whitespace(self) -> None:
        assert canonicalize("Hubble  Space\tTelescope") == "hubble space telescope"

    def test_preserves_punctuation_inside_token(self) -> None:
        # CRISPR-Cas9 / p53 / NF-kB carry meaning in punctuation.
        assert canonicalize("CRISPR-Cas9") == "crispr-cas9"
        assert canonicalize("p53") == "p53"
        assert canonicalize("NF-kB") == "nf-kb"


class TestDedupWithinDoc:
    def test_keeps_highest_confidence(self) -> None:
        m1 = Mention("b", "pytorch", "PyTorch", "software", 0.85)
        m2 = Mention("b", "pytorch", "PyTorch", "software", 0.95)
        out = _dedup_within_doc([m1, m2])
        assert len(out) == 1
        assert out[0].confidence == 0.95

    def test_distinct_types_preserved(self) -> None:
        m1 = Mention("b", "python", "Python", "software", 0.9)
        m2 = Mention("b", "python", "Python", "organism", 0.8)  # snake!
        out = _dedup_within_doc([m1, m2])
        assert len(out) == 2


class TestLabelMap:
    def test_all_ner_labels_have_entity_type(self) -> None:
        for label in NER_LABELS:
            assert label in LABEL_TO_ENTITY_TYPE
            assert LABEL_TO_ENTITY_TYPE[label]


# ---------------------------------------------------------------------------
# Extractor wrapper with stub model
# ---------------------------------------------------------------------------


class TestExtractorPredict:
    def test_filters_below_confidence(self) -> None:
        stub = _StubModel(
            [
                [
                    {"text": "PyTorch", "label": "software", "score": 0.95},
                    {"text": "low-conf", "label": "method", "score": 0.5},
                ]
            ]
        )
        ext = GlinerExtractor(model=stub, confidence=0.7)
        out = ext.predict([PaperInput("b1", "We use PyTorch.")])
        assert len(out) == 1
        assert {m.canonical_name for m in out[0]} == {"pytorch"}

    def test_skips_empty_text(self) -> None:
        stub = _StubModel([[{"text": "X", "label": "software", "score": 0.9}]])
        ext = GlinerExtractor(model=stub, confidence=0.7)
        out = ext.predict(
            [
                PaperInput("b1", ""),
                PaperInput("b2", "Some text here."),
            ]
        )
        assert out[0] == []
        assert {m.canonical_name for m in out[1]} == {"x"}

    def test_unknown_label_dropped(self) -> None:
        stub = _StubModel(
            [
                [
                    {"text": "X", "label": "not-in-label-set", "score": 0.95},
                    {"text": "Y", "label": "software", "score": 0.95},
                ]
            ]
        )
        ext = GlinerExtractor(model=stub, confidence=0.7)
        out = ext.predict([PaperInput("b1", "foo")])
        assert {m.canonical_name for m in out[0]} == {"y"}

    def test_dedup_within_paper(self) -> None:
        stub = _StubModel(
            [
                [
                    {"text": "PyTorch", "label": "software", "score": 0.85},
                    {"text": "  pytorch  ", "label": "software", "score": 0.95},
                ]
            ]
        )
        ext = GlinerExtractor(model=stub, confidence=0.7)
        out = ext.predict([PaperInput("b1", "x")])
        assert len(out[0]) == 1
        assert out[0][0].confidence == 0.95


# ---------------------------------------------------------------------------
# DB integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_dsn() -> str:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set or points at production DB")
    return dsn


@pytest.fixture
def conn(test_dsn: str) -> Iterator[psycopg.Connection]:
    c = psycopg.connect(test_dsn)
    try:
        yield c
    finally:
        c.close()


_FIXTURE_BIBCODES = (
    "9999NER...001",
    "9999NER...002",
    "9999NER...003",
)
_FIXTURE_ENTITY_PREFIX = "ner_test_"


@pytest.fixture
def seed_papers(conn: psycopg.Connection) -> Iterator[None]:
    """Seed three papers, clean entities + document_entities on teardown."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (list(_FIXTURE_BIBCODES),),
        )
        cur.execute(
            "DELETE FROM entities WHERE source = 'gliner' " "AND canonical_name LIKE %s",
            (f"{_FIXTURE_ENTITY_PREFIX}%",),
        )
        cur.execute("DELETE FROM ingest_log WHERE filename LIKE 'ner_pass:abstract:9999NER%'")
        cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (list(_FIXTURE_BIBCODES),))
        for bib in _FIXTURE_BIBCODES:
            cur.execute(
                """
                INSERT INTO papers (bibcode, title, abstract, year)
                VALUES (%s, 'NER fixture', 'fixture body text', 2099)
                """,
                (bib,),
            )
        conn.commit()
    try:
        yield
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
                (list(_FIXTURE_BIBCODES),),
            )
            cur.execute(
                "DELETE FROM entities WHERE source = 'gliner' " "AND canonical_name LIKE %s",
                (f"{_FIXTURE_ENTITY_PREFIX}%",),
            )
            cur.execute("DELETE FROM ingest_log WHERE filename LIKE 'ner_pass:abstract:9999NER%'")
            cur.execute(
                "DELETE FROM papers WHERE bibcode = ANY(%s)",
                (list(_FIXTURE_BIBCODES),),
            )
        conn.commit()


def _stub_with_canonical_names(per_paper: list[list[tuple[str, str, float]]]) -> _StubModel:
    """Build a stub returning prefixed canonical names for safe cleanup."""
    out: list[list[dict[str, Any]]] = []
    for mentions in per_paper:
        out.append(
            [
                {
                    "text": f"{_FIXTURE_ENTITY_PREFIX}{name}",
                    "label": label,
                    "score": score,
                }
                for name, label, score in mentions
            ]
        )
    return _StubModel(out)


def test_process_batch_writes_entities_and_links(
    conn: psycopg.Connection, seed_papers: None
) -> None:
    stub = _stub_with_canonical_names(
        [
            [("pytorch", "software", 0.95), ("imagenet", "dataset", 0.92)],
            [("pytorch", "software", 0.88)],  # same entity as paper 1 -> dedup at entity level
            [],
        ]
    )
    ext = GlinerExtractor(model=stub, confidence=0.7)
    batch = [PaperInput(b, "fixture") for b in _FIXTURE_BIBCODES]

    stats = process_batch(conn, ext, batch, source_version="test/1")
    conn.commit()

    assert stats.papers_seen == 3
    assert stats.papers_with_mentions == 2
    assert stats.mentions_kept == 3
    # Two distinct entities (pytorch, imagenet) shared across two papers
    assert stats.new_entities == 2
    assert stats.upserted_doc_entities == 3

    with conn.cursor() as cur:
        cur.execute(
            "SELECT canonical_name, entity_type, source_version FROM entities "
            "WHERE source='gliner' AND canonical_name LIKE %s ORDER BY 1",
            (f"{_FIXTURE_ENTITY_PREFIX}%",),
        )
        rows = cur.fetchall()
    assert {r[0] for r in rows} == {
        f"{_FIXTURE_ENTITY_PREFIX}pytorch",
        f"{_FIXTURE_ENTITY_PREFIX}imagenet",
    }
    assert all(r[2] == "test/1" for r in rows)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, match_method, tier, confidence FROM document_entities "
            "WHERE bibcode = ANY(%s) ORDER BY bibcode, entity_id",
            (list(_FIXTURE_BIBCODES),),
        )
        de_rows = cur.fetchall()
    assert all(r[1] == "gliner" for r in de_rows)
    assert all(r[2] == NER_TIER for r in de_rows)


def test_run_is_idempotent(conn: psycopg.Connection, seed_papers: None) -> None:
    """A second run with identical predictions must not create duplicate rows."""
    stub = _stub_with_canonical_names(
        [
            [("pytorch", "software", 0.9)],
            [],
            [],
        ]
    )
    ext = GlinerExtractor(model=stub, confidence=0.7)

    run(conn, ext, target="abstract", batch_size=10, since_bibcode="9999NEQ", max_papers=10)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM document_entities WHERE bibcode = ANY(%s)",
            (list(_FIXTURE_BIBCODES),),
        )
        first_count = cur.fetchone()[0]

    # Reset stub for second pass (it's stateless across calls anyway).
    stub.predictions = stub.predictions  # noqa: PLW0127 — explicit no-op for clarity
    run(conn, ext, target="abstract", batch_size=10, since_bibcode="9999NEQ", max_papers=10)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM document_entities WHERE bibcode = ANY(%s)",
            (list(_FIXTURE_BIBCODES),),
        )
        second_count = cur.fetchone()[0]

    assert first_count == second_count, "rerun should be idempotent"


def test_run_skips_completed_checkpoints(conn: psycopg.Connection, seed_papers: None) -> None:
    """A pre-recorded ingest_log entry must cause the batch to be skipped."""
    # Pre-record the checkpoint key for the first batch.
    key = f"ner_pass:abstract:{_FIXTURE_BIBCODES[0]}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_log (filename, records_loaded, edges_loaded, status, finished_at)
            VALUES (%s, 0, 0, 'complete', now())
            ON CONFLICT (filename) DO UPDATE SET status='complete'
            """,
            (key,),
        )
        conn.commit()

    stub = _stub_with_canonical_names(
        [
            [("should_not_be_written", "software", 0.99)],
            [],
            [],
        ]
    )
    # The stub will return predictions; if process_batch ran, we'd see a row.
    ext = GlinerExtractor(model=stub, confidence=0.7)
    run(conn, ext, target="abstract", batch_size=10, since_bibcode="9999NEQ", max_papers=10)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM entities WHERE source='gliner' " "AND canonical_name = %s",
            (f"{_FIXTURE_ENTITY_PREFIX}should_not_be_written",),
        )
        assert cur.fetchone()[0] == 0
