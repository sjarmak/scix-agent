"""Unit + integration tests for the dbl.3 INDUS post-classifier."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from helpers import get_test_dsn

from scix.extract.ner_classifier import (
    NerClassifier,
    _centroid,
    _compose_input,
    _cosine,
    extract_sentence,
)
from scix.extract.ner_classify_pass import process_batch, run

# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------


class TestComposeInput:
    def test_mention_and_context(self) -> None:
        assert _compose_input("PyTorch", "We use PyTorch.") == "PyTorch | We use PyTorch."

    def test_strip_whitespace(self) -> None:
        assert _compose_input("  X  ", " ctx ") == "X | ctx"

    def test_empty_context(self) -> None:
        assert _compose_input("X", "") == "X"

    def test_empty_mention(self) -> None:
        assert _compose_input("", "long context here") == "long context here"


class TestCentroid:
    def test_simple_mean(self) -> None:
        out = _centroid([[1.0, 2.0], [3.0, 4.0]])
        assert out == [2.0, 3.0]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            _centroid([[1.0, 2.0], [3.0]])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _centroid([])


class TestCosine:
    def test_identical(self) -> None:
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_norm_returns_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------


class TestExtractSentence:
    def test_finds_target_sentence(self) -> None:
        text = (
            "We collected data on five species. Among them, Drosophila melanogaster "
            "showed the strongest response. Statistical analysis followed."
        )
        out = extract_sentence(text, "Drosophila melanogaster")
        assert "Drosophila melanogaster" in out
        assert "Statistical analysis" not in out

    def test_missing_mention_falls_back(self) -> None:
        text = "Some text without the mention."
        out = extract_sentence(text, "absent_token")
        assert out  # non-empty fallback
        assert "Some text" in out

    def test_empty_inputs(self) -> None:
        assert extract_sentence("", "x") == ""
        # Mention can be empty (e.g., a missing field) — still return a fallback.
        assert extract_sentence("hello world", "") == "hello world"

    def test_case_insensitive_match(self) -> None:
        text = "We analyzed PYTORCH internals. Then we trained the model."
        out = extract_sentence(text, "pytorch")
        assert "PYTORCH" in out


# ---------------------------------------------------------------------------
# Stub embedder for classifier tests
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Returns deterministic embeddings keyed on the input string.

    Configured per type: every input containing a 'type' marker gets a
    one-hot-ish vector pointing at that type's slot, so the classifier
    will (deterministically) pick the matching type as the winner.
    """

    def __init__(self, type_dim_map: dict[str, int], dim: int = 8) -> None:
        self.type_dim_map = type_dim_map
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            # Each anchor's text begins with its surface form. Match by prefix.
            matched = False
            for type_name, dim_idx in self.type_dim_map.items():
                if type_name in text.lower():
                    vec[dim_idx] = 1.0
                    matched = True
                    break
            if not matched:
                vec[0] = 1.0  # ambiguous → falls into first type's slot
            out.append(vec)
        return out


def _make_anchors_file(tmp_path: Path) -> Path:
    """Build a tiny anchors JSON the stub can recognize by surface text."""
    payload = {
        "version": "test",
        "purpose": "test",
        "anchors": {
            # type "software" anchors all contain the literal "software"
            "software": [
                {"text": "software_one", "context": "we use software_one daily"},
                {"text": "software_two", "context": "deploy software_two on cluster"},
            ],
            "method": [
                {"text": "method_alpha", "context": "apply method_alpha here"},
                {"text": "method_beta", "context": "method_beta improves accuracy"},
            ],
            "dataset": [
                {"text": "dataset_x", "context": "the dataset_x benchmark"},
            ],
        },
    }
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps(payload))
    return p


class TestNerClassifier:
    def test_classify_picks_matching_type(self, tmp_path: Path) -> None:
        anchors = _make_anchors_file(tmp_path)
        stub = _StubEmbedder({"software": 0, "method": 1, "dataset": 2})
        clf = NerClassifier(anchors_path=anchors, embedder=stub)
        # Mention contains "software" so stub returns the software-pointing
        # vector → classifier should agree with predicted_type=software.
        result = clf.classify(
            mention="some software thing",
            context="installed via pip",
            predicted_type="software",
        )
        assert result.classifier_type == "software"
        assert result.agreement is True
        assert result.classifier_score == pytest.approx(1.0)

    def test_disagreement_when_predicted_type_wrong(self, tmp_path: Path) -> None:
        anchors = _make_anchors_file(tmp_path)
        stub = _StubEmbedder({"software": 0, "method": 1, "dataset": 2})
        clf = NerClassifier(anchors_path=anchors, embedder=stub)
        # Mention is recognizably software but GLiNER predicted "method".
        result = clf.classify(
            mention="some software tool",
            context="usage",
            predicted_type="method",
        )
        assert result.classifier_type == "software"
        assert result.predicted_type == "method"
        assert result.agreement is False

    def test_classify_batch_single_call(self, tmp_path: Path) -> None:
        """The vectorized path should issue ONE embed call for N inputs."""
        anchors = _make_anchors_file(tmp_path)
        stub = _StubEmbedder({"software": 0, "method": 1, "dataset": 2})
        clf = NerClassifier(anchors_path=anchors, embedder=stub)
        items = [
            ("a software tool", "ctx", "software"),
            ("our method for X", "ctx", "method"),
            ("the dataset of choice", "ctx", "dataset"),
        ]
        results = clf.classify_batch(items)
        assert len(results) == 3
        # Stub records 2 calls: one for the anchors load, one for the batch.
        assert len(stub.calls) == 2
        # The second call carries all three items.
        assert len(stub.calls[1]) == 3

    def test_centroid_caching(self, tmp_path: Path) -> None:
        """A second classify_batch call must NOT re-embed the anchors."""
        anchors = _make_anchors_file(tmp_path)
        stub = _StubEmbedder({"software": 0, "method": 1, "dataset": 2})
        clf = NerClassifier(anchors_path=anchors, embedder=stub)
        clf.classify("a software tool", "ctx", "software")
        clf.classify("another software tool", "ctx", "software")
        # 1 anchors load + 2 mention batches = 3, NOT 4 (anchors aren't re-loaded).
        assert len(stub.calls) == 3


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
    "9999CLF...001",
    "9999CLF...002",
)
_FIXTURE_ENTITY_PREFIX = "clftest_"


@pytest.fixture
def seed_classify_data(conn: psycopg.Connection) -> Iterator[None]:
    """Seed two papers + their gliner mentions for the classify-pass tests."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (list(_FIXTURE_BIBCODES),),
        )
        cur.execute(
            "DELETE FROM entities WHERE source = 'gliner' AND canonical_name LIKE %s",
            (f"{_FIXTURE_ENTITY_PREFIX}%",),
        )
        cur.execute("DELETE FROM ingest_log WHERE filename LIKE 'ner_classify_pass:9999CLF%'")
        cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (list(_FIXTURE_BIBCODES),))

        # Insert two papers with abstracts that mention the entities.
        cur.execute(
            """
            INSERT INTO papers (bibcode, title, abstract, year)
            VALUES (%s, 'Software paper', 'We built our pipeline using software_alpha for everything.', 2024)
            """,
            (_FIXTURE_BIBCODES[0],),
        )
        cur.execute(
            """
            INSERT INTO papers (bibcode, title, abstract, year)
            VALUES (%s, 'Method paper', 'We applied method_beta to the dataset.', 2024)
            """,
            (_FIXTURE_BIBCODES[1],),
        )

        # Insert two entities (one per paper) and bridge them.
        for canon, etype in (
            (f"{_FIXTURE_ENTITY_PREFIX}software_alpha", "software"),
            (f"{_FIXTURE_ENTITY_PREFIX}method_beta", "method"),
        ):
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source, source_version)
                VALUES (%s, %s, 'gliner', 'test/1')
                RETURNING id
                """,
                (canon, etype),
            )
        conn.commit()

        # Re-fetch ids.
        cur.execute(
            "SELECT id, canonical_name FROM entities WHERE source='gliner' "
            "AND canonical_name LIKE %s",
            (f"{_FIXTURE_ENTITY_PREFIX}%",),
        )
        id_map = {name: eid for eid, name in cur.fetchall()}

        for bib, canon in zip(_FIXTURE_BIBCODES, id_map.keys(), strict=False):
            cur.execute(
                """
                INSERT INTO document_entities
                    (bibcode, entity_id, link_type, confidence, match_method, tier)
                VALUES (%s, %s, 'mentions', 0.9, 'gliner', 4)
                """,
                (bib, id_map[canon]),
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
                "DELETE FROM entities WHERE source = 'gliner' AND canonical_name LIKE %s",
                (f"{_FIXTURE_ENTITY_PREFIX}%",),
            )
            cur.execute("DELETE FROM ingest_log WHERE filename LIKE 'ner_classify_pass:9999CLF%'")
            cur.execute(
                "DELETE FROM papers WHERE bibcode = ANY(%s)",
                (list(_FIXTURE_BIBCODES),),
            )
        conn.commit()


def test_process_batch_writes_evidence(
    conn: psycopg.Connection, seed_classify_data: None, tmp_path: Path
) -> None:
    """Round-trip: classifier sees the seeded mentions and updates evidence."""
    anchors = _make_anchors_file(tmp_path)
    stub = _StubEmbedder({"software": 0, "method": 1, "dataset": 2})
    clf = NerClassifier(anchors_path=anchors, embedder=stub)

    # Mirror the keyset query the driver uses.
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT de.bibcode, de.entity_id, de.link_type, de.tier, de.confidence, "
            "       de.evidence, e.canonical_name, e.entity_type, p.abstract, p.title "
            "FROM document_entities de "
            "JOIN entities e ON e.id = de.entity_id "
            "JOIN papers p ON p.bibcode = de.bibcode "
            "WHERE de.bibcode = ANY(%s) ORDER BY de.bibcode",
            (list(_FIXTURE_BIBCODES),),
        )
        rows = cur.fetchall()

    stats = process_batch(conn, clf, rows)
    conn.commit()

    assert stats.rows_seen == 2
    assert stats.agreements == 2  # both should match (canonical_names contain type word)
    assert stats.disagreements == 0

    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, evidence FROM document_entities "
            "WHERE bibcode = ANY(%s) ORDER BY bibcode",
            (list(_FIXTURE_BIBCODES),),
        )
        for bib, evidence in cur.fetchall():
            assert evidence is not None
            assert "agreement" in evidence
            assert "classifier_type" in evidence
            assert "classifier_score" in evidence
            assert evidence["agreement"] is True


def test_run_is_idempotent(
    conn: psycopg.Connection, seed_classify_data: None, tmp_path: Path
) -> None:
    """Re-running the driver must not re-process already-judged rows."""
    anchors = _make_anchors_file(tmp_path)
    stub = _StubEmbedder({"software": 0, "method": 1, "dataset": 2})
    clf = NerClassifier(anchors_path=anchors, embedder=stub)

    run(conn, clf, batch_size=10, since_bibcode="9999CLE", max_rows=10)
    conn.commit()
    # Second run should see zero pending rows because evidence ? 'agreement' is now true.
    totals = run(conn, clf, batch_size=10, since_bibcode="9999CLE", max_rows=10)
    assert totals.rows_seen == 0
