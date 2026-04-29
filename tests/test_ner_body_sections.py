"""Tests for the ``target='body_sections'`` path in ner_pass.

Phase 2 body NER reads pre-parsed sections from ``papers_fulltext.sections``
and runs GLiNER on the methods + introduction concatenation only. These
tests cover the pure section-filter helper, the keyset-paginated iterator
against a seeded ``scix_test`` database, and an end-to-end ``run()`` smoke
that wires a stub GLiNER through ``process_batch`` writes.

Production driver invocation is unchanged from the abstract pipeline:

    python scripts/run_ner_pass.py --target body_sections ...
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from helpers import get_test_dsn

from scix.extract.ner_pass import (
    KEPT_BODY_SECTION_ROLES,
    NER_TIER,
    GlinerExtractor,
    concat_filtered_section_text,
    iter_paper_batches,
    run,
)

# ---------------------------------------------------------------------------
# Pure unit tests for concat_filtered_section_text
# ---------------------------------------------------------------------------


class TestConcatFilteredSectionText:
    def test_keeps_methods_and_introduction(self) -> None:
        sections = [
            {"heading": "Abstract", "level": 1, "text": "ABS", "offset": 0},
            {"heading": "1. Introduction", "level": 1, "text": "INTRO", "offset": 10},
            {"heading": "2. Methods", "level": 1, "text": "METH", "offset": 20},
            {"heading": "3. Results", "level": 1, "text": "RES", "offset": 30},
            {"heading": "4. Conclusions", "level": 1, "text": "CONC", "offset": 40},
            {"heading": "References", "level": 1, "text": "REF", "offset": 50},
        ]
        out = concat_filtered_section_text(sections)
        assert "INTRO" in out
        assert "METH" in out
        assert "RES" not in out
        assert "CONC" not in out
        assert "ABS" not in out
        assert "REF" not in out
        # Sections joined with blank-line separator in document order.
        assert out == "INTRO\n\nMETH"

    def test_handles_motivation_alias(self) -> None:
        sections = [
            {"heading": "Motivation", "level": 1, "text": "MOTI", "offset": 0},
            {"heading": "Methodology", "level": 1, "text": "MET", "offset": 10},
        ]
        # motivation → background, methodology → method (both kept).
        out = concat_filtered_section_text(sections)
        assert out == "MOTI\n\nMET"

    def test_skips_empty_text(self) -> None:
        sections = [
            {"heading": "Introduction", "level": 1, "text": "  ", "offset": 0},
            {"heading": "Methods", "level": 1, "text": "M", "offset": 10},
        ]
        assert concat_filtered_section_text(sections) == "M"

    def test_skips_unknown_headings(self) -> None:
        sections = [
            {"heading": "Foo Bar", "level": 1, "text": "FB", "offset": 0},
            {"heading": "Methods", "level": 1, "text": "M", "offset": 10},
        ]
        # Foo Bar classifies to ``other`` — dropped.
        assert concat_filtered_section_text(sections) == "M"

    def test_empty_input(self) -> None:
        assert concat_filtered_section_text([]) == ""
        assert concat_filtered_section_text(None) == ""
        # Defensive: malformed JSON string.
        assert concat_filtered_section_text("not-json") == ""
        # Non-list payload.
        assert concat_filtered_section_text({"sections": []}) == ""

    def test_accepts_json_string(self) -> None:
        sections = [
            {"heading": "Methods", "level": 1, "text": "M", "offset": 10},
        ]
        assert concat_filtered_section_text(json.dumps(sections)) == "M"

    def test_custom_role_filter(self) -> None:
        # Operator-supplied role filter — e.g. methods only.
        sections = [
            {"heading": "Introduction", "level": 1, "text": "I", "offset": 0},
            {"heading": "Methods", "level": 1, "text": "M", "offset": 10},
        ]
        out = concat_filtered_section_text(sections, roles=frozenset({"method"}))
        assert out == "M"

    def test_section_dropped_if_text_missing(self) -> None:
        sections = [
            {"heading": "Methods", "level": 1, "offset": 10},  # no text key
            {"heading": "Methods", "level": 1, "text": "M", "offset": 20},
        ]
        assert concat_filtered_section_text(sections) == "M"

    def test_kept_body_section_roles_constant(self) -> None:
        # Guard the bead spec: methods + introduction (== background) only.
        assert KEPT_BODY_SECTION_ROLES == frozenset({"method", "background"})


# ---------------------------------------------------------------------------
# Stub model (mirrors test_ner_pass.py's _StubModel)
# ---------------------------------------------------------------------------


class _StubModel:
    """Mimics the GLiNER public surface used by ``GlinerExtractor.predict``."""

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
        return self.predictions[: len(texts)]


# ---------------------------------------------------------------------------
# DB integration tests
# ---------------------------------------------------------------------------


_FIXTURE_BIBCODES = (
    "9999BSEC..001",
    "9999BSEC..002",
    "9999BSEC..003",
)
_FIXTURE_ENTITY_PREFIX = "ner_bsec_test_"


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


def _seed_paper(cur: psycopg.Cursor, bibcode: str) -> None:
    cur.execute(
        """
        INSERT INTO papers (bibcode, title, abstract, year)
        VALUES (%s, 'fixture title', 'fixture abstract', 2099)
        ON CONFLICT (bibcode) DO NOTHING
        """,
        (bibcode,),
    )


def _seed_fulltext(
    cur: psycopg.Cursor,
    bibcode: str,
    sections: list[dict[str, Any]],
) -> None:
    cur.execute(
        """
        INSERT INTO papers_fulltext
            (bibcode, source, sections, inline_cites, parser_version)
        VALUES (%s, 'test', %s::jsonb, '[]'::jsonb, 'test/v1')
        ON CONFLICT (bibcode) DO UPDATE SET sections = EXCLUDED.sections
        """,
        (bibcode, json.dumps(sections)),
    )


@pytest.fixture
def seed_fulltext_papers(conn: psycopg.Connection) -> Iterator[None]:
    """Seed three papers + papers_fulltext rows; clean up on teardown."""
    sections_paper1 = [
        {"heading": "Abstract", "level": 1, "text": "abstract text", "offset": 0},
        {"heading": "1. Introduction", "level": 1, "text": "We motivate.", "offset": 20},
        {"heading": "2. Methods", "level": 1, "text": "We use PyTorch.", "offset": 40},
        {"heading": "3. Results", "level": 1, "text": "Results table.", "offset": 60},
    ]
    sections_paper2 = [
        # No methods, no introduction — should yield empty text.
        {"heading": "Results", "level": 1, "text": "Only results.", "offset": 0},
        {"heading": "Discussion", "level": 1, "text": "Discussion.", "offset": 20},
    ]
    sections_paper3 = [
        {"heading": "Methodology", "level": 1, "text": "We use ImageNet.", "offset": 0},
    ]
    seeds = {
        _FIXTURE_BIBCODES[0]: sections_paper1,
        _FIXTURE_BIBCODES[1]: sections_paper2,
        _FIXTURE_BIBCODES[2]: sections_paper3,
    }
    with conn.cursor() as cur:
        # Idempotent cleanup (in case a previous run left rows behind).
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (list(_FIXTURE_BIBCODES),),
        )
        cur.execute(
            "DELETE FROM entities WHERE source='gliner' AND canonical_name LIKE %s",
            (f"{_FIXTURE_ENTITY_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM ingest_log WHERE filename LIKE 'ner_pass:body_sections:9999BSEC%'"
        )
        cur.execute(
            "DELETE FROM papers_fulltext WHERE bibcode = ANY(%s)",
            (list(_FIXTURE_BIBCODES),),
        )
        cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (list(_FIXTURE_BIBCODES),))
        for bib, sections in seeds.items():
            _seed_paper(cur, bib)
            _seed_fulltext(cur, bib, sections)
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
                "DELETE FROM entities WHERE source='gliner' AND canonical_name LIKE %s",
                (f"{_FIXTURE_ENTITY_PREFIX}%",),
            )
            cur.execute(
                "DELETE FROM ingest_log WHERE filename LIKE 'ner_pass:body_sections:9999BSEC%'"
            )
            cur.execute(
                "DELETE FROM papers_fulltext WHERE bibcode = ANY(%s)",
                (list(_FIXTURE_BIBCODES),),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode = ANY(%s)",
                (list(_FIXTURE_BIBCODES),),
            )
        conn.commit()


def test_iter_paper_batches_body_sections_yields_filtered_text(
    conn: psycopg.Connection, seed_fulltext_papers: None
) -> None:
    batches = list(
        iter_paper_batches(
            conn,
            target="body_sections",
            batch_size=10,
            since_bibcode="9999BSEC.",
            max_papers=10,
        )
    )
    assert len(batches) == 1
    by_bibcode = {p.bibcode: p.text for p in batches[0]}

    # Paper 1: introduction + methods kept, abstract + results dropped.
    assert "We motivate." in by_bibcode[_FIXTURE_BIBCODES[0]]
    assert "We use PyTorch." in by_bibcode[_FIXTURE_BIBCODES[0]]
    assert "abstract text" not in by_bibcode[_FIXTURE_BIBCODES[0]]
    assert "Results table." not in by_bibcode[_FIXTURE_BIBCODES[0]]

    # Paper 2: no methods/intro → empty text but still yielded so the
    # watermark advances correctly.
    assert by_bibcode[_FIXTURE_BIBCODES[1]] == ""

    # Paper 3: methodology → method role.
    assert by_bibcode[_FIXTURE_BIBCODES[2]] == "We use ImageNet."


def test_iter_paper_batches_invalid_target_raises() -> None:
    """Defensive: misspelled target should fail loudly, not silently."""
    with pytest.raises(ValueError, match="target must be"):
        list(iter_paper_batches(None, target="bogus"))  # type: ignore[arg-type]


def test_run_body_sections_writes_to_document_entities(
    conn: psycopg.Connection, seed_fulltext_papers: None
) -> None:
    """End-to-end: stub GLiNER + body_sections target → document_entities rows."""
    # Build per-paper predictions ALIGNED with the order rows arrive at the
    # extractor. iter_paper_batches yields in bibcode order; the extractor
    # filters out papers whose text is empty before calling the model, so
    # the stub will only see paper 1 and paper 3 (paper 2 has no kept
    # sections). Predictions are returned in active-text order.
    stub = _StubModel(
        [
            # Paper 1 (intro + methods).
            [
                {
                    "text": f"{_FIXTURE_ENTITY_PREFIX}pytorch",
                    "label": "software",
                    "score": 0.95,
                }
            ],
            # Paper 3 (methodology).
            [
                {
                    "text": f"{_FIXTURE_ENTITY_PREFIX}imagenet",
                    "label": "dataset",
                    "score": 0.92,
                }
            ],
        ]
    )
    ext = GlinerExtractor(model=stub, confidence=0.7)

    totals = run(
        conn,
        ext,
        target="body_sections",
        batch_size=10,
        since_bibcode="9999BSEC.",
        max_papers=10,
        source_version="test/body_sections/v1",
    )

    assert totals.papers_seen == 3
    assert totals.papers_with_mentions == 2
    assert totals.mentions_kept == 2
    assert totals.new_entities == 2
    assert totals.upserted_doc_entities == 2

    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, match_method, tier FROM document_entities "
            "WHERE bibcode = ANY(%s) ORDER BY bibcode",
            (list(_FIXTURE_BIBCODES),),
        )
        rows = cur.fetchall()

    assert {r[0] for r in rows} == {_FIXTURE_BIBCODES[0], _FIXTURE_BIBCODES[2]}
    assert all(r[1] == "gliner" for r in rows)
    assert all(r[2] == NER_TIER for r in rows)


def test_run_body_sections_uses_distinct_checkpoint_namespace(
    conn: psycopg.Connection, seed_fulltext_papers: None
) -> None:
    """Body-sections checkpoints must not collide with abstract checkpoints.

    A pre-existing checkpoint for ``ner_pass:abstract:<bibcode>`` should NOT
    cause the body-sections batch to be skipped — the resumability key
    embeds the target.
    """
    # Pre-record the wrong-target checkpoint key.
    abstract_key = f"ner_pass:abstract:{_FIXTURE_BIBCODES[0]}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_log (filename, records_loaded, edges_loaded, status, finished_at)
            VALUES (%s, 0, 0, 'complete', now())
            ON CONFLICT (filename) DO UPDATE SET status='complete'
            """,
            (abstract_key,),
        )
        conn.commit()

    stub = _StubModel(
        [
            [
                {
                    "text": f"{_FIXTURE_ENTITY_PREFIX}pytorch",
                    "label": "software",
                    "score": 0.95,
                }
            ],
            [
                {
                    "text": f"{_FIXTURE_ENTITY_PREFIX}imagenet",
                    "label": "dataset",
                    "score": 0.92,
                }
            ],
        ]
    )
    ext = GlinerExtractor(model=stub, confidence=0.7)

    totals = run(
        conn,
        ext,
        target="body_sections",
        batch_size=10,
        since_bibcode="9999BSEC.",
        max_papers=10,
        source_version="test/body_sections/v1",
    )

    # Mentions written despite the abstract checkpoint being present.
    assert totals.mentions_kept == 2

    # Cleanup: remove the abstract-target sentinel so it doesn't leak across runs.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ingest_log WHERE filename = %s", (abstract_key,))
        conn.commit()
