"""Tests for the cross-discipline ``concepts`` substrate (dbl.1).

Unit tests cover the per-vocabulary parsers via fixture data so they do
not need the network. The integration test runs against ``SCIX_TEST_DSN``
and exercises the full COPY-staging round-trip.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from scix.concept_loaders import gcmd as gcmd_loader
from scix.concept_loaders import openalex as openalex_loader
from scix.concept_loaders import physh as physh_loader
from scix.concepts import (
    Concept,
    ConceptRelationship,
    Vocabulary,
    _pg_text_array,
    load_vocabulary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_production(dsn: str | None) -> bool:
    if not dsn:
        return True
    return "dbname=scix " in f"{dsn} " or dsn.strip().endswith("dbname=scix")


@pytest.fixture
def test_conn():
    """Yield a psycopg connection bound to ``SCIX_TEST_DSN`` only."""
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        pytest.skip("SCIX_TEST_DSN not set; refusing to write to production")
    if _is_production(dsn):
        pytest.skip("SCIX_TEST_DSN points at production; refusing")

    import psycopg

    conn = psycopg.connect(dsn)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


class TestPgTextArray:
    def test_empty(self) -> None:
        assert _pg_text_array(()) == "{}"
        assert _pg_text_array([]) == "{}"

    def test_simple(self) -> None:
        assert _pg_text_array(("Galaxy",)) == '{"Galaxy"}'

    def test_quote_escape(self) -> None:
        out = _pg_text_array(('foo "bar"',))
        assert out == '{"foo \\"bar\\""}'

    def test_backslash_escape(self) -> None:
        out = _pg_text_array(("a\\b",))
        assert out == '{"a\\\\b"}'


class TestOpenAlexParse:
    def test_parses_topic_hierarchy(self) -> None:
        topics = [
            {
                "id": "https://openalex.org/T1",
                "display_name": "Foo",
                "description": "fd",
                "keywords": ["alpha", "beta"],
                "ids": {"wikipedia": "https://en.wikipedia.org/wiki/Foo"},
                "subfield": {
                    "id": "https://openalex.org/subfields/100",
                    "display_name": "SubFoo",
                },
                "field": {
                    "id": "https://openalex.org/fields/10",
                    "display_name": "FieldFoo",
                },
                "domain": {
                    "id": "https://openalex.org/domains/1",
                    "display_name": "Physical Sciences",
                },
            }
        ]
        concepts, rels = openalex_loader._parse(topics)
        ids = {c.concept_id: c for c in concepts}
        assert "https://openalex.org/T1" in ids
        assert ids["https://openalex.org/T1"].alternate_labels == ("alpha", "beta")
        assert ids["https://openalex.org/T1"].level == 3
        assert ids["https://openalex.org/domains/1"].level == 0
        assert ids["https://openalex.org/fields/10"].level == 1
        assert ids["https://openalex.org/subfields/100"].level == 2
        rel_pairs = {(r.parent_id, r.child_id) for r in rels}
        assert (
            "https://openalex.org/domains/1",
            "https://openalex.org/fields/10",
        ) in rel_pairs
        assert (
            "https://openalex.org/subfields/100",
            "https://openalex.org/T1",
        ) in rel_pairs


class TestPhyshParse:
    def test_parses_jsonld_concept(self, tmp_path: Path) -> None:
        items = [
            {
                "@id": "https://doi.org/10.29172/parent",
                "@type": ["http://www.w3.org/2004/02/skos/core#Concept"],
                "http://www.w3.org/2004/02/skos/core#prefLabel": [
                    {"@language": "en", "@value": "Parent"}
                ],
            },
            {
                "@id": "https://doi.org/10.29172/child",
                "@type": ["http://www.w3.org/2004/02/skos/core#Concept"],
                "http://www.w3.org/2004/02/skos/core#prefLabel": [
                    {"@language": "en", "@value": "Child"}
                ],
                "http://www.w3.org/2004/02/skos/core#broader": [
                    {"@id": "https://doi.org/10.29172/parent"}
                ],
            },
        ]
        path = tmp_path / "physh.json.gz"
        path.write_bytes(gzip.compress(json.dumps(items).encode()))
        concepts, rels = physh_loader._parse(path)
        ids = {c.concept_id: c for c in concepts}
        assert ids["https://doi.org/10.29172/parent"].level == 0
        assert ids["https://doi.org/10.29172/child"].level == 1
        assert any(
            r.parent_id == "https://doi.org/10.29172/parent"
            and r.child_id == "https://doi.org/10.29172/child"
            for r in rels
        )


class TestGcmdParse:
    def test_walks_nested_uuid_tree(self, tmp_path: Path) -> None:
        # Top placeholder has no uuid; descends into the real concept tree.
        payload = [
            {
                "broader": None,
                "children": [
                    {
                        "uuid": "u-root",
                        "label": "EARTH SCIENCE",
                        "children": [
                            {
                                "uuid": "u-child",
                                "label": "ATMOSPHERE",
                                "definition": "Atmosphere of Earth",
                                "children": [],
                            }
                        ],
                    }
                ],
            }
        ]
        scheme_file = tmp_path / "sciencekeywords.json"
        scheme_file.write_text(json.dumps(payload))

        concepts: dict[str, Concept] = {}
        rels: list[ConceptRelationship] = []
        gcmd_loader._parse_scheme(scheme_file, "sciencekeywords", concepts, rels)
        rels = gcmd_loader._dedupe_relationships(rels)

        assert "u-root" in concepts
        assert "u-child" in concepts
        assert concepts["u-child"].definition == "Atmosphere of Earth"
        assert concepts["u-child"].level == 1
        assert any(r.parent_id == "u-root" and r.child_id == "u-child" for r in rels)


# ---------------------------------------------------------------------------
# Integration (writes to SCIX_TEST_DSN only)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_load_vocabulary_roundtrip(test_conn) -> None:
    vocab = Vocabulary(
        vocabulary="__test_vocab__",
        name="test",
        license="MIT",
        source_url="https://example.com",
    )
    concepts = [
        Concept(
            vocabulary="__test_vocab__",
            concept_id="A",
            preferred_label="Alpha",
            alternate_labels=("alpha-syn",),
            level=0,
        ),
        Concept(
            vocabulary="__test_vocab__",
            concept_id="B",
            preferred_label="Beta",
            level=1,
        ),
    ]
    rels = [ConceptRelationship(vocabulary="__test_vocab__", parent_id="A", child_id="B")]

    n_c, n_r = load_vocabulary(test_conn, vocab, concepts, rels)
    assert n_c == 2
    assert n_r == 1

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT preferred_label, alternate_labels FROM concepts "
            "WHERE vocabulary=%s AND concept_id=%s",
            ("__test_vocab__", "A"),
        )
        row = cur.fetchone()
        assert row[0] == "Alpha"
        assert "alpha-syn" in row[1]

    # Re-run is idempotent.
    n_c2, n_r2 = load_vocabulary(test_conn, vocab, concepts, rels)
    assert n_c2 == 2
    assert n_r2 == 1
