"""Verify the PostgreSQL schema works: inserts, queries, GIN indexes.

The vector-search coverage that used to live here targeted ``paper_embeddings``,
which ADR-015 dropped; the dense lane serves from Qdrant now (ADR-013). Dense
retrieval is covered by tests/test_qdrant_dense.py and
tests/test_embed_qdrant_store.py.
"""

import os

import psycopg
import pytest
from helpers import is_production_dsn

pytestmark = pytest.mark.integration

DSN = os.environ.get("SCIX_TEST_DSN") or os.environ.get("SCIX_DSN", "dbname=scix")


@pytest.fixture(scope="module")
def conn():
    """Provide a connection to the test database, rolled back after all tests.

    This module INSERTs into papers, citation_edges and extractions. Every
    write is savepoint-wrapped, but a connection to
    production still takes locks on live tables and strands an
    idle-in-transaction session if pytest is killed mid-run, so refuse one.
    """
    if is_production_dsn(DSN):
        pytest.skip(
            "Refusing to run schema write tests against production. "
            "Set SCIX_TEST_DSN to a non-production database."
        )
    with psycopg.connect(DSN) as c:
        c.autocommit = False
        yield c
        c.rollback()


@pytest.fixture(autouse=True)
def _savepoint(conn):
    """Wrap each test in a savepoint so tests don't affect each other."""
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT test_sp")
    yield
    with conn.cursor() as cur:
        cur.execute("ROLLBACK TO SAVEPOINT test_sp")


SAMPLE_PAPER = {
    "bibcode": "2024ApJ...test.001A",
    "title": "A Test Paper on Gravitational Waves",
    "abstract": "We present a novel analysis of gravitational wave signals.",
    "year": 2024,
    "doctype": "article",
    "pub": "The Astrophysical Journal",
    "authors": ["Author, A.", "Author, B."],
    "first_author": "Author, A.",
    "keywords": ["gravitational waves", "LIGO", "signal processing"],
    "arxiv_class": ["astro-ph.HE", "gr-qc"],
    "doi": ["10.3847/test.001"],
    "citation_count": 42,
    "read_count": 150,
    "reference_count": 35,
}


def _insert_paper(cur: psycopg.Cursor, paper: dict) -> None:
    cur.execute(
        """
        INSERT INTO papers (
            bibcode, title, abstract, year, doctype, pub,
            authors, first_author, keywords, arxiv_class, doi,
            citation_count, read_count, reference_count
        ) VALUES (
            %(bibcode)s, %(title)s, %(abstract)s, %(year)s, %(doctype)s, %(pub)s,
            %(authors)s, %(first_author)s, %(keywords)s, %(arxiv_class)s, %(doi)s,
            %(citation_count)s, %(read_count)s, %(reference_count)s
        )
        """,
        paper,
    )


class TestPaperTable:
    def test_insert_and_query(self, conn):
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            cur.execute(
                "SELECT bibcode, title, year FROM papers WHERE bibcode = %s",
                (SAMPLE_PAPER["bibcode"],),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == SAMPLE_PAPER["bibcode"]
            assert row[1] == SAMPLE_PAPER["title"]
            assert row[2] == SAMPLE_PAPER["year"]

    def test_array_fields(self, conn):
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            cur.execute(
                "SELECT authors, keywords, arxiv_class FROM papers WHERE bibcode = %s",
                (SAMPLE_PAPER["bibcode"],),
            )
            row = cur.fetchone()
            assert row[0] == ["Author, A.", "Author, B."]
            assert "LIGO" in row[1]
            assert "gr-qc" in row[2]

    def test_gin_index_array_contains(self, conn):
        """Verify GIN indexes support @> (array contains) queries."""
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            cur.execute("SELECT bibcode FROM papers WHERE authors @> ARRAY['Author, A.']")
            rows = cur.fetchall()
            assert any(r[0] == SAMPLE_PAPER["bibcode"] for r in rows)

            cur.execute("SELECT bibcode FROM papers WHERE keywords @> ARRAY['LIGO']")
            rows = cur.fetchall()
            assert any(r[0] == SAMPLE_PAPER["bibcode"] for r in rows)


class TestCitationEdges:
    def test_forward_and_backward_queries(self, conn):
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            paper2 = {**SAMPLE_PAPER, "bibcode": "2023ApJ...cited.001X", "title": "Cited Paper"}
            _insert_paper(cur, paper2)
            cur.execute(
                "INSERT INTO citation_edges (source_bibcode, target_bibcode) VALUES (%s, %s)",
                (SAMPLE_PAPER["bibcode"], paper2["bibcode"]),
            )
            # Forward: what does the paper cite?
            cur.execute(
                "SELECT target_bibcode FROM citation_edges WHERE source_bibcode = %s",
                (SAMPLE_PAPER["bibcode"],),
            )
            targets = [r[0] for r in cur.fetchall()]
            assert paper2["bibcode"] in targets

            # Backward: what cites this paper?
            cur.execute(
                "SELECT source_bibcode FROM citation_edges WHERE target_bibcode = %s",
                (paper2["bibcode"],),
            )
            sources = [r[0] for r in cur.fetchall()]
            assert SAMPLE_PAPER["bibcode"] in sources


class TestExtractions:
    def test_insert_and_query(self, conn):
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            cur.execute(
                """
                INSERT INTO extractions (bibcode, extraction_type, extraction_version, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (
                    SAMPLE_PAPER["bibcode"],
                    "entities",
                    "v1.0",
                    '{"entities": [{"type": "method", "name": "matched filtering"}]}',
                ),
            )
            cur.execute(
                "SELECT payload->'entities'->0->>'name' FROM extractions WHERE bibcode = %s",
                (SAMPLE_PAPER["bibcode"],),
            )
            assert cur.fetchone()[0] == "matched filtering"
