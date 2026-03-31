"""Verify the PostgreSQL schema works: inserts, queries, vector search, GIN indexes."""

import os

import psycopg
import pytest

DSN = os.environ.get("SCIX_DSN", "dbname=scix")


@pytest.fixture(scope="module")
def conn():
    """Provide a connection to the scix database, rolled back after all tests."""
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


def _vec_to_str(vec: list[float]) -> str:
    """Format a float list as a pgvector literal."""
    return "[" + ",".join(str(v) for v in vec) + "]"


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
            cur.execute("SELECT bibcode, title, year FROM papers WHERE bibcode = %s", (SAMPLE_PAPER["bibcode"],))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == SAMPLE_PAPER["bibcode"]
            assert row[1] == SAMPLE_PAPER["title"]
            assert row[2] == SAMPLE_PAPER["year"]

    def test_array_fields(self, conn):
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            cur.execute("SELECT authors, keywords, arxiv_class FROM papers WHERE bibcode = %s", (SAMPLE_PAPER["bibcode"],))
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

    def test_jsonb_raw_field(self, conn):
        with conn.cursor() as cur:
            paper = {**SAMPLE_PAPER, "bibcode": "2024ApJ...test.002B"}
            _insert_paper(cur, paper)
            cur.execute(
                "UPDATE papers SET raw = %s::jsonb WHERE bibcode = %s",
                ('{"extra_field": "extra_value", "nested": {"key": 1}}', paper["bibcode"]),
            )
            cur.execute("SELECT raw->>'extra_field' FROM papers WHERE bibcode = %s", (paper["bibcode"],))
            assert cur.fetchone()[0] == "extra_value"


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
            cur.execute("SELECT target_bibcode FROM citation_edges WHERE source_bibcode = %s", (SAMPLE_PAPER["bibcode"],))
            targets = [r[0] for r in cur.fetchall()]
            assert paper2["bibcode"] in targets

            # Backward: what cites this paper?
            cur.execute("SELECT source_bibcode FROM citation_edges WHERE target_bibcode = %s", (paper2["bibcode"],))
            sources = [r[0] for r in cur.fetchall()]
            assert SAMPLE_PAPER["bibcode"] in sources


class TestVectorEmbeddings:
    def test_insert_and_cosine_search(self, conn):
        with conn.cursor() as cur:
            _insert_paper(cur, SAMPLE_PAPER)
            # Insert a 768-dim embedding (mostly zeros with a few values for testing)
            embedding = [0.0] * 768
            embedding[0] = 1.0
            embedding[1] = 0.5
            vec_str = _vec_to_str(embedding)
            cur.execute(
                "INSERT INTO paper_embeddings (bibcode, model_name, embedding) VALUES (%s, %s, %s::vector)",
                (SAMPLE_PAPER["bibcode"], "specter2", vec_str),
            )

            # Query: find nearest neighbors by cosine distance
            cur.execute(
                """
                SELECT bibcode, 1 - (embedding <=> %s::vector) AS similarity
                FROM paper_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT 5
                """,
                (vec_str, vec_str),
            )
            rows = cur.fetchall()
            assert len(rows) >= 1
            assert rows[0][0] == SAMPLE_PAPER["bibcode"]
            # Cosine similarity of a vector with itself should be ~1.0
            assert rows[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_different_vectors_ranked_by_similarity(self, conn):
        with conn.cursor() as cur:
            # Insert two papers with different embeddings
            _insert_paper(cur, SAMPLE_PAPER)
            paper2 = {**SAMPLE_PAPER, "bibcode": "2024ApJ...test.003C", "title": "Another Paper"}
            _insert_paper(cur, paper2)

            vec_a = [0.0] * 768
            vec_a[0] = 1.0
            vec_b = [0.0] * 768
            vec_b[0] = 0.5
            vec_b[1] = 0.866  # ~orthogonal-ish

            for bib, vec in [(SAMPLE_PAPER["bibcode"], vec_a), (paper2["bibcode"], vec_b)]:
                cur.execute(
                    "INSERT INTO paper_embeddings (bibcode, model_name, embedding) VALUES (%s, %s, %s::vector)",
                    (bib, "specter2", _vec_to_str(vec)),
                )

            # Query with vec_a — should find SAMPLE_PAPER first
            query_str = _vec_to_str(vec_a)
            cur.execute(
                """
                SELECT bibcode FROM paper_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT 5
                """,
                (query_str,),
            )
            rows = cur.fetchall()
            assert rows[0][0] == SAMPLE_PAPER["bibcode"]


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
