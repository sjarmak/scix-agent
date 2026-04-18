"""Integration tests for M5 — MCP community signals.

Verifies that ``search.explore_community`` accepts a ``signal`` parameter
(citation / semantic / taxonomic), reads the correct ``paper_metrics``
column per signal, joins ``communities`` on ``(signal, resolution,
community_id)``, and that the ``graph_context`` MCP handler surfaces a
per-signal ``communities`` block in its response.

These tests write to ``papers``, ``paper_metrics``, and ``communities``
and therefore require ``SCIX_TEST_DSN`` to be set to a non-production DB.
They SKIP cleanly otherwise.
"""

from __future__ import annotations

import json
import pathlib
import sys
import zlib

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
MIGRATION_051_PATH = REPO_ROOT / "migrations" / "051_community_semantic_columns.sql"
MIGRATION_052_PATH = REPO_ROOT / "migrations" / "052_communities_signal.sql"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import get_test_dsn  # noqa: E402

TEST_DSN = get_test_dsn()

pytestmark = pytest.mark.skipif(
    TEST_DSN is None,
    reason=(
        "SCIX_TEST_DSN is not set or points at production — "
        "mcp-community-signals tests require a dedicated test DB"
    ),
)


TEST_BIBCODE_PREFIX = "MCPCOMMSIG."

# Fixture layout: 5 papers, all in the same citation community (cid=9001),
# same semantic community (cid=9201), and same taxonomic class 'astro-ph.GA'.
CITATION_CID = 9001
SEMANTIC_CID = 9201
TAX_TEXT = "astro-ph.GA"


def _taxonomic_cid(text: str) -> int:
    """Mirror scripts/generate_community_labels._taxonomic_id."""
    return zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF


TAXONOMIC_CID = _taxonomic_cid(TAX_TEXT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    assert TEST_DSN is not None
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migrations(dsn: str) -> None:
    """Apply migrations 051 + 052 idempotently."""
    for path in (MIGRATION_051_PATH, MIGRATION_052_PATH):
        sql = path.read_text()
        with psycopg.connect(dsn) as c:
            c.autocommit = True
            with c.cursor() as cur:
                cur.execute(sql)


def _cleanup(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_metrics WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            # Remove the fixture community rows across all three signals.
            cur.execute(
                "DELETE FROM communities "
                "WHERE (signal, community_id) IN "
                "((%s,%s),(%s,%s),(%s,%s))",
                (
                    "citation", CITATION_CID,
                    "semantic", SEMANTIC_CID,
                    "taxonomic", TAXONOMIC_CID,
                ),
            )


def _seed(dsn: str) -> list[str]:
    bibcodes: list[str] = []
    with psycopg.connect(dsn) as c:
        c.autocommit = False
        with c.cursor() as cur:
            for i in range(5):
                bib = f"{TEST_BIBCODE_PREFIX}{i:03d}"
                bibcodes.append(bib)
                cur.execute(
                    "INSERT INTO papers (bibcode, title, arxiv_class, "
                    "keyword_norm, citation_count) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (bibcode) DO UPDATE SET "
                    "  title = EXCLUDED.title, "
                    "  arxiv_class = EXCLUDED.arxiv_class",
                    (
                        bib,
                        f"fixture paper {bib}",
                        [TAX_TEXT],
                        ["galaxy", "quasar"],
                        100 - i,
                    ),
                )
                cur.execute(
                    "INSERT INTO paper_metrics "
                    "  (bibcode, pagerank, "
                    "   community_id_coarse, community_id_medium, community_id_fine, "
                    "   community_semantic_coarse, community_semantic_medium, "
                    "   community_semantic_fine, community_taxonomic) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (bibcode) DO UPDATE SET "
                    "  pagerank = EXCLUDED.pagerank, "
                    "  community_id_coarse = EXCLUDED.community_id_coarse, "
                    "  community_id_medium = EXCLUDED.community_id_medium, "
                    "  community_id_fine = EXCLUDED.community_id_fine, "
                    "  community_semantic_coarse = EXCLUDED.community_semantic_coarse, "
                    "  community_semantic_medium = EXCLUDED.community_semantic_medium, "
                    "  community_semantic_fine = EXCLUDED.community_semantic_fine, "
                    "  community_taxonomic = EXCLUDED.community_taxonomic",
                    (
                        bib,
                        0.001 * (5 - i),
                        CITATION_CID, CITATION_CID, CITATION_CID,
                        SEMANTIC_CID, SEMANTIC_CID, SEMANTIC_CID,
                        TAX_TEXT,
                    ),
                )
            # Seed labels in communities for each signal at coarse resolution.
            for signal, cid, label, kws in (
                ("citation", CITATION_CID, "cite_label_A",
                 ["cite_kw_alpha", "cite_kw_beta"]),
                ("semantic", SEMANTIC_CID, "sem_label_B",
                 ["sem_kw_alpha", "sem_kw_beta"]),
                ("taxonomic", TAXONOMIC_CID, "astro-ph.GA — galactic",
                 ["galaxy", "formation"]),
            ):
                cur.execute(
                    "INSERT INTO communities "
                    "  (signal, resolution, community_id, label, paper_count, "
                    "   top_keywords) "
                    "VALUES (%s, 'coarse', %s, %s, %s, %s) "
                    "ON CONFLICT (signal, resolution, community_id) DO UPDATE SET "
                    "  label = EXCLUDED.label, "
                    "  paper_count = EXCLUDED.paper_count, "
                    "  top_keywords = EXCLUDED.top_keywords",
                    (signal, cid, label, 5, kws),
                )
            # Citation + semantic also get medium/fine rows so those resolutions
            # populate in graph_context responses.
            for signal, cid in (
                ("citation", CITATION_CID),
                ("semantic", SEMANTIC_CID),
            ):
                for res in ("medium", "fine"):
                    cur.execute(
                        "INSERT INTO communities "
                        "  (signal, resolution, community_id, label, paper_count, "
                        "   top_keywords) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (signal, resolution, community_id) DO UPDATE SET "
                        "  label = EXCLUDED.label, "
                        "  top_keywords = EXCLUDED.top_keywords",
                        (
                            signal, res, cid,
                            f"{signal}_label_{res}",
                            5,
                            [f"{signal}_{res}_kw1", f"{signal}_{res}_kw2"],
                        ),
                    )
        c.commit()
    return bibcodes


@pytest.fixture
def fixture_data(dsn: str, applied_migrations: None):
    _cleanup(dsn)
    bibcodes = _seed(dsn)
    try:
        yield bibcodes
    finally:
        _cleanup(dsn)


# ---------------------------------------------------------------------------
# Test (a) — explore_community per signal returns the right rows
# ---------------------------------------------------------------------------


def test_explore_community_citation_signal(dsn: str, fixture_data: list[str]) -> None:
    from scix import search

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        result = search.explore_community(
            conn, bibcode, resolution="coarse", signal="citation"
        )
    assert result.metadata["signal"] == "citation"
    assert result.metadata["community_id"] == CITATION_CID
    assert result.metadata["label"] == "cite_label_A"
    assert "cite_kw_alpha" in result.metadata["top_keywords"]
    # Siblings: all 5 fixture papers are in this community.
    assert result.total == 5
    returned_bibs = {p["bibcode"] for p in result.papers}
    assert set(fixture_data) == returned_bibs


def test_explore_community_semantic_signal(dsn: str, fixture_data: list[str]) -> None:
    from scix import search

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        result = search.explore_community(
            conn, bibcode, resolution="coarse", signal="semantic"
        )
    assert result.metadata["signal"] == "semantic"
    assert result.metadata["community_id"] == SEMANTIC_CID
    assert result.metadata["label"] == "sem_label_B"
    assert "sem_kw_alpha" in result.metadata["top_keywords"]
    assert result.total == 5


def test_explore_community_taxonomic_signal(dsn: str, fixture_data: list[str]) -> None:
    from scix import search

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        result = search.explore_community(
            conn, bibcode, resolution="coarse", signal="taxonomic"
        )
    assert result.metadata["signal"] == "taxonomic"
    assert result.metadata["community_id"] == TAXONOMIC_CID
    assert result.metadata["community_taxonomic"] == TAX_TEXT
    assert result.metadata["label"] == "astro-ph.GA — galactic"
    assert "galaxy" in result.metadata["top_keywords"]
    # All 5 papers share the same taxonomic class.
    assert result.total == 5


# ---------------------------------------------------------------------------
# Test (b) — invalid signal raises ValueError
# ---------------------------------------------------------------------------


def test_explore_community_invalid_signal_raises(
    dsn: str, fixture_data: list[str]
) -> None:
    from scix import search

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        with pytest.raises(ValueError, match="invalid community signal"):
            search.explore_community(conn, bibcode, signal="bogus")


# ---------------------------------------------------------------------------
# Test (c) — graph_context response includes all three signal blocks
# ---------------------------------------------------------------------------


def test_graph_context_response_includes_all_signal_blocks(
    dsn: str, fixture_data: list[str]
) -> None:
    from scix import mcp_server

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        raw = mcp_server._handle_graph_context(
            conn, {"bibcode": bibcode, "include_community": False}
        )
    data = json.loads(raw)
    communities = data["metadata"]["communities"]

    # All three signals must be present.
    assert set(communities.keys()) == {"citation", "semantic", "taxonomic"}

    # Citation signal: all 3 resolutions populated with labels.
    for res in ("coarse", "medium", "fine"):
        block = communities["citation"][res]
        assert block["community_id"] == CITATION_CID
        assert block["label"] is not None
        assert isinstance(block["top_keywords"], list)
        assert len(block["top_keywords"]) >= 1

    # Semantic signal: all 3 resolutions populated with labels.
    for res in ("coarse", "medium", "fine"):
        block = communities["semantic"][res]
        assert block["community_id"] == SEMANTIC_CID
        assert block["label"] is not None
        assert isinstance(block["top_keywords"], list)

    # Taxonomic signal: only coarse populated, surfacing both integer id and
    # original text label.
    tax_coarse = communities["taxonomic"]["coarse"]
    assert tax_coarse["community_id"] == TAXONOMIC_CID
    assert tax_coarse["community_taxonomic"] == TAX_TEXT
    assert tax_coarse["label"] == "astro-ph.GA — galactic"
    assert "galaxy" in tax_coarse["top_keywords"]


def test_graph_context_with_include_community_uses_signal(
    dsn: str, fixture_data: list[str]
) -> None:
    """When include_community=True, the sibling block should reflect `signal`."""
    from scix import mcp_server

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        raw = mcp_server._handle_graph_context(
            conn,
            {
                "bibcode": bibcode,
                "include_community": True,
                "signal": "taxonomic",
                "resolution": "coarse",
            },
        )
    data = json.loads(raw)
    community = data["community"]
    assert community["metadata"]["signal"] == "taxonomic"
    assert community["metadata"]["community_id"] == TAXONOMIC_CID
    assert community["total"] == 5


def test_graph_context_invalid_signal_returns_error(
    dsn: str, fixture_data: list[str]
) -> None:
    from scix import mcp_server

    bibcode = fixture_data[0]
    with psycopg.connect(dsn) as conn:
        raw = mcp_server._handle_graph_context(
            conn,
            {
                "bibcode": bibcode,
                "include_community": True,
                "signal": "bogus",
            },
        )
    data = json.loads(raw)
    assert "error" in data
    assert "invalid community signal" in data["error"]
