from __future__ import annotations

import json
from unittest.mock import patch

import psycopg
import pytest

from scix.mcp_server import _dispatch_tool
from scix.search import SearchResult
from tests.helpers import get_test_dsn


@pytest.mark.integration
def test_disambiguation_timeout_leaves_connection_usable_for_keyword_search() -> None:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN must identify a non-production database")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

        timeout_observed = False

        def timed_out_disambiguation(*args: object, **kwargs: object) -> None:
            nonlocal timeout_observed
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = 1")
                    cur.execute("SELECT pg_sleep(1)")
            except psycopg.errors.QueryCanceled:
                timeout_observed = True
                raise

        def usable_keyword_search(*args: object, **kwargs: object) -> SearchResult:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone() == (1,)
            return SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 0.0})

        with (
            patch("scix.mcp_server.disambiguate_query", side_effect=timed_out_disambiguation),
            patch("scix.search.lexical_search", side_effect=usable_keyword_search),
        ):
            result = _dispatch_tool(
                conn,
                "search",
                {
                    "query": "text-to-SQL",
                    "mode": "keyword",
                    "filters": {"year_min": 2025, "arxiv_class": "cs.DB"},
                },
            )

        assert json.loads(result)["total"] == 0
        assert timeout_observed
