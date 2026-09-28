from __future__ import annotations

from unittest.mock import patch

import psycopg
import pytest

from scix import mcp_server
from tests.helpers import get_test_dsn


@pytest.mark.integration
def test_auto_seed_timeout_propagates_without_aborting_outer_transaction() -> None:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN must identify a non-production database")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

        def timed_out_search(
            search_conn: psycopg.Connection, *args: object, **kwargs: object
        ) -> None:
            with search_conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = 100")
                cur.execute("SELECT pg_sleep(20)")

        with patch("scix.search.concept_search", side_effect=timed_out_search):
            with pytest.raises(psycopg.errors.QueryCanceled):
                mcp_server._dispatch_tool(conn, "find_gaps", {"query": "x"})

        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)
