"""Tests for per-run PostgreSQL isolation used by the pytest suite."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import pytest
from database_isolation import DatabaseIsolation
from psycopg.conninfo import conninfo_to_dict


@pytest.mark.integration
def test_pytest_process_uses_run_scoped_database() -> None:
    dsn = os.environ.get("SCIX_TEST_DSN")
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    database = conninfo_to_dict(dsn)["dbname"]

    assert isinstance(database, str)
    assert database.startswith("scix_test_run_")


def test_from_dsn_preserves_connection_parameters_and_replaces_database() -> None:
    isolation = DatabaseIsolation.from_dsn(
        "host=db.example port=5433 user=scix password=secret dbname=scix_test",
        run_token="1234_abcd",
    )

    assert isolation.source_database == "scix_test"
    assert isolation.run_database == "scix_test_run_1234_abcd"
    assert conninfo_to_dict(isolation.run_dsn) == {
        "dbname": "scix_test_run_1234_abcd",
        "host": "db.example",
        "password": "secret",
        "port": "5433",
        "user": "scix",
    }
    assert conninfo_to_dict(isolation.admin_dsn)["dbname"] == "postgres"


@pytest.mark.parametrize(
    "dsn",
    [
        "dbname=scix",
        "dbname=another_test",
        "host=localhost",
    ],
)
def test_from_dsn_rejects_non_scix_test_sources(dsn: str) -> None:
    with pytest.raises(ValueError, match="scix_test"):
        DatabaseIsolation.from_dsn(dsn, run_token="1234_abcd")


@pytest.mark.parametrize("run_token", ["", "../../bad", "has spaces", "x" * 60])
def test_from_dsn_rejects_unsafe_run_tokens(run_token: str) -> None:
    with pytest.raises(ValueError, match="run token"):
        DatabaseIsolation.from_dsn("dbname=scix_test", run_token=run_token)


def test_database_isolation_is_immutable() -> None:
    isolation = DatabaseIsolation.from_dsn("dbname=scix_test", run_token="1234_abcd")

    with pytest.raises(FrozenInstanceError):
        isolation.run_database = "changed"  # type: ignore[misc]


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self) -> _RecordingConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: object) -> None:
        self.statements.append(query.as_string(None))


def test_create_and_drop_use_autocommit_admin_connection() -> None:
    isolation = DatabaseIsolation.from_dsn("dbname=scix_test", run_token="1234_abcd")
    connections: list[_RecordingConnection] = []
    calls: list[tuple[str, bool]] = []

    def connect(dsn: str, *, autocommit: bool) -> _RecordingConnection:
        calls.append((dsn, autocommit))
        connection = _RecordingConnection()
        connections.append(connection)
        return connection

    isolation.create(connect=connect)
    isolation.drop(connect=connect)

    assert calls == [(isolation.admin_dsn, True), (isolation.admin_dsn, True)]
    assert connections[0].statements == [
        'CREATE DATABASE "scix_test_run_1234_abcd" TEMPLATE "scix_test"'
    ]
    assert connections[1].statements == [
        'DROP DATABASE IF EXISTS "scix_test_run_1234_abcd" WITH (FORCE)'
    ]
