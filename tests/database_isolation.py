"""Create and remove a private PostgreSQL database for one pytest run."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

_RUN_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{1,32}")
_RUN_DATABASE_PATTERN = re.compile(r"scix_test_run_(?P<pid>[0-9]+)_[a-zA-Z0-9_]{1,32}")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Connection(Protocol):
    def __enter__(self) -> _Connection: ...

    def __exit__(self, *args: object) -> None: ...

    def execute(self, query: object) -> object: ...


Connect = Callable[..., _Connection]


@dataclass(frozen=True)
class DatabaseIsolation:
    """Connection details and lifecycle for a run-scoped test database."""

    source_dsn: str
    source_database: str
    run_dsn: str
    run_database: str
    admin_dsn: str

    @classmethod
    def from_dsn(cls, source_dsn: str, *, run_token: str) -> DatabaseIsolation:
        """Build safe source, run, and maintenance DSNs without connecting."""
        parameters = conninfo_to_dict(source_dsn)
        source_database = parameters.get("dbname")
        if not isinstance(source_database, str) or not source_database.startswith("scix_test"):
            raise ValueError("SCIX_TEST_DSN must name a scix_test database")
        if _RUN_TOKEN_PATTERN.fullmatch(run_token) is None:
            raise ValueError("pytest database run token is unsafe")

        run_database = f"scix_test_run_{run_token}"
        if _RUN_DATABASE_PATTERN.fullmatch(run_database) is None:
            raise ValueError("pytest database run token is too long")
        return cls(
            source_dsn=source_dsn,
            source_database=source_database,
            run_dsn=make_conninfo(source_dsn, dbname=run_database),
            run_database=run_database,
            admin_dsn=make_conninfo(source_dsn, dbname="postgres"),
        )

    def create(self, *, connect: Connect = psycopg.connect) -> None:
        """Clone the configured schema database into this run's database."""
        with connect(self.admin_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                    sql.Identifier(self.run_database),
                    sql.Identifier(self.source_database),
                )
            )

    def drop_stale(
        self,
        *,
        pid_is_alive: Callable[[int], bool] = _pid_is_alive,
        connect: Connect = psycopg.connect,
    ) -> None:
        with connect(self.admin_dsn, autocommit=True) as connection:
            databases = connection.execute(
                sql.SQL(
                    "SELECT database.datname, "
                    "EXISTS (SELECT 1 FROM pg_stat_activity AS activity "
                    "WHERE activity.datname = database.datname) "
                    "FROM pg_database AS database "
                    "WHERE database.datname LIKE 'scix_test_run_%'"
                )
            )
            for database, has_connections in databases:
                match = _RUN_DATABASE_PATTERN.fullmatch(database)
                if match is None or has_connections or pid_is_alive(int(match.group("pid"))):
                    continue
                connection.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database))
                )

    def drop(self, *, connect: Connect = psycopg.connect) -> None:
        """Remove this run's database, terminating leaked test connections."""
        if _RUN_DATABASE_PATTERN.fullmatch(self.run_database) is None:
            raise ValueError("refusing to drop an unsafe pytest database name")
        with connect(self.admin_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(self.run_database)
                )
            )
