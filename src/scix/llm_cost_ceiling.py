"""LLM cost ceiling enforcement (PRD §D3c).

Per-day and per-query USD caps enforced via a Postgres-backed ledger.
Default caps:
    - Per day:   $50.00
    - Per query: $0.01

The ledger table is `llm_cost_ledger(day DATE PK, total_usd NUMERIC,
call_count INT)` and is created idempotently on first use.

Usage:

    if not check_and_reserve(estimated_cost_usd):
        # Budget exceeded; fall back to a cheaper path or bail.
        return
    try:
        result = call_llm(...)
    finally:
        record_actual(actual_cost_usd)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import psycopg

from scix.db import DEFAULT_DSN, get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DAILY_CAP_USD: float = float(os.environ.get("SCIX_LLM_DAILY_CAP_USD", "50.0"))
DEFAULT_PER_QUERY_CAP_USD: float = float(os.environ.get("SCIX_LLM_PER_QUERY_CAP_USD", "0.01"))

#: Rough Haiku-tier pricing ($ per 1M tokens). Used only for ceiling
#: estimation; actual costs should come from the upstream API response.
HAIKU_INPUT_PER_1M_TOKENS: float = 0.25
HAIKU_OUTPUT_PER_1M_TOKENS: float = 1.25


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_cost_ledger (
    day         DATE    PRIMARY KEY,
    total_usd   NUMERIC(12, 6) NOT NULL DEFAULT 0,
    call_count  INTEGER NOT NULL DEFAULT 0
)
"""

# Thread-local last-reservation amount, so `record_actual` can compute the
# delta from estimate -> actual and adjust the ledger accordingly.
_state = threading.local()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostCaps:
    """Immutable caps bundle."""

    daily_usd: float = DEFAULT_DAILY_CAP_USD
    per_query_usd: float = DEFAULT_PER_QUERY_CAP_USD


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_per_1m: float = HAIKU_INPUT_PER_1M_TOKENS,
    output_per_1m: float = HAIKU_OUTPUT_PER_1M_TOKENS,
) -> float:
    """Return a conservative USD cost estimate for a Haiku-tier call."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    cost = (prompt_tokens / 1_000_000.0) * input_per_1m + (
        completion_tokens / 1_000_000.0
    ) * output_per_1m
    return float(cost)


def ensure_ledger(conn: psycopg.Connection) -> None:
    """Create the ledger table if it does not already exist."""
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
    conn.commit()


def check_and_reserve(
    estimated_cost_usd: float,
    *,
    dsn: Optional[str] = None,
    per_query_cap: float = DEFAULT_PER_QUERY_CAP_USD,
    daily_cap: float = DEFAULT_DAILY_CAP_USD,
) -> bool:
    """Reserve budget for an LLM call.

    Returns True if the call is within both caps AND the reservation has been
    persisted to the ledger; False otherwise. A False return means the caller
    MUST NOT make the call.

    Caller contract: after a successful reservation, call :func:`record_actual`
    with the real cost so the ledger can adjust the delta.
    """
    if estimated_cost_usd < 0:
        raise ValueError("estimated_cost_usd must be non-negative")

    if estimated_cost_usd > per_query_cap:
        logger.warning(
            "LLM call rejected: estimated %.6f > per-query cap %.6f",
            estimated_cost_usd,
            per_query_cap,
        )
        return False

    conn = get_connection(dsn or DEFAULT_DSN, autocommit=False)
    try:
        ensure_ledger(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT total_usd FROM llm_cost_ledger " "WHERE day = CURRENT_DATE FOR UPDATE"
            )
            row = cur.fetchone()
            current = float(row[0]) if row is not None else 0.0

            projected = current + float(estimated_cost_usd)
            if projected > daily_cap:
                conn.rollback()
                logger.warning(
                    "LLM call rejected: day total %.6f + est %.6f > daily cap %.6f",
                    current,
                    estimated_cost_usd,
                    daily_cap,
                )
                return False

            cur.execute(
                """
                INSERT INTO llm_cost_ledger (day, total_usd, call_count)
                VALUES (CURRENT_DATE, %s, 1)
                ON CONFLICT (day) DO UPDATE
                    SET total_usd  = llm_cost_ledger.total_usd + EXCLUDED.total_usd,
                        call_count = llm_cost_ledger.call_count + 1
                """,
                (Decimal(str(estimated_cost_usd)),),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _state.last_reservation_usd = float(estimated_cost_usd)
    return True


def record_actual(
    actual_cost_usd: float,
    *,
    dsn: Optional[str] = None,
) -> None:
    """Adjust the ledger to reflect the actual (vs estimated) cost of the
    most recent :func:`check_and_reserve` call on this thread.

    If no reservation has been made, treats the entire amount as new charge.
    """
    if actual_cost_usd < 0:
        raise ValueError("actual_cost_usd must be non-negative")

    last_reserved = float(getattr(_state, "last_reservation_usd", 0.0))
    _state.last_reservation_usd = 0.0
    delta = float(actual_cost_usd) - last_reserved

    if delta == 0:
        return

    conn = get_connection(dsn or DEFAULT_DSN, autocommit=False)
    try:
        ensure_ledger(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_cost_ledger (day, total_usd, call_count)
                VALUES (CURRENT_DATE, %s, 0)
                ON CONFLICT (day) DO UPDATE
                    SET total_usd = llm_cost_ledger.total_usd + EXCLUDED.total_usd
                """,
                (Decimal(str(delta)),),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_daily_total_usd(*, dsn: Optional[str] = None) -> float:
    """Return today's accumulated USD total, or 0 if no calls yet."""
    conn = get_connection(dsn or DEFAULT_DSN, autocommit=False)
    try:
        ensure_ledger(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT total_usd FROM llm_cost_ledger WHERE day = CURRENT_DATE")
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    return float(row[0]) if row else 0.0


__all__ = [
    "DEFAULT_DAILY_CAP_USD",
    "DEFAULT_PER_QUERY_CAP_USD",
    "HAIKU_INPUT_PER_1M_TOKENS",
    "HAIKU_OUTPUT_PER_1M_TOKENS",
    "CostCaps",
    "estimate_cost_usd",
    "ensure_ledger",
    "check_and_reserve",
    "record_actual",
    "get_daily_total_usd",
]
