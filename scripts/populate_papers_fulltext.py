#!/usr/bin/env python3
"""Populate ``papers_fulltext`` from the full-text parser ladder.

Driver for D1 of :file:`docs/prd/prd_structural_fulltext_parsing.md`.

Iterates :table:`papers` in bibcode-sorted chunks, builds a
:class:`~scix.sources.route.RouteInput` for each bibcode, dispatches
through :func:`~scix.sources.route.route_fulltext_request`, invokes the
matching tier parser, and bulk-writes parsed rows into
:table:`papers_fulltext` via COPY. Parse failures are recorded in
:table:`papers_fulltext_failures` with R15 exponential backoff
(24h / 3d / 7d / 30d).

Safety
------
The script refuses to run against the production DSN unless both
``--allow-prod`` is passed AND the ``INVOCATION_ID`` environment variable
is set (which ``scix-batch`` / ``systemd-run --scope`` set automatically).
See the "Memory isolation — coexisting with gascity" section of
:file:`CLAUDE.md` for the rationale.

Usage
-----

.. code-block:: bash

    # Against scix_test:
    SCIX_DSN="dbname=scix_test" python scripts/populate_papers_fulltext.py

    # Against production (via scix-batch for memory isolation):
    scix-batch python scripts/populate_papers_fulltext.py --allow-prod

    # Resume after a crash / partial run:
    python scripts/populate_papers_fulltext.py --resume-from 2024ApJ...900A...1X
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix import db as _db  # noqa: E402
from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402
from scix.sources import ads_body_parser  # noqa: E402
from scix.sources.route import (  # noqa: E402
    RouteDecision,
    RouteInput,
    route_fulltext_request,
)

logger = logging.getLogger("populate_papers_fulltext")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 10_000

# R15 exponential backoff: attempt -> delay.
_BACKOFF_LADDER: Mapping[int, timedelta] = {
    1: timedelta(hours=24),
    2: timedelta(days=3),
    3: timedelta(days=7),
}
_BACKOFF_DEFAULT: timedelta = timedelta(days=30)
# Non-retry reasons: record a far-future retry_after so we don't loop.
_FAR_FUTURE_RETRY: timedelta = timedelta(days=365)

# parser_version pins. For Tier 1 we reuse ADS parser. For sibling clones
# we synthesize a tag; the sibling's own ``source`` is preserved in the row.
ADS_PARSER_VERSION: str = ads_body_parser.PARSER_VERSION
AR5IV_PARSER_VERSION: str = "ar5iv_html@v1"

_COPY_COLS: tuple[str, ...] = (
    "bibcode",
    "source",
    "sections",
    "inline_cites",
    "figures",
    "tables",
    "equations",
    "parser_version",
)
_COPY_SQL: str = f"COPY papers_fulltext ({', '.join(_COPY_COLS)}) FROM STDIN"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriverConfig:
    """Immutable driver configuration (post-argparse)."""

    dsn: str
    chunk_size: int
    resume_from: str | None
    limit: int | None
    allow_prod: bool
    # When set, the driver runs in *reparse* mode: instead of selecting
    # new papers that lack a papers_fulltext row, it selects existing
    # rows whose ``parser_version`` matches this value (and whose
    # ``source`` is ``'ads_body'``) and re-runs the parser, UPSERTing
    # the result via ``INSERT ... ON CONFLICT (bibcode) DO UPDATE``.
    # Default None preserves the original COPY-based new-row workflow.
    reparse_from_version: str | None = None


@dataclass(frozen=True)
class ParsedRow:
    """A parsed papers_fulltext row ready for COPY."""

    bibcode: str
    source: str
    sections_json: str
    inline_cites_json: str
    figures_json: str
    tables_json: str
    equations_json: str
    parser_version: str


@dataclass
class DriverStats:
    """Mutable counters tracked across a driver run (logged on exit)."""

    seen: int = 0
    wrote: int = 0
    skipped_existing: int = 0
    failures: int = 0
    served_sibling: int = 0
    abstract_only: int = 0
    tier3_skipped: int = 0
    # Number of bibcodes upserted in --reparse-from-version mode. Stays
    # 0 in the default (new-row) mode where ``wrote`` is incremented by
    # the COPY writer instead.
    reparsed: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)

    def bump_tier(self, tier: str) -> None:
        self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "wrote": self.wrote,
            "skipped_existing": self.skipped_existing,
            "failures": self.failures,
            "served_sibling": self.served_sibling,
            "abstract_only": self.abstract_only,
            "tier3_skipped": self.tier3_skipped,
            "reparsed": self.reparsed,
            "tier_counts": dict(self.tier_counts),
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_retry_after(attempts: int, now: datetime) -> datetime:
    """Return the ``retry_after`` timestamp for a failure with ``attempts``.

    R15 exponential backoff: 1 -> +24h, 2 -> +3d, 3 -> +7d, >=4 -> +30d.
    """
    delta = _BACKOFF_LADDER.get(attempts, _BACKOFF_DEFAULT)
    return now + delta


def _first_or_none(values: Any) -> str | None:
    """Return the first element of a list / None / scalar, else None."""
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return values[0] if values else None
    if isinstance(values, str):
        return values or None
    return None


def build_route_input(
    row: Mapping[str, Any],
    *,
    has_fulltext_row: bool,
    sibling_row_source: str | None,
) -> RouteInput:
    """Flatten a ``papers`` row into a :class:`RouteInput`.

    ``papers.doi`` and ``papers.bibstem`` are ``TEXT[]`` — we take the
    first element. ``papers_external_ids.openalex_has_pdf_url`` is
    expected on the row dict (defaulted to False if missing).
    """
    body: str = row.get("body") or ""
    doi = _first_or_none(row.get("doi"))
    doctype = row.get("doctype")
    openalex_has_pdf_url = bool(row.get("openalex_has_pdf_url", False))
    return RouteInput(
        bibcode=row["bibcode"],
        has_fulltext_row=has_fulltext_row,
        sibling_row_source=sibling_row_source,
        has_ads_body=bool(body),
        doctype=doctype,
        doi=doi,
        openalex_has_pdf_url=openalex_has_pdf_url,
    )


def _sections_to_json(sections: Sequence[Any]) -> str:
    """Serialize a list of Section (or similar) dataclasses to JSON."""
    out: list[dict[str, Any]] = []
    for s in sections:
        if dataclasses.is_dataclass(s):
            out.append(dataclasses.asdict(s))
        elif isinstance(s, dict):
            out.append(s)
        else:
            raise TypeError(f"cannot serialize section of type {type(s)!r}")
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Production guard
# ---------------------------------------------------------------------------


class ProdGuardError(SystemExit):
    """Raised (as SystemExit) when prod-guard policy is violated."""


def resolve_prod_guard(
    *,
    dsn: str,
    allow_prod: bool,
    env: Mapping[str, str],
) -> None:
    """Enforce production-safety policy.

    Raises :class:`ProdGuardError` (a SystemExit) if:

    * DSN looks like production AND ``--allow-prod`` was not passed, OR
    * ``--allow-prod`` was passed AND ``INVOCATION_ID`` is not set in
      the environment (i.e. we are not inside a systemd-managed scope).
      This mirrors the guard in ``scripts/recompute_citation_communities.py``;
      ``systemd-run --scope`` sets ``INVOCATION_ID`` but *not* ``SYSTEMD_SCOPE``.

    Callers should let the exception propagate — this is a policy guard,
    not a best-effort.
    """
    if is_production_dsn(dsn) and not allow_prod:
        msg = (
            f"refusing to write to production DSN {redact_dsn(dsn)} — "
            "pass --allow-prod to override"
        )
        logger.error(msg)
        raise ProdGuardError(2)

    if allow_prod and not env.get("INVOCATION_ID"):
        msg = (
            "refusing to run --allow-prod outside a systemd scope. "
            "Invoke via: scix-batch python scripts/populate_papers_fulltext.py ..."
        )
        logger.error(msg)
        raise ProdGuardError(2)


# ---------------------------------------------------------------------------
# Sibling row fetch (stub point for future work)
# ---------------------------------------------------------------------------


def fetch_sibling_row(conn: psycopg.Connection, bibcode: str) -> dict[str, Any] | None:
    """Return a LaTeX-derived sibling ``papers_fulltext`` row, or None.

    A "sibling" is a different bibcode that shares an identifier (DOI,
    arxiv_id) with ``bibcode`` and has an ``ar5iv``/``arxiv_local`` row
    already in ``papers_fulltext``. Full cross-identifier resolution is
    an enhancement — for now we return None if no such row exists for
    the exact bibcode lookup hook, which keeps the driver functional
    and lets tests exercise the branch by monkeypatching this function.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pf.bibcode, pf.source, pf.sections, pf.inline_cites,
                   pf.figures, pf.tables, pf.equations, pf.parser_version
              FROM papers_external_ids x
              JOIN papers_external_ids y
                ON (x.doi IS NOT NULL AND x.doi = y.doi
                    AND y.bibcode <> x.bibcode)
                   OR (x.arxiv_id IS NOT NULL AND x.arxiv_id = y.arxiv_id
                       AND y.bibcode <> x.bibcode)
              JOIN papers_fulltext pf ON pf.bibcode = y.bibcode
             WHERE x.bibcode = %s
               AND pf.source IN ('ar5iv', 'arxiv_local')
             LIMIT 1
            """,
            (bibcode,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d.name for d in cur.description]
        return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Candidate iteration
# ---------------------------------------------------------------------------


def iter_candidate_papers(
    conn: psycopg.Connection,
    config: DriverConfig,
    now: datetime,
) -> Iterator[dict[str, Any]]:
    """Yield dicts of candidate papers in bibcode order.

    Default (new-row) mode — skips:

    * bibcodes ``<= config.resume_from`` (inclusive of the resume key).
    * bibcodes that already have a ``papers_fulltext`` row (idempotency).
    * bibcodes whose latest ``papers_fulltext_failures`` row has
      ``retry_after > now`` (backoff window).

    Reparse mode (``config.reparse_from_version`` is not None) — selects:

    * bibcodes whose ``papers_fulltext.parser_version = OLD_VERSION``
      AND ``source = 'ads_body'``, joined back to ``papers`` for the
      body text. Does NOT apply the retry-after / NOT-EXISTS filters —
      reparse explicitly targets already-parsed rows.

    Uses a server-side cursor so the full result set is streamed rather
    than materialised.
    """
    resume = config.resume_from or ""

    if config.reparse_from_version is not None:
        # Reparse branch: select existing ads_body rows at the old version
        # and join back to papers for the body text.
        params: list[Any] = [config.reparse_from_version, resume]
        sql_parts = [
            "SELECT p.bibcode, p.bibstem, p.doi, p.doctype, p.body,",
            "       COALESCE(x.openalex_has_pdf_url, false) AS openalex_has_pdf_url",
            "  FROM papers_fulltext pf",
            "  JOIN papers p ON p.bibcode = pf.bibcode",
            "  LEFT JOIN papers_external_ids x ON x.bibcode = p.bibcode",
            " WHERE pf.parser_version = %s",
            "   AND pf.source = 'ads_body'",
            "   AND pf.bibcode > %s",
            " ORDER BY pf.bibcode",
        ]
        if config.limit is not None:
            sql_parts.append(" LIMIT %s")
            params.append(config.limit)
    else:
        # Default branch: only consider papers that have a full-text source
        # the driver can use right now: an ADS body (Tier 1) or a
        # LaTeX-derived sibling row already in papers_fulltext (Tier 2).
        # Papers with neither route to abstract_only, which is an expected
        # steady-state for ~half the corpus and should not be scanned on
        # every run or recorded as failures. When Tier 2 ingest lands, the
        # sibling subquery below keeps those bibcodes in scope.
        params = [resume, now]
        sql_parts = [
            "SELECT p.bibcode, p.bibstem, p.doi, p.doctype, p.body,",
            "       COALESCE(x.openalex_has_pdf_url, false) AS openalex_has_pdf_url",
            "  FROM papers p",
            "  LEFT JOIN papers_external_ids x ON x.bibcode = p.bibcode",
            " WHERE p.bibcode > %s",
            "   AND (",
            "       p.body IS NOT NULL",
            "       OR EXISTS (",
            "           SELECT 1",
            "             FROM papers_external_ids x1",
            "             JOIN papers_external_ids x2",
            "               ON (x1.doi IS NOT NULL AND x1.doi = x2.doi",
            "                   AND x2.bibcode <> x1.bibcode)",
            "                  OR (x1.arxiv_id IS NOT NULL AND x1.arxiv_id = x2.arxiv_id",
            "                      AND x2.bibcode <> x1.bibcode)",
            "             JOIN papers_fulltext pf2 ON pf2.bibcode = x2.bibcode",
            "            WHERE x1.bibcode = p.bibcode",
            "              AND pf2.source IN ('ar5iv', 'arxiv_local'))",
            "   )",
            "   AND NOT EXISTS (",
            "       SELECT 1 FROM papers_fulltext pf WHERE pf.bibcode = p.bibcode)",
            "   AND NOT EXISTS (",
            "       SELECT 1 FROM papers_fulltext_failures f",
            "        WHERE f.bibcode = p.bibcode AND f.retry_after > %s)",
            " ORDER BY p.bibcode",
        ]
        if config.limit is not None:
            sql_parts.append(" LIMIT %s")
            params.append(config.limit)

    query = "\n".join(sql_parts)

    # Named (server-side) cursor for memory-efficient streaming.
    with conn.cursor(name="populate_papers_fulltext_iter") as cur:
        cur.itersize = max(config.chunk_size, 1)
        cur.execute(query, params)
        col_names: list[str] | None = None
        for row in cur:
            if col_names is None:
                col_names = [d.name for d in cur.description]
            yield dict(zip(col_names, row))


# ---------------------------------------------------------------------------
# Tier dispatch
# ---------------------------------------------------------------------------


def _build_tier1_row(
    row: Mapping[str, Any],
    parse_fn: Callable[[str, str | None], tuple[list[Any], dict]],
) -> ParsedRow:
    """Invoke the Tier 1 parser and build a ParsedRow."""
    body: str = row.get("body") or ""
    bibstem = _first_or_none(row.get("bibstem"))
    sections, _meta = parse_fn(body, bibstem)
    return ParsedRow(
        bibcode=row["bibcode"],
        source="ads_body",
        sections_json=_sections_to_json(sections),
        inline_cites_json="[]",
        figures_json="[]",
        tables_json="[]",
        equations_json="[]",
        parser_version=ADS_PARSER_VERSION,
    )


def _clone_sibling_row(
    row: Mapping[str, Any],
    sibling: Mapping[str, Any],
) -> ParsedRow:
    """Clone a sibling's papers_fulltext row for the current bibcode.

    Preserves the sibling's ``source`` string (so provenance survives)
    but re-stamps ``parser_version`` with our driver's ar5iv pin.
    """

    def _jsonify(v: Any) -> str:
        if v is None:
            return "[]"
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False)

    return ParsedRow(
        bibcode=row["bibcode"],
        source=str(sibling.get("source", "ar5iv")),
        sections_json=_jsonify(sibling.get("sections")),
        inline_cites_json=_jsonify(sibling.get("inline_cites")),
        figures_json=_jsonify(sibling.get("figures")),
        tables_json=_jsonify(sibling.get("tables")),
        equations_json=_jsonify(sibling.get("equations")),
        parser_version=AR5IV_PARSER_VERSION,
    )


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


def write_batch(conn: psycopg.Connection, rows: Sequence[ParsedRow]) -> int:
    """Bulk-write parsed rows via COPY. Returns rows written."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        with cur.copy(_COPY_SQL) as copy:
            for r in rows:
                copy.write_row(
                    (
                        r.bibcode,
                        r.source,
                        r.sections_json,
                        r.inline_cites_json,
                        r.figures_json,
                        r.tables_json,
                        r.equations_json,
                        r.parser_version,
                    )
                )
    conn.commit()
    return len(rows)


_UPSERT_SQL = """
INSERT INTO papers_fulltext
    (bibcode, source, sections, inline_cites, figures, tables, equations,
     parser_version, parsed_at)
VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, now())
ON CONFLICT (bibcode) DO UPDATE SET
    sections       = EXCLUDED.sections,
    inline_cites   = EXCLUDED.inline_cites,
    parser_version = EXCLUDED.parser_version,
    parsed_at      = now()
"""


def upsert_batch(conn: psycopg.Connection, rows: Sequence[ParsedRow]) -> int:
    """Upsert parsed rows via ``INSERT ... ON CONFLICT DO UPDATE``.

    Used only in ``--reparse-from-version`` mode where existing rows must
    be replaced in-place rather than newly inserted. Returns the number
    of rows upserted.
    """
    if not rows:
        return 0
    params = [
        (
            r.bibcode,
            r.source,
            r.sections_json,
            r.inline_cites_json,
            r.figures_json,
            r.tables_json,
            r.equations_json,
            r.parser_version,
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, params)
    conn.commit()
    return len(rows)


_FAILURE_UPSERT_SQL = """
INSERT INTO papers_fulltext_failures
    (bibcode, parser_version, failure_reason, attempts,
     first_attempt, last_attempt, retry_after)
VALUES (%s, %s, %s, 1, now(), now(), %s)
ON CONFLICT (bibcode) DO UPDATE SET
    parser_version = EXCLUDED.parser_version,
    failure_reason = EXCLUDED.failure_reason,
    attempts       = papers_fulltext_failures.attempts + 1,
    last_attempt   = now(),
    retry_after    = now() + (CASE
        WHEN papers_fulltext_failures.attempts + 1 = 1 THEN interval '24 hours'
        WHEN papers_fulltext_failures.attempts + 1 = 2 THEN interval '3 days'
        WHEN papers_fulltext_failures.attempts + 1 = 3 THEN interval '7 days'
        ELSE interval '30 days' END)
"""


def record_failure(
    conn: psycopg.Connection,
    *,
    bibcode: str,
    parser_version: str,
    reason: str,
    initial_delay: timedelta,
) -> None:
    """Upsert a failure row, advancing attempts on conflict."""
    now = datetime.now(timezone.utc)
    retry_after = now + initial_delay
    with conn.cursor() as cur:
        cur.execute(
            _FAILURE_UPSERT_SQL,
            (bibcode, parser_version, reason, retry_after),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


def _dispatch_one(
    conn: psycopg.Connection,
    row: Mapping[str, Any],
    *,
    parse_fn: Callable[[str, str | None], tuple[list[Any], dict]],
    sibling_fetch: Callable[[psycopg.Connection, str], Mapping[str, Any] | None],
    stats: DriverStats,
) -> ParsedRow | None:
    """Route a single paper and either return a ParsedRow or record a failure.

    Returns ``None`` for rows that should not be written (abstract-only,
    tier3-not-wired, serve-existing short-circuit, or on parser error).
    """
    bibcode = row["bibcode"]

    # Optional sibling probe — a fast single-bibcode query (skippable under
    # test by monkeypatching sibling_fetch to return None).
    sibling_row: Mapping[str, Any] | None = None
    try:
        sibling_row = sibling_fetch(conn, bibcode)
    except Exception as exc:  # noqa: BLE001 — probe must not crash driver
        logger.warning("sibling_fetch failed for %s: %s", bibcode, exc)
        sibling_row = None

    sibling_source = sibling_row.get("source") if sibling_row is not None else None

    inp = build_route_input(
        row,
        has_fulltext_row=False,  # NOT-EXISTS join already filtered
        sibling_row_source=sibling_source,
    )
    decision: RouteDecision = route_fulltext_request(inp)
    stats.bump_tier(decision.tier)

    if decision.tier == "serve_existing":
        # Defensive: the iterator already filters these; log and skip.
        stats.skipped_existing += 1
        return None

    if decision.tier == "serve_sibling":
        assert sibling_row is not None
        stats.served_sibling += 1
        return _clone_sibling_row(row, sibling_row)

    if decision.tier == "tier1_ads_body":
        try:
            return _build_tier1_row(row, parse_fn)
        except Exception as exc:  # noqa: BLE001 — record as failure
            logger.warning("tier1 parse failed for %s: %s", bibcode, exc)
            record_failure(
                conn,
                bibcode=bibcode,
                parser_version=ADS_PARSER_VERSION,
                reason=f"tier1_parse_error:{type(exc).__name__}",
                initial_delay=_BACKOFF_LADDER[1],
            )
            stats.failures += 1
            return None

    if decision.tier == "tier3_docling":
        record_failure(
            conn,
            bibcode=bibcode,
            parser_version=ADS_PARSER_VERSION,
            reason="tier3_not_yet_wired",
            initial_delay=_BACKOFF_DEFAULT,
        )
        stats.tier3_skipped += 1
        stats.failures += 1
        return None

    # abstract_only — unreachable in normal flow because
    # iter_candidate_papers() only yields papers with a Tier 1 body or a
    # LaTeX-derived sibling. Kept as a defensive no-op for rows that slip
    # through (e.g. body becomes NULL between iteration and dispatch, or
    # a sibling row is deleted mid-run). Not recorded as a failure.
    stats.abstract_only += 1
    return None


def run(
    config: DriverConfig,
    *,
    parse_fn: Callable[[str, str | None], tuple[list[Any], dict]] | None = None,
    sibling_fetch: Callable[[psycopg.Connection, str], Mapping[str, Any] | None] | None = None,
    now: datetime | None = None,
) -> DriverStats:
    """Run the driver end-to-end. Returns accumulated stats.

    ``parse_fn`` and ``sibling_fetch`` are injectable for testing; they
    default to the real module-level functions.
    """
    if parse_fn is None:
        parse_fn = ads_body_parser.parse_ads_body
    if sibling_fetch is None:
        sibling_fetch = fetch_sibling_row
    if now is None:
        now = datetime.now(timezone.utc)

    stats = DriverStats()
    batch: list[ParsedRow] = []

    reparse_mode = config.reparse_from_version is not None

    def _flush(conn: psycopg.Connection, pending: list[ParsedRow]) -> None:
        """Write pending rows via the mode-appropriate writer."""
        if not pending:
            return
        if reparse_mode:
            n = upsert_batch(conn, pending)
            stats.reparsed += n
        else:
            n = write_batch(conn, pending)
            stats.wrote += n

    # Two connections: one for the streaming cursor, one for writes/upserts
    # (psycopg named cursors are not compatible with simultaneous DML on the
    # same connection without savepoint gymnastics).
    with _db.get_connection(config.dsn) as read_conn, _db.get_connection(config.dsn) as write_conn:
        write_conn.autocommit = False
        for row in iter_candidate_papers(read_conn, config, now):
            stats.seen += 1
            parsed = _dispatch_one(
                write_conn,
                row,
                parse_fn=parse_fn,
                sibling_fetch=sibling_fetch,
                stats=stats,
            )
            if parsed is not None:
                batch.append(parsed)
                if len(batch) >= config.chunk_size:
                    _flush(write_conn, batch)
                    batch.clear()
        # Final flush.
        if batch:
            _flush(write_conn, batch)
            batch.clear()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Populate papers_fulltext by routing each paper through the "
            "full-text parser ladder. Safe against production DSN unless "
            "--allow-prod is passed from inside a systemd scope."
        )
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help=(
            "PostgreSQL DSN (default: $SCIX_DSN or dbname=scix). "
            "Refuses production unless --allow-prod + INVOCATION_ID (systemd scope)."
        ),
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to target the production DSN (dbname=scix).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="BIBCODE",
        help=(
            "Skip bibcodes <= this value; used to resume after a crash. "
            "Existing papers_fulltext rows are always skipped independently."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            f"Rows per COPY batch (default {DEFAULT_CHUNK_SIZE}, capped at " f"{MAX_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on total papers to consider (default: no limit).",
    )
    parser.add_argument(
        "--reparse-from-version",
        default=None,
        metavar="OLD_VERSION",
        help=(
            "Reparse mode: re-run the parser over papers_fulltext rows whose "
            "parser_version = OLD_VERSION AND source = 'ads_body'. The write "
            "path switches from COPY to INSERT ... ON CONFLICT (bibcode) DO "
            "UPDATE so rows are replaced in-place. Cannot be combined with "
            "the default new-row mode — the iterator picks different rows. "
            "--resume-from still applies as a bibcode lower bound."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging (errors still printed).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    chunk_size = max(1, min(int(args.chunk_size), MAX_CHUNK_SIZE))
    config = DriverConfig(
        dsn=args.dsn,
        chunk_size=chunk_size,
        resume_from=args.resume_from,
        limit=args.limit,
        allow_prod=bool(args.allow_prod),
        reparse_from_version=args.reparse_from_version,
    )

    try:
        resolve_prod_guard(
            dsn=config.dsn,
            allow_prod=config.allow_prod,
            env=os.environ,
        )
    except ProdGuardError as exc:
        return int(exc.code) if exc.code is not None else 2

    logger.info(
        "starting driver: dsn=%s chunk_size=%d resume_from=%r limit=%r"
        " reparse_from_version=%r",
        redact_dsn(config.dsn),
        config.chunk_size,
        config.resume_from,
        config.limit,
        config.reparse_from_version,
    )

    stats = run(config)
    logger.info("driver complete: %s", json.dumps(stats.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
