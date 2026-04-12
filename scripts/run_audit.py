#!/usr/bin/env python3
"""Run the M9 entity-link audit harness end-to-end.

Usage::

    # Fixture mode (seeds scix_test, runs sampler + stub judge, writes report)
    SCIX_TEST_DSN=dbname=scix_test python scripts/run_audit.py --fixture

    # Live mode (read-only sampler over whatever database the DSN points at)
    python scripts/run_audit.py --db-url "dbname=scix_test" --n-per-tier 125

Writes ``build-artifacts/eval_report.md`` with Wilson 95% CIs per tier.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import psycopg

from scix.eval.audit import (
    AuditCandidate,
    sample_stratified,
    write_audit_report,
)
from scix.eval.llm_judge import LinkRow, judge

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = pathlib.Path("build-artifacts/eval_report.md")
DEFAULT_N_PER_TIER = 125

# Fixture seed data — small enough to run quickly in scix_test.
_FIXTURE_BIBCODES = [f"2099FIXT..{i:03d}A" for i in range(12)]
_FIXTURE_TIERS = (1, 2, 4, 5)
_FIXTURE_N_PER_TIER = 6  # → 24 total rows so the report has content per tier


# ---------------------------------------------------------------------------
# Fixture helpers (scix_test only)
# ---------------------------------------------------------------------------


def _guard_not_production(dsn: str) -> None:
    """Refuse to mutate production."""
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() == "scix":
                raise RuntimeError("refusing to run --fixture against production dbname=scix")


def _seed_fixture(conn: psycopg.Connection) -> None:
    """Seed a small (papers, entities, document_entities) fixture.

    Writes to ``document_entities`` are flagged with ``# noqa: resolver-lint``
    because this is a read-only analytics path test harness, not the
    resolver write path enforced by M13.
    """
    with conn.cursor() as cur:
        # Papers — we rely on (bibcode) PK uniqueness.
        cur.executemany(
            """
            INSERT INTO papers (bibcode, title)
            VALUES (%s, %s)
            ON CONFLICT (bibcode) DO NOTHING
            """,
            [(b, f"fixture paper {b}") for b in _FIXTURE_BIBCODES],
        )

        # Entities — canonical_name is usually required; we use a deterministic id range.
        cur.execute("""
            INSERT INTO entities (canonical_name, entity_type, source)
            SELECT 'm9_fixture_entity_' || g::text, 'concept', 'm9_fixture'
            FROM generate_series(1, 8) AS g
            ON CONFLICT DO NOTHING
            """)
        cur.execute("""
            SELECT id FROM entities
            WHERE source = 'm9_fixture'
            ORDER BY id
            LIMIT 8
            """)
        entity_ids = [int(r[0]) for r in cur.fetchall()]
        if not entity_ids:
            raise RuntimeError("fixture: failed to seed m9_fixture entities")

        rows: list[tuple[str, int, str, int, float]] = []
        for tier_idx, tier in enumerate(_FIXTURE_TIERS):
            for i in range(_FIXTURE_N_PER_TIER):
                bib = _FIXTURE_BIBCODES[
                    (tier_idx * _FIXTURE_N_PER_TIER + i) % len(_FIXTURE_BIBCODES)
                ]
                eid = entity_ids[i % len(entity_ids)]
                link_type = f"m9_fixture_tier{tier}"
                rows.append((bib, eid, link_type, tier, 0.5 + 0.01 * i))

        # Insert into document_entities. PK is (bibcode, entity_id, link_type, tier),
        # so link_type differentiates per-tier fixture rows.
        cur.executemany(
            "INSERT INTO document_entities (bibcode, entity_id, link_type, tier, confidence) "  # noqa: resolver-lint
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            rows,
        )

    conn.commit()
    logger.info(
        "seeded fixture: %d papers, %d entities, %d document_entities rows",
        len(_FIXTURE_BIBCODES),
        8,
        len(_FIXTURE_TIERS) * _FIXTURE_N_PER_TIER,
    )


def _cleanup_fixture(conn: psycopg.Connection) -> None:
    """Drop the fixture rows we seeded."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE link_type LIKE 'm9_fixture_tier%'"  # noqa: resolver-lint
        )
        cur.execute("DELETE FROM entities WHERE source = 'm9_fixture'")
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (_FIXTURE_BIBCODES,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _fixture_candidate_filter(rows: list[AuditCandidate]) -> list[AuditCandidate]:
    """Keep only fixture-synthetic bibcodes so --fixture runs stay deterministic."""
    return [c for c in rows if c.bibcode in set(_FIXTURE_BIBCODES)]


def run(
    *,
    dsn: str,
    output_path: pathlib.Path,
    n_per_tier: int,
    fixture: bool,
) -> pathlib.Path:
    conn = psycopg.connect(dsn)
    try:
        if fixture:
            _guard_not_production(dsn)
            _seed_fixture(conn)

        # Sampler runs in its own transaction so setseed() scope is clean.
        candidates = sample_stratified(conn, n_per_tier=n_per_tier, seed=0.42)

        if fixture:
            candidates = _fixture_candidate_filter(candidates)

        link_rows = [
            LinkRow(tier=c.tier, bibcode=c.bibcode, entity_id=c.entity_id) for c in candidates
        ]
        judge_labels = judge(link_rows, use_real=False)

        labels_by_key: dict[tuple[str, int], str] = {
            (lbl.bibcode, lbl.entity_id): lbl.label for lbl in judge_labels
        }

        note = (
            "Judge labels sourced from deterministic stub — "
            "set `ANTHROPIC_API_KEY` + `use_real=True` for real judging."
        )
        out = write_audit_report(
            output_path,
            candidates,
            labels_by_key,
            note=note,
        )
    finally:
        if fixture:
            try:
                _cleanup_fixture(conn)
            except Exception as exc:  # pragma: no cover
                logger.warning("fixture cleanup failed: %s", exc)
        conn.close()
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="psycopg DSN; defaults to $SCIX_TEST_DSN then $SCIX_DSN then dbname=scix_test",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"markdown output path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--n-per-tier",
        type=int,
        default=DEFAULT_N_PER_TIER,
        help="number of rows to sample per distinct tier (default: 125)",
    )
    p.add_argument(
        "--fixture",
        action="store_true",
        help="seed a small fixture into SCIX_TEST_DSN, run end-to-end, then clean up",
    )
    p.add_argument("--verbose", "-v", action="store_true", default=False)
    return p


def _resolve_dsn(cli_dsn: str | None, *, fixture: bool) -> str:
    if cli_dsn:
        return cli_dsn
    dsn = os.environ.get("SCIX_TEST_DSN") or os.environ.get("SCIX_DSN")
    if dsn:
        return dsn
    if fixture:
        return "dbname=scix_test"
    return "dbname=scix_test"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = _resolve_dsn(args.db_url, fixture=args.fixture)
    logger.info("running audit against %s (fixture=%s)", dsn, args.fixture)

    try:
        out = run(
            dsn=dsn,
            output_path=args.output,
            n_per_tier=args.n_per_tier,
            fixture=args.fixture,
        )
    except psycopg.Error as exc:
        logger.error("database error: %s", exc)
        return 2

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
