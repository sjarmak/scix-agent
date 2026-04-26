#!/usr/bin/env python3
"""Batch driver for the nanopub-inspired claim extraction pipeline.

Selects bibcodes from ``papers_fulltext`` (joined with ``papers`` for optional
``arxiv_class`` / ``bibcode_glob`` filters), calls
:func:`scix.claims.extract.extract_claims_for_paper` for each, and prints a
JSON stats summary to stdout::

    {"processed": N, "claims_written": M, "skipped": S, "failed": F}

Idempotency
-----------
Re-running with the same input on the same database does NOT double-count.
The pipeline maintains a unique partial index on ``paper_claims`` so duplicate
inserts are dropped via ``ON CONFLICT DO NOTHING`` (or the SELECT-then-INSERT
fallback in :func:`scix.claims.extract._persist_claim_select_then_insert`).

Safety
------
- ``--llm`` defaults to ``stub`` so the script is safe to dry-run wired up.
  The stub returns no claims, which is useful for verifying selection /
  iteration / DB plumbing without invoking the ``claude`` CLI.
- Per-bibcode exceptions are caught (``failed += 1``) so one bad paper does
  not abort the run. A separate ``--strict`` flag re-raises on first failure
  for debugging.

Examples
--------
Dry-run wiring against scix_test (no claims emitted)::

    python scripts/extract_claims_batch.py \\
        --dsn "dbname=scix_test" --limit 10 --llm stub

Real extraction with the OAuth-authenticated ``claude`` CLI (no API key)::

    scix-batch python scripts/extract_claims_batch.py \\
        --dsn "dbname=scix" --limit 1000 --batch-size 50 \\
        --llm claude-cli --section-roles results,discussion,conclusion \\
        --prompt-version v1 --model-name claude-opus-4-7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

# Allow direct script invocation without `pip install -e .` — same pattern as
# scripts/run_claim_extractor.py and scripts/backfill_citation_intent.py.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scix.claims import (  # noqa: E402  (post-sys.path)
    ClaudeCliLLMClient,
    LLMClient,
    StubLLMClient,
    extract_claims_for_paper,
)
from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    redact_dsn,
)

logger = logging.getLogger("extract_claims_batch")


# ---------------------------------------------------------------------------
# Bibcode selection
# ---------------------------------------------------------------------------


def iter_bibcode_batches(
    conn: Any,
    *,
    limit: int,
    batch_size: int,
    arxiv_class: str | None,
    bibcode_glob: str | None,
) -> Iterator[list[str]]:
    """Yield batches of ``bibcode`` strings from ``papers_fulltext``.

    Filters:
      * ``arxiv_class`` — match if the value is in ``papers.arxiv_class``
        (a TEXT[] column). When provided, papers_fulltext is joined to papers.
      * ``bibcode_glob`` — SQL ``LIKE`` pattern on ``bibcode``
        (e.g. ``"2024%"`` for all 2024 papers).

    Bibcodes are yielded in deterministic order (lexicographic) so re-runs
    cover the same papers.
    """
    if limit <= 0:
        return

    # Build SELECT with optional joins and predicates. We always order by
    # bibcode so the run is reproducible and resumable.
    where_clauses: list[str] = []
    params: list[Any] = []

    join_papers = arxiv_class is not None
    base = (
        "SELECT pf.bibcode FROM papers_fulltext pf "
        + ("JOIN papers p ON p.bibcode = pf.bibcode " if join_papers else "")
    )

    if arxiv_class is not None:
        where_clauses.append("%s = ANY(p.arxiv_class)")
        params.append(arxiv_class)
    if bibcode_glob is not None:
        where_clauses.append("pf.bibcode LIKE %s")
        params.append(bibcode_glob)

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql_text = base + where_sql + " ORDER BY pf.bibcode LIMIT %s"
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql_text, tuple(params))
        rows = cur.fetchall()

    bibcodes = [row[0] for row in rows]
    if not bibcodes:
        return

    for start in range(0, len(bibcodes), batch_size):
        yield bibcodes[start : start + batch_size]


def fetch_paper_sections(conn: Any, bibcode: str) -> Sequence[dict[str, Any]] | None:
    """Return the ``sections`` JSONB list for a bibcode, or None if missing.

    Returns:
      * ``None`` if no row exists in ``papers_fulltext``.
      * ``[]`` if the row exists but ``sections`` is empty/null. The caller
        treats this as "skipped".
      * A list of section dicts otherwise.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sections FROM papers_fulltext WHERE bibcode = %s",
            (bibcode,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    sections = row[0]
    if sections is None:
        return []
    if not isinstance(sections, list):
        # Defensive: papers_fulltext.sections is JSONB NOT NULL but psycopg may
        # return dict for malformed rows; treat anything non-list as empty.
        return []
    return sections


# ---------------------------------------------------------------------------
# LLMClient construction
# ---------------------------------------------------------------------------


def build_llm_client(
    *,
    kind: str,
    stub_claims_json: str | None,
) -> LLMClient:
    """Construct an LLMClient based on the ``--llm`` flag.

    ``kind == "stub"``  -> :class:`StubLLMClient`. If ``stub_claims_json``
        is provided it's parsed as a JSON list; the same response is returned
        for every paragraph (via ``default``). Otherwise the stub returns ``[]``.
    ``kind == "claude-cli"`` -> :class:`ClaudeCliLLMClient` shells out to the
        OAuth-authenticated ``claude`` binary (no paid-API SDK).
    """
    if kind == "stub":
        if stub_claims_json:
            try:
                parsed = json.loads(stub_claims_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"--llm-stub-claims-json is not valid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise SystemExit(
                    "--llm-stub-claims-json must be a JSON array of claim dicts"
                )
            return StubLLMClient(default=parsed)
        return StubLLMClient(default=[])
    if kind == "claude-cli":
        return ClaudeCliLLMClient()
    raise SystemExit(f"unknown --llm value: {kind!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the nanopub-inspired claim extraction pipeline over a batch "
            "of papers from papers_fulltext. Prints a JSON stats line to stdout."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Bibcodes processed per progress-log batch (default: 50).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Cap total bibcodes selected from papers_fulltext (default: 100).",
    )
    p.add_argument(
        "--arxiv-class",
        default=None,
        help=(
            "Filter to papers whose papers.arxiv_class array contains this "
            "value (e.g. 'astro-ph.GA'). Omit to include all papers."
        ),
    )
    p.add_argument(
        "--bibcode-glob",
        default=None,
        help=(
            "SQL LIKE pattern on bibcode (e.g. '2024%%' for 2024 papers). "
            "Omit to include all bibcodes."
        ),
    )
    p.add_argument(
        "--section-roles",
        default=None,
        help=(
            "Comma-separated whitelist of section roles to process "
            "(e.g. 'results,discussion,conclusion'). Omit to process all "
            "sections (the pipeline default)."
        ),
    )
    p.add_argument(
        "--prompt-version",
        default="v1",
        help="Stored verbatim into paper_claims.extraction_prompt_version (default: v1).",
    )
    p.add_argument(
        "--model-name",
        default="claude-opus-4-7",
        help="Stored verbatim into paper_claims.extraction_model "
        "(default: claude-opus-4-7).",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help=(
            "Database DSN. Defaults to env SCIX_TEST_DSN, then SCIX_DSN, "
            "then 'dbname=scix' (the project default)."
        ),
    )
    p.add_argument(
        "--llm",
        choices=("stub", "claude-cli"),
        default="stub",
        help=(
            "LLMClient implementation. 'stub' (default) uses StubLLMClient — "
            "produces NO claims; useful for testing wiring without invoking "
            "the claude CLI. 'claude-cli' shells out to the OAuth-authenticated "
            "claude binary (no paid-API SDK)."
        ),
    )
    p.add_argument(
        "--llm-stub-claims-json",
        default=None,
        help=(
            "JSON-encoded list of claim dicts to return from StubLLMClient on "
            "EVERY paragraph (test-only — keeps the harness hermetic). "
            "Ignored when --llm != stub."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Re-raise on the first per-bibcode failure (debugging). "
        "Default: log and continue (failed counter increments).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _resolve_dsn(explicit: str | None) -> str:
    """DSN resolution order: --dsn > SCIX_TEST_DSN > SCIX_DSN > DEFAULT_DSN."""
    if explicit:
        return explicit
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


def _parse_section_roles(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    roles = [r.strip() for r in raw.split(",") if r.strip()]
    return roles or None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # JSON summary goes to stdout — keep logs separate
    )

    dsn = _resolve_dsn(args.dsn)
    section_roles = _parse_section_roles(args.section_roles)

    llm = build_llm_client(
        kind=args.llm,
        stub_claims_json=args.llm_stub_claims_json,
    )

    logger.info(
        "starting batch (dsn=%s, limit=%d, batch_size=%d, llm=%s, "
        "arxiv_class=%s, bibcode_glob=%s, section_roles=%s, prompt_version=%s, "
        "model_name=%s)",
        redact_dsn(dsn),
        args.limit,
        args.batch_size,
        args.llm,
        args.arxiv_class,
        args.bibcode_glob,
        section_roles,
        args.prompt_version,
        args.model_name,
    )

    stats = {
        "processed": 0,
        "claims_written": 0,
        "skipped": 0,
        "failed": 0,
    }

    conn = get_connection(dsn)
    try:
        batch_no = 0
        for batch in iter_bibcode_batches(
            conn,
            limit=args.limit,
            batch_size=args.batch_size,
            arxiv_class=args.arxiv_class,
            bibcode_glob=args.bibcode_glob,
        ):
            batch_no += 1
            logger.info("batch %d: %d bibcodes", batch_no, len(batch))

            for bibcode in batch:
                stats["processed"] += 1
                try:
                    sections = fetch_paper_sections(conn, bibcode)
                    if not sections:
                        stats["skipped"] += 1
                        logger.debug(
                            "skipping %s — no sections in papers_fulltext", bibcode
                        )
                        continue
                    n_inserted = extract_claims_for_paper(
                        conn,
                        bibcode,
                        sections,
                        llm,
                        prompt_version=args.prompt_version,
                        model_name=args.model_name,
                        section_roles=section_roles,
                    )
                    stats["claims_written"] += int(n_inserted)
                except Exception as exc:  # noqa: BLE001 — fail-soft per acceptance criterion
                    if args.strict:
                        raise
                    stats["failed"] += 1
                    logger.warning(
                        "extraction failed for %s: %s", bibcode, exc, exc_info=False
                    )
                    # Roll back the aborted transaction so the next bibcode can
                    # run on a clean connection state.
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass

            logger.info(
                "progress: processed=%d claims_written=%d skipped=%d failed=%d",
                stats["processed"],
                stats["claims_written"],
                stats["skipped"],
                stats["failed"],
            )
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    # Final summary: stdout's last non-empty line MUST be parseable JSON.
    sys.stdout.write(json.dumps(stats) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
