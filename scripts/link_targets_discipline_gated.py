#!/usr/bin/env python3
"""Tier-3 discipline-gated SsODNet target linker — bead xz4.p2v.

Tier-1 keyword and Tier-2 abstract Aho-Corasick scans treat all entities
uniformly. Short SsODNet minor-body names (Ceres, Vesta, Eros, Juno,
Hygiea) collide with common English words and other-domain entities, so
the safety filters used by Tier-2 (``ambiguity_class IN
('unique','domain_safe','homograph')``) suppress them across the whole
corpus — only ~15K of 32M papers carry any target link today.

This tier relaxes the safety filter inside a planetary-science cohort:

* The cohort gate (``config/planetary_cohort.yaml``) restricts scanning
  to papers where short asteroid/comet/moon names are unlikely to be
  false positives — selected ``arxiv_class`` values, planetary
  bibstems, and a list of keyword sentinels.
* Within the cohort, every entity with ``entity_type='target' AND
  source='ssodnet'`` (and not demoted to ``link_policy='llm_only'``) is
  added to the automaton — including names with NULL ``ambiguity_class``
  that Tier 2 skips.
* Surfaces shorter than ``min_surface_length_unconditional`` still
  require an additional disambiguator token in the same title+abstract
  text, to catch e.g. ``Sun`` / ``Earth`` / ``Mars`` in non-planetary
  contexts.

Output rows go to ``document_entities`` with::

    link_type    = 'target_gated_match'
    tier         = 3
    tier_version = 1
    match_method = 'aho_corasick_planetary_cohort'

PK ``(bibcode, entity_id, link_type, tier)`` keeps these orthogonal to
existing tier-1/tier-2 writes; reruns are idempotent via ``ON CONFLICT
DO NOTHING``.

Usage::

    # Dry run on the test DB:
    SCIX_TEST_DSN=dbname=scix_test \\
      python scripts/link_targets_discipline_gated.py --dry-run -v

    # Production run (heavy — wrap in scix-batch):
    scix-batch python scripts/link_targets_discipline_gated.py --allow-prod
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence

import psycopg
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.aho_corasick import (  # noqa: E402
    AhocorasickAutomaton,
    EntityRow,
    LinkCandidate,
    build_automaton,
    link_abstract,
)
from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINK_TYPE: str = "target_gated_match"
TIER: int = 3
TIER_VERSION: int = 1
MATCH_METHOD: str = "aho_corasick_planetary_cohort"

DEFAULT_CONFIG_PATH: pathlib.Path = REPO_ROOT / "config" / "planetary_cohort.yaml"

PAPER_BATCH_SIZE: int = 512

# Confidence calibration — calibrated heuristics, not a learned model. The
# tier-3 confidence is meant to be comparable to tier-2's flat 0.85 only
# in the relative sense (higher = stronger match). u11 / M9 will replace
# this with a calibrated logistic regression.
CONFIDENCE_BASE: float = 0.70
CONFIDENCE_LONG_NAME_BONUS: float = 0.10  # surface_len >= 8
CONFIDENCE_VERY_LONG_NAME_BONUS: float = 0.05  # surface_len >= 12
CONFIDENCE_CANONICAL_BONUS: float = 0.05  # match was canonical_name (not alias)
CONFIDENCE_CONTEXT_BONUS: float = 0.05  # disambiguator token co-present
CONFIDENCE_REPEAT_BONUS: float = 0.05  # >=2 distinct hits for same entity
CONFIDENCE_MIN: float = 0.65
CONFIDENCE_MAX: float = 0.95

LONG_NAME_LEN: int = 8
VERY_LONG_NAME_LEN: int = 12

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortConfig:
    """Cohort gate + safety knobs from ``config/planetary_cohort.yaml``."""

    arxiv_classes: tuple[str, ...]
    bibstems: tuple[str, ...]
    keyword_sentinels: tuple[str, ...]
    min_surface_length_unconditional: int
    short_name_disambiguators: tuple[str, ...]
    stop_words: frozenset[str]
    extra_stop_words: tuple[str, ...]


def _load_stop_words(
    paths: Sequence[str], extras: Sequence[str]
) -> frozenset[str]:
    """Build a lowercase stop-word set from external word-list files plus
    operator-supplied extras.

    Each path is read line-by-line; lines that look like English words
    (alphabetic, ≥2 chars) are added. Possessive suffixes (``'s``) are
    stripped. Missing files are silently skipped — a missing system
    dictionary should not break tier-3, only weaken precision.
    """
    out: set[str] = set()
    for raw in paths:
        path = pathlib.Path(raw)
        if not path.exists():
            logger.warning("stop-words file not found: %s — skipping", path)
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    word = line.strip().split("'", 1)[0]
                    if len(word) >= 2 and word.isalpha():
                        out.add(word.lower())
        except OSError as exc:
            logger.warning("could not read stop-words file %s: %s", path, exc)
    for extra in extras:
        if extra:
            out.add(extra.strip().lower())
    return frozenset(out)


def load_cohort_config(path: pathlib.Path = DEFAULT_CONFIG_PATH) -> CohortConfig:
    """Parse ``planetary_cohort.yaml`` into a :class:`CohortConfig`."""
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    stop_paths = data.get("stop_words_files", []) or []
    extras = data.get("stop_words_extra", []) or []
    stop_words = _load_stop_words(stop_paths, extras)

    return CohortConfig(
        arxiv_classes=tuple(data.get("arxiv_classes", []) or []),
        bibstems=tuple(data.get("bibstems", []) or []),
        keyword_sentinels=tuple(
            (kw or "").lower() for kw in (data.get("keyword_sentinels", []) or [])
        ),
        min_surface_length_unconditional=int(data.get("min_surface_length_unconditional", 6)),
        short_name_disambiguators=tuple(
            (tok or "").lower() for tok in (data.get("short_name_disambiguators", []) or [])
        ),
        stop_words=stop_words,
        extra_stop_words=tuple((s or "").lower() for s in extras),
    )


# ---------------------------------------------------------------------------
# Entity fetching
# ---------------------------------------------------------------------------


_CANONICAL_SQL = """
    SELECT e.id,
           e.canonical_name AS surface,
           e.canonical_name,
           COALESCE(e.ambiguity_class::text, 'unknown') AS ambiguity_class
      FROM entities e
     WHERE e.entity_type = 'target'
       AND e.source = 'ssodnet'
       AND (e.link_policy IS NULL OR e.link_policy <> 'llm_only')
       AND (e.ambiguity_class IS NULL OR e.ambiguity_class <> 'banned')
"""

_ALIAS_SQL = """
    SELECT e.id,
           ea.alias AS surface,
           e.canonical_name,
           COALESCE(e.ambiguity_class::text, 'unknown') AS ambiguity_class
      FROM entities e
      JOIN entity_aliases ea ON ea.entity_id = e.id
     WHERE e.entity_type = 'target'
       AND e.source = 'ssodnet'
       AND (e.link_policy IS NULL OR e.link_policy <> 'llm_only')
       AND (e.ambiguity_class IS NULL OR e.ambiguity_class <> 'banned')
"""


def _is_excluded_surface(
    surface: str,
    *,
    stop_words: frozenset[str],
    min_canonical_len: int,
) -> bool:
    """Return True if ``surface`` should be excluded from the automaton.

    Filters in priority order:

    1. Stop-words: if the lower-case surface is in the curated stop-word
       set (English dictionary + scientific acronyms), drop it. This is
       what kills "The", "NOT", "May", "Field", "Apollo", etc. — words
       with legitimate asteroid namesakes that drown the signal in
       English-text false positives.
    2. Minimum length: short single-token surfaces (less than
       ``min_canonical_len`` chars) are dropped unconditionally because
       they collide with too many surnames, abbreviations, and
       designations even within the planetary cohort. Multi-token names
       (``"Comet Hale-Bopp"``) and designations containing digits
       (``"2005 VA"``) bypass the length check — the digit pattern is
       distinctive enough on its own.
    """
    s = surface.strip()
    if not s:
        return True
    if s.lower() in stop_words:
        return True
    has_digit = any(ch.isdigit() for ch in s)
    multi_token = (" " in s) or ("-" in s) or ("/" in s)
    if not has_digit and not multi_token and len(s) < min_canonical_len:
        return True
    return False


def fetch_target_entity_rows(
    conn: psycopg.Connection,
    *,
    stop_words: frozenset[str] = frozenset(),
    min_canonical_len: int = 7,
) -> list[EntityRow]:
    """Pull canonical + alias surfaces for every SsODNet target entity.

    Unlike Tier-2's ``fetch_entity_rows``, this includes entities with
    NULL ``ambiguity_class`` — the discipline gate is the disambiguator.
    Banned entities and entities demoted to ``link_policy='llm_only'``
    are excluded.

    The ``stop_words`` set drops every surface (canonical or alias) that
    matches a common English word. The ``min_canonical_len`` floor drops
    short single-token names that collide with surnames, abbreviations,
    and designations. Multi-token names (``"Comet Hale-Bopp"``) and
    designations with digits (``"2005 VA"``) bypass the length check.

    Every kept row is emitted with ``ambiguity_class='unique'`` so that
    the upstream :func:`scix.aho_corasick.link_abstract` does not apply
    its homograph long-form gate.
    """
    rows: list[EntityRow] = []
    excluded_canonical = 0
    excluded_alias = 0
    kept_entities: set[int] = set()

    with conn.cursor() as cur:
        cur.execute(_CANONICAL_SQL)
        for entity_id, surface, canonical, _ambig in cur.fetchall():
            if _is_excluded_surface(
                surface,
                stop_words=stop_words,
                min_canonical_len=min_canonical_len,
            ):
                excluded_canonical += 1
                continue
            rows.append(
                EntityRow(
                    entity_id=int(entity_id),
                    surface=surface,
                    canonical_name=canonical,
                    ambiguity_class="unique",
                    is_alias=False,
                )
            )
            kept_entities.add(int(entity_id))

        cur.execute(_ALIAS_SQL)
        for entity_id, surface, canonical, _ambig in cur.fetchall():
            # An alias is only useful if its parent entity survived the
            # canonical filter — otherwise we'd inject a false positive
            # via a stop-word alias of an already-suppressed entity.
            if int(entity_id) not in kept_entities:
                excluded_alias += 1
                continue
            if _is_excluded_surface(
                surface,
                stop_words=stop_words,
                min_canonical_len=min_canonical_len,
            ):
                excluded_alias += 1
                continue
            rows.append(
                EntityRow(
                    entity_id=int(entity_id),
                    surface=surface,
                    canonical_name=canonical,
                    ambiguity_class="unique",
                    is_alias=True,
                )
            )

    logger.info(
        "entity filter: kept %d surfaces (%d entities); "
        "dropped %d canonical + %d alias surfaces",
        len(rows),
        len(kept_entities),
        excluded_canonical,
        excluded_alias,
    )
    return rows


# ---------------------------------------------------------------------------
# Cohort SQL
# ---------------------------------------------------------------------------


def _build_cohort_sql(cfg: CohortConfig, bibcode_prefix: Optional[str]) -> tuple[str, list[Any]]:
    """Compose the SQL + params that streams ``(bibcode, title, abstract)``
    for every paper that matches the cohort gate.

    Returns the SQL text and the parameter list as a 2-tuple.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if cfg.arxiv_classes:
        clauses.append("arxiv_class && %s")
        params.append(list(cfg.arxiv_classes))
    if cfg.bibstems:
        clauses.append("bibstem && %s")
        params.append(list(cfg.bibstems))
    if cfg.keyword_sentinels:
        # Build a normalized lower-case-on-both-sides match against the
        # papers.keywords text[] column.
        clauses.append(
            "EXISTS (SELECT 1 FROM unnest(COALESCE(keywords, ARRAY[]::text[])) k "
            "         WHERE lower(k) = ANY(%s))"
        )
        params.append(list(cfg.keyword_sentinels))

    if not clauses:
        raise ValueError("planetary cohort config produced an empty cohort filter")

    where = "(" + " OR ".join(clauses) + ")"
    sql = (
        "SELECT bibcode, COALESCE(title, ''), COALESCE(abstract, '') "
        "  FROM papers WHERE " + where
    )

    if bibcode_prefix:
        sql += " AND bibcode LIKE %s"
        params.append(bibcode_prefix + "%")

    return sql, params


def iter_cohort_paper_batches(
    conn: psycopg.Connection,
    cfg: CohortConfig,
    *,
    bibcode_prefix: Optional[str] = None,
    batch_size: int = PAPER_BATCH_SIZE,
) -> Iterator[list[tuple[str, str, str]]]:
    """Yield ``[(bibcode, title, abstract), ...]`` batches for the cohort.

    Uses a server-side cursor so the driver does not buffer the whole
    cohort into Python memory — even 1M-row cohorts stream in 512-row
    chunks.
    """
    sql, params = _build_cohort_sql(cfg, bibcode_prefix)

    with conn.cursor(name="tier3_cohort_papers") as cur:
        cur.itersize = batch_size
        cur.execute(sql, params)
        batch: list[tuple[str, str, str]] = []
        for bibcode, title, abstract in cur:
            batch.append((bibcode, title or "", abstract or ""))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ---------------------------------------------------------------------------
# Per-paper linking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatedLink:
    """One discipline-gated link emitted to the writer."""

    bibcode: str
    entity_id: int
    confidence: float
    matched_surface: str
    is_alias: bool
    surface_len: int
    repeat_count: int
    context_hit: bool
    field_seen: tuple[str, ...]


def _has_disambiguator(text_lower: str, disambiguators: Sequence[str]) -> bool:
    """Return True if any disambiguator token appears as a whole word in ``text_lower``.

    The token list is short (<= ~50) and texts are abstract+title (a few
    KB), so a per-token whole-word regex is fast enough — and avoids
    spurious substring matches like ``orbital`` matching ``orbitals``.
    """
    for tok in disambiguators:
        if not tok:
            continue
        # multi-word tokens like "kuiper belt" — substring match with
        # space-bounded check is sufficient.
        if " " in tok or "-" in tok:
            if tok in text_lower:
                return True
            continue
        # single-word: word-boundary regex
        pattern = r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text_lower):
            return True
    return False


def link_paper(
    bibcode: str,
    title: str,
    abstract: str,
    automaton: AhocorasickAutomaton,
    cfg: CohortConfig,
) -> list[GatedLink]:
    """Run the AC scan against ``title + " " + abstract`` and apply the
    short-name disambiguator gate + confidence scoring.
    """
    if not title and not abstract:
        return []

    # Scan the concatenated text so a designation in the title still
    # benefits from co-occurring disambiguators in the abstract.
    text = (title + "\n" + abstract).strip()
    if not text:
        return []

    raw = link_abstract(text, automaton)
    if not raw:
        return []

    text_lower = text.lower()
    has_disambig = _has_disambiguator(text_lower, cfg.short_name_disambiguators)

    # Group hits by entity so we can count repeats per paper and pick the
    # highest-confidence surface form.
    by_entity: dict[int, list[LinkCandidate]] = {}
    for cand in raw:
        by_entity.setdefault(cand.entity_id, []).append(cand)

    out: list[GatedLink] = []
    for entity_id, cands in by_entity.items():
        # Pick the longest matched surface as the representative — longer
        # surfaces are far stronger signal than aliases like "Io".
        cands.sort(key=lambda c: (len(c.matched_surface), not c.is_alias), reverse=True)
        rep = cands[0]
        surface_len = len(rep.matched_surface)

        # Short-name gate: surfaces below the unconditional threshold
        # require a disambiguator token co-present in title+abstract.
        if surface_len < cfg.min_surface_length_unconditional and not has_disambig:
            continue

        # Field seen — used by evidence + downstream eval.
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        field_seen: list[str] = []
        for c in cands:
            ms = c.matched_surface.lower()
            if ms in title_lower:
                field_seen.append("title")
            if ms in abstract_lower:
                field_seen.append("abstract")
        field_seen_unique = tuple(sorted(set(field_seen)))

        # Confidence
        score = CONFIDENCE_BASE
        if surface_len >= LONG_NAME_LEN:
            score += CONFIDENCE_LONG_NAME_BONUS
        if surface_len >= VERY_LONG_NAME_LEN:
            score += CONFIDENCE_VERY_LONG_NAME_BONUS
        if not rep.is_alias:
            score += CONFIDENCE_CANONICAL_BONUS
        if has_disambig:
            score += CONFIDENCE_CONTEXT_BONUS
        if len(cands) >= 2:
            score += CONFIDENCE_REPEAT_BONUS
        score = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, score))

        out.append(
            GatedLink(
                bibcode=bibcode,
                entity_id=entity_id,
                confidence=round(score, 4),
                matched_surface=rep.matched_surface,
                is_alias=rep.is_alias,
                surface_len=surface_len,
                repeat_count=len(cands),
                context_hit=has_disambig,
                field_seen=field_seen_unique,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


_INSERT_SQL = """
    INSERT INTO document_entities (
        bibcode, entity_id, link_type, tier, tier_version,
        confidence, match_method, evidence
    )
    VALUES (
        %(bibcode)s, %(entity_id)s, %(link_type)s, %(tier)s, %(tier_version)s,
        %(confidence)s, %(match_method)s, %(evidence)s::jsonb
    )
    ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
"""  # noqa: resolver-lint (transitional; tier-3 owns its own writes)


def _evidence_json(link: GatedLink) -> str:
    return json.dumps(
        {
            "matched_surface": link.matched_surface,
            "is_alias": link.is_alias,
            "surface_len": link.surface_len,
            "repeat_count": link.repeat_count,
            "context_hit": link.context_hit,
            "field_seen": list(link.field_seen),
            "tier3_confidence_source": "discipline_gated_heuristic",
        },
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Stats / summary
# ---------------------------------------------------------------------------


@dataclass
class GatedStats:
    """Mutable counters captured during the run."""

    papers_scanned: int = 0
    papers_with_links: int = 0
    candidates_generated: int = 0
    rows_inserted: int = 0
    short_name_dropped: int = 0
    per_arxiv_class_yield: dict[str, int] = field(default_factory=dict)
    per_bibstem_yield: dict[str, int] = field(default_factory=dict)


def _format_wall_time(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def write_summary(
    stats: GatedStats,
    output_path: pathlib.Path,
    *,
    wall_seconds: float,
    dry_run: bool = False,
    cohort_size: Optional[int] = None,
) -> None:
    """Write a Markdown summary, including per-discipline yield."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wall_str = _format_wall_time(wall_seconds)
    mode_label = " (DRY RUN)" if dry_run else ""

    lines = [
        f"# Tier-3 Discipline-Gated Target Linker Summary{mode_label}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Papers scanned | {stats.papers_scanned:,} |",
        f"| Papers with at least one link | {stats.papers_with_links:,} |",
        f"| Candidate links generated | {stats.candidates_generated:,} |",
        f"| Rows inserted | {stats.rows_inserted:,} |",
        f"| Short-name candidates dropped (no disambiguator) | {stats.short_name_dropped:,} |",
    ]
    if cohort_size is not None:
        lines.append(f"| Cohort size (defined) | {cohort_size:,} |")
    lines.append(f"| Wall time | {wall_str} |")
    lines.append("")

    if stats.per_arxiv_class_yield:
        lines.append("## Yield by arXiv class")
        lines.append("")
        lines.append("| arxiv_class | papers with links |")
        lines.append("| --- | --- |")
        for ac, n in sorted(stats.per_arxiv_class_yield.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {ac} | {n:,} |")
        lines.append("")

    if stats.per_bibstem_yield:
        lines.append("## Yield by bibstem")
        lines.append("")
        lines.append("| bibstem | papers with links |")
        lines.append("| --- | --- |")
        for bs, n in sorted(stats.per_bibstem_yield.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {bs} | {n:,} |")
        lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", output_path)


# ---------------------------------------------------------------------------
# Per-paper-yield bookkeeping
# ---------------------------------------------------------------------------


def _fetch_paper_facets(
    conn: psycopg.Connection, bibcodes: Sequence[str]
) -> dict[str, tuple[list[str], list[str]]]:
    """Return ``{bibcode: (arxiv_class, bibstem)}`` for the given bibcodes.

    Used to attribute per-discipline yield in the run summary. Empty
    arrays come back as ``[]``.
    """
    if not bibcodes:
        return {}
    out: dict[str, tuple[list[str], list[str]]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, COALESCE(arxiv_class, ARRAY[]::text[]), "
            "       COALESCE(bibstem, ARRAY[]::text[]) "
            "  FROM papers WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
        for bibcode, arxiv_class, bibstem in cur.fetchall():
            out[bibcode] = (list(arxiv_class or []), list(bibstem or []))
    return out


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(
    conn: psycopg.Connection,
    *,
    cfg: CohortConfig,
    bibcode_prefix: Optional[str] = None,
    dry_run: bool = False,
    commit_interval_batches: int = 0,
) -> GatedStats:
    """Run the full discipline-gated tier-3 pass against ``conn``.

    Parameters
    ----------
    conn
        Open psycopg connection.
    cfg
        Cohort definition + safety knobs.
    bibcode_prefix
        Optional LIKE prefix for tests/shards.
    dry_run
        Roll back instead of committing.
    commit_interval_batches
        Commit every N paper batches (0 = end of run only).
    """
    logger.info("Fetching SsODNet target entities (stop-words: %d, min_len: %d)...",
                len(cfg.stop_words), cfg.min_surface_length_unconditional)
    rows = fetch_target_entity_rows(
        conn,
        stop_words=cfg.stop_words,
        min_canonical_len=cfg.min_surface_length_unconditional,
    )
    logger.info("  -> %d surface forms across %d entities",
                len(rows), len({r.entity_id for r in rows}))
    if not rows:
        logger.warning("no SsODNet target entities found; nothing to link")
        return GatedStats()

    automaton = build_automaton(rows)
    logger.info("Built automaton over %d surfaces", len(automaton))

    stats = GatedStats()

    write_conn = psycopg.connect(conn.info.dsn)
    batches_since_commit = 0
    log_interval = 5_000

    try:
        with write_conn.pipeline(), write_conn.cursor() as insert_cur:
            for batch in iter_cohort_paper_batches(
                conn, cfg, bibcode_prefix=bibcode_prefix
            ):
                stats.papers_scanned += len(batch)
                batches_since_commit += 1

                # Per-discipline attribution: only fetch facets for
                # bibcodes that have at least one link, since the cohort
                # SQL did not project arxiv_class/bibstem.
                paper_links: list[tuple[str, list[GatedLink]]] = []
                for bibcode, title, abstract in batch:
                    links = link_paper(bibcode, title, abstract, automaton, cfg)
                    if links:
                        paper_links.append((bibcode, links))

                if paper_links:
                    bibcodes = [b for b, _ in paper_links]
                    facets = _fetch_paper_facets(conn, bibcodes)
                    for bibcode, links in paper_links:
                        stats.papers_with_links += 1
                        stats.candidates_generated += len(links)
                        ac_list, bs_list = facets.get(bibcode, ([], []))
                        for link in links:
                            insert_cur.execute(
                                _INSERT_SQL,  # noqa: resolver-lint
                                {
                                    "bibcode": link.bibcode,
                                    "entity_id": link.entity_id,
                                    "link_type": LINK_TYPE,
                                    "tier": TIER,
                                    "tier_version": TIER_VERSION,
                                    "confidence": link.confidence,
                                    "match_method": MATCH_METHOD,
                                    "evidence": _evidence_json(link),
                                },
                            )
                            stats.rows_inserted += 1
                        # Attribute the paper (not each link) to its
                        # arxiv classes / bibstems for yield reporting.
                        for ac in ac_list:
                            stats.per_arxiv_class_yield[ac] = (
                                stats.per_arxiv_class_yield.get(ac, 0) + 1
                            )
                        for bs in bs_list:
                            stats.per_bibstem_yield[bs] = (
                                stats.per_bibstem_yield.get(bs, 0) + 1
                            )

                if stats.papers_scanned % log_interval < len(batch):
                    logger.info(
                        "  progress: %d papers scanned, %d papers linked, %d rows pending",
                        stats.papers_scanned,
                        stats.papers_with_links,
                        stats.rows_inserted,
                    )

                if (
                    commit_interval_batches > 0
                    and not dry_run
                    and batches_since_commit >= commit_interval_batches
                ):
                    write_conn.commit()
                    batches_since_commit = 0

        if dry_run:
            write_conn.rollback()
            logger.info("DRY RUN — rolled back %d tier-3 rows", stats.rows_inserted)
        else:
            write_conn.commit()
            logger.info(
                "Committed %d tier-3 rows across %d papers",
                stats.rows_inserted,
                stats.papers_with_links,
            )
    finally:
        write_conn.close()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", type=str, default=None, help="DSN override")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Cohort config YAML (default: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--bibcode-prefix",
        type=str,
        default=None,
        help="Only link papers whose bibcode starts with this string (test fixture)",
    )
    parser.add_argument(
        "--commit-interval-batches",
        type=int,
        default=0,
        help="Commit every N paper batches (0 = commit only at end)",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Allow running against the production database",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.db_url or os.environ.get("SCIX_TEST_DSN") or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    cfg = load_cohort_config(args.config)
    logger.info(
        "cohort: %d arxiv_classes, %d bibstems, %d keyword sentinels (min_len=%d)",
        len(cfg.arxiv_classes),
        len(cfg.bibstems),
        len(cfg.keyword_sentinels),
        cfg.min_surface_length_unconditional,
    )

    conn = get_connection(dsn)
    t0 = time.monotonic()
    try:
        stats = run(
            conn,
            cfg=cfg,
            bibcode_prefix=args.bibcode_prefix,
            dry_run=args.dry_run,
            commit_interval_batches=args.commit_interval_batches,
        )
        wall_seconds = time.monotonic() - t0
    finally:
        conn.close()

    verb = "would insert" if args.dry_run else "inserted"
    print(
        f"tier-3 discipline-gated: scanned {stats.papers_scanned:,} cohort papers, "
        f"{verb} {stats.rows_inserted:,} rows "
        f"({stats.papers_with_links:,} papers linked)"
    )

    summary_path = REPO_ROOT / "build-artifacts" / "tier3_target_gated_summary.md"
    write_summary(stats, summary_path, wall_seconds=wall_seconds, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
