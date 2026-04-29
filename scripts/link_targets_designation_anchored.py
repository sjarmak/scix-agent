#!/usr/bin/env python3
"""Tier-3 designation-anchored SsODNet target linker — bead xz4.9.

Pivot from the name-anchored matching used in xz4.p2v (which capped at
~0.25-0.40 precision against the ≥0.90 gate) to **designation-anchored**
matching. Every SsODNet target entity has at least one alias of shape
``(NNNN) Name``, ``YYYY LL``, ``Comet C/YYYY LL``, or ``NP/Name``. Real
asteroid citations in the literature use these designations; namesake
collisions ("Spitzer Space Telescope", "Apollo program") almost never do.

Algorithm
---------

1. Pull every SsODNet target entity + its aliases, classify each surface
   as ``name`` or ``designation`` via :data:`_DESIGNATION_RE`.

2. Build TWO Aho-Corasick automata:

   * ``name_automaton``: name-shaped surfaces (``Spitzer``, ``Apollo``,
     ``Ceres``).
   * ``designation_automaton``: designation-shaped surfaces (``(2160)``,
     ``(2160) Spitzer``, ``2014 KZ113``, ``C/2017 K2``).

3. Per cohort paper, scan title+abstract with both automata. Collect:

   * ``name_hits``: ``{entity_id, ...}`` from the name scan.
   * ``desig_hits``: ``{entity_id, ...}`` from the designation scan.

4. For each entity, decide whether to emit:

   * If the entity has at least one **name-shaped** surface (the
     canonical name itself or a non-designation alias), emit only if the
     entity is in ``name_hits`` AND ``desig_hits`` (co-presence rule).
   * If the entity is **designation-only** (canonical_name is itself a
     designation, e.g. provisional asteroid ``2014 KZ113``), emit if in
     ``desig_hits`` alone — every surface is designation-shape, so name
     co-presence is impossible.

This rejects "Spitzer Space Telescope" papers that have ``Spitzer`` but
no ``(2160)``, while accepting "asteroid (2160) Spitzer" papers that
mention both. For provisional designations, the name is itself a
designation and a single match suffices.

Output rows go to ``document_entities`` with::

    link_type    = 'target_designation_anchored'
    tier         = 3
    tier_version = 2
    match_method = 'aho_corasick_designation_anchored'

PK ``(bibcode, entity_id, link_type, tier)`` keeps these orthogonal to
the (now-rolled-back) tier-3 ``target_gated_match`` rows from xz4.p2v.

Usage::

    # Dry run on the test DB:
    SCIX_TEST_DSN=dbname=scix_test \\
      python scripts/link_targets_designation_anchored.py --dry-run -v

    # Production run (heavy — wrap in scix-batch):
    scix-batch python scripts/link_targets_designation_anchored.py --allow-prod
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

# Reuse the cohort config + paper iterator from the p2v driver. The cohort
# gate + paper streaming is orthogonal to the linking rule — only the
# anchor logic changes between p2v and xz4.9.
from link_targets_discipline_gated import (  # noqa: E402
    CohortConfig,
    DEFAULT_CONFIG_PATH,
    PAPER_BATCH_SIZE,
    iter_cohort_paper_batches,
    load_cohort_config,
    _fetch_paper_facets,
    _format_wall_time,
)

# Make the p2v module importable when this script is run as `python
# scripts/link_targets_designation_anchored.py` from the repo root.
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINK_TYPE: str = "target_designation_anchored"
TIER: int = 3
TIER_VERSION: int = 2
MATCH_METHOD: str = "aho_corasick_designation_anchored"

# Confidence — see CONFIDENCE_NOTES.md if/when a calibrated model lands.
# Designation co-presence is a strong signal on its own, so the base is
# higher than p2v's 0.70.
CONFIDENCE_BASE: float = 0.85
CONFIDENCE_NAME_COPRESENT_BONUS: float = 0.05  # named entity AND name hit
CONFIDENCE_REPEAT_BONUS: float = 0.05  # >=2 distinct designation hits
CONFIDENCE_MIN: float = 0.80
CONFIDENCE_MAX: float = 0.95

# Designation-shape regex. A surface is "designation" iff the FULL string
# matches one of these patterns (anchored at both ends to avoid spurious
# substring classification).
#
# Patterns covered:
#   (NNNN)              -> "(2160)"
#   (NNNN) Name         -> "(2160) Spitzer", "(690713) 2014 KZ113"
#   NNNN Name           -> very rare, also caught by the year pattern
#   YYYY LL[NN]         -> "2014 KZ113", "1999 AP10", "2003 EH1"
#   NP/Name             -> "1P/Halley", "45P/Honda"
#   C/YYYY LL[NN]       -> "C/2017 K2", "C/1995 O1"
#   D/YYYY LL[NN]       -> "D/1993 F2"
#   P/YYYY LL[NN]       -> "P/2010 A2"
#   X/YYYY LL[NN]       -> "X/1106 C1"
#   A/YYYY LL[NN]       -> "A/2017 U7" (active/asteroid-like)
#   I/YYYY LL[NN]       -> "I/2017 U1" (interstellar)
_DESIGNATION_RE: re.Pattern[str] = re.compile(
    r"""
    ^
    (?:
        \(\d+\)(?:\s+\S.*)?                      # (NNNN) optional Name suffix
        | \d{4}\s+[A-Z]{1,3}\d{0,4}              # YYYY LL[NN]
        | \d{1,4}P/\S.*                          # NP/Name
        | [CDPIXA]/\d{4}\s+[A-Z]{1,3}\d{0,4}     # X/YYYY LL[NN]
    )
    $
    """,
    re.VERBOSE,
)


def is_designation_shape(surface: str) -> bool:
    """Return True iff ``surface`` looks like an asteroid/comet designation.

    Examples (True):
        ``(2160)``, ``(2160) Spitzer``, ``2014 KZ113``, ``C/2017 K2``,
        ``45P/Honda``, ``1P/Halley``.

    Examples (False):
        ``Spitzer``, ``Apollo``, ``Ceres``, ``Hayabusa``, ``Spitzer
        Space Telescope`` (the latter is too long for the regex anchor
        but is also not a designation).
    """
    return bool(_DESIGNATION_RE.match(surface.strip()))


# Sub-pattern: YYYY LL[NN] (without the optional comet prefix). Used at
# match time to tell whether a candidate is one of the year+letter
# designations that collides with date+preposition English phrases like
# ``2020 by``, ``2023 to``, ``2024 in``. AC matching is case-
# insensitive, so we re-check the original-case text at the match span
# for these and require the letter portion to be all upper-case.
_YEAR_LETTER_DESIGNATION_RE: re.Pattern[str] = re.compile(
    r"^\d{4}\s+[A-Z]{1,3}\d{0,4}$"
)
_COMET_LETTER_DESIGNATION_RE: re.Pattern[str] = re.compile(
    r"^[CDPIXA]/\d{4}\s+[A-Z]{1,3}\d{0,4}$"
)


def _passes_designation_case_filter(
    surface: str, text: str, start: int, end: int
) -> bool:
    """Return True iff a YYYY LL[NN] / X/YYYY LL[NN] match is upper-case.

    Real asteroid designations are upper-case in the literature
    (``2003 EH1``, ``C/2017 K2``). The lower-case forms (``2003 eh1``)
    are vanishingly rare in real text, but ``2003 to`` and ``2020 by``
    are extremely common date+preposition phrases. AC scanning is
    case-insensitive, so we re-check the original-case text and drop
    any letters-not-upper match. Other designation shapes — ``(NNNN)``,
    ``(NNNN) Name``, ``NP/Name`` — are not subject to this filter.
    """
    if not (
        _YEAR_LETTER_DESIGNATION_RE.match(surface)
        or _COMET_LETTER_DESIGNATION_RE.match(surface)
    ):
        return True
    original = text[start:end]
    letters = "".join(ch for ch in original if ch.isalpha())
    return bool(letters) and letters.isupper()


# ---------------------------------------------------------------------------
# Entity fetching
# ---------------------------------------------------------------------------


_FETCH_SQL = """
    SELECT e.id,
           e.canonical_name,
           ea.alias
      FROM entities e
      LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
     WHERE e.entity_type = 'target'
       AND e.source = 'ssodnet'
       AND (e.link_policy IS NULL OR e.link_policy <> 'llm_only')
       AND (e.ambiguity_class IS NULL OR e.ambiguity_class <> 'banned')
"""


@dataclass(frozen=True)
class FetchedSurfaces:
    """Per-entity surface inventory.

    ``name_surfaces`` and ``designation_surfaces`` are disjoint and their
    union is every distinct surface (canonical + aliases) seen for the
    entity. ``has_name_anchor`` is True iff at least one name-shaped
    surface exists (canonical or alias) — entities without one are
    designation-only and bypass the co-presence rule.
    """

    entity_id: int
    canonical_name: str
    name_surfaces: tuple[str, ...]
    designation_surfaces: tuple[str, ...]
    has_name_anchor: bool


def fetch_target_surfaces(conn: psycopg.Connection) -> list[FetchedSurfaces]:
    """Pull SsODNet target entities and classify each surface.

    Returns one :class:`FetchedSurfaces` per entity, with name vs
    designation surfaces split. Banned entities and ``llm_only``
    entities are filtered out at the SQL layer.
    """
    by_entity: dict[int, dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(_FETCH_SQL)
        for entity_id, canonical, alias in cur.fetchall():
            ent = by_entity.setdefault(
                int(entity_id),
                {
                    "canonical": canonical,
                    "names": set(),
                    "designations": set(),
                },
            )
            for surface in (canonical, alias):
                if surface is None:
                    continue
                if is_designation_shape(surface):
                    ent["designations"].add(surface)
                else:
                    ent["names"].add(surface)

    out: list[FetchedSurfaces] = []
    for entity_id, ent in by_entity.items():
        names = tuple(sorted(ent["names"]))
        desigs = tuple(sorted(ent["designations"]))
        out.append(
            FetchedSurfaces(
                entity_id=entity_id,
                canonical_name=ent["canonical"],
                name_surfaces=names,
                designation_surfaces=desigs,
                has_name_anchor=bool(names),
            )
        )
    logger.info(
        "fetched %d entities: %d named, %d designation-only",
        len(out),
        sum(1 for s in out if s.has_name_anchor),
        sum(1 for s in out if not s.has_name_anchor),
    )
    return out


def _name_min_length_filter(
    surface: str, min_len: int
) -> bool:
    """Drop very-short single-token name surfaces.

    ``Io``, ``Ra`` collide with English words; we keep them out of the
    name automaton even at the cost of recall, since the co-presence
    requirement still needs the name set to be reasonably clean.
    Multi-token names (``Hale-Bopp``) and surfaces with digits bypass
    the floor.
    """
    s = surface.strip()
    if not s:
        return False
    if len(s) < min_len and " " not in s and "-" not in s and "/" not in s:
        return False
    return True


def build_entity_rows(
    surfaces: Sequence[FetchedSurfaces],
    *,
    name_min_len: int = 4,
) -> tuple[list[EntityRow], list[EntityRow]]:
    """Split entity surfaces into name-automaton and designation-automaton rows.

    Each surface gets ``ambiguity_class='unique'`` so the upstream AC
    layer skips its homograph long-form gate — designation co-presence
    is our disambiguator, not the long-form rule.
    """
    name_rows: list[EntityRow] = []
    desig_rows: list[EntityRow] = []
    for ent in surfaces:
        for surface in ent.name_surfaces:
            if not _name_min_length_filter(surface, name_min_len):
                continue
            name_rows.append(
                EntityRow(
                    entity_id=ent.entity_id,
                    surface=surface,
                    canonical_name=ent.canonical_name,
                    ambiguity_class="unique",
                    is_alias=(surface != ent.canonical_name),
                )
            )
        for surface in ent.designation_surfaces:
            desig_rows.append(
                EntityRow(
                    entity_id=ent.entity_id,
                    surface=surface,
                    canonical_name=ent.canonical_name,
                    ambiguity_class="unique",
                    is_alias=(surface != ent.canonical_name),
                )
            )
    return name_rows, desig_rows


# ---------------------------------------------------------------------------
# Per-paper linking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredLink:
    """One designation-anchored link emitted to the writer."""

    bibcode: str
    entity_id: int
    confidence: float
    matched_designation: str
    matched_name: Optional[str]
    name_copresent: bool
    designation_repeat: int
    field_seen: tuple[str, ...]


def _scan_with_automaton(
    text: str, automaton: AhocorasickAutomaton
) -> list[LinkCandidate]:
    """Thin wrapper around :func:`link_abstract` that returns its raw output.

    Kept as a separate helper so tests can stub the AC scan without
    monkeypatching the upstream module.
    """
    return link_abstract(text, automaton)


def link_paper(
    bibcode: str,
    title: str,
    abstract: str,
    name_automaton: Optional[AhocorasickAutomaton],
    designation_automaton: AhocorasickAutomaton,
    entity_index: dict[int, FetchedSurfaces],
) -> list[AnchoredLink]:
    """Run the dual scan + co-presence gate against ``title + abstract``.

    Parameters
    ----------
    bibcode
        Paper key.
    title, abstract
        Source text (lowercased internally by the AC layer).
    name_automaton
        Automaton over name-shaped surfaces. May be ``None`` if no
        named entities survived the filter (purely defensive).
    designation_automaton
        Automaton over designation-shaped surfaces. Required.
    entity_index
        ``{entity_id: FetchedSurfaces}`` so we can look up
        ``has_name_anchor`` per entity in O(1).
    """
    if not (title or abstract):
        return []
    text = (title + "\n" + abstract).strip()
    if not text:
        return []

    # Scan with both automata.
    desig_cands = _scan_with_automaton(text, designation_automaton)
    # Drop year+letter / comet+letter candidates whose letters are
    # lowercase in the original text — those are date+preposition
    # phrases ("2020 by", "2023 to") not asteroid designations.
    desig_cands = [
        c
        for c in desig_cands
        if _passes_designation_case_filter(c.matched_surface, text, c.start, c.end)
    ]
    if not desig_cands:
        return []
    name_cands = (
        _scan_with_automaton(text, name_automaton) if name_automaton else []
    )

    # Group hits by entity.
    desig_by_entity: dict[int, list[LinkCandidate]] = {}
    for c in desig_cands:
        desig_by_entity.setdefault(c.entity_id, []).append(c)
    name_by_entity: dict[int, list[LinkCandidate]] = {}
    for c in name_cands:
        name_by_entity.setdefault(c.entity_id, []).append(c)

    title_lower = title.lower()
    abstract_lower = abstract.lower()

    out: list[AnchoredLink] = []
    for entity_id, dcands in desig_by_entity.items():
        ent = entity_index.get(entity_id)
        if ent is None:
            # Defensive: an entity in the automaton not in the index
            # means a row was added without classification. Skip.
            continue
        ncands = name_by_entity.get(entity_id, [])

        # Co-presence rule: named entities require both a name hit AND
        # a designation hit. Designation-only entities pass with a
        # designation hit alone.
        if ent.has_name_anchor and not ncands:
            continue

        # Pick the longest matched designation as the representative.
        dcands.sort(key=lambda c: len(c.matched_surface), reverse=True)
        rep_d = dcands[0]
        rep_n = ncands[0].matched_surface if ncands else None

        # Field-seen — for downstream eval cards.
        seen: set[str] = set()
        for c in (*dcands, *ncands):
            ms = c.matched_surface.lower()
            if ms in title_lower:
                seen.add("title")
            if ms in abstract_lower:
                seen.add("abstract")
        field_seen = tuple(sorted(seen))

        # Confidence.
        score = CONFIDENCE_BASE
        if ent.has_name_anchor and ncands:
            score += CONFIDENCE_NAME_COPRESENT_BONUS
        if len(dcands) >= 2:
            score += CONFIDENCE_REPEAT_BONUS
        score = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, score))

        out.append(
            AnchoredLink(
                bibcode=bibcode,
                entity_id=entity_id,
                confidence=round(score, 4),
                matched_designation=rep_d.matched_surface,
                matched_name=rep_n,
                name_copresent=bool(ncands),
                designation_repeat=len(dcands),
                field_seen=field_seen,
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


def _evidence_json(link: AnchoredLink) -> str:
    return json.dumps(
        {
            "matched_designation": link.matched_designation,
            "matched_name": link.matched_name,
            "name_copresent": link.name_copresent,
            "designation_repeat": link.designation_repeat,
            "field_seen": list(link.field_seen),
            "tier3_confidence_source": "designation_anchored_heuristic",
        },
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Stats / summary
# ---------------------------------------------------------------------------


@dataclass
class AnchoredStats:
    """Mutable counters captured during the run."""

    papers_scanned: int = 0
    papers_with_links: int = 0
    candidates_generated: int = 0
    rows_inserted: int = 0
    name_only_rejected: int = 0
    desig_only_emitted: int = 0
    per_arxiv_class_yield: dict[str, int] = field(default_factory=dict)
    per_bibstem_yield: dict[str, int] = field(default_factory=dict)


def write_summary(
    stats: AnchoredStats,
    output_path: pathlib.Path,
    *,
    wall_seconds: float,
    dry_run: bool = False,
    n_named: int = 0,
    n_desig_only: int = 0,
) -> None:
    """Write a Markdown run summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wall_str = _format_wall_time(wall_seconds)
    mode_label = " (DRY RUN)" if dry_run else ""

    lines = [
        f"# Tier-3 Designation-Anchored Target Linker Summary{mode_label}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Papers scanned | {stats.papers_scanned:,} |",
        f"| Papers with at least one link | {stats.papers_with_links:,} |",
        f"| Candidate links generated | {stats.candidates_generated:,} |",
        f"| Rows inserted | {stats.rows_inserted:,} |",
        f"| Named entities (require name+designation copresence) | {n_named:,} |",
        f"| Designation-only entities (designation alone suffices) | {n_desig_only:,} |",
        f"| Wall time | {wall_str} |",
        "",
    ]

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
# Main run
# ---------------------------------------------------------------------------


def run(
    conn: psycopg.Connection,
    *,
    cfg: CohortConfig,
    bibcode_prefix: Optional[str] = None,
    dry_run: bool = False,
    commit_interval_batches: int = 0,
    name_min_len: int = 4,
) -> tuple[AnchoredStats, int, int]:
    """Run the dual-automaton designation-anchored pass.

    Returns ``(stats, n_named, n_desig_only)`` so the caller can
    surface the entity counts in the run summary.
    """
    logger.info("Fetching SsODNet target entities + aliases ...")
    surfaces = fetch_target_surfaces(conn)
    n_named = sum(1 for s in surfaces if s.has_name_anchor)
    n_desig_only = sum(1 for s in surfaces if not s.has_name_anchor)

    name_rows, desig_rows = build_entity_rows(surfaces, name_min_len=name_min_len)
    logger.info(
        "  -> %d name surfaces (%d named entities), %d designation surfaces "
        "(%d total entities)",
        len(name_rows),
        n_named,
        len(desig_rows),
        len(surfaces),
    )
    if not desig_rows:
        logger.warning("no designation surfaces; nothing to link")
        return AnchoredStats(), n_named, n_desig_only

    name_automaton = build_automaton(name_rows) if name_rows else None
    designation_automaton = build_automaton(desig_rows)
    logger.info(
        "Built automata: %d name surfaces, %d designation surfaces",
        len(name_automaton) if name_automaton else 0,
        len(designation_automaton),
    )

    entity_index: dict[int, FetchedSurfaces] = {s.entity_id: s for s in surfaces}

    stats = AnchoredStats()
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

                paper_links: list[tuple[str, list[AnchoredLink]]] = []
                for bibcode, title, abstract in batch:
                    links = link_paper(
                        bibcode,
                        title,
                        abstract,
                        name_automaton,
                        designation_automaton,
                        entity_index,
                    )
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
                            if link.name_copresent:
                                pass  # already counted via name_copresent bonus
                            else:
                                stats.desig_only_emitted += 1
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

    return stats, n_named, n_desig_only


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
        help="Only link papers whose bibcode starts with this string",
    )
    parser.add_argument(
        "--commit-interval-batches",
        type=int,
        default=0,
        help="Commit every N paper batches (0 = commit only at end)",
    )
    parser.add_argument(
        "--name-min-len",
        type=int,
        default=4,
        help="Minimum length for single-token name surfaces (default 4)",
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
        "cohort: %d arxiv_classes, %d bibstems, %d keyword sentinels",
        len(cfg.arxiv_classes),
        len(cfg.bibstems),
        len(cfg.keyword_sentinels),
    )

    conn = get_connection(dsn)
    t0 = time.monotonic()
    try:
        stats, n_named, n_desig_only = run(
            conn,
            cfg=cfg,
            bibcode_prefix=args.bibcode_prefix,
            dry_run=args.dry_run,
            commit_interval_batches=args.commit_interval_batches,
            name_min_len=args.name_min_len,
        )
        wall_seconds = time.monotonic() - t0
    finally:
        conn.close()

    verb = "would insert" if args.dry_run else "inserted"
    print(
        f"tier-3 designation-anchored: scanned {stats.papers_scanned:,} cohort papers, "
        f"{verb} {stats.rows_inserted:,} rows "
        f"({stats.papers_with_links:,} papers linked)"
    )

    summary_path = REPO_ROOT / "build-artifacts" / "tier3_target_designation_summary.md"
    write_summary(
        stats,
        summary_path,
        wall_seconds=wall_seconds,
        dry_run=args.dry_run,
        n_named=n_named,
        n_desig_only=n_desig_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
