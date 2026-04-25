#!/usr/bin/env python3
"""Entity-enrichment value-props eval harness.

Scores the six enrichment value props defined in the entity-enrichment PRD
against hand-curated gold sets:

1. alias_expansion         — HST ↔ Hubble Space Telescope
2. ontology_expansion      — "exoplanet" → HD 209458b, TRAPPIST-1b
3. disambiguation          — Hubble-mission vs Edwin-Hubble
4. type_filter             — entity_type='instrument' → only instruments
5. specific_entity         — entity_id=X → only papers mentioning X
6. community_expansion     — entity in community C → siblings in C

## Judge invocation

The judge is a Claude Code subagent invoked via ``claude -p`` as a
subprocess (OAuth-authenticated; *no* paid Anthropic API, *no*
``anthropic`` SDK import). Tests mock :func:`subprocess.run`.

If ``claude`` is not on ``PATH`` the script fails fast — silent
degradation here would let an eval produce all-zero scores that look
like "the feature is broken" when in fact the judge never ran.

## Output

- Per-query judge results → JSONL at
  ``.claude/prd-build-artifacts/eval-d4-<timestamp>.jsonl`` with schema
  ``{prop, query, rubric_score, judge_rationale, ...}``.
- Rolled-up report → ``docs/eval/entity_value_props_2026-04.md``
  (overwritten when ``--write-report`` is set; otherwise a static
  template lives there).

## Retrieval adapter

Retrieval lives behind :class:`RetrievalBackend`. The default
implementation calls :func:`scix.search.hybrid_search` with alias +
ontology expansion enabled. Tests inject a stub so the test suite runs
without a live database.

## CLI

::

    python scripts/eval_entity_value_props.py \\
        --props all --db scix_test --dry-run

    python scripts/eval_entity_value_props.py \\
        --props alias ontology --write-report

Exit codes: 0 on success, 1 on misconfiguration (e.g. missing ``claude``
binary, missing gold set), 2 on runtime failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# src/ layout — let the script run from a checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - defensive
    yaml = None  # Deferred error; we raise a helpful message in `load_gold_set`.

logger = logging.getLogger("eval_entity_value_props")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROPS: tuple[str, ...] = (
    "alias_expansion",
    "ontology_expansion",
    "disambiguation",
    "type_filter",
    "specific_entity",
    "community_expansion",
)
"""The six value props, in canonical order."""

# CLI short names → canonical prop names.
CLI_PROP_ALIASES: dict[str, str] = {
    "alias": "alias_expansion",
    "alias_expansion": "alias_expansion",
    "ontology": "ontology_expansion",
    "ontology_expansion": "ontology_expansion",
    "disambig": "disambiguation",
    "disambiguation": "disambiguation",
    "type": "type_filter",
    "type_filter": "type_filter",
    "specific": "specific_entity",
    "specific_entity": "specific_entity",
    "community": "community_expansion",
    "community_expansion": "community_expansion",
}

RUBRIC_MIN: int = 0
RUBRIC_MAX: int = 3

DEFAULT_DB: str = os.environ.get("SCIX_DSN", "dbname=scix")
DEFAULT_ARTIFACT_DIR: Path = Path(".claude/prd-build-artifacts")
DEFAULT_REPORT_PATH: Path = Path("docs/eval/entity_value_props_2026-04.md")
DEFAULT_GOLD_DIR: Path = Path("data/eval/entity_value_props")

DEFAULT_JUDGE_TIMEOUT_S: int = 120
DEFAULT_TOP_K: int = 10
DEFAULT_JUDGE_RETRIES: int = 3
DEFAULT_JUDGE_RETRY_BASE_DELAY_S: float = 5.0


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldQuery:
    """One entry from a gold-set YAML file.

    ``extra`` carries prop-specific fields (expected_instances,
    intended_sense, seed_entity, etc.) verbatim so the judge prompt can
    surface them.
    """

    prop: str
    query_id: str
    query: str
    expectation: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalDoc:
    """A single retrieval hit — what the judge sees."""

    bibcode: str
    title: str
    snippet: str = ""


@dataclass(frozen=True)
class JudgeResult:
    """One judge verdict for one query."""

    prop: str
    query_id: str
    query: str
    rubric_score: int
    judge_rationale: str
    retrieval_count: int


# ---------------------------------------------------------------------------
# Gold-set loading
# ---------------------------------------------------------------------------


def load_gold_set(path: Path) -> list[GoldQuery]:
    """Parse a YAML gold set into :class:`GoldQuery` records.

    Raises:
        FileNotFoundError: If the YAML file is missing.
        ValueError: If the YAML is malformed or missing required fields.
        RuntimeError: If PyYAML is not installed.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load gold sets. Install with "
            "`pip install pyyaml` (or add to dev extras)."
        )
    if not path.exists():
        raise FileNotFoundError(f"gold set not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    prop = data.get("prop")
    queries = data.get("queries")
    if not isinstance(prop, str) or prop not in PROPS:
        raise ValueError(f"{path}: 'prop' must be one of {PROPS}, got {prop!r}")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path}: 'queries' must be a non-empty list")

    out: list[GoldQuery] = []
    reserved = {"id", "query", "expectation"}
    for raw in queries:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: every query entry must be a mapping")
        qid = raw.get("id")
        query = raw.get("query")
        expectation = raw.get("expectation", "")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{path}: every query entry needs a non-empty 'id'")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}: query {qid!r} needs a non-empty 'query' string")
        extra = {k: v for k, v in raw.items() if k not in reserved}
        out.append(
            GoldQuery(
                prop=prop,
                query_id=qid,
                query=query.strip(),
                expectation=str(expectation).strip(),
                extra=extra,
            )
        )
    return out


def load_all_gold_sets(gold_dir: Path, props: Iterable[str]) -> dict[str, list[GoldQuery]]:
    """Load one YAML per prop from ``gold_dir``. Returns a mapping keyed by prop."""
    out: dict[str, list[GoldQuery]] = {}
    for prop in props:
        yaml_path = gold_dir / f"{prop}.yaml"
        out[prop] = load_gold_set(yaml_path)
    return out


# ---------------------------------------------------------------------------
# Retrieval backends
# ---------------------------------------------------------------------------


class RetrievalBackend(Protocol):
    """Any callable that turns a GoldQuery into a list of RetrievalDoc."""

    def retrieve(self, gold: GoldQuery, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievalDoc]: ...


@dataclass
class StubRetrievalBackend:
    """Deterministic in-memory backend for unit tests and dry-runs.

    Returns one synthetic document per query. The snippet echoes the
    expectation so the stub judge output is traceable in JSONL.
    """

    max_docs: int = 3

    def retrieve(self, gold: GoldQuery, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievalDoc]:
        n = min(self.max_docs, top_k)
        return [
            RetrievalDoc(
                bibcode=f"stub.{gold.prop}.{gold.query_id}.{i}",
                title=f"Stub result {i} for {gold.query!r}",
                snippet=gold.expectation[:200],
            )
            for i in range(n)
        ]


@dataclass
class HybridSearchBackend:
    """Live backend — delegates to :func:`scix.search.hybrid_search`.

    ``dsn`` defaults to the CLI value (or ``SCIX_DSN``). The connection
    is re-used across queries. Callers should close it via
    :meth:`close` when done.

    Alias expansion and ontology-parser expansion are enabled by
    default because several of the six value-props directly test them.
    """

    dsn: str
    enable_alias_expansion: bool = True
    enable_ontology_parser: bool = True
    _conn: Any | None = None  # psycopg.Connection, typed loosely for test mocking

    def _get_conn(self) -> Any:
        if self._conn is None:
            from scix.db import get_connection

            self._conn = get_connection(self.dsn)
        return self._conn

    def retrieve(self, gold: GoldQuery, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievalDoc]:
        # Late import so the script can be --dry-run without the full
        # search stack installed.
        from scix.search import hybrid_search

        conn = self._get_conn()
        result = hybrid_search(
            conn,
            query_text=gold.query,
            query_embedding=None,  # lexical / RRF only; judges don't need vector
            top_n=top_k,
            enable_alias_expansion=self.enable_alias_expansion,
            enable_ontology_parser=self.enable_ontology_parser,
        )
        papers: list[dict[str, Any]] = getattr(result, "papers", []) or []
        return [
            RetrievalDoc(
                bibcode=str(p.get("bibcode", "")),
                title=str(p.get("title", "")),
                snippet=str(p.get("abstract", "") or "")[:400],
            )
            for p in papers[:top_k]
        ]

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


@dataclass
class CommunityExpansionBackend:
    """Retrieval backend that implements the ``community_expansion`` value prop.

    For queries whose prop is ``community_expansion``, the backend:

    1. Resolves ``gold.extra['seed_entity']`` to an ``entities.id`` via
       case-insensitive match against ``entities.canonical_name`` or
       ``entity_aliases.alias``.
    2. Picks the modal ``paper_metrics.community_semantic_medium`` across
       all papers currently linked to that entity via
       ``document_entities`` (ignoring ``NULL`` and the ``-1`` outlier
       sentinel).
    3. Returns the top-``top_k`` papers in that community ordered by
       ``paper_metrics.pagerank DESC NULLS LAST``, excluding papers
       already linked to the seed entity so the judge sees genuine
       community *siblings* rather than seed-adjacent papers.

    For every other prop the backend delegates to ``inner``.

    Caveat: this is a narrow fix wired into the eval harness only —
    ``src/scix/search.py`` is intentionally untouched (see bead
    scix_experiments-xz4.1.34). Semantic communities are topic-based;
    if the seed entity's papers span multiple topics, the modal
    community can be broad (e.g. "observational astronomy") rather than
    mission-specific.
    """

    inner: RetrievalBackend
    dsn: str
    _conn: Any | None = None  # psycopg.Connection

    def _get_conn(self) -> Any:
        if self._conn is None:
            from scix.db import get_connection

            self._conn = get_connection(self.dsn)
        return self._conn

    def _resolve_entity_id(self, name: str) -> int | None:
        """Resolve a seed entity name to ``entities.id``.

        Tries canonical_name first, then entity_aliases. Both matches
        are case-insensitive (the schema has ``idx_entities_canonical_lower``
        and ``idx_entity_aliases_lower`` btrees on the lowered forms).
        When multiple entities match (e.g. two 'LIGO' rows), the one
        with the most paper links wins — the community-expansion
        rationale prefers the heavier-weight node.
        """
        conn = self._get_conn()
        lowered = name.strip().lower()
        if not lowered:
            return None
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, COUNT(de.bibcode) AS n
                FROM entities e
                LEFT JOIN document_entities de ON de.entity_id = e.id
                WHERE lower(e.canonical_name) = %s
                GROUP BY e.id
                ORDER BY n DESC
                LIMIT 1
                """,
                (lowered,),
            )
            row = cur.fetchone()
            if row is not None:
                return int(row[0])
            cur.execute(
                """
                SELECT e.id, COUNT(de.bibcode) AS n
                FROM entity_aliases a
                JOIN entities e ON e.id = a.entity_id
                LEFT JOIN document_entities de ON de.entity_id = e.id
                WHERE lower(a.alias) = %s
                GROUP BY e.id
                ORDER BY n DESC
                LIMIT 1
                """,
                (lowered,),
            )
            row = cur.fetchone()
        return int(row[0]) if row is not None else None

    def _modal_community(self, entity_id: int) -> int | None:
        """Return the most-populated ``community_semantic_medium`` for the entity.

        Ignores NULL and ``-1`` (outlier sentinel). Returns ``None`` if
        no papers linked to the entity have a semantic community assigned.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pm.community_semantic_medium, COUNT(*) AS n
                FROM document_entities de
                JOIN paper_metrics pm ON pm.bibcode = de.bibcode
                WHERE de.entity_id = %s
                  AND pm.community_semantic_medium IS NOT NULL
                  AND pm.community_semantic_medium <> -1
                GROUP BY pm.community_semantic_medium
                ORDER BY n DESC
                LIMIT 1
                """,
                (entity_id,),
            )
            row = cur.fetchone()
        return int(row[0]) if row is not None else None

    def _community_siblings(
        self, community_id: int, seed_entity_id: int, top_k: int
    ) -> list[RetrievalDoc]:
        """Return ``top_k`` papers in the community, excluding seed-linked papers."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.bibcode, p.title, COALESCE(p.abstract, '')
                FROM papers p
                JOIN paper_metrics pm ON pm.bibcode = p.bibcode
                LEFT JOIN document_entities de_seed
                  ON de_seed.bibcode = p.bibcode AND de_seed.entity_id = %s
                WHERE pm.community_semantic_medium = %s
                  AND de_seed.bibcode IS NULL
                ORDER BY pm.pagerank DESC NULLS LAST
                LIMIT %s
                """,
                (seed_entity_id, community_id, top_k),
            )
            rows = cur.fetchall()
        return [
            RetrievalDoc(
                bibcode=str(bibcode),
                title=str(title or ""),
                snippet=str(abstract or "")[:400],
            )
            for bibcode, title, abstract in rows
        ]

    def retrieve(self, gold: GoldQuery, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievalDoc]:
        if gold.prop != "community_expansion":
            return self.inner.retrieve(gold, top_k=top_k)

        seed = gold.extra.get("seed_entity")
        if not isinstance(seed, str) or not seed.strip():
            logger.info("community_expansion: query %r has no seed_entity", gold.query_id)
            return []
        entity_id = self._resolve_entity_id(seed)
        if entity_id is None:
            logger.info(
                "community_expansion: seed %r (query %r) does not resolve to any entity",
                seed,
                gold.query_id,
            )
            return []
        community_id = self._modal_community(entity_id)
        if community_id is None:
            logger.info(
                "community_expansion: entity_id=%d (query %r) has no semantic community",
                entity_id,
                gold.query_id,
            )
            return []
        return self._community_siblings(community_id, entity_id, top_k=top_k)

    def close(self) -> None:
        # The inner backend owns the hybrid_search connection; we only
        # close our own.
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        close_inner = getattr(self.inner, "close", None)
        if callable(close_inner):
            close_inner()


@dataclass
class SpecificEntityBackend:
    """Retrieval backend that implements the ``specific_entity`` value prop.

    For queries whose prop is ``specific_entity``, the backend:

    1. Resolves ``gold.extra['entity_name']`` to an ``entities.id`` via
       case-insensitive match against ``entities.canonical_name`` first,
       then ``entity_aliases.alias``. On ambiguity (multiple rows with
       the same surface form), the entity with the most paper links
       wins — matches :class:`CommunityExpansionBackend` policy.
    2. Returns the top-``top_k`` papers linked to that entity via
       ``document_entities``, ordered by ``paper_metrics.pagerank
       DESC NULLS LAST``. This guarantees results actually mention the
       resolved entity_id rather than colliding on surface form.

    For every other prop the backend delegates to ``inner``.

    Caveat (matches :class:`CommunityExpansionBackend`): this is a
    narrow fix wired into the eval harness only — ``src/scix/search.py``
    is intentionally untouched (see bead scix_experiments-xz4.1.39).
    Adding ``specific_entity`` to MCP ``search`` is a separate decision.
    """

    inner: RetrievalBackend
    dsn: str
    _conn: Any | None = None  # psycopg.Connection

    def _get_conn(self) -> Any:
        if self._conn is None:
            from scix.db import get_connection

            self._conn = get_connection(self.dsn)
        return self._conn

    def _resolve_entity_id(
        self, name: str, *, entity_type: str | None = None
    ) -> int | None:
        """Resolve an entity name to ``entities.id``.

        Searches both canonical_name and entity_aliases case-insensitively
        in a single union, then picks the row with the most paper links.
        When ``entity_type`` is provided, only entities of that type are
        considered — this prevents e.g. 'ALMA' from binding to the
        planetary target ``Alma`` when the eval gold says the type is
        'instrument'. If the typed search yields nothing, falls back to
        the untyped search so we still resolve when the gold's type hint
        is wrong/missing.
        """
        conn = self._get_conn()
        lowered = name.strip().lower()
        if not lowered:
            return None

        with conn.cursor() as cur:
            for type_filter in ([entity_type] if entity_type else []) + [None]:
                params: list[Any] = [lowered, lowered]
                type_clause = ""
                if type_filter is not None:
                    type_clause = " AND e.entity_type = %s"
                    params.append(type_filter)
                cur.execute(
                    f"""
                    WITH candidates AS (
                        SELECT e.id, e.entity_type
                        FROM entities e
                        WHERE lower(e.canonical_name) = %s
                        UNION
                        SELECT e.id, e.entity_type
                        FROM entity_aliases a
                        JOIN entities e ON e.id = a.entity_id
                        WHERE lower(a.alias) = %s
                    )
                    SELECT c.id, COUNT(de.bibcode) AS n
                    FROM candidates c
                    JOIN entities e ON e.id = c.id
                    LEFT JOIN document_entities de ON de.entity_id = c.id
                    WHERE TRUE{type_clause}
                    GROUP BY c.id
                    ORDER BY n DESC, c.id ASC
                    LIMIT 1
                    """,
                    tuple(params),
                )
                row = cur.fetchone()
                if row is not None:
                    return int(row[0])
        return None

    def _entity_papers(self, entity_id: int, top_k: int) -> list[RetrievalDoc]:
        """Return ``top_k`` papers linked to the entity, ranked by pagerank.

        ``document_entities`` may carry multiple rows per (bibcode, entity_id)
        when distinct link_types fire — e.g. an existing tier-2
        ``abstract_match`` row plus a part_of-inheritance ``inherited`` row
        from ``scripts/backfill_part_of_inheritance.py``. The DISTINCT ON
        keeps each bibcode at most once so the judge doesn't see duplicates.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bibcode, title, abstract FROM (
                    SELECT DISTINCT ON (p.bibcode)
                           p.bibcode,
                           p.title,
                           COALESCE(p.abstract, '') AS abstract,
                           pm.pagerank
                      FROM document_entities de
                      JOIN papers p ON p.bibcode = de.bibcode
                      LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode
                     WHERE de.entity_id = %s
                  ORDER BY p.bibcode
                ) s
                ORDER BY pagerank DESC NULLS LAST
                LIMIT %s
                """,
                (entity_id, top_k),
            )
            rows = cur.fetchall()
        return [
            RetrievalDoc(
                bibcode=str(bibcode),
                title=str(title or ""),
                snippet=str(abstract or "")[:400],
            )
            for bibcode, title, abstract in rows
        ]

    def retrieve(self, gold: GoldQuery, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievalDoc]:
        if gold.prop != "specific_entity":
            return self.inner.retrieve(gold, top_k=top_k)

        name = gold.extra.get("entity_name")
        if not isinstance(name, str) or not name.strip():
            logger.info("specific_entity: query %r has no entity_name", gold.query_id)
            return []
        entity_type_hint = gold.extra.get("entity_type")
        type_str = entity_type_hint if isinstance(entity_type_hint, str) else None
        entity_id = self._resolve_entity_id(name, entity_type=type_str)
        if entity_id is None:
            logger.info(
                "specific_entity: name %r (query %r) does not resolve to any entity",
                name,
                gold.query_id,
            )
            return []
        return self._entity_papers(entity_id, top_k=top_k)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        close_inner = getattr(self.inner, "close", None)
        if callable(close_inner):
            close_inner()


# ---------------------------------------------------------------------------
# Judge invocation (claude -p subprocess)
# ---------------------------------------------------------------------------


class JudgeBackend(Protocol):
    """Any callable that scores one (query, retrieval) pair."""

    def judge(self, gold: GoldQuery, docs: list[RetrievalDoc]) -> tuple[int, str]: ...


@dataclass
class StubJudge:
    """Deterministic in-memory judge for tests.

    Returns ``fixed_score`` with a canned rationale. ``calls`` records
    every invocation for assertion.
    """

    fixed_score: int = 2
    rationale: str = "stub"
    calls: list[tuple[GoldQuery, list[RetrievalDoc]]] = field(default_factory=list)

    def judge(self, gold: GoldQuery, docs: list[RetrievalDoc]) -> tuple[int, str]:
        self.calls.append((gold, list(docs)))
        return self.fixed_score, self.rationale


@dataclass
class ClaudeSubprocessJudge:
    """Judge backend that shells out to ``claude -p --output-format=json``.

    Alternative: an in-session Claude Code subagent could be invoked via
    the ``Agent`` tool. We prefer the subprocess path here because the
    script is designed to run as a standalone eval (including in CI /
    cron), where there is no surrounding Claude Code session to host an
    Agent-tool call.

    Fails fast with a clear error if ``claude`` is missing from ``PATH``.

    Attributes:
        claude_binary: CLI binary name / absolute path.
        timeout_s: Per-invocation wall-clock cap. Timeouts propagate as
            a ``RuntimeError`` rather than being swallowed.
    """

    claude_binary: str = "claude"
    timeout_s: int = DEFAULT_JUDGE_TIMEOUT_S
    retries: int = DEFAULT_JUDGE_RETRIES
    retry_base_delay_s: float = DEFAULT_JUDGE_RETRY_BASE_DELAY_S

    def __post_init__(self) -> None:
        if shutil.which(self.claude_binary) is None:
            raise FileNotFoundError(
                f"'{self.claude_binary}' not found on PATH. "
                "The eval harness needs the Claude Code CLI to invoke the "
                "judge subagent. Install Claude Code or pass "
                "--no-judge / use a mock dispatcher in tests."
            )

    def _build_prompt(self, gold: GoldQuery, docs: list[RetrievalDoc]) -> str:
        retrieval_block = "\n".join(
            f"- [{i + 1}] {d.bibcode}: {d.title}" + (f"\n    {d.snippet}" if d.snippet else "")
            for i, d in enumerate(docs)
        ) or "(no results returned)"

        extras = (
            "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in gold.extra.items())
            or "(no extras)"
        )

        return (
            "You are evaluating one of the six entity-enrichment value props "
            "for a scientific-literature retrieval system (NASA ADS / SciX).\n\n"
            f"Value prop: {gold.prop}\n"
            f"Query id: {gold.query_id}\n"
            f"Query text: {gold.query}\n"
            f"Query metadata:\n{extras}\n\n"
            f"Expectation / success criterion:\n{gold.expectation}\n\n"
            f"Retrieval top results (bibcode and title):\n{retrieval_block}\n\n"
            "Score on this rubric (ordinal 0-3):\n"
            "  0 = fails — the value prop is not delivered.\n"
            "  1 = partial — one or two relevant results at most.\n"
            "  2 = mostly works — majority of results satisfy the expectation.\n"
            "  3 = works correctly — retrieval cleanly delivers the value prop.\n\n"
            'Respond with exactly one JSON object on a single line: '
            '{"score": <0-3>, "rationale": "<one or two sentences>"}. '
            "No other text."
        )

    def judge(self, gold: GoldQuery, docs: list[RetrievalDoc]) -> tuple[int, str]:
        prompt = self._build_prompt(gold, docs)
        last_err: str = ""
        attempts = max(1, self.retries)
        for attempt in range(1, attempts + 1):
            try:
                # nosec B603 — deliberate exec of the Claude CLI with no shell
                # expansion. ``prompt`` is a single argv entry.
                completed = subprocess.run(
                    [self.claude_binary, "-p", "--output-format=json", prompt],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                last_err = f"timed out after {self.timeout_s}s"
            except FileNotFoundError as exc:
                raise RuntimeError(f"claude binary disappeared: {exc}") from exc
            else:
                if completed.returncode == 0:
                    return _parse_judge_stdout(completed.stdout)
                last_err = (
                    f"exited {completed.returncode} "
                    f"stderr={completed.stderr[:200]!r} "
                    f"stdout={completed.stdout[:200]!r}"
                )

            if attempt < attempts:
                # Exponential backoff: 5s, 10s, 20s, ...
                delay = self.retry_base_delay_s * (2 ** (attempt - 1))
                logger.warning(
                    "claude -p attempt %d/%d failed for %s (%s); retrying in %.1fs",
                    attempt,
                    attempts,
                    gold.query_id,
                    last_err,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"claude -p failed after {attempts} attempts for {gold.query_id!r}: {last_err}"
        )


def _parse_judge_stdout(raw: str) -> tuple[int, str]:
    """Parse the ``claude -p --output-format=json`` wrapper output.

    Claude Code's JSON output wraps the assistant reply in an envelope::

        {"type": "result", "result": "<reply text>", ...}

    We extract ``result`` and parse our own inner JSON
    (``{"score": int, "rationale": str}``). Falls back to treating the
    whole stdout as the reply if no envelope is detected (so raw-reply
    modes still work).
    """
    if not raw or not raw.strip():
        raise RuntimeError("empty judge stdout")

    text = raw.strip()
    inner_text = text
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict):
        # claude -p --output-format=json envelope
        result = envelope.get("result")
        if isinstance(result, str) and result.strip():
            inner_text = result.strip()

    # Scan for a {"score": int, "rationale": str} object anywhere in the text.
    payload = _extract_score_payload(inner_text)
    if payload is None:
        raise RuntimeError(f"no parseable score payload in judge response: {inner_text[:200]!r}")
    score_raw = payload.get("score")
    rationale_raw = payload.get("rationale", payload.get("reason", ""))
    if isinstance(score_raw, bool) or not isinstance(score_raw, int):
        raise RuntimeError(f"score must be int, got {score_raw!r}")
    if not (RUBRIC_MIN <= score_raw <= RUBRIC_MAX):
        raise RuntimeError(f"score {score_raw} out of [{RUBRIC_MIN}, {RUBRIC_MAX}]")
    rationale = str(rationale_raw)[:1000] if rationale_raw is not None else ""
    return score_raw, rationale


def _extract_score_payload(text: str) -> dict[str, Any] | None:
    """Scan ``text`` for a JSON object carrying a ``score`` field.

    Handles:
      - clean JSON on a single line;
      - JSON embedded in prose;
      - JSON in a ```json fenced block.

    Returns the parsed object or ``None``.
    """
    # Try the whole text first.
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and "score" in obj:
        return obj

    # Fenced-block fallback: ```json\n{...}\n```
    start = text.find("```")
    while start != -1:
        end = text.find("```", start + 3)
        if end == -1:
            break
        block = text[start + 3 : end]
        # drop the language tag if present
        if "\n" in block:
            block = block.split("\n", 1)[1]
        try:
            obj = json.loads(block.strip())
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "score" in obj:
            return obj
        start = text.find("```", end + 3)

    # Brace-scan fallback: first balanced {...} containing "score".
    depth = 0
    begin: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and begin is not None:
                candidate = text[begin : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and "score" in obj:
                    return obj
                begin = None
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropSummary:
    """Rolled-up stats for one value prop."""

    prop: str
    n: int
    mean: float
    stderr: float
    scores: tuple[int, ...]


def summarize_prop(prop: str, results: list[JudgeResult]) -> PropSummary:
    """Compute mean and standard error for a prop's results.

    Standard error uses the sample standard deviation divided by
    ``sqrt(n)``. For ``n == 1`` the stderr is ``0.0`` (no
    within-sample variance), mirroring the convention in the PRD
    writeup template.
    """
    scores = [r.rubric_score for r in results if r.prop == prop]
    n = len(scores)
    if n == 0:
        return PropSummary(prop=prop, n=0, mean=0.0, stderr=0.0, scores=())
    mean = statistics.fmean(scores)
    if n < 2:
        stderr = 0.0
    else:
        stderr = statistics.stdev(scores) / math.sqrt(n)
    return PropSummary(prop=prop, n=n, mean=mean, stderr=stderr, scores=tuple(scores))


def overall_score(summaries: list[PropSummary]) -> float:
    """Mean of per-prop means. Props with n=0 are skipped.

    Weighting per-prop (not per-query) keeps a prop with 50 queries
    from dominating a prop with 10 — the point of this eval is the
    balance across value props, not bulk query count.
    """
    usable = [s.mean for s in summaries if s.n > 0]
    if not usable:
        return 0.0
    return statistics.fmean(usable)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, results: Iterable[JudgeResult]) -> None:
    """Write judge results to a JSONL file (one JSON object per line).

    Schema per line: ``prop``, ``query`` (query_id — per acceptance
    criterion #4), ``query_text``, ``rubric_score``, ``judge_rationale``,
    ``retrieval_count``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in results:
            row = {
                "prop": r.prop,
                "query": r.query_id,
                "query_text": r.query,
                "rubric_score": r.rubric_score,
                "judge_rationale": r.judge_rationale,
                "retrieval_count": r.retrieval_count,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(
    summaries: list[PropSummary],
    per_prop_results: dict[str, list[JudgeResult]],
    *,
    run_timestamp: str,
) -> str:
    """Render the markdown writeup.

    Includes:
      - overall score
      - summary table {prop, n, mean, stderr}
      - per-prop section with query-level scores + rationales
    """
    lines: list[str] = []
    lines.append("# Entity Enrichment Value Props Eval — 2026-04")
    lines.append("")
    lines.append(
        f"_Generated by `scripts/eval_entity_value_props.py` on {run_timestamp}._"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    overall = overall_score(summaries)
    lines.append(f"Overall score: {overall:.2f} / 3.0")
    lines.append("")
    lines.append("| Prop | N | Mean | StdErr |")
    lines.append("|---|---|---|---|")
    for s in summaries:
        lines.append(f"| {s.prop} | {s.n} | {s.mean:.2f} | {s.stderr:.2f} |")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s.prop}")
        lines.append("")
        lines.append(f"N = {s.n}, mean = {s.mean:.2f}, stderr = {s.stderr:.2f}")
        lines.append("")
        results = per_prop_results.get(s.prop, [])
        if not results:
            lines.append("_No results recorded for this prop._")
            lines.append("")
            continue
        lines.append("| Query ID | Score | Rationale |")
        lines.append("|---|---|---|")
        for r in results:
            rationale = (r.judge_rationale or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.query_id} | {r.rubric_score} | {rationale} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_report(path: Path, body: str) -> None:
    """Write the rendered report to ``path`` (parent dirs created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------


def run_eval(
    gold_by_prop: dict[str, list[GoldQuery]],
    retrieval: RetrievalBackend,
    judge: JudgeBackend,
    *,
    top_k: int = DEFAULT_TOP_K,
    on_result: Callable[[JudgeResult], None] | None = None,
) -> list[JudgeResult]:
    """Run the retrieval + judge pipeline over all gold queries.

    ``on_result`` is invoked once per produced result — used to stream
    rows to the JSONL artifact as the eval runs rather than buffering.
    """
    out: list[JudgeResult] = []
    for prop in PROPS:
        queries = gold_by_prop.get(prop, [])
        for gold in queries:
            docs = retrieval.retrieve(gold, top_k=top_k)
            score, rationale = judge.judge(gold, docs)
            result = JudgeResult(
                prop=prop,
                query_id=gold.query_id,
                query=gold.query,
                rubric_score=int(score),
                judge_rationale=rationale,
                retrieval_count=len(docs),
            )
            out.append(result)
            if on_result is not None:
                on_result(result)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_props(cli_values: list[str]) -> list[str]:
    if not cli_values or "all" in cli_values:
        return list(PROPS)
    out: list[str] = []
    for v in cli_values:
        canonical = CLI_PROP_ALIASES.get(v)
        if canonical is None:
            raise SystemExit(f"unknown prop {v!r}; valid: {sorted(CLI_PROP_ALIASES)} or 'all'")
        if canonical not in out:
            out.append(canonical)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_entity_value_props",
        description="Entity-enrichment value-props eval harness.",
    )
    parser.add_argument(
        "--props",
        nargs="+",
        default=["all"],
        choices=sorted(set(CLI_PROP_ALIASES) | {"all"}),
        metavar="PROP",
        help="value props to evaluate; use 'all' for every prop (default).",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="Postgres DSN (default: env SCIX_DSN or 'dbname=scix').",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=DEFAULT_GOLD_DIR,
        help="Directory containing per-prop YAML gold sets.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory for JSONL judge-output artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown writeup output path.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Overwrite the markdown writeup at --report-path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieval hits to expose to the judge per query.",
    )
    parser.add_argument(
        "--claude-binary",
        default="claude",
        help="Path to the Claude Code CLI (used only when judging live).",
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=int,
        default=DEFAULT_JUDGE_TIMEOUT_S,
        help=(
            "Per-invocation wall-clock cap on the claude -p judge subprocess. "
            f"Default: {DEFAULT_JUDGE_TIMEOUT_S}s. Bump when judge calls are slow."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load gold sets, print the planned prop list, and exit 0.",
    )
    parser.add_argument(
        "--stub-judge",
        action="store_true",
        help=(
            "Use the stub (non-Claude) judge. Intended for smoke tests — "
            "scores are fixed and meaningless."
        ),
    )
    parser.add_argument(
        "--stub-retrieval",
        action="store_true",
        help=(
            "Use a synthetic in-memory retrieval backend. Avoids DB / search-stack "
            "dependencies so the script can be exercised on any machine."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        selected_props = _resolve_props(args.props)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"[eval_entity_value_props] props: {', '.join(selected_props)}")
    print(f"[eval_entity_value_props] db:    {args.db}")
    print(f"[eval_entity_value_props] gold:  {args.gold_dir}")

    if args.dry_run:
        # Also verify the gold sets parse — surface missing files / schema
        # errors here rather than at live-judge time.
        try:
            gold = load_all_gold_sets(args.gold_dir, selected_props)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[eval_entity_value_props] gold-set validation failed: {exc}", file=sys.stderr)
            return 1
        for prop in selected_props:
            print(f"  - {prop}: {len(gold[prop])} queries")
        print("[eval_entity_value_props] dry run — exiting without running judge")
        return 0

    # Live path.
    try:
        gold = load_all_gold_sets(args.gold_dir, selected_props)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[eval_entity_value_props] could not load gold sets: {exc}", file=sys.stderr)
        return 1

    retrieval: RetrievalBackend
    if args.stub_retrieval:
        retrieval = StubRetrievalBackend()
    else:
        retrieval = HybridSearchBackend(dsn=args.db)
        # Wire prop-aware backends at the harness level. Each delegates
        # to the inner backend for non-matching props, so additions are
        # cost-free for unaffected evals. Intentionally scoped to the
        # eval harness only — src/scix/search.py stays untouched
        # (beads xz4.1.34, xz4.1.39).
        if "community_expansion" in selected_props:
            retrieval = CommunityExpansionBackend(inner=retrieval, dsn=args.db)
        if "specific_entity" in selected_props:
            retrieval = SpecificEntityBackend(inner=retrieval, dsn=args.db)

    judge: JudgeBackend
    if args.stub_judge:
        judge = StubJudge()
    else:
        try:
            judge = ClaudeSubprocessJudge(
                claude_binary=args.claude_binary, timeout_s=args.judge_timeout_s
            )
        except FileNotFoundError as exc:
            print(f"[eval_entity_value_props] {exc}", file=sys.stderr)
            return 1

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    artifact_path = args.artifact_dir / f"eval-d4-{ts}.jsonl"
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    # Append-as-we-go so crashes don't lose partial results.
    artifact_handle = artifact_path.open("w", encoding="utf-8")

    def _append_jsonl(result: JudgeResult) -> None:
        row = {
            "prop": result.prop,
            "query": result.query_id,
            "query_text": result.query,
            "rubric_score": result.rubric_score,
            "judge_rationale": result.judge_rationale,
            "retrieval_count": result.retrieval_count,
        }
        artifact_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifact_handle.flush()

    try:
        results = run_eval(gold, retrieval, judge, top_k=args.top_k, on_result=_append_jsonl)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        logger.exception("eval run failed")
        print(f"[eval_entity_value_props] run failed: {exc}", file=sys.stderr)
        return 2
    finally:
        artifact_handle.close()
        close_fn = getattr(retrieval, "close", None)
        if callable(close_fn):
            close_fn()

    summaries = [summarize_prop(prop, results) for prop in selected_props]
    print("[eval_entity_value_props] summary:")
    for s in summaries:
        print(f"  {s.prop}: n={s.n} mean={s.mean:.2f} stderr={s.stderr:.2f}")
    print(f"[eval_entity_value_props] overall: {overall_score(summaries):.2f} / 3.0")
    print(f"[eval_entity_value_props] artifact: {artifact_path}")

    if args.write_report:
        per_prop: dict[str, list[JudgeResult]] = {p: [] for p in selected_props}
        for r in results:
            per_prop.setdefault(r.prop, []).append(r)
        body = render_report(summaries, per_prop, run_timestamp=ts)
        write_report(args.report_path, body)
        print(f"[eval_entity_value_props] report:   {args.report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
