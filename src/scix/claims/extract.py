"""Nanopub-inspired claim extraction pipeline.

Pipeline shape::

    papers_fulltext row (bibcode, sections JSONB list of {heading, level, text, offset})
        -> per section, optionally filtered by section role
            -> split into paragraphs (preserving paragraph_index within section)
                -> LLMClient.extract(prompt, paragraph_text) -> list[ClaimDict]
                    -> validate (required fields + char-span bounds)
                        -> INSERT into paper_claims (idempotent)

CRITICAL POLICY
---------------
This module MUST NOT import any paid-API SDK. All LLM calls in production flow
through Claude Code OAuth subagents — i.e. the user's already-authenticated
``claude`` CLI. The default :class:`LLMClient` shells out to that CLI via
``subprocess``. Tests use :class:`StubLLMClient` so pytest never touches Claude.

Idempotency
-----------
Re-extracting the same paper does not produce duplicate rows. We rely on a
unique partial index over ``(bibcode, section_index, paragraph_index,
char_span_start, char_span_end, claim_text)`` plus ``ON CONFLICT DO NOTHING``.
The index is created lazily via ``CREATE UNIQUE INDEX IF NOT EXISTS`` on the
first call to :func:`extract_claims_for_paper` (cheap no-op when present).

Provenance contract
-------------------
``char_span_start`` / ``char_span_end`` are absolute, end-exclusive offsets into
the *paragraph_text* passed to the LLM (matching the prompt's contract). They
are stored verbatim in ``paper_claims`` so downstream consumers can recover the
anchor span via ``paragraph_text[char_span_start:char_span_end]``.

Robustness
----------
- Invalid JSON from the LLM -> log warning, skip the paragraph.
- Out-of-bounds char_span -> skip that single claim.
- Missing required claim fields -> skip that single claim.
- Unknown ``claim_type`` -> skip that single claim (the schema CHECK constraint
  would otherwise reject the INSERT and abort the surrounding transaction).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Claim types accepted by the ``paper_claims_claim_type_check`` CHECK constraint
#: in migration 062. Anything outside this set is dropped before INSERT.
VALID_CLAIM_TYPES: frozenset[str] = frozenset(
    {
        "factual",
        "methodological",
        "comparative",
        "speculative",
        "cited_from_other",
    }
)

#: Required fields on each claim object returned by the LLM. Any missing key
#: causes the claim to be skipped.
_REQUIRED_CLAIM_FIELDS: tuple[str, ...] = (
    "claim_text",
    "claim_type",
    "char_span_start",
    "char_span_end",
)

#: SQL for the unique partial index that backs idempotency. Created lazily.
_UNIQUE_INDEX_SQL: str = (
    "CREATE UNIQUE INDEX IF NOT EXISTS "
    "ux_paper_claims_provenance_text "
    "ON paper_claims "
    "(bibcode, section_index, paragraph_index, "
    " char_span_start, char_span_end, md5(claim_text))"
)

#: Heading -> role classifier table. Substring match (case-insensitive). Order
#: matters: more-specific phrases first, generic last.
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("abstract", "abstract"),
    ("introduction", "introduction"),
    ("intro", "introduction"),
    ("background", "introduction"),
    ("motivation", "introduction"),
    ("related work", "related_work"),
    ("prior work", "related_work"),
    ("data", "data"),
    ("observation", "data"),
    ("methodology", "methods"),
    ("methods", "methods"),
    ("method", "methods"),
    ("approach", "methods"),
    ("experiment", "methods"),
    ("result", "results"),
    ("analysis", "results"),
    ("finding", "results"),
    ("evaluation", "results"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    ("summary", "conclusion"),
    ("future work", "conclusion"),
    ("acknowledg", "acknowledgments"),
    ("reference", "references"),
    ("bibliograph", "references"),
    ("appendix", "appendix"),
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ClaimDict(TypedDict, total=False):
    """Shape of a single claim object — both as returned by an LLMClient and
    as persisted into ``paper_claims``.

    ``total=False`` because ``subject``/``predicate``/``object``/``confidence``
    are optional in the prompt schema; the four provenance/required fields are
    enforced by :func:`_validate_claim` instead of by the type system so we can
    fail-soft rather than raise.
    """

    claim_text: str
    claim_type: str
    subject: str | None
    predicate: str | None
    object: str | None
    char_span_start: int
    char_span_end: int
    confidence: float | None


class LLMClient(Protocol):
    """Pluggable LLM wrapper. Implementations must return a list of ClaimDict-like
    mappings or raise; callers tolerate JSON-decode failures by treating them
    as "no claims for this paragraph"."""

    def extract(self, prompt: str, paragraph: str) -> list[ClaimDict]:  # pragma: no cover - Protocol
        """Run claim extraction over ``paragraph`` and return zero or more claims.

        Implementations MAY raise ``json.JSONDecodeError`` if the underlying
        model produced unparseable output; the pipeline catches it and skips
        the paragraph rather than aborting the whole paper.
        """
        ...


# ---------------------------------------------------------------------------
# LLMClient implementations
# ---------------------------------------------------------------------------


class ClaudeCliLLMClient:
    """LLMClient that shells out to the ``claude`` CLI via subprocess.

    The user's existing OAuth-authenticated ``claude`` binary is treated as a
    Claude Code subagent. We intentionally do NOT use any paid-API SDK
    (no API-key path).

    The CLI is invoked with ``-p ""`` (run a one-shot prompt) and the formatted
    prompt is piped to stdin. The CLI's stdout is parsed as JSON.

    Args:
        cli_path: Path to the ``claude`` binary. Defaults to the
            ``SCIX_CLAUDE_CLI`` env var, falling back to literal ``"claude"``.
        timeout_seconds: Hard cap on a single CLI invocation. Per the
            architecture rules, timeouts at trust boundaries are allowed for
            blast-radius control but must propagate failure (we surface a
            JSONDecodeError-equivalent so the pipeline skips the paragraph).
    """

    def __init__(
        self,
        cli_path: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._cli_path: str = cli_path or os.environ.get("SCIX_CLAUDE_CLI", "claude")
        self._timeout: float = timeout_seconds

    def extract(self, prompt: str, paragraph: str) -> list[ClaimDict]:
        """Pipe ``prompt`` to ``claude -p ""`` on stdin, parse stdout as JSON."""
        try:
            completed = subprocess.run(
                [self._cli_path, "-p", ""],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            # Treat a missing binary as a per-paragraph failure rather than a
            # crash: callers may have many papers in flight.
            raise json.JSONDecodeError(
                f"claude CLI not found at {self._cli_path}: {exc}", doc="", pos=0
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise json.JSONDecodeError(
                f"claude CLI timed out after {self._timeout}s", doc="", pos=0
            ) from exc

        if completed.returncode != 0:
            raise json.JSONDecodeError(
                f"claude CLI exited {completed.returncode}: {completed.stderr.strip()[:500]}",
                doc=completed.stdout or "",
                pos=0,
            )

        stdout = (completed.stdout or "").strip()
        # The prompt instructs Claude to return ONLY a JSON array; if the CLI
        # adds wrapping text we still try to find the outermost array.
        return _parse_claims_json(stdout)


class StubLLMClient:
    """In-memory LLMClient that returns canned responses. Test-only.

    Behaviour modes (in order of precedence):

    1. If ``raise_exc`` is set, every ``extract()`` call raises that exception.
    2. If ``responses`` is a callable, it's invoked with ``(prompt, paragraph)``
       and its return value is yielded.
    3. If ``responses`` is a list, calls consume it FIFO; once exhausted,
       returns ``[]``.
    4. Otherwise returns ``default`` for every call.
    """

    def __init__(
        self,
        responses: list[list[ClaimDict]] | None = None,
        default: list[ClaimDict] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._queue: list[list[ClaimDict]] = list(responses) if responses else []
        self._default: list[ClaimDict] = default if default is not None else []
        self._raise_exc: BaseException | None = raise_exc
        self.calls: list[tuple[str, str]] = []

    def extract(self, prompt: str, paragraph: str) -> list[ClaimDict]:
        self.calls.append((prompt, paragraph))
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._queue:
            return self._queue.pop(0)
        return list(self._default)


# ---------------------------------------------------------------------------
# Section role classification
# ---------------------------------------------------------------------------


def classify_section_role(heading: str | None) -> str:
    """Map a section heading to a coarse role.

    Returns ``"other"`` for headings that don't match any known pattern (and for
    None/empty inputs). The classifier is deliberately simple — substring match
    on a lowercased heading — because the section_role.py module the PRD
    referenced does not exist in this branch yet.
    """
    if not heading:
        return "other"
    lowered = heading.strip().lower()
    # Strip leading numbering like "3.1 " or "II. ".
    lowered = re.sub(r"^[ivxlcdm0-9]+[.)\s-]+", "", lowered).strip()
    for needle, role in _ROLE_PATTERNS:
        if needle in lowered:
            return role
    return "other"


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------


def split_paragraphs(section_text: str) -> list[tuple[int, str, int]]:
    """Split section text into paragraphs.

    Returns a list of ``(paragraph_index, paragraph_text, paragraph_start_offset)``
    tuples where ``paragraph_text`` is the trimmed paragraph and
    ``paragraph_start_offset`` is the offset of paragraph_text[0] within
    ``section_text`` (useful only for callers that want section-relative
    offsets; the pipeline itself stores offsets relative to the paragraph,
    matching the prompt's contract).

    Splits on runs of two-or-more newlines. Empty paragraphs are dropped.
    """
    if not section_text:
        return []

    out: list[tuple[int, str, int]] = []
    cursor = 0
    paragraph_idx = 0
    # Find paragraph boundaries (one-or-more blank lines).
    for chunk in re.split(r"\n[ \t]*\n+", section_text):
        # Recover this chunk's start offset by scanning forward from cursor.
        start = section_text.find(chunk, cursor)
        if start < 0:  # pragma: no cover - shouldn't happen
            start = cursor
        cursor = start + len(chunk)
        # Strip leading/trailing whitespace but track the resulting offset.
        stripped = chunk.lstrip()
        leading = len(chunk) - len(stripped)
        stripped = stripped.rstrip()
        if not stripped:
            continue
        out.append((paragraph_idx, stripped, start + leading))
        paragraph_idx += 1
    return out


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "claim_extraction_v1.md"


def _load_prompt_template() -> str:
    """Load the v1 prompt template from prompts/claim_extraction_v1.md.

    The template uses the literal placeholders ``{paper_bibcode}``,
    ``{section_heading}``, ``{section_index}``, ``{paragraph_index}``,
    ``{paragraph_text}``. We don't call ``.format()`` on the raw markdown
    (which contains literal ``{`` in JSON examples) — we substitute manually.
    """
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    # Fallback minimal template so the module is usable in environments that
    # haven't shipped the prompt file yet (e.g. early test fixtures). The
    # placeholders are still substituted; the LLMClient is responsible for
    # interpreting the rest of the prompt.
    return (
        "Extract atomic, nanopub-style claims from the following paragraph.\n\n"
        "paper_bibcode:    {paper_bibcode}\n"
        "section_heading:  {section_heading}\n"
        "section_index:    {section_index}\n"
        "paragraph_index:  {paragraph_index}\n\n"
        'paragraph_text:\n"""\n{paragraph_text}\n"""\n'
    )


def _format_prompt(
    template: str,
    *,
    paper_bibcode: str,
    section_heading: str,
    section_index: int,
    paragraph_index: int,
    paragraph_text: str,
) -> str:
    """Substitute placeholders in a way that's safe even if the template
    contains literal braces (e.g. JSON examples in markdown)."""
    out = template
    out = out.replace("{paper_bibcode}", paper_bibcode)
    out = out.replace("{section_heading}", section_heading)
    out = out.replace("{section_index}", str(section_index))
    out = out.replace("{paragraph_index}", str(paragraph_index))
    out = out.replace("{paragraph_text}", paragraph_text)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _parse_claims_json(raw: str) -> list[ClaimDict]:
    """Parse the LLM's stdout. Tries strict JSON first; falls back to extracting
    the outermost ``[...]`` array if extra text snuck in. Raises
    ``json.JSONDecodeError`` if no valid array can be recovered."""
    text = (raw or "").strip()
    if not text:
        raise json.JSONDecodeError("empty LLM output", doc=text, pos=0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        first = text.find("[")
        last = text.rfind("]")
        if first == -1 or last == -1 or last <= first:
            raise
        parsed = json.loads(text[first : last + 1])
    if not isinstance(parsed, list):
        raise json.JSONDecodeError(
            f"expected JSON array, got {type(parsed).__name__}", doc=text, pos=0
        )
    return [c for c in parsed if isinstance(c, Mapping)]  # type: ignore[misc]


def _validate_claim(
    claim: Mapping[str, Any],
    paragraph_text: str,
) -> ClaimDict | None:
    """Return a normalized ClaimDict, or None if the claim must be skipped.

    Skip reasons (each logged at WARNING):
      - missing required field
      - claim_type not in VALID_CLAIM_TYPES
      - char_span_start / char_span_end not int-coercible
      - char_span out of bounds for paragraph_text
      - char_span_end <= char_span_start
    """
    for key in _REQUIRED_CLAIM_FIELDS:
        if key not in claim:
            logger.warning("claim missing required field %r — skipping", key)
            return None

    claim_type = claim.get("claim_type")
    if claim_type not in VALID_CLAIM_TYPES:
        logger.warning("claim_type %r not in valid set — skipping", claim_type)
        return None

    try:
        start = int(claim["char_span_start"])
        end = int(claim["char_span_end"])
    except (TypeError, ValueError):
        logger.warning(
            "non-integer char_span (start=%r, end=%r) — skipping",
            claim.get("char_span_start"),
            claim.get("char_span_end"),
        )
        return None

    if start < 0 or end > len(paragraph_text) or end <= start:
        logger.warning(
            "char_span out of bounds (start=%d, end=%d, paragraph_len=%d) — skipping",
            start,
            end,
            len(paragraph_text),
        )
        return None

    claim_text = claim.get("claim_text")
    if not isinstance(claim_text, str) or not claim_text.strip():
        logger.warning("claim_text missing or empty — skipping")
        return None

    confidence_raw = claim.get("confidence")
    confidence: float | None
    if confidence_raw is None:
        confidence = None
    else:
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = None

    normalized: ClaimDict = {
        "claim_text": claim_text,
        "claim_type": str(claim_type),
        "subject": _coerce_optional_str(claim.get("subject")),
        "predicate": _coerce_optional_str(claim.get("predicate")),
        "object": _coerce_optional_str(claim.get("object")),
        "char_span_start": start,
        "char_span_end": end,
        "confidence": confidence,
    }
    return normalized


def _coerce_optional_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return str(val)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _ensure_idempotency_index(conn: Any) -> None:
    """Lazily create the unique index that backs ON CONFLICT DO NOTHING.

    We use ``md5(claim_text)`` instead of the raw ``claim_text`` column because
    a single B-tree index entry is capped at ~2700 bytes and arbitrary claim
    text might exceed that. ``md5`` is a 16-byte digest collision-resistant
    enough for de-duplication at our extraction volumes (claims per paper
    typically O(100)).
    """
    with conn.cursor() as cur:
        cur.execute(_UNIQUE_INDEX_SQL)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def extract_claims_for_paper(
    conn: Any,
    bibcode: str,
    sections: Sequence[Mapping[str, Any]],
    llm: LLMClient,
    prompt_version: str,
    model_name: str,
    section_roles: Sequence[str] | None = None,
) -> int:
    """Extract claims from one paper and INSERT them into ``paper_claims``.

    Args:
        conn: An open psycopg connection (or compatible — we call ``.cursor()``,
            ``.commit()``).
        bibcode: Source paper bibcode. FK target into ``papers``.
        sections: Iterable of section dicts shaped like
            ``{heading, level, text, offset}`` — the JSONB list stored on
            ``papers_fulltext.sections``.
        llm: LLMClient implementation. In production this is
            :class:`ClaudeCliLLMClient`; in tests it's :class:`StubLLMClient`.
        prompt_version: Stored verbatim into ``extraction_prompt_version``.
        model_name: Stored verbatim into ``extraction_model``.
        section_roles: Optional whitelist. When provided, only sections whose
            heading classifies into one of these roles are processed. None
            means process all sections.

    Returns:
        The number of NEW rows inserted into ``paper_claims`` (i.e. excluding
        rows skipped by ON CONFLICT DO NOTHING).
    """
    _ensure_idempotency_index(conn)

    template = _load_prompt_template()
    role_filter: frozenset[str] | None = (
        frozenset(section_roles) if section_roles is not None else None
    )

    inserted = 0

    for section_index, section in enumerate(sections):
        heading = str(section.get("heading", "") or "")
        section_text = section.get("text", "") or ""
        if not isinstance(section_text, str):
            section_text = str(section_text)

        if role_filter is not None:
            role = classify_section_role(heading)
            if role not in role_filter:
                continue

        for paragraph_index, paragraph_text, _para_offset in split_paragraphs(section_text):
            prompt = _format_prompt(
                template,
                paper_bibcode=bibcode,
                section_heading=heading,
                section_index=section_index,
                paragraph_index=paragraph_index,
                paragraph_text=paragraph_text,
            )

            try:
                raw_claims: Iterable[Mapping[str, Any]] = llm.extract(prompt, paragraph_text)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "LLM returned invalid JSON for %s section=%d paragraph=%d: %s",
                    bibcode,
                    section_index,
                    paragraph_index,
                    exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — fail-soft per acceptance criterion
                logger.warning(
                    "LLM raised unexpectedly for %s section=%d paragraph=%d: %s",
                    bibcode,
                    section_index,
                    paragraph_index,
                    exc,
                )
                continue

            inserted += _persist_claims(
                conn=conn,
                bibcode=bibcode,
                section_index=section_index,
                paragraph_index=paragraph_index,
                paragraph_text=paragraph_text,
                raw_claims=raw_claims,
                model_name=model_name,
                prompt_version=prompt_version,
            )

    conn.commit()
    return inserted


def _persist_claims(
    *,
    conn: Any,
    bibcode: str,
    section_index: int,
    paragraph_index: int,
    paragraph_text: str,
    raw_claims: Iterable[Mapping[str, Any]],
    model_name: str,
    prompt_version: str,
) -> int:
    """Validate and INSERT each claim. Returns the count of newly-inserted rows."""
    # ON CONFLICT clause must match the unique-index column list verbatim,
    # including the md5(claim_text) expression — otherwise PG raises 42P10
    # ("no unique or exclusion constraint matching ON CONFLICT") and we'd
    # have to fall back to SELECT-then-INSERT.
    insert_sql = (
        "INSERT INTO paper_claims ("
        "  bibcode, section_index, paragraph_index,"
        "  char_span_start, char_span_end,"
        "  claim_text, claim_type,"
        "  subject, predicate, object,"
        "  confidence, extraction_model, extraction_prompt_version"
        ") VALUES ("
        "  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s"
        ") ON CONFLICT (bibcode, section_index, paragraph_index, "
        "  char_span_start, char_span_end, md5(claim_text)) DO NOTHING "
        "RETURNING claim_id"
    )

    inserted = 0
    with conn.cursor() as cur:
        for raw in raw_claims:
            if not isinstance(raw, Mapping):
                logger.warning("non-dict claim payload — skipping: %r", raw)
                continue
            claim = _validate_claim(raw, paragraph_text)
            if claim is None:
                continue

            try:
                cur.execute(
                    insert_sql,
                    (
                        bibcode,
                        section_index,
                        paragraph_index,
                        claim["char_span_start"],
                        claim["char_span_end"],
                        claim["claim_text"],
                        claim["claim_type"],
                        claim.get("subject"),
                        claim.get("predicate"),
                        claim.get("object"),
                        claim.get("confidence"),
                        model_name,
                        prompt_version,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — fail-soft per acceptance criterion
                # ON CONFLICT ON CONSTRAINT requires the constraint to exist;
                # if it doesn't (e.g. older partial schema), fall back to a
                # SELECT-existence check before the INSERT so we still get
                # idempotency without aborting the surrounding transaction.
                logger.warning(
                    "INSERT failed for %s section=%d paragraph=%d claim=%r: %s",
                    bibcode,
                    section_index,
                    paragraph_index,
                    claim["claim_text"][:80],
                    exc,
                )
                # Roll back the savepoint implicitly opened by psycopg on the
                # failed statement so subsequent statements in this txn can
                # still run.
                conn.rollback()
                inserted += _persist_claim_select_then_insert(
                    conn=conn,
                    bibcode=bibcode,
                    section_index=section_index,
                    paragraph_index=paragraph_index,
                    claim=claim,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
                continue

            row = cur.fetchone()
            if row is not None:
                inserted += 1

    return inserted


def _persist_claim_select_then_insert(
    *,
    conn: Any,
    bibcode: str,
    section_index: int,
    paragraph_index: int,
    claim: ClaimDict,
    model_name: str,
    prompt_version: str,
) -> int:
    """Fallback idempotent INSERT: SELECT first, INSERT only if absent."""
    select_sql = (
        "SELECT 1 FROM paper_claims "
        "WHERE bibcode = %s AND section_index = %s AND paragraph_index = %s "
        "  AND char_span_start = %s AND char_span_end = %s AND claim_text = %s "
        "LIMIT 1"
    )
    insert_sql = (
        "INSERT INTO paper_claims ("
        "  bibcode, section_index, paragraph_index,"
        "  char_span_start, char_span_end,"
        "  claim_text, claim_type,"
        "  subject, predicate, object,"
        "  confidence, extraction_model, extraction_prompt_version"
        ") VALUES ("
        "  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s"
        ")"
    )
    with conn.cursor() as cur:
        cur.execute(
            select_sql,
            (
                bibcode,
                section_index,
                paragraph_index,
                claim["char_span_start"],
                claim["char_span_end"],
                claim["claim_text"],
            ),
        )
        if cur.fetchone() is not None:
            return 0
        cur.execute(
            insert_sql,
            (
                bibcode,
                section_index,
                paragraph_index,
                claim["char_span_start"],
                claim["char_span_end"],
                claim["claim_text"],
                claim["claim_type"],
                claim.get("subject"),
                claim.get("predicate"),
                claim.get("object"),
                claim.get("confidence"),
                model_name,
                prompt_version,
            ),
        )
    return 1
