"""Single-persona relevance judge for retrieval evaluation.

Scores ``(query, bibcode, paper_snippet)`` triples on a 0-3 relevance scale
by dispatching to a Claude Code OAuth subagent (``in_domain_researcher``).
No paid Anthropic API; no ``anthropic`` SDK import.

## Why single persona

Three arbitrary personas (senior expert / cross-domain / novice) is
pattern-matched from LLM-judge literature, not principled design. SciX's
target user is an in-domain researcher searching for relevant papers to
their specific question. One well-calibrated persona matching that user is
the correct shape. A second persona is justified only by evidence from
calibration — specifically a systematic gap that a different framing would
close.

## Architecture

- :class:`JudgeTriple`, :class:`JudgeScore` — immutable DTOs.
- :func:`build_snippet` — constructs the persona input: title + abstract +
  first 500 chars of body, honoring the licensing snippet budget
  (:data:`scix.sources.licensing.DEFAULT_SNIPPET_BUDGET`).
- :func:`parse_judge_response` — parses the persona's JSON response into
  :class:`JudgeScore`. Tolerates leading/trailing prose around the JSON
  object; rejects out-of-range or malformed scores.
- :class:`Dispatcher` protocol — any async callable that takes a
  :class:`JudgeTriple` and returns a :class:`JudgeScore`. Two
  implementations ship: :class:`StubDispatcher` (deterministic, for tests)
  and :class:`ClaudeSubprocessDispatcher` (runs ``claude -p`` in a
  subprocess).
- :class:`PersonaJudge` — orchestrates parallel dispatch with a bounded
  semaphore, retry/backoff, and preserves output order.
- :func:`quadratic_weighted_kappa`, :func:`spearman_rho` — calibration
  metrics against a human-labeled seed.

## Relationship to ``llm_judge.py``

:mod:`scix.eval.llm_judge` implements the **binary** link-audit judge
(labels: correct / incorrect / ambiguous) for entity linking. That module
remains unchanged. The persona judge here is for **ordinal relevance**
(scores: 0-3) and is orthogonal in both semantics and callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from scix.sources.licensing import DEFAULT_SNIPPET_BUDGET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_MIN: int = 0
SCORE_MAX: int = 3
ERROR_SENTINEL: int = -1
"""Returned for a triple that exhausted retries — surfaces as a hard error."""

PERSONA_NAME: str = "in_domain_researcher"
"""Alternate (legacy) subagent: SciX-specific scientist framing, JSON output."""

UMBRELA_PERSONA_NAME: str = "umbrela_judge"
"""Default subagent: verbatim UMBRELA rubric (Castorini, Apache-2.0). See
.claude/agents/umbrela_judge.md and arXiv:2406.06519."""

DEFAULT_PERSONA: str = UMBRELA_PERSONA_NAME
"""UMBRELA is the default — published Kendall's τ > 0.87 vs TREC human
assessors gives us a benchmarked baseline. The in_domain_researcher
persona is retained for A/B comparison on the same seed."""

IN_DOMAIN_PROMPT_VERSION: str = "in_domain_researcher-v1"
UMBRELA_PROMPT_VERSION: str = "umbrela_judge-v1"

DEFAULT_PROMPT_VERSION: str = UMBRELA_PROMPT_VERSION
"""Prompt version tag used for drift-watch bookkeeping."""

DEFAULT_MAX_CONCURRENCY: int = 4
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_BASE_S: float = 2.0
DEFAULT_SUBPROCESS_TIMEOUT_S: float = 120.0
"""Timeout for a single ``claude -p`` subprocess call. Propagates an error
when exceeded (see CLAUDE.md — "timeouts must never swallow the result")."""


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeTriple:
    """One input to the persona judge.

    Attributes:
        query: User's research question or retrieval query.
        bibcode: ADS bibcode of the paper being judged.
        snippet: Pre-built snippet text (title + abstract + body excerpt)
            — assemble via :func:`build_snippet` when wiring from DB rows.
    """

    query: str
    bibcode: str
    snippet: str


@dataclass(frozen=True)
class JudgeScore:
    """One output from the persona judge.

    Attributes:
        score: 0-3 relevance score, or :data:`ERROR_SENTINEL` on failure.
        reason: Free-text reasoning from the persona (or error message on
            failure). Truncated to ~1000 chars to keep logs bounded.
            Empty string when the underlying prompt (e.g. UMBRELA) does
            not request a reason.
        triple: Back-reference to the input triple (optional — set by
            :class:`PersonaJudge.run`; ``None`` when the DTO is used
            standalone).
        needs_human_review: Self-reported triage flag. ``True`` when the
            judge flagged the row as borderline / uncertain. ``None`` when
            the prompt does not emit a review flag (e.g. the JSON
            in_domain_researcher format).
    """

    score: int
    reason: str
    triple: JudgeTriple | None = None
    needs_human_review: bool | None = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DispatcherError(Exception):
    """Transient error from a dispatcher call. Retried by :class:`PersonaJudge`."""


# ---------------------------------------------------------------------------
# Snippet construction
# ---------------------------------------------------------------------------


def build_snippet(
    *,
    title: str,
    abstract: str | None,
    body: str | None,
    body_char_budget: int = DEFAULT_SNIPPET_BUDGET,
) -> str:
    """Build a persona-input snippet: title + abstract + first N chars of body.

    Honors the ADR-006 snippet budget by capping the body excerpt at
    ``body_char_budget`` characters (default 500). Title and abstract are
    metadata fields (not full-text) and are included verbatim.

    Args:
        title: Non-empty paper title.
        abstract: Paper abstract. May be ``None`` if unavailable.
        body: Full paper body. May be ``None``. When present, only the first
            ``body_char_budget`` characters are included.
        body_char_budget: Max characters of body text to include.

    Raises:
        ValueError: If ``title`` is empty or whitespace-only.
    """
    if not title or not title.strip():
        raise ValueError("title must be non-empty")
    if body_char_budget < 0:
        raise ValueError("body_char_budget must be non-negative")

    parts: list[str] = [f"Title: {title.strip()}"]
    if abstract and abstract.strip():
        parts.append(f"Abstract: {abstract.strip()}")
    if body and body.strip():
        excerpt = body.strip()[:body_char_budget]
        parts.append(f"Body excerpt (first {body_char_budget} chars): {excerpt}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_response(raw: str) -> JudgeScore:
    """Parse the persona's text response into a :class:`JudgeScore`.

    Expects a JSON object with fields ``score`` (int 0-3) and ``reason``
    (str). Tolerates leading/trailing prose: scans for the last JSON object
    in the text (personas occasionally narrate before the structured answer).

    Args:
        raw: The raw response text from the subagent.

    Returns:
        A :class:`JudgeScore` with ``triple=None`` (caller re-attaches).

    Raises:
        ValueError: If no parseable JSON object is found, or if fields are
            missing / out of range.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")

    candidates = _JSON_OBJECT_RE.findall(raw)
    if not candidates:
        # Try the whole string as a last resort (covers clean JSON on a
        # single line).
        candidates = [raw.strip()]

    last_error: Exception | None = None
    for candidate in reversed(candidates):  # prefer the last JSON object
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(obj, dict):
            continue
        if "score" not in obj or "reason" not in obj:
            continue

        score_raw = obj["score"]
        if not isinstance(score_raw, bool) and isinstance(score_raw, int):
            score = score_raw
        else:
            raise ValueError(f"score must be an int 0-{SCORE_MAX}, got {score_raw!r}")
        if not (SCORE_MIN <= score <= SCORE_MAX):
            raise ValueError(f"score {score} out of range [{SCORE_MIN}, {SCORE_MAX}]")

        reason = obj.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError(f"reason must be a string, got {type(reason).__name__}")

        return JudgeScore(score=score, reason=reason[:1000])

    raise ValueError(f"no parseable judge JSON in response (last error: {last_error})")


_UMBRELA_SCORE_RE = re.compile(
    r"^\s*##\s*final\s*score\s*:\s*([0-3])\s*$", re.IGNORECASE | re.MULTILINE
)
_UMBRELA_REVIEW_RE = re.compile(
    r"^\s*##\s*needs[_ ]human[_ ]review\s*:\s*(true|false)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_umbrela_response(raw: str) -> JudgeScore:
    """Parse the UMBRELA-formatted judge response.

    UMBRELA (verbatim) emits ``##final score: N``. Our pipeline adds one
    adapter line — ``##needs_human_review: true|false`` — after it. Both
    lines must appear on their own line; any surrounding prose is
    tolerated but ignored.

    Args:
        raw: Raw stdout from the subagent.

    Returns:
        :class:`JudgeScore` with ``reason=""`` (UMBRELA does not request
        a reason, per the verbatim rubric) and ``needs_human_review`` set
        from the adapter line.

    Raises:
        ValueError: If either line is missing or malformed.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")

    score_match = _UMBRELA_SCORE_RE.search(raw)
    if score_match is None:
        raise ValueError("no '##final score: N' line found in UMBRELA response")
    score = int(score_match.group(1))

    review_match = _UMBRELA_REVIEW_RE.search(raw)
    if review_match is None:
        raise ValueError(
            "no '##needs_human_review: true|false' line found — "
            "the umbrela_judge output-format addendum was not emitted"
        )
    needs_human_review = review_match.group(1).lower() == "true"

    return JudgeScore(score=score, reason="", needs_human_review=needs_human_review)


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------


class Dispatcher(Protocol):
    """Any async callable that maps a triple to a score.

    Implementations should raise :class:`DispatcherError` for transient
    failures that the orchestrator should retry. Any other exception is
    treated as fatal and recorded as an error score.
    """

    async def judge(self, triple: JudgeTriple) -> JudgeScore: ...


@dataclass
class StubDispatcher:
    """Deterministic in-memory dispatcher for unit tests.

    Records every call on :attr:`calls` for test assertions.

    Attributes:
        fixed_score: Score to return for every input.
        reason: Reason string to attach.
        calls: List of triples received, in call order.
    """

    fixed_score: int = 2
    reason: str = "stub"
    calls: list[JudgeTriple] = field(default_factory=list)

    async def judge(self, triple: JudgeTriple) -> JudgeScore:
        self.calls.append(triple)
        return JudgeScore(score=self.fixed_score, reason=self.reason)


PromptFormatter = Callable[["JudgeTriple", str], str]
ResponseParser = Callable[[str], "JudgeScore"]


@dataclass
class ClaudeSubprocessDispatcher:
    """Dispatches via ``claude -p <prompt>`` as a subprocess (OAuth, no API key).

    Relies on the user's existing Claude Code OAuth session — this is the
    high-throughput path for 500+ triples. The smaller-batch in-session
    ``Agent`` tool path is used when a Claude Code session orchestrates the
    judge directly.

    Defaults to the UMBRELA persona (verbatim Castorini rubric), which has
    published Kendall's τ > 0.87 vs TREC human assessors. The
    ``in_domain_researcher()`` classmethod returns a dispatcher configured
    for the legacy SciX-specific persona (JSON output with a reason field).

    Attributes:
        claude_binary: Path or name of the ``claude`` CLI.
        persona: Subagent name to dispatch to (default
            :data:`DEFAULT_PERSONA`, i.e. UMBRELA).
        timeout_s: Per-call timeout. A timeout propagates as
            :class:`DispatcherError` so the orchestrator can retry.
        prompt_formatter: Builds the text passed to ``claude -p`` from a
            :class:`JudgeTriple` and the persona name. Defaults to the
            UMBRELA formatter.
        response_parser: Parses the subprocess stdout into a
            :class:`JudgeScore`. Defaults to :func:`parse_umbrela_response`.
    """

    claude_binary: str = "claude"
    persona: str = DEFAULT_PERSONA
    timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S
    prompt_formatter: PromptFormatter = field(default=None)  # type: ignore[assignment]
    response_parser: ResponseParser = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prompt_formatter is None:
            self.prompt_formatter = _format_persona_prompt_umbrela
        if self.response_parser is None:
            self.response_parser = parse_umbrela_response

    @classmethod
    def in_domain_researcher(
        cls,
        *,
        claude_binary: str = "claude",
        timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
    ) -> ClaudeSubprocessDispatcher:
        """Factory for the legacy SciX-specific persona (JSON output)."""
        return cls(
            claude_binary=claude_binary,
            persona=PERSONA_NAME,
            timeout_s=timeout_s,
            prompt_formatter=_format_persona_prompt,
            response_parser=parse_judge_response,
        )

    async def judge(self, triple: JudgeTriple) -> JudgeScore:
        prompt = self.prompt_formatter(triple, self.persona)
        try:
            completed = await asyncio.wait_for(
                _run_claude_subprocess(self.claude_binary, prompt),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise DispatcherError(f"claude -p timed out after {self.timeout_s}s") from exc
        except FileNotFoundError as exc:
            # Fatal: binary missing. Don't retry.
            raise RuntimeError(
                f"claude binary not found at {self.claude_binary!r}; "
                "ensure Claude Code CLI is installed"
            ) from exc

        if completed.returncode != 0:
            raise DispatcherError(
                f"claude -p exited {completed.returncode}: " f"stderr={completed.stderr[:500]!r}"
            )

        try:
            return self.response_parser(completed.stdout)
        except ValueError as exc:
            raise DispatcherError(f"unparseable judge response: {exc}") from exc


async def _run_claude_subprocess(binary: str, prompt: str) -> subprocess.CompletedProcess:
    """Run ``{binary} -p <prompt>`` and collect its output.

    Uses ``asyncio.create_subprocess_exec`` to stay non-blocking in the
    orchestrator's event loop. The prompt is passed as a positional
    argument to avoid shell quoting hazards.
    """
    # nosec B603: deliberate subprocess call to the Claude CLI with no shell
    # expansion. The prompt argument is a single argv entry, not interpreted
    # by a shell.
    proc = await asyncio.create_subprocess_exec(
        binary,
        "-p",
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    return subprocess.CompletedProcess(
        args=[binary, "-p"],
        returncode=proc.returncode or 0,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
    )


def _format_persona_prompt(triple: JudgeTriple, persona: str) -> str:
    """Format the user-message prompt for the JSON-output persona
    (:data:`PERSONA_NAME` / ``in_domain_researcher``).

    The subagent emits a JSON object with ``score`` and ``reason``.
    """
    return (
        f"Use the {persona} subagent to score this (query, paper) pair on the "
        f"0-3 relevance scale defined in the subagent definition.\n\n"
        f"Query: {triple.query}\n\n"
        f"Paper (bibcode {triple.bibcode}):\n{triple.snippet}\n\n"
        f"Respond with a single JSON object: "
        f'{{"score": <0-3>, "reason": "<one sentence>"}}.'
    )


def _format_persona_prompt_umbrela(triple: JudgeTriple, persona: str) -> str:
    """Format the prompt for the UMBRELA persona.

    The subagent's prompt file carries the verbatim UMBRELA rubric and
    4-example demonstration. The caller injects only the Query and
    Passage and reminds the subagent of the exact output format.
    """
    return (
        f"Use the {persona} subagent to score this (query, passage) pair "
        f"on the UMBRELA 0-3 relevance scale defined in the subagent "
        f"definition.\n\n"
        f"Query: {triple.query}\n\n"
        f"Passage (bibcode {triple.bibcode}): {triple.snippet}\n\n"
        f"Respond with exactly two lines:\n"
        f"##final score: <0|1|2|3>\n"
        f"##needs_human_review: <true|false>\n"
        f"No other output. No reasoning. No JSON."
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PersonaJudge:
    """Parallel orchestrator over a :class:`Dispatcher`.

    Runs up to ``max_concurrency`` dispatcher calls concurrently with
    exponential-backoff retry. Returns one :class:`JudgeScore` per input
    triple in the same order, attaching the triple to each score.

    Attributes:
        dispatcher: The dispatcher implementation.
        max_concurrency: Max simultaneous in-flight calls (default 4).
        max_retries: Retry budget for transient errors (default 3).
        backoff_base_s: Base delay for exponential backoff (set to 0.0 in
            tests to keep them fast).
    """

    dispatcher: Dispatcher
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_base_s < 0:
            raise ValueError("backoff_base_s must be >= 0")

    async def run(self, triples: Sequence[JudgeTriple]) -> list[JudgeScore]:
        """Score every triple in parallel; preserve input order."""
        if not triples:
            return []

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _judge_one(triple: JudgeTriple) -> JudgeScore:
            async with sem:
                return await self._judge_with_retry(triple)

        tasks = [asyncio.create_task(_judge_one(t)) for t in triples]
        return list(await asyncio.gather(*tasks))

    async def _judge_with_retry(self, triple: JudgeTriple) -> JudgeScore:
        attempt = 0
        while True:
            try:
                score = await self.dispatcher.judge(triple)
            except DispatcherError as exc:
                if attempt >= self.max_retries:
                    logger.warning(
                        "judge failure for %s after %d attempts: %s",
                        triple.bibcode,
                        attempt + 1,
                        exc,
                    )
                    return JudgeScore(
                        score=ERROR_SENTINEL,
                        reason=f"dispatcher error after {attempt + 1} attempts: {exc}",
                        triple=triple,
                    )
                # Exponential backoff
                delay = self.backoff_base_s * (2**attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
                attempt += 1
                continue
            except Exception as exc:  # fatal — don't retry
                logger.error("fatal judge error for %s: %s", triple.bibcode, exc, exc_info=True)
                return JudgeScore(
                    score=ERROR_SENTINEL,
                    reason=f"fatal error: {exc}",
                    triple=triple,
                )
            # Attach triple to score (dispatcher returns it with triple=None).
            # Preserve needs_human_review so UMBRELA triage flags survive
            # the orchestrator hop.
            return JudgeScore(
                score=score.score,
                reason=score.reason,
                triple=triple,
                needs_human_review=score.needs_human_review,
            )


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def quadratic_weighted_kappa(
    human: Sequence[int],
    judge_scores: Sequence[int],
    *,
    min_score: int = SCORE_MIN,
    max_score: int = SCORE_MAX,
) -> float:
    """Cohen's kappa with quadratic weights.

    Appropriate for ordinal scales (our 0-3 scoring) because it penalizes
    larger disagreements more than small ones.

    Formula::

        w_{i,j} = ((i - j) / (N - 1))^2
        kappa = 1 - (sum w_{i,j} * O_{i,j}) / (sum w_{i,j} * E_{i,j})

    Args:
        human: Sequence of ground-truth scores.
        judge_scores: Sequence of judge scores; must match ``human`` in length.
        min_score: Minimum allowed score (default 0).
        max_score: Maximum allowed score (default 3).

    Returns:
        Kappa in roughly ``[-1, 1]``; 1.0 is perfect agreement.

    Raises:
        ValueError: On length mismatch or out-of-range scores.
    """
    if len(human) != len(judge_scores):
        raise ValueError(f"length mismatch: {len(human)} vs {len(judge_scores)}")
    n = len(human)
    if n == 0:
        return 0.0

    for value in (*human, *judge_scores):
        if not (min_score <= value <= max_score):
            raise ValueError(f"score {value} out of range [{min_score}, {max_score}]")

    num_categories = max_score - min_score + 1
    if num_categories < 2:
        raise ValueError("need at least 2 categories for kappa")

    # Confusion matrix O (observed) and marginals
    obs = [[0] * num_categories for _ in range(num_categories)]
    for h, j in zip(human, judge_scores):
        obs[h - min_score][j - min_score] += 1

    human_marginal = [sum(row) for row in obs]
    judge_marginal = [sum(obs[r][c] for r in range(num_categories)) for c in range(num_categories)]

    # Expected matrix (under independence)
    expected = [
        [(human_marginal[r] * judge_marginal[c]) / n for c in range(num_categories)]
        for r in range(num_categories)
    ]

    # Quadratic weights
    denom_w = (num_categories - 1) ** 2
    weights = [
        [((r - c) ** 2) / denom_w for c in range(num_categories)] for r in range(num_categories)
    ]

    num = sum(
        weights[r][c] * obs[r][c] for r in range(num_categories) for c in range(num_categories)
    )
    den = sum(
        weights[r][c] * expected[r][c] for r in range(num_categories) for c in range(num_categories)
    )

    if den == 0:
        # Raters used a single category each — kappa degenerates.
        return 1.0 if num == 0 else 0.0

    return 1.0 - (num / den)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation coefficient.

    Computes Pearson correlation on the rank-transformed inputs, handling
    ties via the average-rank method. Pure Python — no scipy dependency.

    Args:
        x, y: Equal-length numeric sequences.

    Returns:
        Rho in ``[-1, 1]``; returns ``0.0`` for empty or degenerate inputs.

    Raises:
        ValueError: On length mismatch.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    n = len(x)
    if n == 0:
        return 0.0

    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n

    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Assign average ranks to ``values``, handling ties.

    For example, ``[10, 20, 20, 30]`` has ranks ``[1, 2.5, 2.5, 4]``.
    """
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        # Find extent of the tie group
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Public module surface
# ---------------------------------------------------------------------------

__all__ = [
    "ClaudeSubprocessDispatcher",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_PERSONA",
    "DEFAULT_PROMPT_VERSION",
    "Dispatcher",
    "DispatcherError",
    "ERROR_SENTINEL",
    "IN_DOMAIN_PROMPT_VERSION",
    "JudgeScore",
    "JudgeTriple",
    "PERSONA_NAME",
    "PersonaJudge",
    "SCORE_MAX",
    "SCORE_MIN",
    "StubDispatcher",
    "UMBRELA_PERSONA_NAME",
    "UMBRELA_PROMPT_VERSION",
    "build_snippet",
    "parse_judge_response",
    "parse_umbrela_response",
    "quadratic_weighted_kappa",
    "spearman_rho",
]
