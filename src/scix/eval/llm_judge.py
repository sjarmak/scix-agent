"""LLM-judge stub + Cohen's kappa for M9 eval harness.

``judge()`` returns deterministic stub labels by default. If
``ANTHROPIC_API_KEY`` is set AND ``use_real=True`` is passed, a real
Anthropic client is attempted — otherwise we never hit the network.

``cohens_kappa()`` is hand-rolled (no sklearn dep) using the closed-form
formula::

    kappa = (p_o - p_e) / (1 - p_e)

where ``p_o`` is observed agreement and ``p_e`` is chance agreement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

ALLOWED_LABELS: tuple[str, ...] = ("correct", "incorrect", "ambiguous")
_STUB_CYCLE: tuple[str, ...] = ALLOWED_LABELS


@dataclass(frozen=True)
class LinkRow:
    """Minimal view of an entity link for the judge."""

    tier: int
    bibcode: str
    entity_id: int


@dataclass(frozen=True)
class JudgeLabel:
    """One judge decision for a (bibcode, entity_id) pair."""

    bibcode: str
    entity_id: int
    label: str
    rationale: str = ""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _stub_label_for(index: int) -> str:
    """Deterministic cycling stub — used for tests and `--fixture` runs."""
    return _STUB_CYCLE[index % len(_STUB_CYCLE)]


def judge(
    links: Sequence[LinkRow],
    *,
    use_real: bool = False,
    model: str = "claude-opus-4-5",
) -> list[JudgeLabel]:
    """Return one :class:`JudgeLabel` per input link.

    By default (``use_real=False``) returns deterministic stub labels cycled
    through ``ALLOWED_LABELS``. This keeps unit tests hermetic.

    If ``use_real=True`` AND ``ANTHROPIC_API_KEY`` is in the environment,
    we attempt to import ``anthropic`` and call it for real. Any import or
    auth failure falls back to the stub with a warning — tests never hit
    the network regardless.
    """
    if use_real and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _judge_with_anthropic(links, model=model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("real Anthropic judge failed (%s); falling back to stub", exc)

    return [
        JudgeLabel(
            bibcode=link.bibcode,
            entity_id=link.entity_id,
            label=_stub_label_for(i),
            rationale="stub",
        )
        for i, link in enumerate(links)
    ]


def _judge_with_anthropic(
    links: Sequence[LinkRow],
    *,
    model: str,
) -> list[JudgeLabel]:  # pragma: no cover - network path
    """Best-effort real Anthropic call. Not exercised in tests."""
    import anthropic  # type: ignore[import-not-found]

    client = anthropic.Anthropic()
    out: list[JudgeLabel] = []
    for link in links:
        msg = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Is the entity link bibcode={link.bibcode} "
                        f"entity_id={link.entity_id} tier={link.tier} correct? "
                        "Respond with one of: correct, incorrect, ambiguous."
                    ),
                }
            ],
        )
        text = ""
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")
        label = text.strip().lower()
        if label not in ALLOWED_LABELS:
            label = "ambiguous"
        out.append(
            JudgeLabel(
                bibcode=link.bibcode,
                entity_id=link.entity_id,
                label=label,
                rationale=text.strip()[:200],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def cohens_kappa(human: Sequence[str], judge_labels: Sequence[str]) -> float:
    """Return Cohen's kappa for two equal-length label sequences.

    Formula::

        p_o = sum(human[i] == judge[i]) / n
        p_e = sum_k (p_human[k] * p_judge[k])
        kappa = (p_o - p_e) / (1 - p_e)

    Edge cases:
        - Empty inputs return ``0.0`` (no agreement measurable).
        - ``p_e == 1`` (all raters used a single label on both sides) returns
          ``1.0`` if ``p_o == 1`` else ``0.0``.
        - Mismatched lengths raise ``ValueError``.
    """
    if len(human) != len(judge_labels):
        raise ValueError(
            f"human and judge_labels must be the same length: "
            f"{len(human)} vs {len(judge_labels)}"
        )

    n = len(human)
    if n == 0:
        return 0.0

    p_o = sum(1 for h, j in zip(human, judge_labels) if h == j) / n

    categories = set(human) | set(judge_labels)
    p_e = 0.0
    for cat in categories:
        p_h = sum(1 for x in human if x == cat) / n
        p_j = sum(1 for x in judge_labels if x == cat) / n
        p_e += p_h * p_j

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0

    return (p_o - p_e) / (1.0 - p_e)


def labels_from_judge(labels: Iterable[JudgeLabel]) -> list[str]:
    """Flatten :class:`JudgeLabel` objects to plain label strings (for kappa)."""
    return [lbl.label for lbl in labels]
