#!/usr/bin/env python3
"""Claim extraction eval driver.

Loads ``eval/claim_extraction_gold_standard.jsonl``, runs the configured
``LLMClient`` against each paragraph, and computes precision / recall / F1
both globally and stratified by discipline.

Matching policy
---------------
A predicted claim is counted as a true positive against a gold claim when:

1. **Exact-span match**: predicted ``(char_span_start, char_span_end)`` equals
   gold's, AND ``claim_type`` matches.
2. **Jaccard fallback**: token-level Jaccard similarity between predicted and
   gold ``claim_text`` is >= 0.6, AND ``claim_type`` matches. Tokens are
   lowercased ``\\w+`` matches via ``re.findall``.

Each gold claim can match at most one predicted claim, and vice versa
(greedy first-match). Unmatched predicted -> FP. Unmatched gold -> FN.

The script does NOT touch the database. It calls the ``LLMClient`` directly
and matches in memory.

Usage
-----
::

    # Sanity check: stub returns gold claims verbatim -> P=R=F1=1.0
    python scripts/eval_claim_extraction.py --llm stub

    # Real run via the user's authenticated claude CLI
    python scripts/eval_claim_extraction.py --llm claude-cli \\
        --output results/claim_extraction_eval.json

Per repo policy this module MUST NOT import any paid-API SDK; the
``claude-cli`` backend shells out to the user's already-authenticated
``claude`` binary via ``ClaudeCliLLMClient``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# Add src/ to path for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.claims.extract import (  # noqa: E402  (path mutation must precede import)
    ClaudeCliLLMClient,
    LLMClient,
    _format_prompt,
    _load_prompt_template,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_GOLD_PATH: Path = REPO_ROOT / "eval" / "claim_extraction_gold_standard.jsonl"
DEFAULT_OUTPUT_PATH: Path = REPO_ROOT / "results" / "claim_extraction_eval.json"
DEFAULT_MODEL_NAME: str = "claude-cli"
DEFAULT_PROMPT_VERSION: str = "claim_extraction_v1"

#: Jaccard fallback threshold from the PRD acceptance criteria.
JACCARD_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Gold loader
# ---------------------------------------------------------------------------


def load_gold(gold_path: Path) -> list[dict[str, Any]]:
    """Read the gold JSONL file. Each non-empty line is one entry."""
    entries: list[dict[str, Any]] = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - guard
                raise ValueError(
                    f"{gold_path}:{line_no}: invalid JSON: {exc}"
                ) from exc
    return entries


# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    """Lowercased ``\\w+`` token set used by :func:`jaccard`."""
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity over lowercased ``\\w+`` tokens.

    Returns 0.0 when both token sets are empty (avoids divide-by-zero and
    is the conservative answer for the matcher: empty texts are not "the
    same claim").
    """
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


def _claim_match(
    predicted: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> bool:
    """Decide whether ``predicted`` satisfies ``gold`` per the two-tier rule.

    Tier 1: exact span (start, end) AND claim_type equality.
    Tier 2: Jaccard(claim_text) >= 0.6 AND claim_type equality.
    """
    if predicted.get("claim_type") != gold.get("claim_type"):
        return False

    p_start = predicted.get("char_span_start")
    p_end = predicted.get("char_span_end")
    g_start = gold.get("char_span_start")
    g_end = gold.get("char_span_end")
    if (
        p_start is not None
        and p_end is not None
        and p_start == g_start
        and p_end == g_end
    ):
        return True

    return jaccard(
        str(predicted.get("claim_text", "")),
        str(gold.get("claim_text", "")),
    ) >= JACCARD_THRESHOLD


def match_claims(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    """Greedy first-match alignment: each gold claim consumes at most one
    predicted claim, and each predicted claim satisfies at most one gold.

    Returns ``(tp, fp, fn)``.
    """
    used_predicted: set[int] = set()
    matched_gold = 0

    for g in gold:
        for pi, p in enumerate(predicted):
            if pi in used_predicted:
                continue
            if _claim_match(p, g):
                used_predicted.add(pi)
                matched_gold += 1
                break

    tp = matched_gold
    fp = len(predicted) - len(used_predicted)
    fn = len(gold) - matched_gold
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision / recall / F1 with 0/0 -> 0.0."""
    p_denom = tp + fp
    r_denom = tp + fn
    precision = tp / p_denom if p_denom > 0 else 0.0
    recall = tp / r_denom if r_denom > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# LLMClient factory
# ---------------------------------------------------------------------------


def _build_llm_client(kind: str, gold_entries: Sequence[Mapping[str, Any]]) -> LLMClient:
    """Construct an LLMClient for the requested backend.

    For ``stub``, returns a callable-backed StubLLMClient that yields each
    paragraph's gold ``expected_claims`` verbatim — i.e. the sanity check
    where P=R=F1=1.0.
    """
    if kind == "stub":
        # Build a paragraph_text -> claims index. The pipeline calls
        # ``llm.extract(prompt, paragraph)``; the formatted prompt embeds the
        # paragraph text, but here we key on the paragraph itself which is
        # passed through verbatim.
        index: dict[str, list[dict[str, Any]]] = {}
        for entry in gold_entries:
            paragraph = entry["paragraph_text"]
            index[paragraph] = [dict(c) for c in entry["expected_claims"]]

        def _responder(_prompt: str, paragraph: str) -> list[dict[str, Any]]:
            return [dict(c) for c in index.get(paragraph, [])]

        return _CallableStub(_responder)

    if kind == "claude-cli":
        return ClaudeCliLLMClient()

    raise ValueError(f"unknown --llm backend: {kind!r}")


class _CallableStub:
    """LLMClient adapter for a (prompt, paragraph) -> list[claim] callable.

    :class:`StubLLMClient` in ``extract.py`` supports a FIFO queue + default
    but not a per-call function. This thin wrapper plugs a callable into the
    LLMClient Protocol without modifying extract.py.
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.calls: list[tuple[str, str]] = []

    def extract(self, prompt: str, paragraph: str) -> list[dict[str, Any]]:
        self.calls.append((prompt, paragraph))
        return list(self._fn(prompt, paragraph))


# ---------------------------------------------------------------------------
# Eval orchestration
# ---------------------------------------------------------------------------


def run_eval(
    gold_entries: Sequence[Mapping[str, Any]],
    llm: LLMClient,
    *,
    model_name: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Run the eval and return the result dict ready to be JSON-dumped.

    Output keys (per acceptance criterion 4):
        model, prompt_version, n_paragraphs, n_gold_claims, n_predicted,
        tp, fp, fn, precision, recall, f1, per_discipline.
    """
    template = _load_prompt_template()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_gold = 0
    n_predicted = 0

    # discipline -> running tp/fp/fn
    by_discipline: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for entry in gold_entries:
        bibcode = entry.get("bibcode", "")
        section_index = int(entry.get("section_index", 0))
        paragraph_index = int(entry.get("paragraph_index", 0))
        paragraph_text = entry["paragraph_text"]
        gold_claims = list(entry.get("expected_claims", []))
        discipline = str(entry.get("discipline", "unknown"))

        prompt = _format_prompt(
            template,
            paper_bibcode=bibcode,
            section_heading="",
            section_index=section_index,
            paragraph_index=paragraph_index,
            paragraph_text=paragraph_text,
        )

        try:
            predicted_raw = llm.extract(prompt, paragraph_text)
        except Exception as exc:  # noqa: BLE001 — fail-soft per pipeline policy
            logger.warning(
                "LLMClient.extract raised for %s s=%d p=%d: %s — counted as 0 predictions",
                bibcode,
                section_index,
                paragraph_index,
                exc,
            )
            predicted_raw = []

        predicted: list[dict[str, Any]] = [
            dict(p) for p in predicted_raw if isinstance(p, Mapping)
        ]

        tp, fp, fn = match_claims(predicted, gold_claims)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_gold += len(gold_claims)
        n_predicted += len(predicted)

        bucket = by_discipline[discipline]
        bucket["tp"] += tp
        bucket["fp"] += fp
        bucket["fn"] += fn

    overall = compute_metrics(total_tp, total_fp, total_fn)

    per_discipline: dict[str, dict[str, float]] = {}
    for disc, counts in by_discipline.items():
        m = compute_metrics(counts["tp"], counts["fp"], counts["fn"])
        per_discipline[disc] = {
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }

    return {
        "model": model_name,
        "prompt_version": prompt_version,
        "n_paragraphs": len(gold_entries),
        "n_gold_claims": n_gold,
        "n_predicted": n_predicted,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "per_discipline": per_discipline,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claim extraction eval driver (precision/recall/F1 vs gold JSONL).",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Path to gold JSONL (default: {DEFAULT_GOLD_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Free-text model identifier stored in the output JSON.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt-template version stored in the output JSON.",
    )
    parser.add_argument(
        "--llm",
        choices=("stub", "claude-cli"),
        default="stub",
        help="LLM backend: 'stub' returns gold verbatim (sanity); 'claude-cli' shells out.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    gold_path: Path = args.gold_path
    if not gold_path.is_file():
        logger.error("gold-path does not exist: %s", gold_path)
        return 2

    gold_entries = load_gold(gold_path)
    logger.info("Loaded %d gold entries from %s", len(gold_entries), gold_path)

    llm = _build_llm_client(args.llm, gold_entries)

    result = run_eval(
        gold_entries,
        llm,
        model_name=args.model_name,
        prompt_version=args.prompt_version,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    logger.info(
        "Eval complete: P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d) -> %s",
        result["precision"],
        result["recall"],
        result["f1"],
        result["tp"],
        result["fp"],
        result["fn"],
        output_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
