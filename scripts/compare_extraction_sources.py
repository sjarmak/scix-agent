#!/usr/bin/env python3
"""Head-to-head comparison: ADS metadata vs NER vs Haiku entity extraction.

This is unit M3 of the PRD ``docs/prd/prd_entity_extraction_phase2_ner.md``.
Given a gap cohort produced by M2 (``results/metadata_gap_report.json``) and
the WIESP-pinned NER model evaluated in M1 (``scripts/eval_ner_wiesp.py``),
the script draws a stratified sample of papers and runs three extractors:

1. **metadata** — reads the ADS array fields ``facility``/``data``/
   ``keyword_norm`` (zero LLM cost).
2. **ner**     — runs the pinned ``adsabs/nasa-smd-ibm-v0.1_NER_DEAL``
   token-classifier on the abstract.
3. **haiku**   — calls the Anthropic ``claude-haiku-4-5`` Messages API with
   the v3 entity-extraction prompt.

For each method the script computes per-entity-type and aggregate
precision / recall / F1 against a caller-supplied gold-label fixture, plus
USD cost-per-paper, and writes a JSON report to
``results/extraction_head_to_head.json``.

The script is **fully mockable** for offline testing: ``--mock-all`` takes
fixtures for predictions and gold labels and never touches the DB, the NER
model, or the Anthropic API. ``--mock-haiku`` mocks only the LLM call.

A cost gate refuses to call Haiku in non-mock mode unless
``SCIX_HEAD_TO_HEAD_BUDGET_USD`` is set to a positive USD budget.

Recommendation logic (see :func:`decide_recommendation`):
    - Restrict to methods within a 2x cost band of the cheapest method.
    - Within the band, pick the method with highest aggregate F1; tie-break
      by lower cost.
    - Override with ``ensemble`` when both NER and metadata exceed
      F1 >= 0.7 AND dominate disjoint sets of entity types.

Usage::

    # offline self-test
    python scripts/compare_extraction_sources.py \\
        --mock-all \\
        --predictions-fixture tests/data/h2h_preds.json \\
        --gold-fixture        tests/data/h2h_gold.json

    # live run (requires SCIX_HEAD_TO_HEAD_BUDGET_USD set)
    SCIX_HEAD_TO_HEAD_BUDGET_USD=5.00 \\
        python scripts/compare_extraction_sources.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Allow running as a plain script: make src/ importable for downstream calls.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned model identifiers and pricing constants
# ---------------------------------------------------------------------------

#: HuggingFace repo id for the WIESP-aligned NER model (must match M1).
MODEL_NAME: str = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"

#: Full commit SHA — kept in lockstep with ``scripts/eval_ner_wiesp.py``.
#: Refresh both files at the same time. Drift here breaks reproducibility.
MODEL_REVISION: str = "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"

#: Anthropic model identifier for the Haiku extractor.
HAIKU_MODEL: str = "claude-haiku-4-5-20251001"

#: Anthropic listed pricing for Claude Haiku 4.5 (single-call Messages API,
#: not the 50%-off Batches API). Values are USD per million tokens.
#: Source: Anthropic public pricing page, April 2026.
HAIKU_INPUT_USD_PER_MTOK: float = 1.0
HAIKU_OUTPUT_USD_PER_MTOK: float = 5.0

#: Default sample size for the head-to-head comparison.
DEFAULT_SAMPLE_SIZE: int = 500

#: Default paths.
DEFAULT_COHORT_PATH: Path = (
    Path(__file__).resolve().parent.parent / "results" / "metadata_gap_report.json"
)
DEFAULT_OUTPUT_PATH: Path = (
    Path(__file__).resolve().parent.parent / "results" / "extraction_head_to_head.json"
)

#: Deterministic RNG seed used for stratified sampling.
DEFAULT_SAMPLE_SEED: int = 42

#: F1 threshold above which both NER and metadata can earn an ensemble
#: recommendation (when their dominant entity types are disjoint).
F1_ENSEMBLE_THRESHOLD: float = 0.7

#: Methods whose cost-per-paper is more than this many times the cheapest
#: method's cost-per-paper are excluded from the recommendation. Cheaper
#: methods always stay in-band.
COST_BAND_FACTOR: float = 2.0

#: Env var that gates real Haiku spend.
COST_GATE_ENV: str = "SCIX_HEAD_TO_HEAD_BUDGET_USD"

#: Entity types compared in M3. These are the three M2 gap fields plus the
#: corresponding entity-extraction taxonomy buckets.
ENTITY_TYPES: tuple[str, ...] = ("instruments", "datasets", "software")

#: Map ADS metadata fields to entity types (mirrors analyze_metadata_gaps.py).
METADATA_FIELD_TO_ENTITY: dict[str, str] = {
    "facility": "instruments",
    "data": "datasets",
    "keyword_norm": "software",
}

#: Method keys exposed in the per_method report.
METHOD_KEYS: tuple[str, str, str] = ("metadata", "ner", "haiku")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortPaper:
    """A single bibcode pulled from the gap cohort, with optional stratum."""

    bibcode: str
    arxiv_class_primary: str | None = None


@dataclass(frozen=True)
class HaikuUsage:
    """Token usage from a single Haiku call."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class MethodMetrics:
    """Aggregate + per-entity-type metrics for a single extraction method."""

    precision: float
    recall: float
    f1: float
    cost_per_paper_usd: float
    per_entity_type: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "cost_per_paper_usd": self.cost_per_paper_usd,
            "per_entity_type": {
                etype: dict(scores) for etype, scores in self.per_entity_type.items()
            },
        }


# ---------------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------------


def _primary_arxiv_class(arxiv_classes: Sequence[Any] | None) -> str | None:
    """Return the first non-empty arxiv_class entry, lower-cased."""
    if not arxiv_classes:
        return None
    for c in arxiv_classes:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return None


def load_cohort(path: Path) -> list[CohortPaper]:
    """Load the gap cohort from M2's metadata_gap_report.json.

    Bibcodes come from ``gap_cohort_bibcodes_sample`` (and the optional
    sidecar). When the report includes an ``arxiv_class_by_bibcode`` map
    (forward-compatible enrichment), we attach the primary arxiv class to
    each paper for stratification; otherwise the stratum is None and the
    sampler falls back to uniform.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Cohort report {path} must be a JSON object")

    bibcodes: list[str] = list(raw.get("gap_cohort_bibcodes_sample") or [])
    sidecar_path = raw.get("gap_cohort_bibcodes_sidecar_path")
    if sidecar_path:
        sidecar = Path(sidecar_path)
        if sidecar.is_file():
            extra = [
                line.strip()
                for line in sidecar.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            seen = set(bibcodes)
            for bc in extra:
                if bc not in seen:
                    bibcodes.append(bc)
                    seen.add(bc)

    arxiv_map: Mapping[str, Any] = raw.get("arxiv_class_by_bibcode") or {}

    cohort: list[CohortPaper] = []
    for bc in bibcodes:
        if not isinstance(bc, str) or not bc.strip():
            continue
        primary = _primary_arxiv_class(arxiv_map.get(bc))
        cohort.append(CohortPaper(bibcode=bc.strip(), arxiv_class_primary=primary))

    if not cohort:
        raise ValueError(f"Cohort report {path} contains no bibcodes")

    return cohort


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _largest_remainder_allocation(
    bucket_sizes: Mapping[str, int],
    sample_size: int,
) -> dict[str, int]:
    """Allocate ``sample_size`` slots across buckets via Largest Remainder.

    Each bucket gets at minimum ``floor(bucket_size / total * sample_size)``
    slots, then the remaining slots go to the buckets with the largest
    fractional remainder. Allocations are clamped to ``min(alloc, bucket_size)``
    so we never request more samples than a bucket contains.
    """
    total = sum(bucket_sizes.values())
    if total == 0 or sample_size <= 0:
        return {k: 0 for k in bucket_sizes}

    raw: dict[str, float] = {
        k: (size / total) * sample_size for k, size in bucket_sizes.items()
    }
    floor_alloc: dict[str, int] = {k: int(math.floor(v)) for k, v in raw.items()}
    remainders: list[tuple[float, str]] = sorted(
        ((raw[k] - floor_alloc[k], k) for k in raw),
        reverse=True,
    )

    # Remaining slots after floor pass.
    remaining = sample_size - sum(floor_alloc.values())
    alloc = dict(floor_alloc)
    for _, k in remainders:
        if remaining <= 0:
            break
        if alloc[k] < bucket_sizes[k]:
            alloc[k] += 1
            remaining -= 1

    # Final clamp (defensive — shouldn't trigger after the bucket-size guard).
    for k in alloc:
        alloc[k] = min(alloc[k], bucket_sizes[k])

    return alloc


def stratified_sample(
    cohort: Sequence[CohortPaper],
    sample_size: int,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> list[CohortPaper]:
    """Draw a stratified-by-arxiv-class sample, falling back to uniform.

    - When at least one cohort entry has a non-None ``arxiv_class_primary``
      AND there is more than one distinct stratum, sampling is proportional
      across strata using the Largest Remainder Method. Within each stratum
      papers are drawn without replacement using a seeded ``random.Random``.
    - Otherwise (no stratum data, or only one stratum), sampling is uniform
      without replacement, also seeded.
    - If ``sample_size >= len(cohort)``, the entire cohort is returned in a
      deterministic shuffled order.
    """
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    cohort_list = list(cohort)
    if sample_size >= len(cohort_list):
        shuffled = list(cohort_list)
        rng.shuffle(shuffled)
        return shuffled

    # Group by stratum, treating None as fallback.
    buckets: dict[str | None, list[CohortPaper]] = defaultdict(list)
    for paper in cohort_list:
        buckets[paper.arxiv_class_primary].append(paper)

    real_strata = {k: v for k, v in buckets.items() if k is not None}
    use_stratified = len(real_strata) >= 2 and sum(len(v) for v in real_strata.values()) > 0

    if not use_stratified:
        # Uniform sample over the full cohort.
        result = rng.sample(cohort_list, sample_size)
        # Deterministic post-sort by bibcode for reproducible output ordering.
        result.sort(key=lambda p: p.bibcode)
        return result

    # Proportional allocation across real strata.
    # Papers without a stratum get carried as a residual pool used only if
    # the real strata cannot satisfy the requested sample size.
    bucket_sizes = {k: len(v) for k, v in real_strata.items()}
    real_total = sum(bucket_sizes.values())
    allocation = _largest_remainder_allocation(
        bucket_sizes, min(sample_size, real_total)
    )

    sampled: list[CohortPaper] = []
    for stratum, n in allocation.items():
        if n <= 0:
            continue
        sampled.extend(rng.sample(real_strata[stratum], n))

    # Top up from None bucket if needed (rare).
    deficit = sample_size - len(sampled)
    none_pool = buckets.get(None, [])
    if deficit > 0 and none_pool:
        take = min(deficit, len(none_pool))
        sampled.extend(rng.sample(none_pool, take))

    sampled.sort(key=lambda p: p.bibcode)
    return sampled


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _normalize_entity(value: str) -> str:
    """Lower-case + strip whitespace for matching purposes."""
    return value.strip().lower()


def _normalize_pred_map(
    raw: Mapping[str, Mapping[str, Iterable[str]]],
) -> dict[str, dict[str, frozenset[str]]]:
    """Normalize ``{bibcode: {etype: [entity, ...]}}`` to canonical form."""
    result: dict[str, dict[str, frozenset[str]]] = {}
    for bib, by_type in raw.items():
        ent_map: dict[str, frozenset[str]] = {}
        for etype, items in (by_type or {}).items():
            cleaned = {
                _normalize_entity(x)
                for x in (items or [])
                if isinstance(x, str) and x.strip()
            }
            ent_map[etype] = frozenset(cleaned)
        result[bib] = ent_map
    return result


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 with safe divide-by-zero handling."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_method_metrics(
    preds_by_bib: Mapping[str, Mapping[str, Iterable[str]]],
    gold_by_bib: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    entity_types: Sequence[str] = ENTITY_TYPES,
    cost_per_paper_usd: float = 0.0,
) -> MethodMetrics:
    """Aggregate set-overlap precision/recall/F1 per entity type and overall.

    For each ``(bibcode, entity_type)`` we treat predictions and gold as sets
    of normalized entity strings. TP/FP/FN are summed across all bibcodes
    that appear in *either* the predictions or the gold map. Aggregate metrics
    are micro-averaged over entity types.
    """
    preds_norm = _normalize_pred_map(preds_by_bib)
    gold_norm = _normalize_pred_map(gold_by_bib)
    bibs = set(preds_norm) | set(gold_norm)

    per_type_counts: dict[str, dict[str, int]] = {
        etype: {"tp": 0, "fp": 0, "fn": 0} for etype in entity_types
    }
    total_tp = total_fp = total_fn = 0

    for bib in bibs:
        pred_map = preds_norm.get(bib, {})
        gold_map = gold_norm.get(bib, {})
        for etype in entity_types:
            pred_set = pred_map.get(etype, frozenset())
            gold_set = gold_map.get(etype, frozenset())
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            per_type_counts[etype]["tp"] += tp
            per_type_counts[etype]["fp"] += fp
            per_type_counts[etype]["fn"] += fn
            total_tp += tp
            total_fp += fp
            total_fn += fn

    per_entity_type: dict[str, dict[str, float]] = {}
    for etype, counts in per_type_counts.items():
        p, r, f = _prf(counts["tp"], counts["fp"], counts["fn"])
        per_entity_type[etype] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": float(counts["tp"] + counts["fn"]),
        }

    micro_p, micro_r, micro_f = _prf(total_tp, total_fp, total_fn)
    return MethodMetrics(
        precision=micro_p,
        recall=micro_r,
        f1=micro_f,
        cost_per_paper_usd=cost_per_paper_usd,
        per_entity_type=per_entity_type,
    )


def compute_haiku_cost(usages: Sequence[HaikuUsage], n_papers: int) -> float:
    """Compute average USD cost per paper from a list of ``HaikuUsage``."""
    if n_papers <= 0:
        return 0.0
    total_in = sum(u.input_tokens for u in usages)
    total_out = sum(u.output_tokens for u in usages)
    cost = (total_in / 1_000_000.0) * HAIKU_INPUT_USD_PER_MTOK + (
        total_out / 1_000_000.0
    ) * HAIKU_OUTPUT_USD_PER_MTOK
    return cost / n_papers


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _disjoint_entity_dominance(
    ner_per_type: Mapping[str, Mapping[str, float]],
    metadata_per_type: Mapping[str, Mapping[str, float]],
) -> bool:
    """True iff NER and metadata each dominate at least one entity type AND
    those dominated sets do not overlap.

    "Dominate" means strictly higher F1 on that entity type. Ties (or types
    where both are 0) are considered un-dominated.
    """
    ner_wins: set[str] = set()
    md_wins: set[str] = set()
    for etype in set(ner_per_type) | set(metadata_per_type):
        nf = ner_per_type.get(etype, {}).get("f1", 0.0)
        mf = metadata_per_type.get(etype, {}).get("f1", 0.0)
        if nf > mf:
            ner_wins.add(etype)
        elif mf > nf:
            md_wins.add(etype)
    return bool(ner_wins) and bool(md_wins) and ner_wins.isdisjoint(md_wins) is True


def decide_recommendation(
    metrics: Mapping[str, MethodMetrics],
) -> tuple[str, str]:
    """Pick a recommended method from the per-method metrics.

    Logic (documented inline so future readers don't reverse-engineer it):

    1. **Cost band**: identify the cheapest method's ``cost_per_paper_usd``.
       Methods are *in-band* if their cost is at most ``COST_BAND_FACTOR``
       times the minimum, with a special rule: when the cheapest method is
       free (\u200b$0), only other free methods stay in-band — paid methods
       are excluded since "2x of zero" is still zero.
    2. **Highest F1**: among in-band methods, recommend the one with the
       highest aggregate F1. Tie-break by lower cost.
    3. **Ensemble override**: if both ``ner`` and ``metadata`` exceed
       :data:`F1_ENSEMBLE_THRESHOLD` AND they dominate disjoint entity
       types, return ``ensemble``.

    Returns ``(recommendation, rationale)``.
    """
    if not metrics:
        return "metadata", "no metrics provided; defaulting to metadata"

    # Step 3 first — ensemble overrides single-method recommendations because
    # disjoint dominance implies neither single method captures all signal.
    ner = metrics.get("ner")
    md = metrics.get("metadata")
    if (
        ner is not None
        and md is not None
        and ner.f1 >= F1_ENSEMBLE_THRESHOLD
        and md.f1 >= F1_ENSEMBLE_THRESHOLD
        and _disjoint_entity_dominance(ner.per_entity_type, md.per_entity_type)
    ):
        return (
            "ensemble",
            (
                f"NER F1={ner.f1:.3f} and metadata F1={md.f1:.3f} both exceed "
                f"{F1_ENSEMBLE_THRESHOLD:.2f}, and they dominate disjoint "
                "entity types — ensembling captures complementary signal."
            ),
        )

    # Step 1 — cost band filter.
    costs = {k: m.cost_per_paper_usd for k, m in metrics.items()}
    min_cost = min(costs.values())
    if min_cost <= 0.0:
        in_band = {k for k, c in costs.items() if c <= 0.0}
    else:
        ceiling = min_cost * COST_BAND_FACTOR
        in_band = {k for k, c in costs.items() if c <= ceiling}

    candidates = {k: m for k, m in metrics.items() if k in in_band}
    if not candidates:
        # Defensive — at least the cheapest must qualify.
        candidates = dict(metrics)

    # Step 2 — pick highest F1, tie-break by lower cost.
    best_key = max(
        candidates,
        key=lambda k: (candidates[k].f1, -candidates[k].cost_per_paper_usd),
    )
    rationale = (
        f"{best_key} wins on F1={candidates[best_key].f1:.3f} within the "
        f"{COST_BAND_FACTOR:.0f}x cost band of the cheapest method "
        f"(${min_cost:.4f}/paper). In-band methods: "
        f"{sorted(in_band)}."
    )
    return best_key, rationale


# ---------------------------------------------------------------------------
# Cost gate
# ---------------------------------------------------------------------------


def enforce_cost_gate(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> None:
    """Refuse to invoke real Haiku without an explicit USD budget.

    - ``--mock-all``  → no spend possible; gate is a no-op.
    - ``--mock-haiku`` → only Haiku is mocked; the live methods (metadata
      / NER) cost nothing, so the gate is also a no-op.
    - Otherwise the env var :data:`COST_GATE_ENV` must be set to a
      positive float.
    """
    if args.mock_all or args.mock_haiku:
        return
    env = env if env is not None else os.environ
    raw = env.get(COST_GATE_ENV)
    try:
        budget = float(raw) if raw is not None else None
    except ValueError:
        budget = None
    if budget is None or budget <= 0:
        print(
            f"ERROR: live Haiku run requires {COST_GATE_ENV} to be set to a "
            "positive USD budget. Either export the env var (e.g. "
            f"`export {COST_GATE_ENV}=5.00`) or pass --mock-haiku/--mock-all "
            "to run offline.",
            file=sys.stderr,
        )
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Extractor adapters
# ---------------------------------------------------------------------------


def _load_predictions_fixture(
    path: Path,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Load a per-method predictions fixture.

    Schema::

        {
          "metadata": { "<bibcode>": { "instruments": [...], ... }, ... },
          "ner":      { "<bibcode>": { ... }, ... },
          "haiku":    { "<bibcode>": { ... }, ... },
          "haiku_usage": [ { "input_tokens": int, "output_tokens": int }, ... ]
        }
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Predictions fixture {path} must be a JSON object")
    return raw


def _load_gold_fixture(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load gold labels: ``{bibcode: {entity_type: [entity, ...]}}``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Gold fixture {path} must be a JSON object")
    return raw


def _haiku_usages_from_fixture(raw: Mapping[str, Any]) -> list[HaikuUsage]:
    """Pull HaikuUsage list from a predictions-fixture payload."""
    usage_raw = raw.get("haiku_usage") or []
    out: list[HaikuUsage] = []
    for entry in usage_raw:
        if not isinstance(entry, Mapping):
            continue
        out.append(
            HaikuUsage(
                input_tokens=int(entry.get("input_tokens", 0) or 0),
                output_tokens=int(entry.get("output_tokens", 0) or 0),
            )
        )
    return out


def extract_via_metadata_live(
    papers: Sequence[CohortPaper],
    *,
    dsn: str | None,
) -> dict[str, dict[str, list[str]]]:
    """Real metadata extractor — pulls ADS array fields from the DB.

    Reads ``facility``/``data``/``keyword_norm`` for each bibcode and maps them
    to the corresponding entity type. Lazy-imports ``scix.db`` so test runs
    that never call this function don't pay the import cost.
    """
    if not papers:
        return {}
    from scix.db import get_connection  # noqa: WPS433

    bib_list = [p.bibcode for p in papers]
    conn = get_connection(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, facility, data, keyword_norm "
                "FROM papers WHERE bibcode = ANY(%s)",
                (bib_list,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    out: dict[str, dict[str, list[str]]] = {}
    for bib, facility, data, keyword_norm in rows:
        per_type: dict[str, list[str]] = {etype: [] for etype in ENTITY_TYPES}
        for field_name, value in (
            ("facility", facility),
            ("data", data),
            ("keyword_norm", keyword_norm),
        ):
            if not value:
                continue
            etype = METADATA_FIELD_TO_ENTITY[field_name]
            per_type[etype] = [v for v in value if isinstance(v, str) and v.strip()]
        out[bib] = per_type
    return out


def extract_via_ner_live(
    papers: Sequence[CohortPaper],
    *,
    dsn: str | None,
) -> dict[str, dict[str, list[str]]]:
    """Real NER extractor — pinned WIESP model on each abstract.

    Lazy-imports the M1 helpers so this module remains import-cheap. Skips
    silently in M3 with a clear NotImplementedError if invoked from the CLI
    in a non-mock mode that lacks a fitted abstract pipeline. M5 wires this
    up end-to-end; for M3 the head-to-head can run from --predictions-fixture.
    """
    raise NotImplementedError(
        "Live NER extraction is wired through M5 (scripts/run_ner_batch.py). "
        "For the M3 head-to-head, supply --predictions-fixture or "
        "--ner-fixture so the script can score pre-computed predictions."
    )


def extract_via_haiku_live(
    papers: Sequence[CohortPaper],
) -> tuple[dict[str, dict[str, list[str]]], list[HaikuUsage]]:
    """Real Haiku extractor — calls the Anthropic Messages API per paper.

    Lazy-imports ``anthropic`` and ``scix.extract`` so test environments don't
    need the SDK installed. M5 will wire abstract retrieval; M3 fails fast
    with a documented NotImplementedError if invoked without --mock-haiku.
    """
    raise NotImplementedError(
        "Live Haiku extraction is wired through M5 (scripts/run_ner_batch.py). "
        "For the M3 head-to-head, pass --mock-haiku and supply per-paper "
        "predictions via --predictions-fixture."
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_head_to_head(
    cohort: Sequence[CohortPaper],
    *,
    sample_size: int,
    seed: int,
    predictions: Mapping[str, Any],
    gold: Mapping[str, Mapping[str, list[str]]],
    cohort_source: str,
    mock_mode: str,
) -> dict[str, Any]:
    """Compute metrics, cost, and recommendation; return the report dict."""
    sample = stratified_sample(cohort, sample_size=sample_size, seed=seed)
    sample_bibs = [p.bibcode for p in sample]
    n = len(sample_bibs)

    method_metrics: dict[str, MethodMetrics] = {}
    for method in METHOD_KEYS:
        raw = predictions.get(method, {}) or {}
        # Restrict predictions to the sampled bibcodes so per-method metrics
        # are consistent across methods.
        scoped = {bib: raw.get(bib, {}) for bib in sample_bibs}
        gold_scoped = {bib: gold.get(bib, {}) for bib in sample_bibs}
        if method == "haiku":
            cost = compute_haiku_cost(_haiku_usages_from_fixture(predictions), n)
        else:
            cost = 0.0
        method_metrics[method] = compute_method_metrics(
            scoped, gold_scoped, cost_per_paper_usd=cost
        )

    recommendation, rationale = decide_recommendation(method_metrics)

    return {
        "cohort_source": cohort_source,
        "sample_size": n,
        "model_revision": MODEL_REVISION,
        "per_method": {k: v.to_dict() for k, v in method_metrics.items()},
        "recommendation": recommendation,
        "recommendation_rationale": rationale,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_seed": seed,
            "mock_mode": mock_mode,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Head-to-head comparison: ADS metadata vs WIESP NER vs Haiku. "
            f"Live Haiku requires the {COST_GATE_ENV} env var to be set to a "
            "positive USD budget. Use --mock-all for offline self-tests."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Stratified sample size (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    p.add_argument(
        "--cohort-path",
        type=Path,
        default=DEFAULT_COHORT_PATH,
        help=f"Path to M2 gap report (default: {DEFAULT_COHORT_PATH}).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"RNG seed for sampling (default: {DEFAULT_SAMPLE_SEED}).",
    )
    p.add_argument(
        "--mock-all",
        action="store_true",
        help=(
            "Disable every live call; requires --predictions-fixture and "
            "--gold-fixture. Use for offline tests / CI."
        ),
    )
    p.add_argument(
        "--mock-haiku",
        action="store_true",
        help=(
            "Mock only the Haiku LLM call; metadata + NER may still run live. "
            "Skips the cost gate."
        ),
    )
    p.add_argument(
        "--predictions-fixture",
        type=Path,
        default=None,
        help=(
            "JSON fixture with per-method predictions. Shape: "
            "{'metadata': {bibcode: {entity_type: [entity, ...]}}, "
            "'ner': {...}, 'haiku': {...}, 'haiku_usage': [{input_tokens, "
            "output_tokens}]}."
        ),
    )
    p.add_argument(
        "--gold-fixture",
        type=Path,
        default=None,
        help="JSON fixture: {bibcode: {entity_type: [entity, ...]}}.",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN for live metadata lookups (live mode only).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def _resolve_predictions_and_gold(
    args: argparse.Namespace,
    cohort: Sequence[CohortPaper],
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]], str]:
    """Resolve predictions + gold from fixtures (mock paths) or live runs.

    Returns ``(predictions, gold, mock_mode)`` where ``mock_mode`` is one of
    ``"mock_all"``, ``"mock_haiku"``, or ``"live"``.
    """
    if args.mock_all:
        if args.predictions_fixture is None or args.gold_fixture is None:
            raise SystemExit(
                "--mock-all requires both --predictions-fixture and --gold-fixture"
            )
        preds = _load_predictions_fixture(args.predictions_fixture)
        gold = _load_gold_fixture(args.gold_fixture)
        return preds, gold, "mock_all"

    if args.gold_fixture is None:
        raise SystemExit(
            "--gold-fixture is required (live mode still needs gold labels for scoring)"
        )
    gold = _load_gold_fixture(args.gold_fixture)

    metadata_preds = extract_via_metadata_live(cohort, dsn=args.dsn)

    if args.predictions_fixture is not None:
        fixture = _load_predictions_fixture(args.predictions_fixture)
        ner_preds = fixture.get("ner", {})
        haiku_payload = {
            "haiku": fixture.get("haiku", {}),
            "haiku_usage": fixture.get("haiku_usage", []),
        }
    else:
        ner_preds = extract_via_ner_live(cohort, dsn=args.dsn)
        haiku_payload = {}

    if args.mock_haiku:
        if not haiku_payload:
            raise SystemExit(
                "--mock-haiku requires --predictions-fixture providing the "
                "haiku predictions and haiku_usage entries"
            )
        mode = "mock_haiku"
    else:
        haiku_pred, haiku_usage = extract_via_haiku_live(cohort)
        haiku_payload = {"haiku": haiku_pred, "haiku_usage": [u.__dict__ for u in haiku_usage]}
        mode = "live"

    predictions = {
        "metadata": metadata_preds,
        "ner": ner_preds,
        **haiku_payload,
    }
    return predictions, gold, mode


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    enforce_cost_gate(args)

    cohort = load_cohort(args.cohort_path)
    predictions, gold, mock_mode = _resolve_predictions_and_gold(args, cohort)

    report = run_head_to_head(
        cohort,
        sample_size=args.sample_size,
        seed=args.sample_seed,
        predictions=predictions,
        gold=gold,
        cohort_source=str(args.cohort_path),
        mock_mode=mock_mode,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote head-to-head report to %s — recommendation=%s",
        args.output_path,
        report["recommendation"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
