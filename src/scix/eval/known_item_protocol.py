"""Publication protocol helpers for the fusion known-item diagnostic."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

EVALUATION_KIND = "known-item retrieval diagnostic"
Lane = list[tuple[str, float]]


@dataclass(frozen=True)
class BenchmarkQuery:
    """One decile-stratified known-item query."""

    query_id: str
    query: str
    bucket: str
    discipline: str
    gold_bibcodes: tuple[str, ...]
    decile: int


@dataclass(frozen=True)
class QueryLanes:
    """Retrieved candidate lanes, or the error that prevented retrieval."""

    query: BenchmarkQuery
    dense: Lane
    bm25: Lane
    error: str | None


def _query_id(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _required_string(row: dict[str, Any], field: str, line_no: int) -> str:
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_no}: {field} must be a non-empty string")
    return value


def _gold_bibcodes(row: dict[str, Any], line_no: int) -> tuple[str, ...]:
    value = row["gold_bibcodes"]
    if not isinstance(value, list) or not value:
        raise ValueError(f"line {line_no}: gold_bibcodes must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"line {line_no}: gold_bibcodes must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"line {line_no}: gold_bibcodes must not contain duplicates")
    return tuple(value)


def _parse_query(row: Any, line_no: int) -> BenchmarkQuery:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_no}: each JSONL record must be an object")
    required = {"query", "bucket", "discipline", "gold_bibcodes", "decile"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"line {line_no}: missing required fields {sorted(missing)}")
    query = _required_string(row, "query", line_no)
    bucket = _required_string(row, "bucket", line_no)
    discipline = _required_string(row, "discipline", line_no)
    decile = row["decile"]
    if not isinstance(decile, int) or isinstance(decile, bool) or not 0 <= decile <= 9:
        raise ValueError(f"line {line_no}: decile must be an integer from 0 through 9")
    if bucket != "recall_decile":
        raise ValueError(f"line {line_no}: bucket must be 'recall_decile'")
    return BenchmarkQuery(
        query_id=_query_id(row),
        query=query,
        bucket=bucket,
        discipline=discipline,
        gold_bibcodes=_gold_bibcodes(row, line_no),
        decile=decile,
    )


def load_benchmark_queries(path: Path) -> list[BenchmarkQuery]:
    """Load and validate the complete decile-stratified JSONL contract."""
    queries: list[BenchmarkQuery] = []
    query_texts: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
        query = _parse_query(row, line_no)
        normalized_query = query.query.strip().casefold()
        if normalized_query in query_texts:
            raise ValueError(f"line {line_no}: duplicate query text")
        query_texts.add(normalized_query)
        queries.append(query)
    seen = {query.decile for query in queries}
    if seen != set(range(10)):
        raise ValueError(f"gold set must contain all deciles 0..9; found {sorted(seen)}")
    return queries


def split_queries(
    queries: Sequence[BenchmarkQuery], *, tuning_fraction: float, salt: str
) -> tuple[list[BenchmarkQuery], list[BenchmarkQuery]]:
    """Return deterministic, disjoint, per-decile tuning/confirmation splits."""
    if not 0.0 < tuning_fraction < 1.0:
        raise ValueError("tuning_fraction must be between 0 and 1")
    by_decile: dict[int, list[BenchmarkQuery]] = defaultdict(list)
    for query in queries:
        by_decile[query.decile].append(query)
    if set(by_decile) != set(range(10)):
        raise ValueError("queries must cover every decile 0..9")
    tuning: list[BenchmarkQuery] = []
    confirmation: list[BenchmarkQuery] = []
    for decile in range(10):
        rows = sorted(by_decile[decile], key=lambda query: _split_key(salt, query.query_id))
        if len(rows) < 2:
            raise ValueError(f"decile {decile} needs at least two queries for a held-out split")
        cut = max(1, min(len(rows) - 1, round(len(rows) * tuning_fraction)))
        tuning.extend(rows[:cut])
        confirmation.extend(rows[cut:])
    return tuning, confirmation


def _split_key(salt: str, query_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{query_id}".encode()).hexdigest()


def enforce_coverage_gate(lanes: Sequence[QueryLanes], *, minimum: float) -> dict[str, Any]:
    """Reject a run whose successful retrieval fraction misses its declared gate."""
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum coverage must be in (0, 1]")
    successful = [row for row in lanes if row.error is None]
    coverage = len(successful) / len(lanes) if lanes else 0.0
    summary = {
        "required": minimum,
        "observed": coverage,
        "n_queries": len(lanes),
        "n_successful": len(successful),
        "errors": [
            {"query_id": row.query.query_id, "decile": row.query.decile, "error": row.error}
            for row in lanes
            if row.error is not None
        ],
    }
    if coverage < minimum:
        raise RuntimeError(
            f"retrieval coverage {coverage:.2%} is below required {minimum:.2%}; "
            f"{len(lanes) - len(successful)} query error(s)"
        )
    return summary


def paired_bootstrap_delta(
    candidate: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    *,
    metric: str = "ndcg_at_10",
    seed: int = 0,
    samples: int = 2_000,
) -> dict[str, Any]:
    """Return a paired mean difference and deterministic percentile bootstrap CI."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    baseline_by_id = {row["query_id"]: row for row in baseline}
    differences = _paired_differences(candidate, baseline_by_id, metric)
    if not differences:
        return _empty_uncertainty(metric)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)
    )
    return {
        "metric": metric,
        "n_paired": len(differences),
        "mean_delta": sum(differences) / len(differences),
        "ci95_low": bootstrap[int(0.025 * (samples - 1))],
        "ci95_high": bootstrap[int(0.975 * (samples - 1))],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def _paired_differences(
    candidate: Sequence[dict[str, Any]], baseline_by_id: dict[str, Any], metric: str
) -> list[float]:
    return [
        float(row[metric]) - float(baseline_by_id[row["query_id"]][metric])
        for row in candidate
        if row["query_id"] in baseline_by_id
        and row.get(metric) is not None
        and baseline_by_id[row["query_id"]].get(metric) is not None
    ]


def _empty_uncertainty(metric: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "n_paired": 0,
        "mean_delta": None,
        "ci95_low": None,
        "ci95_high": None,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _headline(results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any], float, float]:
    dense = float(results.get("dense_only", {}).get("overall", {}).get("ndcg_at_10", 0.0))
    bm25 = float(results.get("bm25_only", {}).get("overall", {}).get("ndcg_at_10", 0.0))
    hybrids = [
        (name, block) for name, block in results.items() if name not in {"dense_only", "bm25_only"}
    ]
    if len(hybrids) != 1:
        raise ValueError("confirmation results must include exactly one tuning-selected hybrid")
    name, block = hybrids[0]
    return name, block, dense, bm25


def _overall_table(results: dict[str, dict[str, Any]], dense_ndcg: float) -> list[str]:
    rows = sorted(
        results.items(), key=lambda item: -float(item[1]["overall"].get("ndcg_at_10", 0.0))
    )
    lines = [
        "## Overall (ranked by nDCG@10)",
        "",
        "| Fusion config | nDCG@10 | MRR@10 | Recall@10 | Recall@20 | Recall@50 | ΔnDCG vs dense_only |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, block in rows:
        overall = block["overall"]
        values = [
            overall.get(key, 0.0)
            for key in ("ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_20", "recall_at_50")
        ]
        delta = float(overall.get("ndcg_at_10", 0.0)) - dense_ndcg
        lines.append(
            f"| {name} | " + " | ".join(_fmt(value) for value in values) + f" | {delta:+.4f} |"
        )
    return lines + [""]


def _verdict(
    name: str, block: dict[str, Any], dense: float, bm25: float, path: str, count: int
) -> list[str]:
    selected = float(block["overall"].get("ndcg_at_10", 0.0))
    if dense <= bm25:
        text = f"**Premise does NOT reproduce on this gold set.** dense_only nDCG@10 {_fmt(dense)} is below bm25_only {_fmt(bm25)} on `{path}` ({count} queries). The dense lane is not dominant, so `{name}` at {_fmt(selected)} must not be reported as a dense-dominant fusion result."
    elif selected > dense:
        text = f"**Winning fusion: `{name}`** — nDCG@10 {_fmt(selected)} ({selected - dense:+.4f} over dense-alone) on held-out confirmation."
    else:
        text = f"**No fusion beats dense-alone** (dense_only nDCG@10 {_fmt(dense)}). The tuning-selected hybrid `{name}` confirms at {_fmt(selected)}."
    return ["## Verdict", "", text, ""]


def _decile_table(name: str, block: dict[str, Any], dense_block: dict[str, Any]) -> list[str]:
    lines = [
        f"## Per-decile confirmation (`{name}` vs dense_only)",
        "",
        "| Decile | dense nDCG@10 | selected nDCG@10 | dense R@10 | selected R@10 | dense R@20 | selected R@20 | dense R@50 | selected R@50 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    strata = block["by_decile"]
    dense_strata = dense_block["by_decile"]
    keys = ("ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50")
    for decile in strata:
        values = [
            value
            for key in keys
            for value in (dense_strata.get(decile, {}).get(key, 0.0), strata[decile].get(key, 0.0))
        ]
        lines.append(f"| {decile} | " + " | ".join(_fmt(value) for value in values) + " |")
    return lines + [""]


def _uncertainty_table(uncertainty: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Paired uncertainty on held-out confirmation",
        "",
        "| Metric | Paired queries | Mean Δ vs dense | 95% bootstrap CI |",
        "|---|---|---|---|",
    ]
    for metric, block in uncertainty.items():
        lines.append(
            f"| {metric} | {block['n_paired']} | {_fmt(block['mean_delta'])} | [{_fmt(block['ci95_low'])}, {_fmt(block['ci95_high'])}] |"
        )
    return lines + [""]


def _provenance_lines(provenance: dict[str, Any]) -> list[str]:
    return [
        "## Provenance",
        "",
        f"- Gold SHA-256: `{provenance['gold']['sha256']}`",
        f"- Git revision: `{provenance['git_revision']}`",
        f"- Dense model: `{provenance['model']['huggingface_id']}` at `{provenance['model']['revision']}`",
        f"- Qdrant: `{provenance['qdrant']['endpoint']}` / `{provenance['qdrant']['collection']}`",
        f"- Corpus DB/schema: `{provenance['corpus']['database']}` / migration `{provenance['corpus']['schema_migration']}`; estimated papers: {provenance['corpus']['paper_count_estimate']}",
        "",
    ]


def render_markdown(
    results: dict[str, dict[str, Any]],
    *,
    queries_path: str,
    n_queries: int,
    k: int,
    paired_uncertainty: dict[str, dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> str:
    """Render held-out results with explicit interpretation and provenance."""
    name, block, dense, bm25 = _headline(results)
    lines = [
        "# Fusion-calibration known-item diagnostic — v2",
        "",
        f"- Evaluation kind: **{EVALUATION_KIND}** (exact paper titles; not general retrieval evidence)",
        "- Protocol: configurations are selected on tuning data and reported once on held-out confirmation data",
        f"- Gold set: `{queries_path}`",
        f"- Held-out confirmation: {n_queries} queries",
        f"- nDCG/MRR cutoff: {k}; Recall cutoffs: 10, 20, 50",
        "",
    ]
    lines.extend(_overall_table(results, dense))
    lines.extend(_verdict(name, block, dense, bm25, queries_path, n_queries))
    lines.extend(_decile_table(name, block, results.get("dense_only", {})))
    if paired_uncertainty:
        lines.extend(_uncertainty_table(paired_uncertainty))
    if provenance:
        lines.extend(_provenance_lines(provenance))
    return "\n".join(lines)
