"""Benchmark orchestration for the graph-experiment spike (bead vdtd).

Holds the testable parts of the driver:

  * graph-aware bibcode pickers per template
  * harness-result aggregation per variant
  * go/no-go verdict policy
  * markdown report rendering

The CLI shim lives in ``scripts/run_graph_experiment_benchmark.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from scix.graph_experiment.analysis import (
    TraceSummary,
    summarize_traces,
)
from scix.graph_experiment.benchmark import (
    BENCHMARK_TEMPLATES,
    Question,
    QuestionTemplate,
    materialize_templates,
)
from scix.graph_experiment.harness import HarnessConfig, RunResult, run_question

logger = logging.getLogger(__name__)


_METHOD_TOKENS = ("method", "algorithm", "technique", "pipeline")


# --------------------------------------------------------------------------
# graph-aware bibcode pickers
# --------------------------------------------------------------------------


def pick_top_cited(
    graph,
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    min_count: int = 0,
    max_count: int | None = None,
    title_contains: tuple[str, ...] = (),
) -> list[int]:
    """Return vertex ids sorted by citation_count desc, filtered by attrs.

    Skips vertices missing year/title when a filter requires them.
    """
    cands: list[tuple[int, int]] = []
    titles = graph.vs["title"]
    titles_lower = [t.lower() if isinstance(t, str) else "" for t in titles]
    years = graph.vs["year"]
    counts = graph.vs["citation_count"]
    for vid, (y, cc) in enumerate(zip(years, counts)):
        if cc is None or cc < min_count:
            continue
        if max_count is not None and cc > max_count:
            continue
        if min_year is not None and (y is None or y < min_year):
            continue
        if max_year is not None and (y is None or y > max_year):
            continue
        if title_contains and not any(
            tok in titles_lower[vid] for tok in title_contains
        ):
            continue
        cands.append((vid, cc))
    cands.sort(key=lambda t: t[1], reverse=True)
    return [vid for vid, _ in cands]


def _bibcode(graph, vid: int) -> str:
    return graph.vs[vid]["name"]


def build_pickers(graph) -> dict[str, Callable[[QuestionTemplate], tuple[str, ...]]]:
    """Return template_id → picker. Each picker returns a tuple of bibcodes.

    Pickers degrade gracefully: empty tuple → ``materialize_templates``
    skips that template with a warning.
    """

    def t1_direct_citations(_: QuestionTemplate) -> tuple[str, ...]:
        ids = pick_top_cited(graph, min_count=1)
        return (_bibcode(graph, ids[0]),) if ids else ()

    def t1_recent_citers(_: QuestionTemplate) -> tuple[str, ...]:
        ids = pick_top_cited(graph, min_count=50, max_year=2022)
        return (_bibcode(graph, ids[0]),) if ids else ()

    def t1_topical_search(_: QuestionTemplate) -> tuple[str, ...]:
        return ()  # seed_count == 0

    def t2_method_lineage(_: QuestionTemplate) -> tuple[str, ...]:
        # No reference_count on the slice; substitute high-citation papers
        # from 2020+ which are likely to have substantial reference lists.
        ids = pick_top_cited(graph, min_year=2020, min_count=20)
        return (_bibcode(graph, ids[0]),) if ids else ()

    def t2_co_cited(_: QuestionTemplate) -> tuple[str, ...]:
        ids = pick_top_cited(graph, min_count=20, max_count=200)
        return (_bibcode(graph, ids[0]),) if ids else ()

    def t2_bridge_between(_: QuestionTemplate) -> tuple[str, ...]:
        # "Different research areas" is best-effort: pick a high-cited and a
        # different mid-cited paper that are not directly connected.
        top = pick_top_cited(graph, min_count=100)
        mid = pick_top_cited(graph, min_count=20, max_count=80)
        if not top or not mid:
            return ()
        head_id = top[0]
        head_neighbors = set(graph.neighbors(head_id, mode="all"))
        for cand_id in mid:
            if cand_id == head_id or cand_id in head_neighbors:
                continue
            return _bibcode(graph, head_id), _bibcode(graph, cand_id)
        return ()

    def t2_ppr_relevance(_: QuestionTemplate) -> tuple[str, ...]:
        # Triangle anchored on a top-cited paper; fallback to top-3 by
        # citation if no triangle is found in the search budget.
        top = pick_top_cited(graph, min_count=50)[:200]
        if len(top) < 3:
            return ()
        top_set = set(top)
        for head_id in top[:50]:
            head_neighbors = set(graph.neighbors(head_id, mode="all"))
            cluster_neighbors = [n for n in head_neighbors if n in top_set]
            for a in cluster_neighbors:
                a_neighbors = set(graph.neighbors(a, mode="all"))
                for b in cluster_neighbors:
                    if b == a:
                        continue
                    if b in a_neighbors:
                        return tuple(
                            _bibcode(graph, x) for x in (head_id, a, b)
                        )
        return tuple(_bibcode(graph, vid) for vid in top[:3])

    def t3_subgraph_communities(_: QuestionTemplate) -> tuple[str, ...]:
        ids = pick_top_cited(graph, min_count=100)
        if len(ids) < 2:
            return ()
        head_id = ids[0]
        head_neighbors = set(graph.neighbors(head_id, mode="all"))
        for cand_id in ids[1:200]:
            if cand_id == head_id or cand_id in head_neighbors:
                continue
            return _bibcode(graph, head_id), _bibcode(graph, cand_id)
        return tuple(_bibcode(graph, vid) for vid in ids[:2])

    def t3_method_then_application(_: QuestionTemplate) -> tuple[str, ...]:
        ids = pick_top_cited(
            graph, min_count=200, title_contains=_METHOD_TOKENS
        )
        if not ids:
            ids = pick_top_cited(graph, min_count=200)
        return (_bibcode(graph, ids[0]),) if ids else ()

    def t3_cross_community(_: QuestionTemplate) -> tuple[str, ...]:
        ids = pick_top_cited(graph, min_count=100, max_year=2018)
        return (_bibcode(graph, ids[0]),) if ids else ()

    return {
        "t1_direct_citations": t1_direct_citations,
        "t1_recent_citers": t1_recent_citers,
        "t1_topical_search": t1_topical_search,
        "t2_method_lineage": t2_method_lineage,
        "t2_co_cited": t2_co_cited,
        "t2_bridge_between": t2_bridge_between,
        "t2_ppr_relevance": t2_ppr_relevance,
        "t3_subgraph_communities": t3_subgraph_communities,
        "t3_method_then_application": t3_method_then_application,
        "t3_cross_community": t3_cross_community,
    }


def materialize_questions(graph) -> list[Question]:
    pickers = build_pickers(graph)

    def picker(template: QuestionTemplate) -> tuple[str, ...]:
        fn = pickers.get(template.template_id)
        if fn is None:
            logger.warning(
                "no picker for template %s — skipping", template.template_id
            )
            return ()
        return fn(template)

    return materialize_templates(BENCHMARK_TEMPLATES, picker)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def run_all(
    questions: list[Question],
    config: HarnessConfig,
) -> list[RunResult]:
    results: list[RunResult] = []
    for q in questions:
        for variant in ("control", "treatment"):
            logger.info("running %s / %s", q.id, variant)
            t0 = time.time()
            result = run_question(q.prompt, q.id, variant, config)
            logger.info(
                "  done in %.1fs (cost=%s, tokens=%s, error=%s)",
                time.time() - t0,
                result.cost_usd,
                result.total_tokens,
                result.error,
            )
            results.append(result)
    return results


def summaries_by_variant(
    results: list[RunResult], trace_dir: Path
) -> dict[str, TraceSummary]:
    out: dict[str, TraceSummary] = {}
    for variant in ("control", "treatment"):
        sessions = [r.session_id for r in results if r.variant == variant]
        paths = [trace_dir / f"trace_{sid}.jsonl" for sid in sessions]
        existing = [p for p in paths if p.exists()]
        if not existing:
            out[variant] = TraceSummary(
                session_ids=[],
                event_count=0,
                tool_call_counts={},
                error_counts={},
                depth_per_event=[],
                depth_distribution={},
                latency_ms_per_tool={},
            )
            continue
        out[variant] = summarize_traces(existing)
    return out


# --------------------------------------------------------------------------
# verdict policy + report
# --------------------------------------------------------------------------


def go_no_go(comparison: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, rationale) for the bead acceptance report.

    Verdict policy:
      * GO if treatment depth_shift_max >= 2 AND freeform_query_emergence > 0
        AND at least one new graph primitive saw use.
      * INCONCLUSIVE if depth shifted but freeform queries never fired.
      * NO-GO if treatment shows no depth lift over control.
    """
    depth_shift = float(comparison.get("depth_shift_max", 0))
    freeform = int(comparison.get("freeform_query_emergence", 0))
    new_tools = comparison.get("new_tool_usage", {}) or {}

    if depth_shift >= 2 and freeform > 0 and new_tools:
        return (
            "GO",
            "Agents reached deeper queries when given the experimental tools "
            f"(max depth shift {depth_shift:+.0f}, {freeform} freeform queries, "
            f"{len(new_tools)} new primitives used). Recommend Apache AGE "
            "spike on the slice as the next step.",
        )
    if depth_shift > 0:
        return (
            "INCONCLUSIVE",
            f"Depth lifted ({depth_shift:+.0f}) but freeform-query emergence "
            f"is {freeform} and new-tool usage is {len(new_tools)}. Re-run "
            "with a stronger benchmark or relax the harness budget before "
            "deciding on Apache AGE.",
        )
    return (
        "NO-GO",
        f"No depth lift over control (max shift {depth_shift:+.0f}). The "
        "1-hop-heavy workload is intrinsic to the questions or to the agent's "
        "tool-selection priors — Apache AGE migration not justified by this "
        "evidence.",
    )


def render_markdown(
    *,
    verdict: str,
    rationale: str,
    comparison: dict[str, Any],
    questions: list[Question],
    results: list[RunResult],
) -> str:
    control = comparison["control_summary"]
    treatment = comparison["treatment_summary"]
    new_tools = comparison.get("new_tool_usage", {}) or {}
    lines: list[str] = []
    lines.append("# Graph-Experiment Benchmark Report")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append(
        f"- Depth shift (median): {comparison['depth_shift_median']:+.2f}"
    )
    lines.append(f"- Depth shift (max):    {comparison['depth_shift_max']:+.0f}")
    lines.append(
        f"- Freeform queries (treatment): {comparison['freeform_query_emergence']}"
    )
    if new_tools:
        lines.append("- New primitives used in treatment:")
        for tool, n in sorted(new_tools.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - `{tool}`: {n}")
    else:
        lines.append("- New primitives used in treatment: (none)")
    lines.append("")
    lines.append("## Trace summary")
    lines.append("")
    lines.append("| metric | control | treatment |")
    lines.append("|--------|---------|-----------|")
    lines.append(
        f"| events | {control['event_count']} | {treatment['event_count']} |"
    )
    lines.append(
        f"| max depth | {control['max_depth']} | {treatment['max_depth']} |"
    )
    lines.append(
        f"| median depth | {control['median_depth']:.2f} | "
        f"{treatment['median_depth']:.2f} |"
    )
    lines.append("")
    lines.append("## Per-question runs")
    lines.append("")
    lines.append(
        "| question | tier | control cost | treatment cost | control err | treatment err |"
    )
    lines.append(
        "|----------|------|--------------|----------------|-------------|---------------|"
    )
    by_q: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by_q.setdefault(r.question_id, {})[r.variant] = r
    tiers = {q.id: q.tier for q in questions}
    for qid in sorted(by_q):
        ctrl = by_q[qid].get("control")
        trt = by_q[qid].get("treatment")
        cc = f"${ctrl.cost_usd:.4f}" if ctrl and ctrl.cost_usd else "-"
        tc = f"${trt.cost_usd:.4f}" if trt and trt.cost_usd else "-"
        ce = (ctrl.error if ctrl and ctrl.error else "-")[:40]
        te = (trt.error if trt and trt.error else "-")[:40]
        lines.append(
            f"| `{qid}` | {tiers.get(qid, '-')} | {cc} | {tc} | {ce} | {te} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"
