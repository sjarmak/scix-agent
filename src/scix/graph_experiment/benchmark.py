"""Benchmark questions for the graph-experiment spike.

Each question carries a tier (1-hop / 2-3-hop / subgraph-pattern), a textual
prompt, expected hop depth, and a scoring rubric for the LLM judge. Day-3
deliverable.

Two layers:
  * ``Question`` dataclass + JSONL serialization.
  * ``BENCHMARK_TEMPLATES`` — bibcode-agnostic question shells. Concrete
    ``Question`` objects are produced by ``materialize_templates`` once the
    slice is loaded so we can plug in real bibcodes from the slice.

Bibcode-specific questions land in ``data/graph_experiment/benchmark.jsonl``;
templates are the reproducible source. The ``materialize_templates`` step
runs after the slice exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)


Tier = Literal["one_hop_control", "multi_hop_target", "subgraph_pattern_stretch"]


@dataclass(frozen=True)
class Question:
    """A single benchmark question.

    ``expected_hop_depth`` is the minimum hop count an agent must traverse
    to construct a defensible answer. Use this to score whether agents
    reached the depth the question requires — the central experiment metric.
    """

    id: str
    tier: Tier
    prompt: str
    expected_hop_depth: int
    rubric: str
    seed_bibcodes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Question":
        return cls(
            id=payload["id"],
            tier=payload["tier"],
            prompt=payload["prompt"],
            expected_hop_depth=int(payload["expected_hop_depth"]),
            rubric=payload["rubric"],
            seed_bibcodes=tuple(payload.get("seed_bibcodes", ())),
            metadata=dict(payload.get("metadata", {})),
        )


def write_jsonl(path: Path, questions: Iterable[Question]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict()) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[Question]:
    questions: list[Question] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            questions.append(Question.from_dict(json.loads(line)))
    return questions


@dataclass(frozen=True)
class QuestionTemplate:
    """A bibcode-agnostic question shell.

    ``materialize`` plugs concrete seed bibcodes from the loaded slice into
    ``prompt_template`` (using ``str.format(seeds=...)``). Templates with
    ``seed_count == 0`` materialize directly.
    """

    template_id: str
    tier: Tier
    prompt_template: str
    expected_hop_depth: int
    rubric: str
    seed_count: int
    seed_selector: str  # human-readable description of how to pick seeds


# Bibcode-agnostic question shells. Each tier is intentionally over-stocked
# so we can drop poorly-resolving items after a dry run on the slice.
BENCHMARK_TEMPLATES: tuple[QuestionTemplate, ...] = (
    # ---------------------------------------------------------------- tier 1
    QuestionTemplate(
        template_id="t1_direct_citations",
        tier="one_hop_control",
        prompt_template=(
            "List the papers cited by {seeds}. For each, give the title "
            "and year."
        ),
        expected_hop_depth=1,
        rubric=(
            "FULL credit if the agent enumerates direct citations of the "
            "seed paper and returns at least 5 with title and year. PARTIAL "
            "if it returns fewer than 5 or omits attributes. ZERO if it "
            "returns unrelated papers or refuses."
        ),
        seed_count=1,
        seed_selector="any high-citation_count paper from the slice",
    ),
    QuestionTemplate(
        template_id="t1_recent_citers",
        tier="one_hop_control",
        prompt_template=(
            "Which papers from 2023 or later cite {seeds}? Return up to 10."
        ),
        expected_hop_depth=1,
        rubric=(
            "FULL if agent returns recent citing papers with year filter "
            "applied. PARTIAL if year filter ignored. ZERO if agent "
            "confuses cited-by vs cites."
        ),
        seed_count=1,
        seed_selector="paper with citation_count >= 50 and year <= 2022",
    ),
    QuestionTemplate(
        template_id="t1_topical_search",
        tier="one_hop_control",
        prompt_template=(
            "Find recent papers about exoplanet atmospheric retrievals."
        ),
        expected_hop_depth=0,
        rubric=(
            "FULL if agent uses search/concept_search to return on-topic "
            "papers. ZERO if agent uses graph tools without a starting "
            "bibcode (graph tools cannot answer this directly)."
        ),
        seed_count=0,
        seed_selector="(none)",
    ),
    # ---------------------------------------------------------------- tier 2
    QuestionTemplate(
        template_id="t2_method_lineage",
        tier="multi_hop_target",
        prompt_template=(
            "Trace the methodological lineage of {seeds}: which earlier "
            "papers (2 or more hops back through references) likely "
            "influenced its core method? Return the chain."
        ),
        expected_hop_depth=2,
        rubric=(
            "FULL if agent walks 2+ hops backward through citations and "
            "returns a coherent chain. PARTIAL if only direct references "
            "(1 hop). ZERO if agent only returns the seed paper or "
            "unrelated work."
        ),
        seed_count=1,
        seed_selector="paper with reference_count >= 30 and year >= 2020",
    ),
    QuestionTemplate(
        template_id="t2_co_cited",
        tier="multi_hop_target",
        prompt_template=(
            "Find papers that share a citation target with {seeds} (papers "
            "co-cited with it). What recurring topics emerge?"
        ),
        expected_hop_depth=2,
        rubric=(
            "FULL if agent identifies co-cited papers (out -> in pattern) "
            "and abstracts a topical theme. PARTIAL if it returns "
            "co-citations without synthesis. ZERO if it only returns "
            "direct citations or citers."
        ),
        seed_count=1,
        seed_selector="paper with citation_count between 20 and 200",
    ),
    QuestionTemplate(
        template_id="t2_bridge_between",
        tier="multi_hop_target",
        prompt_template=(
            "Find the shortest citation path between {seeds} (two papers "
            "from different research areas). What intermediate papers "
            "bridge them?"
        ),
        expected_hop_depth=3,
        rubric=(
            "FULL if agent uses shortest_path or equivalent to return a "
            "concrete chain. PARTIAL if path is found but bridging papers "
            "are not characterized. ZERO if no path returned or unrelated."
        ),
        seed_count=2,
        seed_selector="two seeds drawn from different first-author surnames "
        "and from arxiv_class subtrees that rarely co-occur",
    ),
    QuestionTemplate(
        template_id="t2_ppr_relevance",
        tier="multi_hop_target",
        prompt_template=(
            "Given the working set {seeds}, what other papers in the corpus "
            "are most relevant to this research thread? Rank the top 20."
        ),
        expected_hop_depth=2,
        rubric=(
            "FULL if agent uses personalized_pagerank or equivalent and "
            "returns ranked results. PARTIAL if results are returned but "
            "without rank/score. ZERO if agent re-runs basic search."
        ),
        seed_count=3,
        seed_selector="three papers all citing each other or one another's "
        "references (a tight cluster)",
    ),
    # ---------------------------------------------------------------- tier 3
    QuestionTemplate(
        template_id="t3_subgraph_communities",
        tier="subgraph_pattern_stretch",
        prompt_template=(
            "Build the citation neighborhood around {seeds} (1-2 hops). "
            "Are there sub-communities within this neighborhood? Describe "
            "each by its dominant topic."
        ),
        expected_hop_depth=2,
        rubric=(
            "FULL if agent extracts a subgraph and identifies multiple "
            "sub-clusters with topic labels. PARTIAL if subgraph extracted "
            "without clustering. ZERO if no subgraph or single flat list."
        ),
        seed_count=2,
        seed_selector="two well-cited papers in adjacent subfields",
    ),
    QuestionTemplate(
        template_id="t3_method_then_application",
        tier="subgraph_pattern_stretch",
        prompt_template=(
            "Starting from the methods paper {seeds}, find application "
            "papers that cite descendants of this method (papers that cite "
            "papers that cite {seeds}). Which scientific domains adopted "
            "this method?"
        ),
        expected_hop_depth=3,
        rubric=(
            "FULL if agent walks the in,in pattern (or equivalent) and "
            "characterises adopting domains. PARTIAL if pattern walked but "
            "domains not labelled. ZERO if agent only returns direct citers."
        ),
        seed_count=1,
        seed_selector="seminal methods paper (citation_count > 200, "
        "title contains method/algorithm/technique/pipeline)",
    ),
    QuestionTemplate(
        template_id="t3_cross_community",
        tier="subgraph_pattern_stretch",
        prompt_template=(
            "Find papers that cite {seeds} but are themselves rarely cited "
            "by other papers in the same research community. What unusual "
            "applications or critiques are surfacing?"
        ),
        expected_hop_depth=3,
        rubric=(
            "FULL if agent identifies citers and qualifies their "
            "embeddedness in the source community. PARTIAL if only citers "
            "returned. ZERO if no citers or off-topic."
        ),
        seed_count=1,
        seed_selector="established result paper from a mature subfield "
        "(citation_count >= 100, year <= 2018)",
    ),
)


def materialize_templates(
    templates: Iterable[QuestionTemplate],
    bibcode_picker,
) -> list[Question]:
    """Materialize templates by drawing concrete bibcodes from the slice.

    ``bibcode_picker(template)`` is a callable returning a tuple of bibcodes
    matching the template's seed_selector. Returns one Question per template
    (or zero if the picker returns an empty tuple — caller decides whether
    to relax the selector).
    """
    out: list[Question] = []
    for template in templates:
        if template.seed_count == 0:
            seeds: tuple[str, ...] = ()
        else:
            seeds = tuple(bibcode_picker(template))
            if len(seeds) != template.seed_count:
                logger.warning(
                    "skipping %s: picker returned %d seeds, wanted %d",
                    template.template_id,
                    len(seeds),
                    template.seed_count,
                )
                continue
        prompt = template.prompt_template.format(seeds=", ".join(seeds) if seeds else "")
        out.append(
            Question(
                id=template.template_id,
                tier=template.tier,
                prompt=prompt,
                expected_hop_depth=template.expected_hop_depth,
                rubric=template.rubric,
                seed_bibcodes=seeds,
                metadata={"seed_selector": template.seed_selector},
            )
        )
    return out
