#!/usr/bin/env python3
"""Rerun the May-29 (gascity orchestration) and June-02 (agentic memory)
lit-review themes through hybrid search now that the dense lane is restored
(Qdrant, 2026-06-11), vs a lexical-only control = the degraded mode both
reviews were actually built in.

Output: results/litreview_rerun_2026-06-11.md
"""

import logging

logging.disable(logging.INFO)

import re
from pathlib import Path

import psycopg

from scix.embed import embed_batch, load_model
from scix.search import hybrid_search

REPO = Path(__file__).resolve().parent.parent
TOP_N = 15

REVIEWS = {
    "agentic_memory": dict(
        doc=REPO / "docs/agentic_memory_litreview_2026-06-02.md",
        queries=[
            "memory systems for LLM agents taxonomy semantic episodic procedural",
            "agent memory architecture external memory store retrieval augmented generation",
            "procedural memory skill library reusable skills LLM agents",
            "reflection experience replay self-improvement LLM agent memory",
            "benchmark for evaluating long-term memory in conversational agents multi-session",
            "evaluation methodology for agent memory retrieval quality",
            "synthetic data generation for memory benchmarks dialogue simulation",
            "memory poisoning security risks LLM agent long-term memory",
            "memory consolidation forgetting obsolescence LLM agents",
        ],
    ),
    "gascity_orchestration": dict(
        doc=REPO / "docs/gascity_lit_review_2026-05-29.md",
        queries=[
            "multi-agent LLM orchestration framework runtime execution traces",
            "typed configuration composition formal semantics agent workflows",
            "failure modes of multi-agent LLM systems",
            "distributed parallel execution agentic workflows scheduling",
            "coordination protocols communication between LLM agents",
            "dynamic role assignment agent team formation LLM",
            "model routing cascade cost-aware LLM inference tiering",
        ],
    ),
}

BIB_RE = re.compile(r"20[0-9]{2}arXiv[0-9]{6}[A-Z]?")


def main() -> None:
    model, tok = load_model("indus", device="auto")
    out: list[str] = [
        "# Lit-Review Rerun — dense lane restored (2026-06-11)\n",
        "Both reviews were built in the lexical-only window (dense lane down "
        "since late May). Per theme-query: hybrid (dense+BM25, now) vs "
        "lexical-only (the mode the reviews ran in). `NEW` = paper absent "
        "from the review's cited corpus.\n",
    ]
    summary_rows: list[str] = []

    with psycopg.connect("dbname=scix") as conn:
        for name, spec in REVIEWS.items():
            review_bibs = set(BIB_RE.findall(spec["doc"].read_text()))
            out.append(f"\n## {name} (review corpus: {len(review_bibs)} arXiv ids)\n")
            total_new_dense = 0
            for q in spec["queries"]:
                vec = embed_batch(model, tok, [q], pooling="mean")[0]
                hyb = hybrid_search(conn, q, list(vec), top_n=TOP_N)
                lex = hybrid_search(conn, q, None, top_n=TOP_N)
                hyb_bibs = [p["bibcode"] for p in hyb.papers]
                lex_bibs = [p["bibcode"] for p in lex.papers]
                dense_only = [b for b in hyb_bibs if b not in set(lex_bibs)]
                new_dense = [b for b in dense_only if b not in review_bibs]
                total_new_dense += len(new_dense)
                titles = {p["bibcode"]: (p.get("title") or "")[:80] for p in hyb.papers}
                years = {p["bibcode"]: p.get("year") for p in hyb.papers}

                out.append(f"\n### “{q}”\n")
                out.append(
                    f"- top-{TOP_N} overlap hybrid∩lexical: "
                    f"{len(set(hyb_bibs) & set(lex_bibs))}/{TOP_N} | "
                    f"dense-added: {len(dense_only)} | "
                    f"dense-added & not in review: {len(new_dense)}"
                )
                for b in dense_only:
                    mark = "NEW" if b in new_dense else "in-review"
                    out.append(f"  - [{mark}] {b} ({years.get(b)}) {titles.get(b, '')}")
            summary_rows.append(f"| {name} | {len(spec['queries'])} | {total_new_dense} |")

    out.insert(
        2,
        "\n| review | queries | dense-surfaced papers not in review |\n|---|---|---|\n"
        + "\n".join(summary_rows)
        + "\n",
    )
    dest = REPO / "results/litreview_rerun_2026-06-11.md"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest}")
    print("\n".join(summary_rows))


if __name__ == "__main__":
    main()
