#!/usr/bin/env python3
"""Re-compute eval precision after the INDUS post-classifier filter (dbl.3).

Reads the existing 414 hand-judged mentions (207 pre-1990 +
207 modern), pulls each row's classifier verdict from
document_entities.evidence, and reports:

  * filtered precision = (correct AND agreement) / agreement-true count
  * unfiltered precision (baseline, sanity check)
  * recall trade = agreement-true count / total
  * per-type breakdown

Sample sources:
  - docs/eval/dbl3_ner_precision_sample_200.jsonl       (pre-1990 sample)
  - docs/eval/dbl3_ner_precision_judgments_200.jsonl    (judge verdicts)
  - docs/eval/dbl3_ner_precision_sample_200_modern.jsonl
  - docs/eval/dbl3_ner_precision_judgments_200_modern.jsonl

Modern sample mentions are NOT in document_entities (they were generated
ad-hoc in-memory from a fresh GLiNER pull on year>=2010 papers — see
scripts/sample_ner_eval_modern.py). For those, this script re-runs the
classifier in-process on the (mention, sentence) pairs from the sample
file and joins against the judge verdicts. The pre-1990 mentions DO live
in document_entities (Phase 1 wrote them in bibcode order) so we read
their classifier verdict from the DB.

Output: docs/eval/dbl3_ner_classifier_filtered_eval.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.extract.ner_classifier import NerClassifier, extract_sentence

_SAMPLES = [
    {
        "label": "pre-1990",
        "sample_path": "docs/eval/dbl3_ner_precision_sample_200.jsonl",
        "judgments_path": "docs/eval/dbl3_ner_precision_judgments_200.jsonl",
        "source": "db",
    },
    {
        "label": "modern (year>=2010)",
        "sample_path": "docs/eval/dbl3_ner_precision_sample_200_modern.jsonl",
        "judgments_path": "docs/eval/dbl3_ner_precision_judgments_200_modern.jsonl",
        "source": "in_memory",
    },
]

_OUT_PATH = Path("docs/eval/dbl3_ner_classifier_filtered_eval.md")


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _join_judgments(samples: list[dict], judgments: list[dict]) -> list[dict]:
    """Inner-join sample → judgment by mention_id; carry both sets of fields."""
    by_id = {j["mention_id"]: j for j in judgments}
    joined: list[dict] = []
    for s in samples:
        j = by_id.get(s["mention_id"])
        if j is None:
            continue
        joined.append({**s, **j})
    return joined


def _classifier_verdict_db(
    conn: psycopg.Connection, mention_ids: list[str]
) -> dict[str, dict | None]:
    """Pull classifier_type + agreement from document_entities.evidence.

    mention_id format used in the pre-1990 sample is "{bibcode}:{entity_id}".
    Returns dict keyed on mention_id.
    """
    out: dict[str, dict | None] = {mid: None for mid in mention_ids}
    by_pair: dict[tuple[str, int], str] = {}
    for mid in mention_ids:
        bib, eid = mid.rsplit(":", 1)
        by_pair[(bib, int(eid))] = mid

    if not by_pair:
        return out

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, entity_id, evidence
            FROM document_entities
            WHERE (bibcode, entity_id) IN (
                SELECT * FROM unnest(%s::text[], %s::int[]) AS t(b, e)
            )
              AND match_method = 'gliner'
            """,
            (
                [b for (b, _) in by_pair],
                [e for (_, e) in by_pair],
            ),
        )
        for bib, eid, ev in cur.fetchall():
            mid = by_pair.get((bib, eid))
            if mid is None:
                continue
            if ev and "agreement" in ev:
                out[mid] = {
                    "classifier_type": ev.get("classifier_type"),
                    "classifier_score": ev.get("classifier_score"),
                    "agreement": bool(ev.get("agreement")),
                }
    return out


def _classifier_verdict_in_memory(
    samples: list[dict], classifier: NerClassifier
) -> dict[str, dict]:
    """For modern sample (not in DB), re-run classifier on (mention, sentence_context)."""
    items: list[tuple[str, str, str]] = []
    for s in samples:
        sentence = extract_sentence(s["abstract"], s.get("surface_text") or s["canonical_name"])
        items.append(
            (
                s.get("surface_text") or s["canonical_name"],
                sentence,
                s["entity_type"],
            )
        )
    results = classifier.classify_batch(items)
    return {
        s["mention_id"]: {
            "classifier_type": r.classifier_type,
            "classifier_score": r.classifier_score,
            "agreement": r.agreement,
        }
        for s, r in zip(samples, results, strict=True)
    }


def _summarize(label: str, joined: list[dict], verdicts: dict[str, dict | None]) -> dict:
    """Compute precision metrics + per-type table for one sample set."""
    n_total = len(joined)
    n_with_verdict = 0
    n_correct_unfiltered = 0
    n_kept = 0  # agreement=true
    n_correct_kept = 0
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    # per_type[t] = [n_total, n_correct_unfiltered, n_kept, n_correct_kept]

    for row in joined:
        mid = row["mention_id"]
        is_correct = bool(row.get("correct"))
        v = verdicts.get(mid)
        n_correct_unfiltered += 1 if is_correct else 0
        et = row.get("entity_type", "?")
        per_type[et][0] += 1
        per_type[et][1] += 1 if is_correct else 0
        if v is None:
            continue
        n_with_verdict += 1
        if v["agreement"]:
            n_kept += 1
            per_type[et][2] += 1
            if is_correct:
                n_correct_kept += 1
                per_type[et][3] += 1

    return {
        "label": label,
        "n_total": n_total,
        "n_with_verdict": n_with_verdict,
        "n_correct_unfiltered": n_correct_unfiltered,
        "n_kept": n_kept,
        "n_correct_kept": n_correct_kept,
        "per_type": {t: tuple(v) for t, v in per_type.items()},
    }


def _format_summary(summaries: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# DBL3 NER — INDUS Classifier Filtered Eval (2026-04-25)\n")
    lines.append(
        "Re-eval of the two earlier 207-mention judgments after applying the "
        "`evidence->>'agreement' = 'true'` filter from the INDUS post-classifier. "
        "The hypothesis: filtering out mentions where the classifier disagrees "
        "with GLiNER's predicted type lifts effective precision over the "
        "80% bar at the cost of some recall.\n"
    )

    for s in summaries:
        coverage = (s["n_with_verdict"] / s["n_total"] * 100) if s["n_total"] else 0
        unfilt_prec = (s["n_correct_unfiltered"] / s["n_total"] * 100) if s["n_total"] else 0
        kept_prec = (s["n_correct_kept"] / s["n_kept"] * 100) if s["n_kept"] else 0
        recall = (s["n_kept"] / s["n_total"] * 100) if s["n_total"] else 0

        lines.append(f"## {s['label']}\n")
        lines.append(f"- mentions in sample:           **{s['n_total']}**")
        lines.append(f"- with classifier verdict:      {s['n_with_verdict']} ({coverage:.1f}%)")
        lines.append(
            f"- unfiltered precision:         **{unfilt_prec:.1f}%** "
            f"({s['n_correct_unfiltered']}/{s['n_total']}) — baseline"
        )
        lines.append(f"- agreement=true (kept):        {s['n_kept']} ({recall:.1f}% recall)")
        lines.append(
            f"- **filtered precision:         {kept_prec:.1f}%** "
            f"({s['n_correct_kept']}/{s['n_kept']}) — vs 80% bar"
        )
        lines.append("")

        lines.append("### Per-type")
        lines.append("| type | n | unfilt prec | kept | filt prec |")
        lines.append("|---|---|---|---|---|")
        for et in sorted(s["per_type"].keys()):
            n_tot, n_corr, n_kept, n_corr_kept = s["per_type"][et]
            up = (n_corr / n_tot * 100) if n_tot else 0
            fp = (n_corr_kept / n_kept * 100) if n_kept else 0
            lines.append(
                f"| {et} | {n_tot} | {up:.0f}% | {n_kept} | "
                f"{fp:.0f}% ({n_corr_kept}/{n_kept}) |"
            )
        lines.append("")

    # Combined headline
    if summaries:
        tot_n = sum(s["n_total"] for s in summaries)
        tot_kept = sum(s["n_kept"] for s in summaries)
        tot_corr_kept = sum(s["n_correct_kept"] for s in summaries)
        tot_corr_unfilt = sum(s["n_correct_unfiltered"] for s in summaries)
        tot_with_verdict = sum(s["n_with_verdict"] for s in summaries)
        unfilt = (tot_corr_unfilt / tot_n * 100) if tot_n else 0
        kept_p = (tot_corr_kept / tot_kept * 100) if tot_kept else 0
        recall = (tot_kept / tot_n * 100) if tot_n else 0
        cov = (tot_with_verdict / tot_n * 100) if tot_n else 0
        lines.append("## Combined")
        lines.append(f"- total mentions:               {tot_n}")
        lines.append(f"- total classified:             {tot_with_verdict} ({cov:.1f}%)")
        lines.append(f"- unfiltered precision:         {unfilt:.1f}% ({tot_corr_unfilt}/{tot_n})")
        lines.append(f"- recall after filter:          {recall:.1f}% ({tot_kept}/{tot_n})")
        lines.append(
            f"- **filtered precision:         {kept_p:.1f}%** ({tot_corr_kept}/{tot_kept})"
        )
        verdict = "PASS" if kept_p >= 80 else "FAIL"
        lines.append(f"- **vs 80% bar:                {verdict}**")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    conn = psycopg.connect("dbname=scix")
    classifier = NerClassifier(device="cuda", embed_batch_size=16)

    summaries: list[dict] = []

    for cfg in _SAMPLES:
        samples = _load_jsonl(cfg["sample_path"])
        judgments = _load_jsonl(cfg["judgments_path"])
        joined = _join_judgments(samples, judgments)

        if cfg["source"] == "db":
            verdicts = _classifier_verdict_db(conn, [r["mention_id"] for r in joined])
        else:
            verdicts = _classifier_verdict_in_memory(joined, classifier)

        summary = _summarize(cfg["label"], joined, verdicts)
        summaries.append(summary)
        print(
            f"{cfg['label']:24} n={summary['n_total']:3d} "
            f"verdicts={summary['n_with_verdict']:3d} "
            f"kept={summary['n_kept']:3d} "
            f"filt_prec={(summary['n_correct_kept']/max(summary['n_kept'],1)*100):.1f}%"
        )

    conn.close()

    report = _format_summary(summaries)
    _OUT_PATH.write_text(report)
    print(f"\nWrote -> {_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
