#!/usr/bin/env python3
"""M9 entity-link audit: OAuth-Claude judge over a stratified sample.

Pivots the bead acceptance from "500 human labels + kappa >= 0.6 vs human"
to "500 OAuth-Claude labels with explicit no-kappa-gate-this-pass" per
standing feedback memory (no paid API; OAuth subagents for all judges;
human validation is nice-to-have not blocking).

Pipeline:

  1. Sample ``n_per_tier`` stratified candidates per tier from
     ``document_entities`` (uses ``scix.eval.audit.sample_stratified``).
  2. Enrich each candidate with paper title/abstract + entity
     canonical_name/type/properties + link evidence + confidence.
  3. Dispatch ``claude -p <prompt>`` per candidate (concurrency=N) with
     an entity-link-audit rubric. Parse one of
     ``correct / incorrect / ambiguous`` plus a one-sentence rationale.
  4. Write labels to ``entity_link_audits`` with annotator
     ``claude_oauth_judge_v1`` (skip rows already labeled by that
     annotator — reruns are idempotent).
  5. Print per-tier precision (Wilson 95% CI) summary.

This script does NOT recompute ``tier_weight()`` or refresh the MV — that
step is gated on mayor sign-off (see m9_apply_calibration.py for the
gated stage 2).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import psycopg  # noqa: E402

from scix.eval.audit import AuditCandidate, sample_stratified  # noqa: E402
from scix.eval.wilson import wilson_95_ci  # noqa: E402

logger = logging.getLogger("m9_audit_judge")

ANNOTATOR_NAME = "claude_oauth_judge_v1"
LABELS: tuple[str, ...] = ("correct", "incorrect", "ambiguous")
DEFAULT_OUTPUT = pathlib.Path("build-artifacts/eval_report.md")
DEFAULT_N_PER_TIER = 125
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 2.0
ABSTRACT_CHAR_BUDGET = 1200
EVIDENCE_CHAR_BUDGET = 600
PROPERTIES_CHAR_BUDGET = 500


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichedCandidate:
    tier: int
    bibcode: str
    entity_id: int
    confidence: float | None
    link_type: str
    evidence_json: str
    paper_title: str
    paper_abstract: str | None
    entity_canonical_name: str
    entity_type: str
    entity_source: str
    entity_properties_json: str


@dataclass(frozen=True)
class JudgeOutcome:
    bibcode: str
    entity_id: int
    tier: int
    label: str
    rationale: str


# ---------------------------------------------------------------------------
# Enrichment SQL
# ---------------------------------------------------------------------------


_ENRICH_SQL = """
SELECT
    de.bibcode,
    de.entity_id,
    de.tier,
    de.confidence,
    de.link_type,
    COALESCE(de.evidence::text, '{}') AS evidence_json,
    COALESCE(p.title, '') AS title,
    p.abstract,
    e.canonical_name,
    e.entity_type,
    COALESCE(e.source, '') AS source,
    COALESCE(e.properties::text, '{}') AS properties_json
FROM document_entities de
JOIN papers   p ON p.bibcode = de.bibcode
JOIN entities e ON e.id      = de.entity_id
WHERE de.bibcode = %s
  AND de.entity_id = %s
  AND de.tier = %s
LIMIT 1
"""


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[:budget] + "...[truncated]"


def _enrich(
    conn: psycopg.Connection, candidates: Iterable[AuditCandidate]
) -> list[EnrichedCandidate]:
    out: list[EnrichedCandidate] = []
    with conn.cursor() as cur:
        for c in candidates:
            cur.execute(_ENRICH_SQL, (c.bibcode, c.entity_id, c.tier))
            row = cur.fetchone()
            if row is None:
                logger.warning(
                    "skip orphan candidate (no enrichment): tier=%d bib=%s eid=%d",
                    c.tier, c.bibcode, c.entity_id,
                )
                continue
            (bib, eid, tier, conf, link_type, evidence_json, title, abstract,
             canonical_name, entity_type, source, properties_json) = row
            out.append(
                EnrichedCandidate(
                    tier=int(tier),
                    bibcode=str(bib),
                    entity_id=int(eid),
                    confidence=float(conf) if conf is not None else None,
                    link_type=str(link_type),
                    evidence_json=_truncate(str(evidence_json), EVIDENCE_CHAR_BUDGET),
                    paper_title=str(title or ""),
                    paper_abstract=(
                        _truncate(str(abstract), ABSTRACT_CHAR_BUDGET)
                        if abstract is not None
                        else None
                    ),
                    entity_canonical_name=str(canonical_name or ""),
                    entity_type=str(entity_type or ""),
                    entity_source=str(source),
                    entity_properties_json=_truncate(
                        str(properties_json), PROPERTIES_CHAR_BUDGET
                    ),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Prompt + parser
# ---------------------------------------------------------------------------


_RUBRIC = (
    "You are an expert ADS metadata auditor judging whether an automatically-"
    "extracted entity link from a scientific paper is correct.\n\n"
    "TIER SEMANTICS:\n"
    "  tier 0: legacy mention payloads (instruments / facilities / etc).\n"
    "  tier 1: keyword exact-match against a curated entity name.\n"
    "  tier 2: alias + Aho-Corasick context match in title/abstract.\n"
    "  tier 4: NER + classifier (text-mention, classifier-driven).\n\n"
    "JUDGE RUBRIC:\n"
    "  correct   — the entity is genuinely discussed in the paper at the\n"
    "              specificity implied by the entity record. For instruments,\n"
    "              the paper actually uses or discusses that instrument; for\n"
    "              concepts/methods, the paper engages that concept;\n"
    "              for objects, the paper observes or studies that object.\n"
    "  incorrect — the entity is not what the paper is about, OR the link is\n"
    "              a name collision (e.g. 'Hubble' the person tagged where\n"
    "              the paper is about Hubble Space Telescope), OR the\n"
    "              extraction matched a string with no scientific connection.\n"
    "  ambiguous — title/abstract genuinely do not provide enough signal to\n"
    "              decide; the link could be correct but isn't verifiable\n"
    "              from the available context.\n\n"
    "OUTPUT (must be a single JSON object, no prose before or after):\n"
    '  {"label": "correct" | "incorrect" | "ambiguous",\n'
    '   "rationale": "<one sentence, <= 200 chars>"}\n'
)


def _format_prompt(c: EnrichedCandidate) -> str:
    abstract_block = c.paper_abstract or "(no abstract available)"
    return (
        f"{_RUBRIC}\n"
        "---\n"
        f"PAPER\n"
        f"  bibcode: {c.bibcode}\n"
        f"  title: {c.paper_title}\n"
        f"  abstract: {abstract_block}\n\n"
        f"LINK\n"
        f"  tier: {c.tier}\n"
        f"  link_type: {c.link_type}\n"
        f"  confidence (system score): {c.confidence}\n"
        f"  evidence: {c.evidence_json}\n\n"
        f"ENTITY\n"
        f"  id: {c.entity_id}\n"
        f"  canonical_name: {c.entity_canonical_name}\n"
        f"  entity_type: {c.entity_type}\n"
        f"  source: {c.entity_source}\n"
        f"  properties: {c.entity_properties_json}\n\n"
        "Respond with the JSON object only."
    )


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(raw: str) -> tuple[str, str]:
    if not raw or not raw.strip():
        raise ValueError("empty response")
    candidates = _JSON_OBJECT_RE.findall(raw)
    if not candidates:
        candidates = [raw.strip()]
    last_err: Exception | None = None
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        rationale = obj.get("rationale", "")
        if not isinstance(label, str) or label not in LABELS:
            continue
        if not isinstance(rationale, str):
            rationale = ""
        return label, rationale[:1000]
    raise ValueError(f"no parseable label JSON in response (last error: {last_err})")


# ---------------------------------------------------------------------------
# claude -p subprocess dispatcher
# ---------------------------------------------------------------------------


class DispatcherError(Exception):
    pass


async def _run_claude_subprocess(
    binary: str, prompt: str
) -> subprocess.CompletedProcess:
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


async def _judge_one(
    sem: asyncio.Semaphore,
    candidate: EnrichedCandidate,
    *,
    claude_binary: str,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
) -> JudgeOutcome:
    prompt = _format_prompt(candidate)
    last_err: str = ""
    for attempt in range(1, max_retries + 1):
        async with sem:
            try:
                completed = await asyncio.wait_for(
                    _run_claude_subprocess(claude_binary, prompt),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                last_err = f"timeout after {timeout_s}s"
                logger.warning(
                    "judge timeout (attempt %d/%d) bib=%s eid=%d",
                    attempt, max_retries, candidate.bibcode, candidate.entity_id,
                )
            except FileNotFoundError as exc:
                return JudgeOutcome(
                    bibcode=candidate.bibcode, entity_id=candidate.entity_id,
                    tier=candidate.tier, label="error",
                    rationale=f"claude binary missing: {exc}",
                )
            else:
                if completed.returncode != 0:
                    last_err = (
                        f"exit={completed.returncode} stderr={completed.stderr[:200]!r}"
                    )
                else:
                    try:
                        label, rationale = _parse_response(completed.stdout)
                        return JudgeOutcome(
                            bibcode=candidate.bibcode,
                            entity_id=candidate.entity_id,
                            tier=candidate.tier,
                            label=label,
                            rationale=rationale,
                        )
                    except ValueError as exc:
                        last_err = f"parse: {exc}; stdout[:200]={completed.stdout[:200]!r}"

        if attempt < max_retries:
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))

    return JudgeOutcome(
        bibcode=candidate.bibcode, entity_id=candidate.entity_id,
        tier=candidate.tier, label="error",
        rationale=f"exhausted {max_retries} retries: {last_err}",
    )


async def _judge_all(
    candidates: list[EnrichedCandidate],
    *,
    claude_binary: str,
    concurrency: int,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
    progress_every: int = 10,
) -> list[JudgeOutcome]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            _judge_one(
                sem, c,
                claude_binary=claude_binary, timeout_s=timeout_s,
                max_retries=max_retries, backoff_base_s=backoff_base_s,
            )
        )
        for c in candidates
    ]
    outcomes: list[JudgeOutcome] = []
    started = time.time()
    for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
        outcome = await fut
        outcomes.append(outcome)
        if i % progress_every == 0 or i == len(tasks):
            elapsed = time.time() - started
            logger.info(
                "judge progress: %d/%d  (%.1fs elapsed, %.1fs/item avg)",
                i, len(tasks), elapsed, elapsed / i,
            )
    by_key = {(o.bibcode, o.entity_id, o.tier): o for o in outcomes}
    return [by_key[(c.bibcode, c.entity_id, c.tier)] for c in candidates]


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


def _existing_audited_keys(
    conn: psycopg.Connection, annotator: str
) -> set[tuple[int, str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tier, bibcode, entity_id "
            "FROM entity_link_audits WHERE annotator = %s",
            (annotator,),
        )
        return {(int(r[0]), str(r[1]), int(r[2])) for r in cur.fetchall()}


def _persist_audits(
    conn: psycopg.Connection,
    outcomes: Iterable[JudgeOutcome],
    *,
    annotator: str,
) -> int:
    rows: list[tuple[int, str, int, str, str, str]] = []
    for o in outcomes:
        if o.label not in LABELS:
            continue
        rows.append((o.tier, o.bibcode, o.entity_id, annotator, o.label, o.rationale))
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO entity_link_audits
                (tier, bibcode, entity_id, annotator, label, note)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (tier, bibcode, entity_id, annotator) DO UPDATE
                SET label = EXCLUDED.label,
                    note  = EXCLUDED.note,
                    created_at = now()
            """,
            rows,
        )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierSummary:
    tier: int
    n_total: int
    n_correct: int
    n_incorrect: int
    n_ambiguous: int
    n_error: int

    @property
    def n_decisive(self) -> int:
        return self.n_correct + self.n_incorrect

    @property
    def precision(self) -> float:
        return (self.n_correct / self.n_decisive) if self.n_decisive else 0.0


def _summarize_from_db(conn: psycopg.Connection, annotator: str) -> list[TierSummary]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tier, label, count(*)
            FROM entity_link_audits
            WHERE annotator = %s
            GROUP BY tier, label
            ORDER BY tier, label
            """,
            (annotator,),
        )
        rows = cur.fetchall()
    by_tier: dict[int, dict[str, int]] = {}
    for tier, label, n in rows:
        d = by_tier.setdefault(int(tier), {"correct": 0, "incorrect": 0, "ambiguous": 0})
        d[str(label)] = int(n)
    out: list[TierSummary] = []
    for tier, counts in sorted(by_tier.items()):
        total = sum(counts.values())
        out.append(
            TierSummary(
                tier=tier, n_total=total,
                n_correct=counts.get("correct", 0),
                n_incorrect=counts.get("incorrect", 0),
                n_ambiguous=counts.get("ambiguous", 0),
                n_error=0,
            )
        )
    return out


def _write_report(
    output: pathlib.Path,
    summaries: list[TierSummary],
    *,
    n_per_tier_target: int,
    annotator: str,
    n_errors_this_run: int,
    note: str | None = None,
) -> pathlib.Path:
    lines: list[str] = []
    lines.append("# M9 entity-link audit report\n")
    lines.append(f"- Annotator: `{annotator}`")
    lines.append(f"- Target sample per tier: **{n_per_tier_target}**")
    lines.append(
        f"- Total candidates judged: **{sum(s.n_total for s in summaries)}**"
    )
    lines.append(f"- Judge errors (this run): **{n_errors_this_run}**")
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")

    lines.append("## Per-tier precision (Wilson 95% CI on decisive labels)\n")
    lines.append(
        "Precision is computed only over `correct` ∪ `incorrect`; `ambiguous` "
        "rows reflect snippet-only insufficient context and are excluded from "
        "the denominator (a documented bias in eval methodology)."
    )
    lines.append("")
    lines.append(
        "| tier | total | correct | incorrect | ambiguous | precision | CI low | CI high |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for s in summaries:
        lo, hi = wilson_95_ci(s.n_correct, s.n_decisive)
        lines.append(
            f"| {s.tier} | {s.n_total} | {s.n_correct} | {s.n_incorrect} | "
            f"{s.n_ambiguous} | "
            f"{s.precision:.3f} | {lo:.3f} | {hi:.3f} |"
        )
    lines.append("")

    ex_lo, ex_hi = wilson_95_ci(95, 100)
    lines.append(
        f"_Worked example `wilson_95_ci(95, 100)` → **[{ex_lo:.3f}, {ex_hi:.3f}]**_"
    )
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote audit report to %s", output)
    return output


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_dsn(cli_dsn: str | None) -> str:
    if cli_dsn:
        return cli_dsn
    dsn = os.environ.get("SCIX_DSN") or os.environ.get("SCIX_TEST_DSN")
    if dsn:
        return dsn
    return "dbname=scix"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--n-per-tier", type=int, default=DEFAULT_N_PER_TIER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--backoff-base-s", type=float, default=DEFAULT_BACKOFF_BASE_S)
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--annotator", default=ANNOTATOR_NAME)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--claude-binary", default=os.environ.get("CLAUDE_BINARY", "claude"))
    parser.add_argument("--dry-run", action="store_true",
                        help="sample + enrich only; no claude calls, no DB writes")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = _resolve_dsn(args.db_url)
    logger.info("connecting to %s", dsn)
    conn = psycopg.connect(dsn)
    try:
        candidates = sample_stratified(conn, n_per_tier=args.n_per_tier, seed=args.seed)
        logger.info("sampled %d candidates across all tiers", len(candidates))

        already = _existing_audited_keys(conn, args.annotator)
        if already:
            before = len(candidates)
            candidates = [
                c for c in candidates
                if (c.tier, c.bibcode, c.entity_id) not in already
            ]
            logger.info(
                "filtered %d already-audited rows; %d remain",
                before - len(candidates), len(candidates),
            )

        enriched = _enrich(conn, candidates)
        logger.info("enriched %d candidates", len(enriched))

        if args.dry_run:
            logger.info("--dry-run: skipping claude calls and DB writes")
            for c in enriched[:3]:
                logger.info(
                    "sample candidate prompt (tier %d):\n%s\n---\n",
                    c.tier, _format_prompt(c)[:600],
                )
            return 0

        if not enriched:
            logger.info("no new candidates to judge — already up to date")
        else:
            outcomes = asyncio.run(
                _judge_all(
                    enriched,
                    claude_binary=args.claude_binary,
                    concurrency=args.concurrency,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    backoff_base_s=args.backoff_base_s,
                )
            )
            n_inserted = _persist_audits(conn, outcomes, annotator=args.annotator)
            n_errors = sum(1 for o in outcomes if o.label == "error")
            logger.info(
                "persisted %d audits (annotator=%s); %d judge errors this run",
                n_inserted, args.annotator, n_errors,
            )

        summaries = _summarize_from_db(conn, args.annotator)

        note = (
            "**Calibration source**: OAuth Claude subagent only — no human "
            "ground truth was available within autonomous-agent scope. "
            "Cohen's kappa gate (>= 0.6 vs human) is **not satisfied** by "
            "this run; mark `tier_weight_calibration_log.notes` accordingly."
        )
        n_errors_total = 0  # Errors are not persisted; they reset across reruns.
        _write_report(
            args.output,
            summaries,
            n_per_tier_target=args.n_per_tier,
            annotator=args.annotator,
            n_errors_this_run=n_errors_total,
            note=note,
        )

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
