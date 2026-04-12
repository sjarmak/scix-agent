"""Stratified sampler and audit-report writer for M9 eval harness.

``sample_stratified`` pulls a uniform random sample of ``n_per_tier`` rows
for each distinct tier present in ``document_entities``. If a tier has
fewer than ``n_per_tier`` rows the full tier is returned.

The sampler reads from the base ``document_entities`` table — NOT from
``document_entities_canonical``. This keeps the read path out of the
single-entry-point lint (M13) while still serving M9 analytics.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import psycopg

from scix.eval.wilson import wilson_95_ci

logger = logging.getLogger(__name__)

DEFAULT_N_PER_TIER = 125


@dataclass(frozen=True)
class AuditCandidate:
    """One row drawn from ``document_entities`` for human/LLM audit."""

    tier: int
    bibcode: str
    entity_id: int
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


_LIST_TIERS_SQL = """
SELECT DISTINCT tier
FROM document_entities
ORDER BY tier
"""


def _list_tiers(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(_LIST_TIERS_SQL)
        return [int(r[0]) for r in cur.fetchall()]


_SAMPLE_TIER_SQL = """
SELECT tier, bibcode, entity_id, confidence
FROM document_entities
WHERE tier = %s
ORDER BY random()
LIMIT %s
"""


def sample_stratified(
    conn: psycopg.Connection,
    n_per_tier: int = DEFAULT_N_PER_TIER,
    *,
    seed: float | None = None,
) -> list[AuditCandidate]:
    """Return up to ``n_per_tier`` random rows per distinct tier.

    Samples uniformly from ``document_entities`` per tier. If ``seed`` is
    provided, we issue ``SELECT setseed(%s)`` first so subsequent
    ``random()`` calls within the same transaction are reproducible.

    Tiers with fewer rows than ``n_per_tier`` are returned in full.

    Args:
        conn: live psycopg connection. Must NOT point at production when
            used in tests (use ``SCIX_TEST_DSN``).
        n_per_tier: target sample size per tier (default 125 → 500 across
            the four canonical tiers).
        seed: optional float in ``[-1, 1]`` passed to postgres ``setseed``
            for deterministic ordering.

    Returns:
        Flat list of :class:`AuditCandidate` across all tiers.
    """
    if n_per_tier <= 0:
        return []

    out: list[AuditCandidate] = []
    with conn.cursor() as cur:
        if seed is not None:
            # postgres requires seed in [-1, 1]
            clamped = max(-1.0, min(1.0, float(seed)))
            cur.execute("SELECT setseed(%s)", (clamped,))

        tiers = _list_tiers(conn)
        for tier in tiers:
            cur.execute(_SAMPLE_TIER_SQL, (tier, n_per_tier))
            for row in cur.fetchall():
                out.append(
                    AuditCandidate(
                        tier=int(row[0]),
                        bibcode=str(row[1]),
                        entity_id=int(row[2]),
                        confidence=(float(row[3]) if row[3] is not None else None),
                    )
                )

    logger.info(
        "sample_stratified: drew %d candidates across %d tier(s) (n_per_tier=%d)",
        len(out),
        len({c.tier for c in out}),
        n_per_tier,
    )
    return out


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierStats:
    tier: int
    total: int
    correct: int

    @property
    def precision(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


def _tier_stats(
    candidates: Sequence[AuditCandidate],
    labels_by_key: dict[tuple[str, int], str],
) -> list[TierStats]:
    """Group candidates by tier, count correct labels, return sorted stats."""
    buckets: dict[int, list[str]] = {}
    for c in candidates:
        key = (c.bibcode, c.entity_id)
        label = labels_by_key.get(key)
        if label is None:
            continue
        buckets.setdefault(c.tier, []).append(label)

    stats: list[TierStats] = []
    for tier, labels in sorted(buckets.items()):
        correct = sum(1 for lab in labels if lab == "correct")
        stats.append(TierStats(tier=tier, total=len(labels), correct=correct))
    return stats


def write_audit_report(
    output_path: pathlib.Path,
    candidates: Iterable[AuditCandidate],
    labels_by_key: dict[tuple[str, int], str],
    *,
    title: str = "Entity-link audit report (M9)",
    note: str | None = None,
) -> pathlib.Path:
    """Render per-tier Wilson CIs to ``output_path`` as markdown."""
    candidate_list = list(candidates)
    stats = _tier_stats(candidate_list, labels_by_key)

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"- Total candidates sampled: **{len(candidate_list)}**")
    lines.append(f"- Total labeled: **{sum(s.total for s in stats)}**")
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")

    lines.append("## Per-tier precision (Wilson 95% CI)\n")
    lines.append("| tier | correct | total | precision | CI low | CI high |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in stats:
        lo, hi = wilson_95_ci(s.correct, s.total)
        lines.append(
            f"| {s.tier} | {s.correct} | {s.total} | " f"{s.precision:.3f} | {lo:.3f} | {hi:.3f} |"
        )
    lines.append("")

    # Worked example so readers can verify wilson is wired up.
    ex_lo, ex_hi = wilson_95_ci(95, 100)
    lines.append(f"_Worked example `wilson_95_ci(95, 100)` → **[{ex_lo:.3f}, {ex_hi:.3f}]**_")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote audit report to %s", output_path)
    return output_path
