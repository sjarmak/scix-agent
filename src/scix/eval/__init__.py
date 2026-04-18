"""Evaluation harness.

Two judge surfaces live here:

1. **Binary link audit** (:mod:`scix.eval.llm_judge`) — labels entity-link
   decisions as ``correct / incorrect / ambiguous``. Used by M9.
2. **Ordinal relevance** (:mod:`scix.eval.persona_judge`) — scores
   ``(query, paper)`` pairs on a 0-3 scale via the OAuth
   ``in_domain_researcher`` subagent. Used for retrieval calibration.

Public API:

- :func:`wilson_95_ci` — Wilson 95% binomial confidence interval.
- :func:`sample_stratified` — stratified sampler over ``document_entities.tier``.
- :func:`judge` — binary LLM-judge that returns per-link labels (stub by default).
- :func:`cohens_kappa` — inter-annotator agreement metric.
- :class:`AuditCandidate`, :class:`LinkRow`, :class:`JudgeLabel` — DTOs.
- :class:`PersonaJudge`, :class:`JudgeScore`, :class:`JudgeTriple` — ordinal judge DTOs.
- :func:`quadratic_weighted_kappa`, :func:`spearman_rho` — ordinal calibration metrics.
"""

from __future__ import annotations

from scix.eval.audit import (
    AuditCandidate,
    sample_stratified,
    write_audit_report,
)
from scix.eval.llm_judge import (
    JudgeLabel,
    LinkRow,
    cohens_kappa,
    judge,
)
from scix.eval.persona_judge import (
    JudgeScore,
    JudgeTriple,
    PersonaJudge,
    build_snippet,
    quadratic_weighted_kappa,
    spearman_rho,
)
from scix.eval.wilson import wilson_95_ci

__all__ = [
    "AuditCandidate",
    "JudgeLabel",
    "JudgeScore",
    "JudgeTriple",
    "LinkRow",
    "PersonaJudge",
    "build_snippet",
    "cohens_kappa",
    "judge",
    "quadratic_weighted_kappa",
    "sample_stratified",
    "spearman_rho",
    "wilson_95_ci",
    "write_audit_report",
]
