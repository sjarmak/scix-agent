"""Evaluation harness for entity linking (M9).

Public API:

- :func:`wilson_95_ci` — Wilson 95% binomial confidence interval.
- :func:`sample_stratified` — stratified sampler over ``document_entities.tier``.
- :func:`judge` — LLM-judge that returns per-link labels (stub by default).
- :func:`cohens_kappa` — inter-annotator agreement metric.
- :class:`AuditCandidate`, :class:`LinkRow`, :class:`JudgeLabel` — DTOs.
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
from scix.eval.wilson import wilson_95_ci

__all__ = [
    "AuditCandidate",
    "JudgeLabel",
    "LinkRow",
    "cohens_kappa",
    "judge",
    "sample_stratified",
    "wilson_95_ci",
    "write_audit_report",
]
