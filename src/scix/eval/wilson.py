"""Wilson 95% confidence interval for a binomial proportion.

Ported from ``scripts/audit_tier1.py`` (u06) so the M9 eval harness can
import the canonical implementation. The spec anchor is ``wilson_95_ci(95,
100) -> [0.887, 0.978]`` (tolerance ±0.005 to cover z-constant choice).
"""

from __future__ import annotations

import math

# z for 95% two-sided normal-approx: 1.959963984540054 (commonly 1.96)
_Z_95 = 1.959963984540054


def wilson_95_ci(successes: int, total: int) -> tuple[float, float]:
    """Return the Wilson 95% confidence interval for a binomial proportion.

    >>> lo, hi = wilson_95_ci(95, 100)
    >>> round(lo, 3), round(hi, 3)
    (0.887, 0.978)

    Degenerate cases:
        - ``total == 0`` returns ``(0.0, 1.0)`` (no information).
        - ``successes < 0`` or ``successes > total`` raises ``ValueError``.
    """
    if total <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be in [0, {total}], got {successes}")

    n = float(total)
    p = successes / n
    z = _Z_95
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom

    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)
