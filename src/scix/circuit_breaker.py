"""Time-budget circuit breaker — PRD §M10 / u13.

Protects the incremental daily-sync pipeline from exceeding its 5-minute
linking budget. The breaker is a 3-state FSM (closed → open → half_open)
that enforces a wall-clock budget once :meth:`start` is called:

* ``closed``     — normal operation, :meth:`check` is a no-op while elapsed
                   is under budget.
* ``open``       — budget exceeded or :meth:`trip` called explicitly.
                   :meth:`check` raises :class:`CircuitBreakerOpen` and all
                   subsequent work should short-circuit.
* ``half_open``  — transitional probe state entered by :meth:`half_open_probe`
                   after a cooldown; the next :meth:`check` either flips
                   back to closed (if we have budget again) or re-opens.

The breaker is deliberately dependency-free so the unit tests in
``tests/test_circuit_breaker.py`` can exercise it without a database or
any scix imports. Callers that need to persist trip counts (e.g. the
incremental-sync runner that writes to the ``alerts`` table) inspect
:attr:`trip_count` after the run completes.

Usage::

    breaker = CircuitBreaker(budget_seconds=300.0)
    breaker.start()
    try:
        for paper in papers:
            breaker.check()           # raises if budget exhausted
            process(paper)
    except CircuitBreakerOpen:
        logger.warning("budget exhausted after %d papers", n)
        # graceful-degradation path: advance watermark, skip the rest

The breaker is NOT thread-safe. Callers that need concurrent access
should wrap the mutating methods in a lock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreakerOpen(Exception):
    """Raised by :meth:`CircuitBreaker.check` when the budget is exhausted.

    Callers catch this to drop into graceful-degradation mode.
    """


@dataclass
class CircuitBreaker:
    """Wall-clock time-budget circuit breaker.

    Parameters
    ----------
    budget_seconds:
        Maximum wall-clock seconds the breaker permits between
        :meth:`start` and the first :meth:`check` that would exceed the
        budget. Values <= 0 trip immediately on the first check after
        ``start``.
    clock:
        Monotonic clock injection point for deterministic tests. Defaults
        to :func:`time.monotonic`.
    """

    budget_seconds: float
    clock: Callable[[], float] = field(default=time.monotonic)
    state: State = "closed"
    trip_count: int = 0
    _started_at: float | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start (or restart) the wall-clock budget.

        This is the only way to move the breaker back into ``closed``
        from a fresh run; :meth:`reset` is available for tests that want
        to clear ``trip_count`` as well.
        """
        self._started_at = self.clock()
        self.state = "closed"

    def reset(self) -> None:
        """Clear state fully — used by tests. Does NOT restart the clock.

        After :meth:`reset` the breaker is ``closed`` with no start time;
        call :meth:`start` before :meth:`check` to begin metering again.
        """
        self.state = "closed"
        self.trip_count = 0
        self._started_at = None

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        return self.state == "open"

    def is_closed(self) -> bool:
        return self.state == "closed"

    def is_half_open(self) -> bool:
        return self.state == "half_open"

    def elapsed(self) -> float:
        """Seconds since :meth:`start`. Returns 0.0 if not started."""
        if self._started_at is None:
            return 0.0
        return self.clock() - self._started_at

    def remaining(self) -> float:
        """Seconds of budget remaining. May be negative if exhausted."""
        return self.budget_seconds - self.elapsed()

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def check(self) -> None:
        """Assert we still have budget. Raise :class:`CircuitBreakerOpen`
        if the budget is exhausted or the breaker is already open.

        In ``half_open`` state this performs a single probe: if we have
        budget the breaker flips to ``closed``; if not, it re-opens.
        """
        if self.state == "open":
            raise CircuitBreakerOpen(
                f"circuit breaker open (budget={self.budget_seconds:.3f}s, "
                f"elapsed={self.elapsed():.3f}s)"
            )

        if self._started_at is None:
            # Budget metering hasn't begun — treat as unlimited. Callers
            # who forget to call start() simply get no enforcement.
            return

        if self.elapsed() >= self.budget_seconds:
            self._open()
            raise CircuitBreakerOpen(
                f"circuit breaker tripped (budget={self.budget_seconds:.3f}s, "
                f"elapsed={self.elapsed():.3f}s)"
            )

        if self.state == "half_open":
            # Probe succeeded — restore normal operation.
            self.state = "closed"

    def trip(self) -> None:
        """Force the breaker open. Increments :attr:`trip_count`."""
        self._open()

    def half_open_probe(self) -> None:
        """Transition ``open`` → ``half_open`` so the next :meth:`check`
        acts as a probe. No-op if the breaker is already closed.

        The caller is responsible for calling :meth:`start` again (or
        leaving the existing start-time in place) before probing — the
        probe uses the same elapsed-time rule as a normal check.
        """
        if self.state == "open":
            self.state = "half_open"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self.state != "open":
            self.trip_count += 1
        self.state = "open"
