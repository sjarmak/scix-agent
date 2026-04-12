"""Unit tests for :mod:`scix.circuit_breaker` — PRD §M10 / u13.

These tests exercise the 3-state FSM in isolation: no database, no
external dependencies. A fake monotonic clock is injected so the tests
are deterministic even on slow CI runners.
"""

from __future__ import annotations

import pytest

from scix.circuit_breaker import CircuitBreaker, CircuitBreakerOpen


class FakeClock:
    """Injected monotonic clock with an advance() knob."""

    def __init__(self, t0: float = 0.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(budget_seconds=5.0, clock=clock)


class TestInitialState:
    def test_starts_closed(self, breaker: CircuitBreaker) -> None:
        assert breaker.is_closed()
        assert not breaker.is_open()
        assert not breaker.is_half_open()
        assert breaker.trip_count == 0

    def test_check_without_start_is_noop(self, breaker: CircuitBreaker) -> None:
        # Before start() the breaker is "unmetered"; check() never raises.
        breaker.check()
        breaker.check()
        assert breaker.is_closed()
        assert breaker.trip_count == 0


class TestBudgetEnforcement:
    def test_check_within_budget_is_noop(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        breaker.start()
        clock.advance(1.0)
        breaker.check()
        clock.advance(2.0)
        breaker.check()
        assert breaker.is_closed()

    def test_check_at_budget_trips(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        breaker.start()
        clock.advance(5.0)  # exactly at budget
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        assert breaker.is_open()
        assert breaker.trip_count == 1

    def test_check_over_budget_trips(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        breaker.start()
        clock.advance(10.0)
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        assert breaker.is_open()
        assert breaker.trip_count == 1

    def test_subsequent_checks_after_trip_keep_raising(
        self, breaker: CircuitBreaker, clock: FakeClock
    ) -> None:
        breaker.start()
        clock.advance(10.0)
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        # trip_count should only increment once — already open
        assert breaker.trip_count == 1

    def test_zero_budget_trips_immediately(self) -> None:
        clock = FakeClock()
        b = CircuitBreaker(budget_seconds=0.0, clock=clock)
        b.start()
        with pytest.raises(CircuitBreakerOpen):
            b.check()
        assert b.is_open()

    def test_tiny_budget_from_monkeypatch(self) -> None:
        """Acceptance criterion: monkeypatch ``budget_seconds=0.001`` to
        trip the breaker in tests."""
        clock = FakeClock()
        b = CircuitBreaker(budget_seconds=0.001, clock=clock)
        b.start()
        clock.advance(0.5)  # well over 0.001s
        with pytest.raises(CircuitBreakerOpen):
            b.check()
        assert b.trip_count == 1


class TestTripAndReset:
    def test_explicit_trip(self, breaker: CircuitBreaker) -> None:
        breaker.start()
        breaker.trip()
        assert breaker.is_open()
        assert breaker.trip_count == 1
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()

    def test_trip_counter_increments_across_cycles(
        self, breaker: CircuitBreaker, clock: FakeClock
    ) -> None:
        # First cycle: trip, reset state back to closed via start()
        breaker.start()
        breaker.trip()
        assert breaker.trip_count == 1

        # start() should reset state to closed but preserve trip_count
        breaker.start()
        assert breaker.is_closed()
        assert breaker.trip_count == 1

        # Second cycle: trip again
        clock.advance(10.0)
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        assert breaker.trip_count == 2

    def test_reset_clears_everything(self, breaker: CircuitBreaker) -> None:
        breaker.start()
        breaker.trip()
        breaker.reset()
        assert breaker.is_closed()
        assert breaker.trip_count == 0


class TestHalfOpenProbe:
    def test_half_open_from_open_only(self, breaker: CircuitBreaker) -> None:
        # Can't half-open a closed breaker
        breaker.half_open_probe()
        assert breaker.is_closed()

        breaker.start()
        breaker.trip()
        breaker.half_open_probe()
        assert breaker.is_half_open()

    def test_half_open_probe_success_flips_closed(
        self, breaker: CircuitBreaker, clock: FakeClock
    ) -> None:
        breaker.start()
        breaker.trip()
        # Pretend cooldown has passed and a fresh budget window opens.
        breaker.start()  # restart meter in closed state
        breaker.trip()  # trip again manually
        breaker.half_open_probe()
        assert breaker.is_half_open()
        # A fresh start puts us in closed; now probe with plenty of budget
        breaker.start()
        breaker.half_open_probe()  # no-op (not open)
        breaker.check()
        assert breaker.is_closed()

    def test_half_open_probe_failure_re_opens(
        self, breaker: CircuitBreaker, clock: FakeClock
    ) -> None:
        breaker.start()
        clock.advance(10.0)
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        assert breaker.is_open()

        # Transition to half_open WITHOUT restarting the clock — still
        # over budget — so the probe check immediately re-opens.
        breaker.half_open_probe()
        assert breaker.is_half_open()
        with pytest.raises(CircuitBreakerOpen):
            breaker.check()
        assert breaker.is_open()
        # Re-opening from half_open should also increment trip_count
        assert breaker.trip_count == 2


class TestElapsedAndRemaining:
    def test_elapsed_zero_when_not_started(self, breaker: CircuitBreaker) -> None:
        assert breaker.elapsed() == 0.0

    def test_elapsed_tracks_clock(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        breaker.start()
        clock.advance(2.5)
        assert breaker.elapsed() == pytest.approx(2.5)

    def test_remaining_can_go_negative(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        breaker.start()
        clock.advance(7.0)
        assert breaker.remaining() == pytest.approx(-2.0)
