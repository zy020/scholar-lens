import pytest
import time
from scholar_lens.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_failures_below_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=60)
        cb.record_failure()
        assert not cb.allow_request()

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.01)
        assert cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_closes_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        time.sleep(0.01)
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        time.sleep(0.01)
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_closed_circuit_allows_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        assert cb.allow_request()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
