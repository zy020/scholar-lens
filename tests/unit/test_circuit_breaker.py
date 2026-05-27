import asyncio
import pytest
from scholar_lens.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_starts_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_stays_closed_under_failures_below_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=60)
        await cb.record_failure()
        assert not await cb.allow_request()

    @pytest.mark.asyncio
    async def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.01)
        assert await cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_success_closes_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        await cb.record_failure()
        await asyncio.sleep(0.01)
        await cb.allow_request()
        await cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        await cb.record_failure()
        await asyncio.sleep(0.01)
        await cb.allow_request()
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_closed_circuit_allows_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        assert await cb.allow_request()

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED
