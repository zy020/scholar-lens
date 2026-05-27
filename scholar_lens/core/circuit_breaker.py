from __future__ import annotations

import asyncio
import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """3 consecutive failures → open circuit → cooldown → half-open → test request.

    Thread-safe and async-safe via asyncio.Lock.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    async def record_success(self) -> bool:
        async with self._lock:
            was_open = self.state == CircuitState.HALF_OPEN
            self._failure_count = 0
            self.state = CircuitState.CLOSED
        return was_open  # True if this success recovered from a failure state

    async def record_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
