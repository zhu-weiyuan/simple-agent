"""Resilience primitives for upstream LLM calls."""
from __future__ import annotations

import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when an upstream call is rejected by an open circuit."""


class CircuitBreaker:
    """Thread-safe consecutive-failure circuit breaker.

    One request is allowed as a probe after ``recovery_timeout``. A successful
    probe closes the breaker; a failed probe reopens it.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must not be negative")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._advance_to_half_open_if_ready()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def allow_request(self) -> bool:
        """Return whether a call may reach the upstream, reserving a probe."""
        with self._lock:
            self._advance_to_half_open_if_ready()
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def release_probe(self) -> None:
        """Release a cancelled half-open probe without changing circuit state."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            if self._state is CircuitState.HALF_OPEN:
                self._open()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._open()

    def _advance_to_half_open_if_ready(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False

@dataclass(frozen=True)
class ToolCircuitSnapshot:
    """A stable, JSON-friendly view of one tool's shared circuit."""

    state: str
    consecutive_failures: int


class ToolCircuitManager:
    """Per-tool process-wide circuit management for tool handlers.

    The manager is intentionally owned by :class:`QueryEngine`, which is a
    long-lived singleton in the HTTP app.  That makes a circuit shared across
    requests without leaking state across unrelated engine instances or tests.
    Only *transient* failures are reported to this manager; bad parameters,
    permission denials and deterministic business errors must never take a
    healthy tool offline.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def _get(self, tool_name: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(tool_name)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                    clock=self._clock,
                )
                self._breakers[tool_name] = breaker
            return breaker

    def allow_request(self, tool_name: str) -> Tuple[bool, ToolCircuitSnapshot]:
        breaker = self._get(tool_name)
        allowed = breaker.allow_request()
        return allowed, self.snapshot(tool_name)

    def record_success(self, tool_name: str) -> ToolCircuitSnapshot:
        breaker = self._get(tool_name)
        breaker.record_success()
        return self.snapshot(tool_name)

    def record_transient_failure(self, tool_name: str) -> ToolCircuitSnapshot:
        breaker = self._get(tool_name)
        breaker.record_failure()
        return self.snapshot(tool_name)

    def release_probe(self, tool_name: str) -> None:
        self._get(tool_name).release_probe()

    def snapshot(self, tool_name: str) -> ToolCircuitSnapshot:
        breaker = self._get(tool_name)
        return ToolCircuitSnapshot(
            state=breaker.state.value,
            consecutive_failures=breaker.consecutive_failures,
        )

    def snapshots(self) -> Dict[str, ToolCircuitSnapshot]:
        with self._lock:
            names = list(self._breakers)
        return {name: self.snapshot(name) for name in names}
