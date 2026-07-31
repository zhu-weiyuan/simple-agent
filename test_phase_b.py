import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent.resilience import CircuitBreaker, CircuitState


def test_circuit_opens_and_recovers_with_one_probe():
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10, clock=lambda: now[0])
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()
    now[0] = 10.0
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request()
    assert not breaker.allow_request()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0

