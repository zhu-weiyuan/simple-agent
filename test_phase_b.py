import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent.core.engine import QueryEngine
from my_agent.core.hooks import HookPoint, HookRegistry
from my_agent.resilience import CircuitBreaker, CircuitState
from my_agent.tools.registry import ToolRegistry


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


def test_stream_methods_propagate_context_to_hooks():
    hooks = HookRegistry()
    request_ids = []
    hooks.register(HookPoint.LLM_START, lambda _ctx, **kw: request_ids.append(kw["data"].get("request_id")))
    engine = QueryEngine("system", tool_registry=ToolRegistry(), hooks=hooks)

    class Msg:
        role = "assistant"
        content = "ok"
        tool_calls = []
    class Choice:
        message = Msg()
        finish_reason = "stop"
    class Response:
        choices = [Choice()]

    engine.set_llm(lambda _messages, _schemas: Response())
    assert "ok" in "".join(engine.run_stream("hello", context={"request_id": "stream-id"}))
    assert request_ids == ["stream-id"]


def test_context_usage_logs_thresholds(caplog):
    engine = QueryEngine("system", tool_registry=ToolRegistry())
    engine.session.config.max_tokens = 10
    engine.session.messages[0].content = "x" * 40
    with caplog.at_level(logging.INFO):
        assert engine._track_context_usage() > 0.8
    assert "Dumb Zone" in caplog.text


def test_degraded_response_contract():
    from app import get_degraded_response
    response = get_degraded_response("hello", "req-123")
    assert response["status"] == "degraded"
    assert response["code"] == "LLM_UNAVAILABLE"
    assert response["request_id"] == "req-123"


def test_plain_stream_rejects_open_circuit_without_calling_upstream():
    from app import _stream_plain_reply

    class Engine:
        class Session:
            def append(self, _message):
                raise AssertionError("open circuit must not mutate session")
        session = Session()

    class Breaker:
        def allow_request(self):
            return False

    class Agent:
        engine = Engine()
        circuit_breaker = Breaker()

    event = "".join(_stream_plain_reply(Agent(), "hello", "stream-req"))
    assert '"code": "LLM_UNAVAILABLE"' in event
    assert '"request_id": "stream-req"' in event


def test_plain_stream_releases_cancelled_half_open_probe():
    from app import _stream_plain_reply

    class Session:
        messages = []

        def append(self, _message):
            pass

    class Engine:
        session = Session()

        @staticmethod
        def _dedup_assistant(messages):
            return messages

    class Breaker:
        released = 0

        def allow_request(self):
            return True

        def release_probe(self):
            self.released += 1

    class LLM:
        @staticmethod
        def chat_stream(_messages):
            yield "first"
            yield "second"

    class Agent:
        engine = Engine()
        circuit_breaker = Breaker()
        llm = LLM()

    stream = _stream_plain_reply(Agent(), "hello", "stream-req")
    next(stream)
    stream.close()
    assert Agent.circuit_breaker.released == 1


def test_plain_stream_records_failure_and_hides_upstream_detail():
    from app import _stream_plain_reply

    class Session:
        def __init__(self):
            self.messages = []

        def append(self, message):
            self.messages.append(message)

    class Engine:
        session = Session()

        @staticmethod
        def _dedup_assistant(messages):
            return messages

    class Breaker:
        failures = 0
        successes = 0

        def allow_request(self):
            return True

        def record_failure(self):
            self.failures += 1

        def record_success(self):
            self.successes += 1

    class LLM:
        @staticmethod
        def chat_stream(_messages):
            raise RuntimeError("provider secret detail")
            yield  # pragma: no cover

    class Agent:
        engine = Engine()
        circuit_breaker = Breaker()
        llm = LLM()

    event = "".join(_stream_plain_reply(Agent(), "hello", "stream-req"))
    assert Agent.circuit_breaker.failures == 1
    assert Agent.circuit_breaker.successes == 0
    assert '"code": "LLM_UNAVAILABLE"' in event
    assert "provider secret detail" not in event
