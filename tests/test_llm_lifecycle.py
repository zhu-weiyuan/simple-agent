import threading

import pytest

from my_agent.llm import LLMClient


def test_llm_client_uses_bounded_per_attempt_timeout(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    import my_agent.llm as module
    monkeypatch.setattr(module, "DEFAULT_TIMEOUT", 3.0)
    monkeypatch.setattr(module, "TOTAL_DEADLINE", 3.0)
    monkeypatch.setattr(module.requests, "post", lambda *a, **kw: calls.append(kw) or Response())
    assert LLMClient(base_url="http://llm", api_key="x").chat([])["choices"][0]["message"]["content"] == "ok"
    assert calls[0]["timeout"] <= 3.0


def test_llm_stream_closes_response_when_generator_is_closed(monkeypatch):
    class Response:
        status_code = 200
        closed = False
        def raise_for_status(self):
            return None
        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}'
            yield b'data: [DONE]'
        def close(self):
            self.closed = True

    response = Response()
    import my_agent.llm as module
    monkeypatch.setattr(module.requests, "post", lambda *a, **kw: response)
    stream = LLMClient(base_url="http://llm").chat_stream([])
    assert next(stream) == "a"
    stream.close()
    assert response.closed
