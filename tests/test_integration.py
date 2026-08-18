# -*- coding: utf-8 -*-
"""
Integration tests for SimpleAgent API endpoints.

Tests the full request/response flow using FastAPI's TestClient.
No running server needed — everything is tested in-process.

Run: pytest test_integration.py -v
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set up auth for tests (*** is a valid key for test endpoints)
# Force-set these before any module imports
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "test-key")
os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "https://api.example.test")
os.environ["API_KEYS"] = "***,test-key,integration-test"
os.environ["DEV_AUTH_BYPASS"] = "0"  # app_prod honors .env; force auth on


def _get_app():
    """Import and return the FastAPI app instance."""
    from app_prod import app
    return app


# ── Auth Header Helper ─────────────────────────────────────

AUTH_HEADERS = {"X-API-Key": "***"}


# ── Health & Metrics (public endpoints) ────────────────────

def test_health_endpoint():
    """GET /api/health should return system status."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "agent" in data
        assert "version" in data


def test_metrics_prometheus_format():
    """GET /api/metrics should return Prometheus text format."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "# HELP" in text or "# TYPE" in text


def test_observability_json():
    """GET /api/observability should return structured JSON."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/observability")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ── Auth Tests ─────────────────────────────────────────────

def test_chat_requires_auth():
    """POST /api/chat without API key should return 401."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 401


def test_chat_rejects_empty_message():
    """POST /api/chat with empty message should return 400."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"message": "   "}, headers=AUTH_HEADERS)
        assert resp.status_code == 400


# ── Security & Sentiment ───────────────────────────────────

def test_security_scan_detects_pii():
    """POST /api/security/scan should detect PII in phone numbers."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/security/scan",
            json={"text": "我的电话是13800138000"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should flag PII or indicate safety status
        assert "pii" in data or "safe" in data


def test_sentiment_detects_emotion():
    """POST /api/sentiment should return emotion analysis."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/sentiment",
            json={"text": "我很生气！"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "emotion" in data or "sentiment" in data


def test_summary_generates_ticket():
    """POST /api/summary should return conversation summary."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        messages = [
            {"role": "user", "content": "我的订单怎么还没到？"},
            {"role": "assistant", "content": "让我帮您查一下物流信息。"},
        ]
        resp = client.post(
            "/api/summary",
            json={"messages": messages},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "category" in data or "summary" in data


# ── Memory & Tools ─────────────────────────────────────────

def test_memory_stats():
    """GET /api/memory/stats should return memory statistics."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/memories", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


def test_tools_list():
    """GET /api/tools should list available tools."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/tools", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)


# ── Response Quality ───────────────────────────────────────

def test_response_includes_timing_header():
    """API responses should include X-Response-Time header."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert "X-Response-Time" in resp.headers


def test_json_responses_have_correct_content_type():
    """JSON endpoints should set application/json content type."""
    from fastapi.testclient import TestClient
    app = _get_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert "application/json" in resp.headers.get("content-type", "")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
