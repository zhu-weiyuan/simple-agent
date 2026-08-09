import os
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from my_agent.auth import AuthMiddleware


def _request(headers=None):
    return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()], "query_string": b"", "path": "/api/chat"})


def test_jwt_round_trip_and_identity(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    token = AuthMiddleware.create_access_token("alice", "acme")
    request = _request({"Authorization": f"Bearer {token}"})
    assert AuthMiddleware.verify(request)
    assert request.state.auth_subject == "alice"
    assert request.state.auth_tenant_id == "acme"


def test_invalid_jwt_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    with pytest.raises(HTTPException) as exc:
        AuthMiddleware.verify(_request({"Authorization": "Bearer invalid"}))
    assert exc.value.status_code == 401


def test_api_key_compatibility(monkeypatch):
    monkeypatch.setenv("API_KEYS", "legacy-key")
    request = _request({"X-API-Key": "legacy-key"})
    assert AuthMiddleware.verify(request)
    assert request.state.auth_scheme == "api_key"


def test_example_jwt_secret_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "replace-with-a-long-random-secret")
    with pytest.raises(ValueError, match="JWT_SECRET"):
        AuthMiddleware.create_access_token("alice")


def test_expired_jwt_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("JWT_ACCESS_TTL_SECONDS", "-1")
    token = AuthMiddleware.create_access_token("alice")
    with pytest.raises(HTTPException) as exc:
        AuthMiddleware.verify(_request({"Authorization": f"Bearer {token}"}))
    assert exc.value.status_code == 401
