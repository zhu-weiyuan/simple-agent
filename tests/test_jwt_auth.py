import os
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from my_agent.auth import AuthMiddleware, AuthError, get_jwt_secret, issue_token, decode_token


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


# ── get_jwt_secret minimum length validation ──────────────────


def test_weak_secret_rejected_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "short")
    with pytest.raises(AuthError, match="at least 32 bytes"):
        get_jwt_secret()


def test_weak_secret_rejected_via_environment(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "short")
    with pytest.raises(AuthError, match="at least 32 bytes"):
        get_jwt_secret()


def test_weak_secret_allowed_in_development(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "short")
    # No ENVIRONMENT or APP_ENV set → development mode
    assert get_jwt_secret() == "short"


def test_long_secret_accepted_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    assert get_jwt_secret() == "x" * 32


def test_empty_secret_returns_dev_default(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    assert get_jwt_secret() == "dev-insecure-jwt-secret-change-me-0000000000000000"


def test_empty_secret_rejected_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(AuthError, match="must be configured"):
        get_jwt_secret()


def test_issue_token_uses_get_jwt_secret_validation(monkeypatch):
    """issue_token() calls get_jwt_secret(), so weak secrets are caught."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "weak")
    with pytest.raises(AuthError, match="at least 32 bytes"):
        issue_token("user-123")


def test_decode_token_uses_get_jwt_secret_validation(monkeypatch):
    """decode_token() calls get_jwt_secret(), so weak secrets are caught."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "weak")
    with pytest.raises(AuthError, match="at least 32 bytes"):
        decode_token("some.token.here")
