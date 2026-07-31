"""Tests for API authentication middleware."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_agent.auth import AuthMiddleware, auth_required


def test_is_public_endpoint_health():
    """Health endpoint should be public."""
    assert AuthMiddleware.is_public_endpoint("/api/health") == True
    assert AuthMiddleware.is_public_endpoint("/health") == True


def test_is_public_endpoint_metrics():
    """Metrics endpoint should be public."""
    assert AuthMiddleware.is_public_endpoint("/api/metrics") == True


def test_is_private_endpoint_chat():
    """Chat endpoint should require auth."""
    assert AuthMiddleware.is_public_endpoint("/api/chat") == False


def test_verify_valid_bearer_token():
    """Valid Bearer token (JWT) should pass verification."""
    import jwt as pyjwt
    from my_agent.auth import ALGORITHM
    secret = "test-secret-key-that-is-at-least-32-chars"
    token = pyjwt.encode({"sub": "user1", "exp": 9999999999}, secret, algorithm=ALGORITHM)
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {token}"}
    request.query_params = {}
    
    with patch.dict(os.environ, {"JWT_SECRET": secret}):
        result = AuthMiddleware.verify(request)
        assert result == True


def test_verify_valid_x_api_key():
    """Valid X-API-Key header should pass."""
    request = MagicMock()
    request.headers = {"X-API-Key": "valid-key"}
    request.query_params = {}
    
    with patch.dict(os.environ, {"API_KEYS": "valid-key"}):
        result = AuthMiddleware.verify(request)
        assert result == True


def test_verify_invalid_token():
    """Invalid token should raise 401."""
    request = MagicMock()
    request.headers = {"Authorization": "Bearer invalid"}
    request.query_params = {}
    
    with patch.dict(os.environ, {"API_KEYS": "valid-key"}):
        try:
            AuthMiddleware.verify(request)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 401


def test_verify_no_auth():
    """No auth headers should raise 401."""
    request = MagicMock()
    request.headers = {}
    request.query_params = {}
    
    with patch.dict(os.environ, {"API_KEYS": "valid-key"}):
        try:
            AuthMiddleware.verify(request)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 401


def test_multiple_api_keys():
    """Multiple API keys (comma-separated) should work."""
    request = MagicMock()
    request.headers = {"X-API-Key": "key2"}
    request.query_params = {}
    
    with patch.dict(os.environ, {"API_KEYS": "key1,key2,key3"}):
        result = AuthMiddleware.verify(request)
        assert result == True
