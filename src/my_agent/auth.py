"""
API authentication middleware for FastAPI — JWT + API Key.

Usage:
    from my_agent.auth import auth_required
    @app.post("/api/chat")
    @auth_required
    async def chat(...): ...
"""

import os
from typing import Optional
from fastapi import HTTPException, Request


class AuthMiddleware:
    """API key authentication for FastAPI."""
    
    @staticmethod
    def verify(request: Request) -> bool:
        """Verify API key from header or query parameter.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if authenticated
            
        Raises:
            HTTPException: 401 if authentication fails
        """
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if AuthMiddleware._validate_key(token):
                return True
        
        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key", "")
        if api_key_header and AuthMiddleware._validate_key(api_key_header):
            return True
        
        # Check query parameter (for backward compatibility)
        api_key_param = request.query_params.get("api_key", "")
        if api_key_param and AuthMiddleware._validate_key(api_key_param):
            return True
        
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )
    
    @staticmethod
    def _validate_key(api_key: str) -> bool:
        """Validate API key against stored keys."""
        valid_keys = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
        return api_key in valid_keys
    
    @staticmethod
    def is_public_endpoint(path: str) -> bool:
        """Check if endpoint should be publicly accessible."""
        public_paths = ["/health", "/api/health", "/api/metrics"]
        return path in public_paths


def auth_required(func):
    """Decorator to require API key authentication.
    
    Usage:
        @app.post("/api/chat")
        @auth_required
        async def chat(request: Request, ...): ...
    """
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find the Request object in arguments
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request:
            raise HTTPException(status_code=401, detail="Request not found")
        
        if AuthMiddleware.is_public_endpoint(request.url.path):
            return await func(*args, **kwargs)
        
        AuthMiddleware.verify(request)
        return await func(*args, **kwargs)
    
    return wrapper
