# -*- coding: utf-8 -*-
"""
my_agent.auth — 用户身份 (轻量登录 + JWT 签发/校验 + user_id 解析)

设计要点
--------
- 首登即注册 (username → 稳定 user_id)，可选密码 (pbkdf2_hmac, stdlib)。
- JWT 优先使用 PyJWT (守卫导入)，缺失时降级到内置 HS256 实现 (hmac + hashlib +
  base64)，保证纯 stdlib 环境下签发/校验依然可用、单测无需三方依赖。
- request 级 user_id 解析优先级: JWT ``sub`` > ``X-User-Id`` 头 > 匿名。

安全说明
--------
- JWT_SECRET 取自环境变量 ``JWT_SECRET``；生产必须显式配置 (>=32 字符)。
- 密码使用 pbkdf2_hmac(sha256, 200k 迭代) + 每用户随机盐，不落明文。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ── PyJWT 守卫导入 (缺失时用内置 HS256) ───────────────────────
try:  # pragma: no cover - 依赖是否安装取决于环境
    import jwt as _pyjwt  # type: ignore
    _HAS_PYJWT = True
except Exception:  # noqa: BLE001
    _pyjwt = None  # type: ignore
    _HAS_PYJWT = False

ALGORITHM = "HS256"
DEFAULT_TTL_SECONDS = int(os.environ.get("JWT_TTL_SECONDS", str(7 * 24 * 3600)))
ANONYMOUS_USER_ID = "anonymous"

_PBKDF2_ITERATIONS = 200_000


class AuthError(Exception):
    """认证/令牌校验失败。"""


class AuthMiddleware:
    """FastAPI compatibility facade over the JWT and API-key helpers."""

    @staticmethod
    def create_access_token(username: str, tenant_id: str = "", ttl_seconds: Optional[int] = None) -> str:
        """Issue a JWT using the legacy username/tenant API."""
        if os.environ.get("APP_ENV") == "production":
            secret = os.environ.get("JWT_SECRET", "")
            if not secret or secret == "replace-with-a-long-random-secret" or len(secret) < 32:
                raise ValueError("JWT_SECRET must be a real secret of at least 32 characters")
        if ttl_seconds is None:
            ttl_seconds = int(os.environ.get("JWT_ACCESS_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
        return issue_token(
            username,
            username=username,
            ttl_seconds=ttl_seconds,
            extra_claims={"tenant_id": tenant_id} if tenant_id else None,
        )

    @staticmethod
    def verify(request) -> bool:
        """Validate bearer JWT, API key, or legacy query API key."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            try:
                claims = decode_token(auth_header[7:].strip())
            except AuthError as exc:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
            request.state.auth_subject = str(claims.get("sub", ""))
            request.state.auth_tenant_id = claims.get("tenant_id", "")
            request.state.auth_scheme = "jwt"
            return True

        provided = request.headers.get("X-API-Key", "") or request.query_params.get("api_key", "")
        keys = {item.strip() for item in os.environ.get("API_KEYS", "").split(",") if item.strip()}
        if provided and provided in keys:
            request.state.auth_scheme = "api_key"
            return True
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key")

    @staticmethod
    def is_public_endpoint(path: str) -> bool:
        return path in {"/health", "/healthz", "/api/health", "/api/ready", "/api/metrics"}


def auth_required(func):
    """Legacy decorator for FastAPI handlers using ``Request`` injection."""
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = next((value for value in list(args) + list(kwargs.values())
                        if hasattr(value, "headers") and hasattr(value, "state")), None)
        if request is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Request not found")
        if not AuthMiddleware.is_public_endpoint(request.url.path):
            AuthMiddleware.verify(request)
        return await func(*args, **kwargs)

    return wrapper


# ── 密钥获取 ─────────────────────────────────────────────────
def get_jwt_secret() -> str:
    """读取 JWT 密钥。

    生产 (ENVIRONMENT=production) 下要求显式配置；开发缺省给一个固定 dev 值，
    避免本地起不来 (与 DEV_AUTH_BYPASS 语义一致)。
    """
    secret = os.environ.get("JWT_SECRET", "")
    if secret:
        return secret
    if os.environ.get("ENVIRONMENT") == "production":
        raise AuthError("JWT_SECRET must be configured in production")
    return "dev-insecure-jwt-secret-change-me-0000000000000000"


# ── base64url helpers ────────────────────────────────────────
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


# ── user_id 派生 (稳定, 与用户名一一对应) ─────────────────────
def derive_user_id(username: str) -> str:
    """由用户名确定性派生 user_id (稳定, 大小写/空白归一)。"""
    norm = (username or "").strip().lower()
    if not norm:
        raise AuthError("username must not be empty")
    digest = hashlib.sha256(f"user:{norm}".encode("utf-8")).hexdigest()
    return f"u-{digest[:24]}"


# ── 密码哈希 (pbkdf2, stdlib) ─────────────────────────────────
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """返回 ``pbkdf2$<iter>$<salt_hex>$<hash_hex>`` 格式串。"""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常数时间校验密码。"""
    try:
        scheme, iters_s, salt_hex, hash_hex = stored.split("$", 3)
        if scheme != "pbkdf2":
            return False
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(dk, expected)


# ── JWT 签发/校验 ────────────────────────────────────────────
def issue_token(
    user_id: str,
    username: str = "",
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """签发 JWT。sub=user_id。"""
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "iat": now,
        "exp": now + int(ttl_seconds),
        "jti": uuid.uuid4().hex,
    }
    if extra_claims:
        payload.update(extra_claims)
    secret = get_jwt_secret()

    if _HAS_PYJWT:
        token = _pyjwt.encode(payload, secret, algorithm=ALGORITHM)
        # PyJWT>=2 返回 str
        return token if isinstance(token, str) else token.decode("ascii")
    return _encode_hs256(payload, secret)


def decode_token(token: str) -> Dict[str, Any]:
    """校验并解出 payload；失败抛 AuthError。"""
    if not token:
        raise AuthError("empty token")
    secret = get_jwt_secret()
    if _HAS_PYJWT:
        try:
            return _pyjwt.decode(token, secret, algorithms=[ALGORITHM])
        except Exception as e:  # noqa: BLE001 - 统一为 AuthError
            raise AuthError(f"invalid token: {e}") from e
    return _decode_hs256(token, secret)


# ── 内置 HS256 (PyJWT 缺失时) ─────────────────────────────────
def _encode_hs256(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": ALGORITHM, "typ": "JWT"}
    h = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signing_input = f"{h}.{p}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"


def _decode_hs256(token: str, secret: str) -> Dict[str, Any]:
    try:
        h, p, s = token.split(".")
    except ValueError as e:
        raise AuthError("malformed token") from e
    signing_input = f"{h}.{p}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_encode(expected), s):
        raise AuthError("signature mismatch")
    try:
        payload = json.loads(_b64url_decode(p))
    except (ValueError, json.JSONDecodeError) as e:
        raise AuthError("payload decode error") from e
    exp = payload.get("exp")
    if exp is not None and int(time.time()) > int(exp):
        raise AuthError("token expired")
    return payload


# ── request user_id 解析 (JWT sub > X-User-Id > anon) ─────────
def resolve_user_id_from_headers(headers: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """从 (小写化) 请求头解析 user_id。

    返回 (user_id, claims)。claims 为 JWT 载荷 (匿名/仅头时为空 dict)。
    headers 键应为小写。
    """
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        try:
            claims = decode_token(token)
            sub = claims.get("sub")
            if sub:
                return str(sub), claims
        except AuthError:
            pass  # 令牌无效 → 继续降级
    x_user = headers.get("x-user-id", "").strip()
    if x_user:
        return x_user, {}
    return ANONYMOUS_USER_ID, {}


# ── 用户存储 (SQLite, 复用 conversations.db) ─────────────────
@dataclass
class UserRecord:
    user_id: str
    username: str
    has_password: bool


class UserStore:
    """轻量用户表 (首登即注册, 可选密码)。

    直接复用一个 SqliteConversationStore 兼容的连接工厂 (``_get_conn``)，
    或独立开库。仅依赖 stdlib sqlite3。
    """

    def __init__(self, db_path: str = "conversations.db") -> None:
        import sqlite3
        self._sqlite3 = sqlite3
        self.db_path = db_path
        self._persistent_conn = None
        if db_path == ":memory:":
            self._persistent_conn = self._new_conn()
        self._ensure_table()

    def _new_conn(self):
        conn = self._sqlite3.connect(self.db_path)
        conn.row_factory = self._sqlite3.Row
        return conn

    def _conn(self):
        return self._persistent_conn or self._new_conn()

    def _user_columns(self, conn) -> set[str]:
        return {row[1] for row in conn.execute("PRAGMA table_info(users)")}

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_login_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            # Existing installations may have a legacy multi-tenant users table.
            # Add optional columns where possible and keep required legacy columns
            # populated in login_or_register below.
            cols = self._user_columns(conn)
            for col, decl in (
                ("user_id", "TEXT"),
                ("username", "TEXT"),
                ("password_hash", "TEXT DEFAULT ''"),
                ("created_at", "TEXT"),
                ("last_login_at", "TEXT"),
            ):
                if col not in cols:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
            conn.commit()

    def login_or_register(
        self, username: str, password: Optional[str] = None
    ) -> UserRecord:
        """首登即注册；已存在且设了密码则校验。返回 UserRecord。"""
        username = (username or "").strip()
        if not username:
            raise AuthError("username required")
        user_id = derive_user_id(username)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                pw_hash = hash_password(password) if password else ""
                columns = self._user_columns(conn)
                values = {"user_id": user_id, "username": username, "password_hash": pw_hash}
                if "id" in columns:
                    values["id"] = user_id
                if "tenant_id" in columns:
                    values["tenant_id"] = "default"
                if "display_name" in columns:
                    values["display_name"] = username
                names = list(values)
                conn.execute(
                    f"INSERT INTO users ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
                    [values[name] for name in names],
                )
                conn.commit()
                return UserRecord(user_id, username, bool(pw_hash))
            # 已存在
            stored_hash = row["password_hash"] or ""
            if stored_hash:
                if not password or not verify_password(password, stored_hash):
                    raise AuthError("invalid credentials")
            elif password:
                # 之前无密码, 本次带密码 → 补设密码
                conn.execute(
                    "UPDATE users SET password_hash = ? WHERE user_id = ?",
                    (hash_password(password), user_id),
                )
                stored_hash = "set"
            conn.execute(
                "UPDATE users SET last_login_at = datetime('now') WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return UserRecord(user_id, row["username"], bool(stored_hash))

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return UserRecord(row["user_id"], row["username"], bool(row["password_hash"]))

    def close(self) -> None:
        if self._persistent_conn is not None:
            try:
                self._persistent_conn.close()
            finally:
                self._persistent_conn = None
