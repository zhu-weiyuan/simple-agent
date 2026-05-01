"""检查 OpenAI SDK 实际发送什么"""
import os
from openai import OpenAI

# 设置代理来捕获请求 - 用 mock
from unittest.mock import patch, MagicMock
import httpx

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "xxx"),
    base_url="http://localhost:8080",
)

# 拦截 HTTPX 的 request 方法
original_send = client._client._send_single_request

def debug_send(request):
    print(f"URL: {request.url}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    if hasattr(request, 'content'):
        print(f"Body: {request.content.decode()[:200]}")
    elif hasattr(request, 'read'):
        content = request.read()
        print(f"Body: {content.decode()[:200] if content else 'empty'}")
    return original_send(request)

try:
    with patch.object(client._client, '_send_single_request', debug_send):
        resp = client.chat.completions.create(
            model="Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
except Exception as e:
    print(f"\nException (expected): {e}")
