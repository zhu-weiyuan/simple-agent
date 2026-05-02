# -*- coding: utf-8 -*-
"""对比 SDK 和 requests 的请求差异"""
import os
import json
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY", "xxx")
client = OpenAI(
    api_key=api_key,
    base_url="http://localhost:8080",
)

# Build the exact request like SDK does
req_body = {
    "model": "Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10,
}

# Test 1: SDK direct call
print("=== Test 1: SDK direct ===")
try:
    resp = client.chat.completions.create(
        model="Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    )
    print(f"Success: {resp.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: SDK with api_key="" (empty)
print("\n=== Test 2: SDK empty key ===")
client2 = OpenAI(api_key="", base_url="http://localhost:8080")
try:
    resp2 = client2.chat.completions.create(
        model="Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    )
    print(f"Success: {resp2.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: SDK with no auth header (override)
print("\n=== Test 3: Override auth to empty ===")
client3 = OpenAI(api_key="none", base_url="http://localhost:8080")
# Try removing the Authorization header via a hook
import httpx
original_build_request = client3._client._send_single_request

def patched_build(*args, **kwargs):
    # Can't easily modify headers here, let's just try a raw call
    return original_build_request(*args, **kwargs)

# Test 4: Try with model aliases
print("\n=== Test 4: Short model name ===")
try:
    resp4 = client.chat.completions.create(
        model="Qwen3.6",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    )
    print(f"Success: {resp4.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
