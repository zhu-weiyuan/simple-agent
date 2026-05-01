"""直接测试 OpenAI SDK vs requests 请求的区别"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("测试 1: requests POST /chat/completions")
print("=" * 60)

import requests
data = {
    "model": "Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10,
}
headers = {"Content-Type": "application/json"}

# Without /v1/
r = requests.post("http://localhost:8080/chat/completions", json=data)
print(f"POST /chat/completions -> {r.status_code}")
if r.status_code == 200:
    print(r.json()["choices"][0]["message"]["content"])

# With /v1/
r2 = requests.post("http://localhost:8080/v1/chat/completions", json=data)
print(f"POST /v1/chat/completions -> {r2.status_code}")
if r2.status_code == 200:
    print(r2.json()["choices"][0]["message"]["content"])

print()
print("=" * 60)
print("测试 2: OpenAI SDK (base_url=http://localhost:8080)")
print("=" * 60)

from openai import OpenAI
sdk_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "xxx"),
    base_url="http://localhost:8080",
)
print(f"SDK base_url: {sdk_client.base_url}")
try:
    resp = sdk_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "default"),
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    )
    print(f"SDK success: {resp.choices[0].message.content}")
except Exception as e:
    print(f"SDK error: {e}")

print()
print("=" * 60)
print("测试 3: OpenAI SDK (base_url=http://localhost:8080/v1)")
print("=" * 60)

sdk_client2 = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "xxx"),
    base_url="http://localhost:8080/v1",
)
print(f"SDK base_url: {sdk_client2.base_url}")
try:
    resp2 = sdk_client2.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "default"),
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    )
    print(f"SDK success: {resp2.choices[0].message.content}")
except Exception as e:
    print(f"SDK error: {e}")
