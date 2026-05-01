"""底层 httpx 直接发请求，模拟 SDK 的完整行为"""
import httpx

# Exact headers that the SDK sends (from our earlier debug output)
headers = {
    "host": "localhost:8080",
    "accept-encoding": "gzip, deflate, br, zstd",
    "connection": "keep-alive",
    "accept": "application/json",
    "content-type": "application/json",
    "user-agent": "OpenAI/Python 1.109.1",
    "authorization": "Bearer xxx",
    "x-stainless-lang": "python",
    "x-stainless-package-version": "1.109.1",
    "x-stainless-os": "Windows",
    "x-stainless-arch": "other:amd64",
    "x-stainless-runtime": "CPython",
    "x-stainless-runtime-version": "3.11.9",
}

body = '{"messages":[{"role":"user","content":"Hi"}],"model":"Qwen3.6-35B-A3B-APEX-I-Quality.gguf","max_tokens":10}'

with httpx.Client(base_url="http://localhost:8080") as client:
    req = client.build_request(
        "POST",
        "/chat/completions",
        content=body,
        headers=headers,
    )
    print(f"URL: {req.url}")
    
    # Send the exact request
    resp = client.send(req)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:300]}")

# Also test with /v1/ path
print("\n--- Testing with /v1/chat/completions ---")
with httpx.Client(base_url="http://localhost:8080") as client:
    req = client.build_request(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers=headers,
    )
    resp = client.send(req)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:300]}")
