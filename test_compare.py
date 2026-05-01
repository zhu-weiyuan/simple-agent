"""对比 requests 和 httpx 的差异"""
import requests
import httpx

body = '{"messages":[{"role":"user","content":"Hi"}],"model":"Qwen3.6-35B-A3B-APEX-I-Quality.gguf","max_tokens":10}'
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer xxx",
}

print("=== requests ===")
r = requests.post("http://localhost:8080/chat/completions", data=body, headers=headers)
print(f"Status: {r.status_code}")
print(f"Body: {r.text[:200]}")

print("\n=== httpx (sync) ===")
with httpx.Client() as client:
    req = client.build_request("POST", "http://localhost:8080/chat/completions", content=body, headers=headers)
    resp = client.send(req)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:200]}")

print("\n=== httpx (stream=False) ===")
with httpx.Client(stream=False) as client:
    req = client.build_request("POST", "http://localhost:8080/chat/completions", content=body, headers=headers)
    resp = client.send(req)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:200]}")
