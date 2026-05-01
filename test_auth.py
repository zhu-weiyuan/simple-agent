"""测试 Auth header 是否影响响应"""
import requests

data = {
    "model": "Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10,
}

print("Without Authorization header:")
r = requests.post("http://localhost:8080/chat/completions", json=data)
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    print(f"  Content: {r.json()['choices'][0]['message']['content']}")

print("\nWith Authorization: Bearer xxx:")
headers = {"Content-Type": "application/json", "Authorization": "Bearer xxx"}
r2 = requests.post("http://localhost:8080/chat/completions", json=data, headers=headers)
print(f"  Status: {r2.status_code}")
if r2.status_code == 401:
    print(f"  Body: {r2.text[:200]}")
elif r2.status_code == 200:
    print(f"  Content: {r2.json()['choices'][0]['message']['content']}")
