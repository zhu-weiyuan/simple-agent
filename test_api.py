# -*- coding: utf-8 -*-
import requests, json, sys
sys.stdout = open(1, 'w', encoding='utf-8')

resp = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "你好，介绍一下你自己", "chatId": "test2"},
    timeout=120
)
print(f"Status: {resp.status_code}")
data = resp.json()
reply = data.get('reply', 'N/A')
print(f"Reply length: {len(reply)} chars")
print(f"Reply preview: {reply[:300]}")
enhanced = data.get('enhanced', {})
print(f"\nEnhanced modules:")
for k, v in enhanced.items():
    print(f"  {k}: {v}")
