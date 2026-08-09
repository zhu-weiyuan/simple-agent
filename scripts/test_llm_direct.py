# -*- coding: utf-8 -*-
import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')

body = {
    'model': 'D:\\download\\KAT-Coder-V2.5-Dev-APEX-I-Quality.gguf',
    'messages': [{'role': 'user', 'content': '你好，请用一句简短的话回答。'}],
    'max_tokens': 100,
    'temperature': 0.7
}
r = urllib.request.Request('http://localhost:8080/v1/chat/completions',
    data=json.dumps(body).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST')
try:
    resp = json.loads(urllib.request.urlopen(r, timeout=60).read())
    content = resp['choices'][0]['message']['content']
    print(f'Reply: [{content}]')
    usage = resp.get('usage', {})
    print(f'Tokens: {usage}')
except Exception as e:
    print(f'Error: {str(e)[:100]}')
