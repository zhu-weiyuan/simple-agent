# -*- coding: utf-8 -*-
"""Start simple-agent with local llama.cpp"""
import subprocess, os, sys, time

BASE = r'C:\Users\Administrator\.openclaw\workspace\simple-agent'
os.chdir(BASE)

env = os.environ.copy()
env['OPENAI_API_KEY'] = 'sk-local'
env['OPENAI_BASE_URL'] = 'http://localhost:8080/v1'
env['OPENAI_MODEL'] = 'D:\\download\\KAT-Coder-V2.5-Dev-APEX-I-Quality.gguf'
env['API_KEYS'] = 'test-key-123'
env['JWT_SECRET'] = 'test-jwt'
env['RATE_LIMIT_REQUESTS'] = '300'
env['RATE_LIMIT_WINDOW'] = '60'

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000', '--workers', '1'],
    cwd=BASE,
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print('Server PID:', proc.pid)
time.sleep(8)

# Quick test
import urllib.request, json
h = {'X-API-Key': '***'}
try:
    r = urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/health', headers=h), timeout=10)
    d = json.loads(r.read())
    print(f'OK: agent={d.get("agent")} model={d.get("llm",{}).get("model","")[:40]} reachable={d.get("llm",{}).get("reachable")}')
except Exception as e:
    print(f'FAIL: {str(e)[:100]}')
