# -*- coding: utf-8 -*-
"""Verify all endpoints are working"""
import sys, json, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
KEY = 'test-key-123'

def g(path, timeout=10):
    r = urllib.request.Request('http://localhost:8000' + path, headers={'X-API-Key': KEY})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

# Health
h = g('/api/health')
print('=== HEALTH ===')
print(f'Model: {h.get("llm",{}).get("model","")[:40]}')
print(f'Reachable: {h.get("llm",{}).get("reachable")}')
print(f'Sessions: {h.get("sessions")}')
print(f'Inflight: {h.get("inflight")}')
print(f'Requests: {h.get("requests")}')
print(f'System: CPU={h.get("system",{}).get("cpu_percent")}% Mem={h.get("system",{}).get("memory_percent")}%')

# Tools
t = g('/api/tools')
print(f'\n=== TOOLS: {len(t.get("tools",[]))} ===')
for tool in t['tools']:
    print(f'  - {tool["name"]}')

# Costs
c = g('/api/costs')
print(f'\n=== COSTS ===')
print(json.dumps(c, ensure_ascii=False)[:300])

# Observability
o = g('/api/observability')
print(f'\n=== OBSERVABILITY ===')
print(f'Counters: {json.dumps(o.get("counters",{}), ensure_ascii=False)}')

print('\nAll endpoints working! Dashboard ready at http://localhost:8000/dashboard.html')
