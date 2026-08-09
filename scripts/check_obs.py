# -*- coding: utf-8 -*-
"""Verify observability data"""
import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
H = {'X-API-Key': 'test-key-123'}

def g(path):
    r = urllib.request.Request(f'http://localhost:8000{path}', headers=H)
    return json.loads(urllib.request.urlopen(r, timeout=5).read())

h = g('/api/health')
print('=== Health ===')
print(f'  Agent: {h.get("agent")} v{h.get("version")}')
print(f'  LLM: {h.get("llm",{}).get("model")} | Reachable: {h.get("llm",{}).get("reachable")}')
print(f'  Requests: {json.dumps(h.get("requests",{}))}')
print(f'  Uptime: {h.get("uptime_seconds")}s')

o = g('/api/observability')
print('\n=== Observability ===')
for k, v in sorted(o.get('counters',{}).items()):
    print(f'  📊 {k}: {v}')
for k, v in o.get('histograms',{}).items():
    print(f'  📈 {k}: count={v.get("count")}, avg={v.get("avg_ms")}ms, total={v.get("sum_ms")}ms')

r = urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/metrics', headers=H), timeout=5)
txt = r.read().decode('utf-8')
lines = [l for l in txt.split('\n') if l and not l.startswith('#')]
print('\n=== Prometheus Metrics ===')
for l in lines:
    print(f'  {l}')
