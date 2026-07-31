# -*- coding: utf-8 -*-
"""Quick check + feed E2E data"""
import sys, json, urllib.request, urllib.error, time
sys.stdout.reconfigure(encoding='utf-8')
KEY = 'test-key-123'

def g(path, timeout=10):
    r = urllib.request.Request('http://localhost:8000' + path, headers={'X-API-Key': KEY})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def post(path, data, timeout=300):
    hdrs = {'Content-Type': 'application/json', 'X-API-Key': KEY}
    req = urllib.request.Request('http://localhost:8000' + path, data=json.dumps(data).encode(), headers=hdrs, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {'_err': e.code, '_body': e.read().decode()[:200]}
    except Exception as e:
        return {'_err': str(e)[:80]}

# 1. Health
h = g('/api/health', timeout=10)
print('=== HEALTH ===')
print(f'LLM: {h.get("llm",{}).get("model","")[:40]}')
print(f'Reachable: {h.get("llm",{}).get("reachable")}')
print(f'Sessions: {h.get("sessions")}')
print(f'Inflight: {h.get("inflight")}')
print(f'Requests: {h.get("requests")}')

# 2. Tools
print('\n=== TOOLS ===')
t = g('/api/tools', timeout=10)
print(f'{len(t.get("tools",[]))} tools:')
for tool in t['tools']:
    print(f'  {tool["name"]}')

# 3. Costs
print('\n=== COSTS ===')
c = g('/api/costs', timeout=10)
print(json.dumps(c, ensure_ascii=False)[:300])

# 4. Observability
print('\n=== OBSERVABILITY ===')
o = g('/api/observability', timeout=10)
print(f'Counters: {json.dumps(o.get("counters",{}), ensure_ascii=False)}')
hist = o.get('histograms', {})
for k, v in hist.items():
    print(f'  {k}: count={v.get("count")}, avg_ms={v.get("avg_ms")}')

# 5. Feed data via local LLM (slow but user wants it)
print('\n=== FEEDING (local LLM, slow) ===')
prompts = [
    "你好，帮我看看当前目录有哪些文件",
    "请介绍一下你自己",
    "现在几点了，帮我算一下 15*37",
    "帮我读取README.md文件",
]
suc = 0
for i, p in enumerate(prompts):
    start = time.time()
    r = post('/api/chat', {'message': p, 'stream': False}, timeout=300)
    elapsed = time.time() - start
    if '_err' in r:
        print(f'  [{i+1}] ERR {r["_err"]}: {p[:30]} ({elapsed:.0f}s)')
    else:
        suc += 1
        reply = r.get('reply', '')
        print(f'  [{i+1}] OK {p[:30]} -> {reply[:40]} ({elapsed:.0f}s)')
    time.sleep(1)

print(f'\n{suc}/{len(prompts)} done')

# Final observability
o2 = g('/api/observability', timeout=10)
print(f'\nFinal counters: {json.dumps(o2.get("counters",{}), ensure_ascii=False)}')
for k, v in o2.get('histograms',{}).items():
    print(f'  {k}: count={v.get("count")}, avg_ms={v.get("avg_ms")}')
