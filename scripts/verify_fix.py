# -*- coding: utf-8 -*-
"""Verify fixed endpoints and feed E2E data"""
import sys, json, urllib.request, urllib.error, time
sys.stdout.reconfigure(encoding='utf-8')

KEY = 'test-key-123'
BASE = 'http://localhost:8000'

def g(path, timeout=10):
    r = urllib.request.Request(BASE + path, headers={'X-API-Key': KEY})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def api(method, path, data=None, timeout=300):
    hdrs = {'Content-Type': 'application/json', 'X-API-Key': KEY}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(BASE + path, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {'_err': e.code, '_body': e.read().decode()[:200]}
    except Exception as e:
        return {'_err': type(e).__name__, '_msg': str(e)[:80]}

# 1. Test all endpoints
print('=== /api/health ===')
h = g('/api/health')
llm_model = h.get('llm',{}).get('model','')[:50]
print(f'Model: {llm_model}')
print(f'Reachable: {h.get("llm",{}).get("reachable")}')
print(f'Sessions: {h.get("sessions",{})}')
print(f'Inflight: {h.get("inflight")}')
print(f'Requests: {h.get("requests",{})}')

print('\n=== /api/tools ===')
t = g('/api/tools')
tool_list = t.get('tools', [])
print(f'{len(tool_list)} tools')
for tool in tool_list:
    desc = tool.get('description', '')[:60]
    print(f'  - {tool["name"]}: {desc}')

print('\n=== /api/costs ===')
c = g('/api/costs')
print(json.dumps(c, ensure_ascii=False, indent=2)[:400])

print('\n=== /api/observability ===')
o = g('/api/observability')
print(f'Counters: {json.dumps(o.get("counters",{}), ensure_ascii=False)}')
for k, v in o.get('histograms',{}).items():
    print(f'  {k}: count={v.get("count")} avg={v.get("avg_ms"):.0f}ms')

# 2. Feed some requests via local LLM
print('\n=== Feeding with local LLM ===')
prompts = [
    "你好，帮我看看当前目录有哪些文件",
    "请介绍一下你自己的功能",
    "帮我算一下 15 * 37 等于多少",
    "现在几点了",
]
success = 0
for i, prompt in enumerate(prompts):
    start = time.time()
    chat = api('POST', '/api/chat', {'message': prompt})
    elapsed = time.time() - start
    if '_err' in chat:
        print(f'  [{i+1}] ❌ {chat.get("_err")}: {prompt[:30]} ({elapsed:.0f}s)')
    else:
        success += 1
        reply = chat.get('reply','')
        print(f'  [{i+1}] ✅ {prompt[:35]} -> {reply[:50]} ({elapsed:.0f}s)')

# 3. Final check
print(f'\n=== {success}/{len(prompts)} succeeded ===')
o2 = g('/api/observability')
print(f'Chat requests: {o2.get("counters",{}).get("chat_requests",0)}')
for k, v in o2.get('histograms',{}).items():
    print(f'  {k}: count={v.get("count")} avg={v.get("avg_ms"):.0f}ms')

print('\nDone. Open http://localhost:8000/dashboard.html')
