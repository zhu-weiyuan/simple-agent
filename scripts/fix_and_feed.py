# -*- coding: utf-8 -*-
"""Start simple-agent with local llama.cpp, then feed E2E data"""
import subprocess, os, sys, time, json, urllib.request, urllib.error

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

# Kill any existing server on :8000
import subprocess
r = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if ':8000' in line and 'LISTENING' in line:
        pid = line.strip().split()[-1]
        print(f'Killing PID {pid} on :8000')
        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
        time.sleep(2)

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000', '--workers', '1'],
    cwd=BASE,
    env=env,
    stdout=open('server_stdout.log', 'w', encoding='utf-8'),
    stderr=open('server_stderr.log', 'w', encoding='utf-8'),
)
print('Server PID:', proc.pid)
time.sleep(8)

def api(method, path, data=None, timeout=120):
    hdrs = {'Content-Type': 'application/json', 'X-API-Key': '***'}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(f'http://localhost:8000{path}', data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        return {'_err': e.code, '_body': body}
    except Exception as e:
        return {'_err': type(e).__name__, '_msg': str(e)[:80]}

# Health check
h = api('GET', '/api/health', timeout=5)
print('Health:', h.get('status') or h.get('ok'), '| Model:', h.get('llm',{}).get('model','?')[:50])

# Test tools endpoint
t = api('GET', '/api/tools', timeout=5)
if '_err' in t:
    print(f'/api/tools: {t["_err"]}')
else:
    print(f'/api/tools: {len(t.get("tools",[]))} tools registered')

# Test costs endpoint
c = api('GET', '/api/costs', timeout=5)
if '_err' in c:
    print(f'/api/costs: {c["_err"]}')
else:
    print(f'/api/costs: {json.dumps(c, ensure_ascii=False)[:150]}')

# Quick observability check
o = api('GET', '/api/observability', timeout=5)
if '_err' not in o:
    print(f'/api/observability: counters={o.get("counters",{})}')

# Feed E2E prompts (use local LLM - will be slow but it's what user wants)
prompts = [
    "你好，帮我看看当前目录有哪些文件",
    "请介绍一下你自己",
    "现在几点了",
    "帮我算一下 15 * 37",
    "1加到100等于多少",
    "用中文写一首关于秋天的短诗",
    "什么是FastAPI",
    "帮我读取README.md文件",
    "我项目里有哪些.py文件",
    "帮我看看我的代码有多少行",
    "你有什么功能",
    "帮我查看桌面上的项目文档",
]

success, errors = 0, 0
print(f'\nFeeding {len(prompts)} requests with local LLM...')
print('(Local 35B is slow, each call may take 10-30s)')
for i, prompt in enumerate(prompts):
    start = time.time()
    chat = api('POST', '/api/chat', {'message': prompt, 'stream': False}, timeout=300)
    elapsed = time.time() - start
    if '_err' in chat:
        errors += 1
        print(f'  [{i+1}] ❌ {chat["_err"]}: {prompt[:30]} ({elapsed:.0f}s)')
        if errors >= 3:
            print('Too many errors, stopping')
            break
    else:
        success += 1
        reply = chat.get('reply', '')
        print(f'  [{i+1}] ✅ {prompt[:35]} → {reply[:40]}... ({elapsed:.0f}s)')

print(f'\n=== {success} success, {errors} errors ===')

# Final observability check
print('\n=== Final Dashboard Data ===')
h = api('GET', '/api/health', timeout=5)
print(f'Agent: {h.get("agent")} v{h.get("version")}')
print(f'LLM: {h.get("llm",{}).get("model","")[:50]} reachable={h.get("llm",{}).get("reachable")}')
print(f'Sessions: {h.get("sessions",{})}')
print(f'Requests: {h.get("requests",{})}')
print(f'System: CPU={h.get("system",{}).get("cpu_percent")}% Mem={h.get("system",{}).get("memory_percent")}%')

o = api('GET', '/api/observability', timeout=5)
if '_err' not in o:
    print(f'Counters: {json.dumps(o.get("counters",{}), ensure_ascii=False)}')
    for k, v in o.get('histograms',{}).items():
        print(f'  {k}: count={v.get("count")} avg={v.get("avg_ms"):.0f}ms')

print('\n🎉 Open http://localhost:8000/dashboard.html to see the dashboard')
