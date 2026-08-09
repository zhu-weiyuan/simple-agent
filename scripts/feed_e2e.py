# -*- coding: utf-8 -*-
"""Start simple-agent with DeepSeek, feed E2E data for observability"""
import subprocess, os, sys, time, json, urllib.request, urllib.error

BASE = r'C:\Users\Administrator\.openclaw\workspace1\simple-agent'
os.chdir(BASE)

# DeepSeek config
DS_KEY = 'sk-fa769b6617ea4f5d9fdb5532c5a0e05a'
DS_BASE = 'https://api.deepseek.com/v1'
DS_MODEL = 'deepseek-chat'

env = os.environ.copy()
env['OPENAI_API_KEY'] = DS_KEY
env['OPENAI_BASE_URL'] = DS_BASE
env['OPENAI_MODEL'] = DS_MODEL
env['API_KEYS'] = 'test-key-123'
env['JWT_SECRET'] = 'test-jwt'
env['RATE_LIMIT_REQUESTS'] = '300'
env['RATE_LIMIT_WINDOW'] = '60'

# Start server
proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000', '--workers', '1'],
    cwd=BASE,
    env=env,
    stdout=open('server_stdout.log', 'w', encoding='utf-8'),
    stderr=open('server_stderr.log', 'w', encoding='utf-8'),
)
print('Server PID:', proc.pid)
time.sleep(10)

# Health check
def api(method, path, data=None, key='test-key-123'):
    hdrs = {'Content-Type': 'application/json', 'X-API-Key': key}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(f'http://localhost:8000{path}', data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {'_err': e.code, '_body': e.read().decode()[:200]}
    except Exception as e:
        return {'_err': str(e)[:80]}

# Test health
h = api('GET', '/healthz')
print('Health:', h.get('status'), '| LLM reachable:', h.get('llm', {}).get('reachable'))

# ===== Feed E2E data =====
test_prompts = [
    "你好，我想查询今天的天气",
    "帮我算一下 125 * 37 等于多少",
    "你能读取桌面上的 test.txt 文件吗",
    "你好，请介绍一下你自己有什么功能",
    "现在几点了",
    "帮我看看当前目录有哪些文件",
    "你好啊，今天心情不错",
    "Python中怎么用列表推导式生成1到10的平方",
    "你能帮我算一下 (15 + 27) * 3 等于多少",
    "我想知道当前的系统时间",
    "帮我查一下昨天的日志文件",
    "用中文写一首关于秋天的短诗",
    "1加到100等于多少",
    "翻译'Hello World'成中文",
    "帮我看看C盘根目录有哪些文件夹",
    "什么是FastAPI，有什么用",
    "Linux和Windows有什么区别",
    "写一个Python函数来计算斐波那契数列",
    "今天星期几？",
    "用Python读取CSV文件的标准步骤是什么",
    "123456789 * 987654321 等于多少",
    "帮我列一下当前目录下所有.py文件",
    "用中文解释什么是API",
    "我会写Python，帮我出三道面试题",
    "什么是Docker，它解决了什么问题",
    "帮我看看我的项目文件结构",
    "请用中文写一首现代诗",
    "现在北京时间是多少",
    "sqrt(144) 等于多少",
    "你能帮我写一个简单的Flask应用吗",
]

print(f"\n=== Feeding {len(test_prompts)} requests ===")
success = 0
errors = 0
for i, prompt in enumerate(test_prompts):
    try:
        chat = api('POST', '/api/chat', {'message': prompt, 'stream': False})
        if '_err' in chat:
            errors += 1
            print(f'  [{i+1}] ERR {chat["_err"]}: {prompt[:30]}')
            if errors >= 3:
                print('Too many errors, stopping feed')
                break
        else:
            success += 1
            reply = chat.get('reply', '')
            print(f'  [{i+1}] ✅ {prompt[:35]} → {reply[:45]}')
        time.sleep(0.3)  # rate limit
    except Exception as e:
        errors += 1
        print(f'  [{i+1}] EXC: {str(e)[:50]}')

print(f"\n=== Results: {success} success, {errors} errors ===")

# Check observability data
print("\n=== Observability Dashboard Data ===")
obs = api('GET', '/api/observability')
if '_err' in obs:
    print(f'Error: {obs}')
else:
    cs = obs.get('counters', {})
    hs = obs.get('histograms', {})
    print(f'Counters: {json.dumps(cs, indent=2, ensure_ascii=False)}')
    for k, v in hs.items():
        print(f'Histo [{k}]: count={v.get("count")}, avg_ms={v.get("avg_ms")}')

# Check metrics endpoint
print("\n=== /api/metrics (first 20 lines) ===")
metrics_resp = api('GET', '/api/metrics')
if isinstance(metrics_resp, dict) and '_err' in metrics_resp:
    print(f'Error: {metrics_resp["_err"]}')
else:
    lines = metrics_resp.get('text', str(metrics_resp)) if isinstance(metrics_resp, dict) else str(metrics_resp)
    for line in str(lines).split('\n')[:25]:
        print(line)

# Check health for session/system data
print("\n=== Health (system + sessions) ===")
h = api('GET', '/api/health')
print(f'Sessions: {h.get("sessions", {})}')
print(f'System CPU: {h.get("system", {}).get("cpu_percent")}%')
print(f'LLM: {h.get("llm", {}).get("model")} reachable={h.get("llm", {}).get("reachable")}')
