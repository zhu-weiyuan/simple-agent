# -*- coding: utf-8 -*-
"""Feed E2E data to running simple-agent (server already up on :8000)"""
import sys, json, urllib.request, urllib.error, time
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://localhost:8000'

def api(method, path, data=None):
    hdrs = {'Content-Type': 'application/json', 'X-API-Key': 'test-key-123'}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(f'{BASE}{path}', data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {'_err': e.code, '_body': e.read().decode()[:200]}
    except Exception as e:
        return {'_err': type(e).__name__, '_msg': str(e)[:80]}

# Check health
h = api('GET', '/api/health')
print(f'Health: agent={h.get("agent")} | LLM={h.get("llm",{}).get("reachable")}')

# E2E prompts
prompts = [
    # File browsing (demonstrates list_files)
    "帮我看看当前目录有哪些文件",
    "C盘根目录有哪些文件夹，帮我列出来",
    "我的项目目录下有哪些文件",
    # General questions
    "你好，请介绍一下你自己",
    "现在几点了",
    "帮我算一下 15 * 37 等于多少",
    "1加到100等于多少",
    "sqrt(144) 等于多少",
    "用中文写一首关于秋天的短诗",
    "什么是FastAPI，它有什么用",
    # Tool usage
    "帮我读取当前目录下的README.md文件",
    "你能执行PowerShell命令吗？帮我看看进程列表",
    "翻译Hello World成中文",
    "写一个Python函数来计算斐波那契数列",
    # Business scenarios
    "帮我查看桌面上的项目文档",
    "我项目里有哪些.py文件",
    "帮我计算一下如果每天学2小时，一个月能学多少小时",
    "你能读取config.json文件吗",
    "帮我算一下25的平方根是多少",
    "给我讲个笑话",
    # More file browsing
    "我想看看我的文档文件夹里有什么",
    "当前目录下有没有日志文件",
    # Knowledge
    "解释一下什么是REST API",
    "Python和Java的主要区别是什么",
    "Docker和虚拟机的区别是什么",
    "什么是Git，常用的命令有哪些",
    # Quick queries
    "你好啊",
    "今天天气怎么样？帮我查一下",
    "帮我看看我的代码有多少行",
    "你有什么功能",
]

success, errors = 0, 0
print(f'\nFeeding {len(prompts)} requests...')
for i, prompt in enumerate(prompts):
    try:
        chat = api('POST', '/api/chat', {'message': prompt, 'stream': False})
        if '_err' in chat:
            errors += 1
            if errors >= 5:
                print(f'  [{i+1}] ❌ {chat["_err"]}: {prompt[:30]} (abort)', flush=True)
                break
            print(f'  [{i+1}] ❌ {chat["_err"]}: {prompt[:30]}', flush=True)
        else:
            success += 1
            reply = chat.get('reply', '')
            print(f'  [{i+1}] ✅ {prompt[:35]} → {reply[:50]}', flush=True)
        time.sleep(0.5)
    except Exception as e:
        errors += 1
        print(f'  [{i+1}] ⚠ {str(e)[:50]}', flush=True)

print(f'\n=== Done: {success} success, {errors} errors ===')

# Show observability data
print('\n=== Observability Data ===')
obs = api('GET', '/api/observability')
if '_err' in obs:
    print(f'Error: {obs}')
else:
    print(f'Counters: {json.dumps(obs.get("counters",{}), ensure_ascii=False, indent=2)}')
    for k, v in obs.get('histograms',{}).items():
        print(f'  Histo [{k}]: count={v.get("count")} avg={v.get("avg_ms")}ms')

print('\n=== Active Sessions from Health ===')
h = api('GET', '/api/health')
print(f'Sessions: {json.dumps(h.get("sessions",{}), ensure_ascii=False)}')
print(f'Requests: {json.dumps(h.get("requests",{}), ensure_ascii=False)}')
print(f'System: CPU={h.get("system",{}).get("cpu_percent")}% Mem={h.get("system",{}).get("memory_percent")}%')
print(f'LLM reachable: {h.get("llm",{}).get("reachable")}')

print('\n🎉 Done! Open http://localhost:8000/dashboard.html to see the observability panel')
