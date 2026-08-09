# -*- coding: utf-8 -*-
"""Feed a few requests to generate observability data"""
import sys, json, urllib.request, urllib.error, time
sys.stdout.reconfigure(encoding='utf-8')
H = {'X-API-Key': 'test-key-123'}

def post(path, data, timeout=300):
    req = urllib.request.Request('http://localhost:8000' + path,
        data=json.dumps(data).encode(), headers={**H, 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {'_err': e.code, '_body': e.read().decode()[:200]}
    except Exception as e:
        return {'_err': str(e)[:80]}

prompts = [
    "你好，请帮我看看当前目录有哪些文件",
    "现在几点了",
    "帮我算一下 15 * 37 等于多少",
]

suc = 0
for i, p in enumerate(prompts):
    print('[%d/3] %s ...' % (i+1, p[:30]), flush=True)
    start = time.time()
    chat = post('/api/chat', {'message': p, 'stream': False})
    elapsed = time.time() - start
    if '_err' in chat:
        print('  ERROR %s (%ds)' % (chat.get('_err'), elapsed), flush=True)
    else:
        suc += 1
        reply = chat.get('reply', '')
        print('  OK (%ds) -> %s' % (elapsed, reply[:60]), flush=True)

# Show results
import urllib.request
o = json.loads(urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/observability', headers=H), timeout=5).read())
print('\n=== Observability After Feed ===')
print('Counters:', json.dumps(o.get('counters',{}), ensure_ascii=False))
print('Histograms:', json.dumps({k:{'count':v['count']} for k,v in o.get('histograms',{}).items()}, ensure_ascii=False))
print('Done! Dashboard at http://localhost:8000/dashboard.html')
