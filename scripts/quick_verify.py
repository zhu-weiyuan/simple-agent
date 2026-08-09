# -*- coding: utf-8 -*-
import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
H = {'X-API-Key': 'test-key-123'}

for path in ['/api/health','/api/tools','/api/costs','/api/observability']:
    r = urllib.request.Request('http://localhost:8000' + path, headers=H)
    resp = json.loads(urllib.request.urlopen(r, timeout=15).read())
    if path == '/api/health':
        s = resp.get('sessions',{})
        l = resp.get('llm',{})
        print('HEALTH: sessions=%s, llm_reachable=%s' % (s, l.get('reachable')))
    elif path == '/api/tools':
        tools = [t['name'] for t in resp.get('tools',[])]
        print('TOOLS: %d registered: %s' % (len(tools), tools))
    elif path == '/api/costs':
        print('COSTS: by_model=%s' % list(resp.get('by_model',{}).keys()))
    elif path == '/api/observability':
        print('OBSERV: counters=%s' % resp.get('counters',{}))
print('ALL OK')
