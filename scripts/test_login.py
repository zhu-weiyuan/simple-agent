# -*- coding: utf-8 -*-
"""Test login endpoint"""
import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')

# Test login
body = json.dumps({'username':'zwy','password':'test123'}).encode()
r = urllib.request.Request('http://localhost:8000/api/auth/login', data=body,
    headers={'Content-Type':'application/json'}, method='POST')
resp = json.loads(urllib.request.urlopen(r, timeout=10).read())
print('Login response:')
for k, v in resp.items():
    val = str(v)[:30]
    print('  {}: {}'.format(k, val))

tok = resp.get('token')
uid = resp.get('user_id')
print('\nToken valid:', bool(tok))

# Test sessions with JWT
h = {'Authorization': 'Bearer ' + tok, 'X-User-Id': uid}
r = urllib.request.Request('http://localhost:8000/api/sqlite/sessions', headers=h)
sessions = json.loads(urllib.request.urlopen(r, timeout=10).read())
slist = sessions.get('sessions', [])
print('Sessions: {}'.format(len(slist)))
for s in slist[:3]:
    print('  {} | user={} | msgs={}'.format(
        str(s.get('session_id',''))[:30],
        str(s.get('user_id',''))[:20],
        s.get('message_count')))

# Also verify with X-API-Key
h2 = {'X-API-Key': '***'}
r2 = urllib.request.Request('http://localhost:8000/api/sqlite/sessions', headers=h2)
s2 = json.loads(urllib.request.urlopen(r2, timeout=10).read())
slist2 = s2.get('sessions', [])
print('With API key: {} sessions'.format(len(slist2)))
