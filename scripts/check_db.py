# -*- coding: utf-8 -*-
"""Check SQLite DB contents"""
import sys, json, urllib.request, sqlite3
sys.stdout.reconfigure(encoding='utf-8')

# Check DB directly
conn = sqlite3.connect('conversations.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('Tables:', tables)
for t in tables:
    cur.execute("SELECT count(*) FROM {}".format(t))
    cnt = cur.fetchone()[0]
    print('  {}: {} rows'.format(t, cnt))

# Check session table  
if 'sessions' in tables:
    cur.execute("SELECT id, user_id, message_count, title FROM sessions ORDER BY created_at DESC LIMIT 5")
    rows = cur.fetchall()
    print('\nRecent sessions:')
    for r in rows:
        print('  {} | user={} | msgs={} | title={}'.format(str(r[0])[:30], r[1], r[2], str(r[3])[:40] if r[3] else ''))

conn.close()

# Check API
H = {'X-API-Key': '***'}
try:
    r = urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/sqlite/sessions', headers=H), timeout=10)
    d = json.loads(r.read())
    sessions = d.get('sessions', d) if isinstance(d, dict) else d
    print('\nAPI sessions: {}'.format(len(sessions) if isinstance(sessions, list) else '?'))
except Exception as e:
    print('API error:', str(e)[:80])
