# -*- coding: utf-8 -*-
"""Check messages to see original user data"""
import sys, sqlite3
sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('conversations.db')
cur = conn.cursor()

# Check sessions with their real user IDs
cur.execute("""
    SELECT s.id, s.user_id, s.message_count, s.created_at, u.username
    FROM sessions s
    LEFT JOIN users u ON s.user_id = u.id
    ORDER BY s.created_at DESC
    LIMIT 10
""")
rows = cur.fetchall()
print('All sessions (recent 10):')
for r in rows:
    print('  {} | userid={} | username={} | msgs={} | at={}'.format(
        str(r[0])[:30], str(r[1])[:24], r[4] or '-', r[2], str(r[3])[:19]))

# Check users table
cur.execute("SELECT id, username, created_at FROM users ORDER BY created_at")
users = cur.fetchall()
print('\nUsers:')
for u in users:
    print('  {} | username={} | created={}'.format(str(u[0])[:30], u[1] or '(none)', str(u[2])[:19] if u[2] else '?'))

# Sample messages
cur.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT 5")
msgs = cur.fetchall()
print('\nLast 5 messages:')
for m in msgs:
    print('  {}: {}'.format(m[0], str(m[1])[:80]))

conn.close()
