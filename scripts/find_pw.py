# -*- coding: utf-8 -*-
"""Test which password matches the stored hash."""
import sqlite3, hashlib, hmac
from my_agent.auth import hash_password

conn = sqlite3.connect('conversations.db')
row = conn.execute('SELECT password_hash FROM users WHERE username=?', ('zwy',)).fetchone()
pw = row[0]
print('Stored hash:', pw)

parts = pw.split('$')
salt = bytes.fromhex(parts[2])
expected = bytes.fromhex(parts[3])
iters = int(parts[1])
print('Salt:', salt.hex())
print('Expected:', expected.hex())
print('Iterations:', iters)
print()

candidates = ['test123', 'zwy123', 'Test123', 'Test123!', '123456', 'admin123', 'password123', 'zwy', '']
for pwd in candidates:
    dk = hashlib.pbkdf2_hmac('sha256', pwd.encode('utf-8'), salt, iters)
    match = hmac.compare_digest(dk, expected)
    print(f'  {pwd!r}: {match}')
    if match:
        print('>>> FOUND!')

# Generate a fresh hash so we can compare
fresh = hash_password('test123')
print('\nFresh hash for test123:', fresh)
conn.close()
