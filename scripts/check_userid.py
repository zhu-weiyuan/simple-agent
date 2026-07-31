# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
from my_agent.auth import derive_user_id

uid = derive_user_id('zwy')
print('derive_user_id(zwy):', uid)

import sqlite3
conn = sqlite3.connect('conversations.db')
cur = conn.cursor()
cur.execute("SELECT id, username FROM users WHERE username=?", ('zwy',))
row = cur.fetchone()
if row:
    print('DB user id:', row[0], 'username:', row[1])
    print('Match:', row[0] == uid)
else:
    print('No zwy user in DB')
conn.close()
