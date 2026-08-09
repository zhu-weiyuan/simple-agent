# -*- coding: utf-8 -*-
import sys
src = open('app.py','r',encoding='utf-8').read()
checks = ['/api/tools','/api/costs','active_session','inflight','http://localhost:8080']
for c in checks:
    status = 'OK' if c in src else 'MISSING'
    print(f'{c}: {status}')
