# -*- coding: utf-8 -*-
import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://localhost:8000'

tests = [
    ('正常内容', 'Python 是由 Guido van Rossum 于 1991 年发布的。'),
    ('明显幻觉', 'Python 3.12 强制要求所有变量声明类型，不声明就报错。'),
    ('过度自信', 'Python 将在 2027 年被 AI 完全取代，这是毫无疑问的。'),
    ('正常技术', 'Python 3.12 改进了错误消息和 f-string 语法。'),
]

for name, text in tests:
    r = requests.post(f'{BASE}/api/hallucination/detect', json={'text': text})
    d = r.json()
    icon = '🚨' if d['is_hallucination'] else '✅'
    print(f'{icon} {name}: hallucination={d["is_hallucination"]}, type={d["hallucination_type"]}, confidence={d["confidence"]}')
    if d.get('correction_suggestion'):
        print(f'   建议: {d["correction_suggestion"]}')
    print()
