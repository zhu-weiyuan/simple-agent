import sqlite3

con = sqlite3.connect("prompts.db")
cur = con.cursor()

print("== 所有表 ==")
for r in cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
    print("表:", r[0])
    print("  DDL:", r[1])
    print()

print("== prompts 表统计 ==")
print("总行数:", cur.execute("SELECT COUNT(*) FROM prompts").fetchone()[0])
print("唯一 content 数:", cur.execute("SELECT COUNT(DISTINCT content) FROM prompts").fetchone()[0])
print("version 范围:", cur.execute("SELECT MIN(version), MAX(version) FROM prompts").fetchone())
print("gray_release 分布:", cur.execute("SELECT gray_release, COUNT(*) FROM prompts GROUP BY gray_release").fetchall())
print("is_active 分布:", cur.execute("SELECT is_active, COUNT(*) FROM prompts GROUP BY is_active").fetchall())
print()
print("== 时间分布 ==")
for r in cur.execute("SELECT substr(created_at,1,10) d, COUNT(*) FROM prompts GROUP BY d ORDER BY d"):
    print(r[0], r[1])
print()
print("== 内容样例 ==")
for r in cur.execute("SELECT id, version, content, gray_release, created_at, is_active FROM prompts LIMIT 5"):
    print(r)
print()
print("== 内容长度 ==")
print("最短:", cur.execute("SELECT MIN(LENGTH(content)) FROM prompts").fetchone()[0])
print("最长:", cur.execute("SELECT MAX(LENGTH(content)) FROM prompts").fetchone()[0])
print("平均:", round(cur.execute("SELECT AVG(LENGTH(content)) FROM prompts").fetchone()[0], 1))
