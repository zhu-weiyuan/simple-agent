import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.environ["OPENAI_API_KEY"] = "your_key_here"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8080"
os.environ["OPENAI_MODEL"] = "Qwen3.6-27B-IQ4_NL.gguf"

from my_agent import SimpleAgent

agent = SimpleAgent()
questions = [
    "你好，请简单介绍一下",
    "现在几点了？",
    "计算 123 * 456",
]

results = []
for q in questions:
    try:
        r = agent.run(q)
        results.append({"q": q, "status": "ok", "len": len(r)})
        print(f"OK [{len(r)} chars]: {q}")
    except Exception as e:
        results.append({"q": q, "status": "error", "msg": str(e)[:100]})
        print(f"ERR: {q} -> {e}")

agent.close()
print(json.dumps(results, ensure_ascii=False, indent=2))
print("DONE")
