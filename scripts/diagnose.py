# -*- coding: utf-8 -*-
"""
scripts/diagnose.py — simple-agent 一键真实环境诊断（启动服务器之前运行）。
用法（Windows，项目根目录）:
    python scripts\\diagnose.py
    python scripts\\diagnose.py --server http://127.0.0.1:8000   # 附带在线服务检查
设计约束：
  * 仅用 stdlib + 项目已有依赖；所有三方 import 全部 try/except 守卫。
  * 任何单项检查失败都不会让脚本崩溃 — 输出 [FAIL]/[SKIP] + 中文修复提示。
  * 退出码 = FAIL 数（上限 250），供 CI / 批处理判断。
"""
from __future__ import annotations
import argparse, asyncio, importlib, io, json, logging, os, shutil  # noqa: E401
import sqlite3, sys, tempfile, time, traceback, urllib.error, urllib.request  # noqa: E401
# ── 项目根目录定位 + cwd 归一（conversations.db 等相对路径依赖 cwd）──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = os.path.join(ROOT, "src")
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 输出基础设施：ANSI 颜色（Win10+ 启用 VT）、结果收集  # ══════
def _enable_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:  # Win10+ 打开虚拟终端转义序列
        import ctypes
        k32 = ctypes.windll.kernel32
        h, mode = k32.GetStdHandle(-11), ctypes.c_uint32()
        if not k32.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k32.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False
_ANSI = _enable_ansi()
_COLORS = {"PASS": "\x1b[32m", "FAIL": "\x1b[31m", "WARN": "\x1b[33m", "SKIP": "\x1b[36m"}
RESULTS: list = []          # (status, group, name, detail)
_GROUP_ORDER = ["环境", "依赖", "模块导入", "数据层", "外部服务", "引擎冒烟", "HTTP层"]
def _emit(status: str, group: str, name: str, detail: str = "", hint: str = "") -> None:
    RESULTS.append((status, group, name, detail))
    tag = f"[{status}]"
    if _ANSI:
        tag = f"{_COLORS.get(status, '')}{tag}\x1b[0m"
    line = f"  {tag} {name}"
    if detail:
        line += f" — {detail}"
    if hint:
        line += f" (修复: {hint})"
    try:
        print(line)
    except UnicodeEncodeError:  # Windows GBK 控制台兜底
        print(line.encode("gbk", errors="replace").decode("gbk"))
def ok(g, n, d=""):   _emit("PASS", g, n, d)
def fail(g, n, d="", hint=""): _emit("FAIL", g, n, d, hint)
def warn(g, n, d="", hint=""): _emit("WARN", g, n, d, hint)
def skip(g, n, d="", hint=""): _emit("SKIP", g, n, d, hint)
def section(title: str) -> None:
    print(f"\n=== {title} ===")
def guarded(group: str, name: str, hint: str = ""):
    """装饰器：单项检查隔离，异常 → FAIL（绝不让脚本崩溃）。"""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                fail(group, name, f"{type(exc).__name__}: {exc}", hint)
                return None
        return wrapper
    return deco
def compact_tb(exc: BaseException, frames: int = 4) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return " | ".join("".join(lines).strip().splitlines()[-frames:])
# A. 环境  # ══════
def load_dotenv_fallback(path: str) -> int:
    """手动解析 .env（python-dotenv 缺席时兜底），返回加载条数。"""
    count = 0
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value); count += 1
    return count
def _mask(v: str) -> str:
    return f"{v[:6]}...{v[-4:]}({len(v)}字符)" if len(v) > 12 else "***"
@guarded("环境", "A组整体")
def check_env() -> None:
    g = "环境"; section("A. 环境")
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        ok(g, "Python 版本", f"{v.major}.{v.minor}.{v.micro}")
    else:
        fail(g, "Python 版本", f"{v.major}.{v.minor} < 3.10", "安装 Python 3.10+（项目使用 3.10+ 语法）")
    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        fail(g, ".env 文件", "项目根目录不存在 .env", "创建 .env 并配置 OPENAI_API_KEY / OPENAI_BASE_URL 等（参考 DEPLOYMENT.md）")
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            ok(g, ".env 加载", "python-dotenv")
        except ImportError:
            n = load_dotenv_fallback(env_path)
            warn(g, ".env 加载", f"手动解析 {n} 条（python-dotenv 未装）", "pip install python-dotenv")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        fail(g, "OPENAI_API_KEY", "未设置或为空", "在 .env 里配置 OPENAI_API_KEY（本地 llama.cpp 可填 sk-local-dummy）")
    else:
        ok(g, "OPENAI_API_KEY", f"已设置 ({_mask(key)})")
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base:
        warn(g, "OPENAI_BASE_URL", "未设置，将用默认 http://localhost:8080/v1", "在 .env 里显式配置（须含 /v1）")
    elif not base.startswith(("http://", "https://")):
        fail(g, "OPENAI_BASE_URL", f"格式非法: {base}", "应形如 http://127.0.0.1:8080/v1（含协议头）")
    elif not base.rstrip("/").endswith("/v1"):
        warn(g, "OPENAI_BASE_URL", f"{base} 不以 /v1 结尾", "约定 base_url 含 /v1（llm 客户端在其后拼 /chat/completions）")
    else:
        ok(g, "OPENAI_BASE_URL", base)
    dev = os.environ.get("DEV_AUTH_BYPASS", "").strip()
    if dev in ("1", "true", "yes"):
        warn(g, "DEV_AUTH_BYPASS", "=1 开发模式已跳过鉴权", "上线前移除，改用 API_KEYS 鉴权")
    elif dev:
        ok(g, "DEV_AUTH_BYPASS", f"={dev}（未启用旁路）")
    else:
        skip(g, "DEV_AUTH_BYPASS", "未设置（按 API_KEYS 鉴权）")
    for var in ("OPENAI_MODEL", "LLM_PROFILE", "API_KEYS", "MY_AGENT_API_KEY",
                "MY_AGENT_BASE_URL", "MY_AGENT_MODEL", "WORKERS", "REQUEST_TIMEOUT_SECONDS"):
        val = os.environ.get(var, "").strip()
        if val:
            ok(g, f"可选 {var}", _mask(val) if var in ("API_KEYS", "MY_AGENT_API_KEY") else val)
        else:
            skip(g, f"可选 {var}", "未设置（使用内部默认值）")
# B. 依赖  # ══════
def _mod_version(name: str) -> str:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        try:
            return getattr(importlib.import_module(name), "__version__", "?")
        except Exception:
            return "?"
@guarded("依赖", "B组整体")
def check_deps() -> dict:
    g = "依赖"; section("B. 依赖")
    present: dict = {}
    required = [("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("httpx", "httpx"), ("dotenv", "python-dotenv")]
    optional = [("psutil", "psutil"), ("redis", "redis"), ("tiktoken", "tiktoken"), ("pytest", "pytest")]
    for mod, pkg in required:
        try:
            importlib.import_module(mod)
            present[mod] = True; ok(g, f"必需 {mod}", f"v{_mod_version(pkg)}")
        except Exception as exc:
            present[mod] = False
            fail(g, f"必需 {mod}", f"{type(exc).__name__}: {exc}", f"pip install {pkg}")
    for mod, pkg in optional:
        try:
            importlib.import_module(mod)
            present[mod] = True; ok(g, f"可选 {mod}", f"v{_mod_version(pkg)}")
        except Exception:
            present[mod] = False; skip(g, f"可选 {mod}", "未安装", f"需要时 pip install {pkg}")
    return present
# C. 模块导入  # ══════
_CORE_EXACT = {"my_agent", "my_agent.session_manager", "my_agent.cost_tracker",
               "my_agent.gateway", "my_agent.router", "my_agent.observability",
               "my_agent.metrics", "my_agent.llm", "my_agent.memory", "my_agent.memory.sqlite_store"}
_CORE_PREFIX = ("my_agent.core", "my_agent.tools", "my_agent.types")
def _is_core(mod: str) -> bool:
    return mod in _CORE_EXACT or any(mod == p or mod.startswith(p + ".") for p in _CORE_PREFIX)
@guarded("模块导入", "C组整体")
def check_imports(deps: dict) -> bool:
    g = "模块导入"; section("C. 模块导入 (src/my_agent/**.py + app_prod)")
    core_ok = True
    mods = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(SRC, "my_agent")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            mod = os.path.relpath(os.path.join(dirpath, fn), SRC)[:-3].replace(os.sep, ".")
            mods.append(mod[:-len(".__init__")] if mod.endswith(".__init__") else mod)
    for mod in sorted(set(mods)):
        label = "核心" if _is_core(mod) else "外围"
        try:
            importlib.import_module(mod)
            ok(g, f"[{label}] {mod}")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if label == "核心":
                core_ok = False
                fail(g, f"[核心] {mod}", detail, "核心模块必须可导入；缺依赖则 pip install，语法错误按提示修改")
            else:
                warn(g, f"[外围] {mod}", detail, "外围/遗留模块（sentiment/summary/bridge/a2a 等），缺依赖不阻塞核心链路")
    if deps.get("fastapi"):
        try:
            importlib.import_module("app_prod")
            ok(g, "app_prod")
        except Exception as exc:
            core_ok = False
            fail(g, "app_prod", compact_tb(exc, 3), "服务入口无法导入，服务器将无法启动")
    else:
        skip(g, "app_prod", "fastapi 未安装，跳过", "pip install fastapi uvicorn")
    # 诊断脚本静默 my_agent 的 INFO 日志（LRU 驱逐等噪音）
    logging.getLogger("my_agent").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    return core_ok
# D. 数据层  # ══════
_TMP_SCHEMA = ("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT DEFAULT '',"
               " created_at TEXT NOT NULL DEFAULT (datetime('now')),"
               " updated_at TEXT NOT NULL DEFAULT (datetime('now')),"
               " message_count INTEGER DEFAULT 0, metadata TEXT DEFAULT '{}');"
               "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
               " session_id TEXT NOT NULL,"
               " role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),"
               " content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')),"
               " metadata TEXT DEFAULT '{}', sequence_no INTEGER NOT NULL DEFAULT 0,"
               " FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,"
               " UNIQUE(session_id, sequence_no));")
@guarded("数据层", "D组整体")
def check_data(core_ok: bool) -> None:
    g = "数据层"; section("D. 数据层")
    # conversations.db schema 报告（只读，不写真实库）
    db_path = os.environ.get("CONVERSATIONS_DB", "conversations.db")
    if not os.path.isfile(db_path):
        skip(g, "conversations.db", f"{db_path} 尚不存在（首次启动会自动创建）")
    else:
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)")]; conn.close()
            seq = "含 sequence_no（旧版 schema，代码已兼容）" if "sequence_no" in cols else "无 sequence_no（新版 schema）"
            ok(g, "conversations.db", f"{db_path} 可读，{len(tables)} 张表"
               f"({', '.join(tables[:6])}{'...' if len(tables) > 6 else ''})；messages {seq}")
        except Exception as exc:
            fail(g, "conversations.db", f"{type(exc).__name__}: {exc}", "文件可能损坏或被占用，关闭占用进程或删除重建")
    # 临时库 roundtrip（模拟带 sequence_no UNIQUE 的旧 schema，绝不写真实库）
    tmpdir = tempfile.mkdtemp(prefix="diag_simple_agent_")
    try:
        Store = None
        try:
            from my_agent.memory.sqlite_store import SqliteConversationStore as Store
        except Exception as exc:
            skip(g, "SqliteConversationStore 回环", f"模块导入失败: {exc}", "先修复 C 组核心导入")
            skip(g, "SessionManager LRU", "依赖 SqliteConversationStore", "先修复 C 组核心导入")
        if Store is not None:
            try:
                tmp_db = os.path.join(tmpdir, "diag.db")
                conn = sqlite3.connect(tmp_db)
                conn.executescript(_TMP_SCHEMA); conn.commit(); conn.close()
                store = Store(tmp_db)
                snap1 = [{"role": "system", "content": "诊断系统提示"},
                         {"role": "user", "content": "你好"},
                         {"role": "assistant", "content": "你好！有什么可以帮你？"}]
                store.replace_session_messages("diag-s1", snap1)
                # 第二次整体替换：命中 sequence_no UNIQUE 约束的迁移路径
                store.replace_session_messages("diag-s1", snap1 + [{"role": "user", "content": "再见"}])
                store.add_message("diag-s1", "tool", "工具结果探针")
                rows = store.load_session_messages("diag-s1")
                roles = [r["role"] for r in rows]
                if len(rows) == 5 and roles == ["system", "user", "assistant", "user", "tool"] and rows[3]["content"] == "再见":
                    ok(g, "SqliteConversationStore 回环",
                       f"临时库(带 sequence_no UNIQUE) 快照替换×2 + add_message + 读回 {len(rows)} 条，顺序正确")
                else:
                    fail(g, "SqliteConversationStore 回环",
                         f"读回 {len(rows)} 条 roles={roles}（期望 5 条 system/user/assistant/user/tool）",
                         "检查 replace_session_messages / _next_seq 的 sequence_no 兼容逻辑")
                store.close()
            except Exception as exc:
                fail(g, "SqliteConversationStore 回环", compact_tb(exc, 3),
                     "sequence_no 兼容路径出错：检查 sqlite_store.py 的 _has_sequence_no 分支")
            # SessionManager get_or_create + LRU 驱逐冒烟（独立临时库）
            try:
                from my_agent.session_manager import SessionManager
                from my_agent.types.message import Message
                store2 = Store(os.path.join(tmpdir, "diag_sm.db"))
                sm = SessionManager("诊断系统提示", store=store2, max_sessions=2)
                for sid in ("diag-a", "diag-b", "diag-c"):
                    _, state = sm.get_or_create(sid)
                    state.append(Message.user(f"来自 {sid} 的消息"))
                evicted = store2.load_session_messages("diag-a")
                if sm.active_count != 2:
                    fail(g, "SessionManager LRU", f"active_count={sm.active_count}（期望 2）", "检查 _evict_if_needed")
                elif not evicted:
                    fail(g, "SessionManager LRU", "diag-a 被驱逐但未持久化到 store", "检查 _persist / replace_session_messages")
                else:
                    _, restored = sm.get_or_create("diag-a")
                    if any("diag-a" in (m.content or "") for m in restored.messages):
                        ok(g, "SessionManager LRU", f"3 会话/上限 2 → 驱逐并落库 {len(evicted)} 条，重取 diag-a 恢复成功")
                    else:
                        fail(g, "SessionManager LRU", "驱逐会话重新载入后消息丢失", "检查 _load_from_store 的 metadata.openai 解码")
                store2.close()
            except Exception as exc:
                fail(g, "SessionManager LRU", compact_tb(exc, 3), "检查 session_manager.py 与 types.session")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    # prompts.db
    pdb = os.environ.get("PROMPT_DB_PATH", "runtime/prompts.db")
    if not os.path.isfile(pdb):
        skip(g, "prompts.db", f"{pdb} 尚不存在（prompt_registry 首次使用时创建）")
    else:
        try:
            conn = sqlite3.connect(pdb, timeout=5)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]; conn.close()
            ok(g, "prompts.db", f"{pdb} 可读，{len(tables)} 张表")
        except Exception as exc:
            fail(g, "prompts.db", f"{type(exc).__name__}: {exc}", "文件可能损坏，删除后由 prompt_registry 重建")
# E. 外部服务  # ══════
def _http_post_json(url: str, payload: dict, headers: dict, timeout: float) -> tuple:
    """stdlib POST，返回 (status, body_dict_or_text)。"""
    headers = dict(headers or {}); headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except Exception:
                return resp.status, body[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:200]
@guarded("外部服务", "E组整体")
def check_services(deps: dict) -> bool:
    g = "外部服务"; section("E. 外部服务（网络探测）")
    llm_ok = False
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = (os.environ.get("OPENAI_BASE_URL", "").strip() or "http://localhost:8080/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not key:
        skip(g, "LLM chat ping", "OPENAI_API_KEY 未设置", "配置 .env 后重跑（本地 llama.cpp 可填 sk-local-dummy）")
    else:
        try:
            t0 = time.perf_counter()
            status, body = _http_post_json(
                f"{base}/chat/completions",
                {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "ping，回复 ok"}]},
                {"Authorization": f"Bearer {key}"},
                timeout=float(os.environ.get("DIAG_LLM_TIMEOUT", "30")))
            ms = (time.perf_counter() - t0) * 1000
            if status == 200:
                reply = ""
                if isinstance(body, dict):
                    m = (body.get("choices") or [{}])[0].get("message", {})
                    reply = (m.get("content") or m.get("reasoning_content") or "")[:40]
                llm_ok = True
                ok(g, "LLM chat ping", f"{model} {ms:.0f}ms 回复={reply!r}")
            elif status == 401:
                fail(g, "LLM chat ping", f"401 未授权 ({ms:.0f}ms)", "OPENAI_API_KEY 无效；本地 llama.cpp 通常不校验 key，云端需有效 key")
            elif status == 404:
                fail(g, "LLM chat ping", f"404 ({base}/chat/completions)", "OPENAI_BASE_URL 路径不对，确认以 /v1 结尾")
            else:
                fail(g, "LLM chat ping", f"HTTP {status}: {str(body)[:120]}", "检查 base_url / OPENAI_MODEL / 服务是否已启动")
        except Exception as exc:
            fail(g, "LLM chat ping", f"{type(exc).__name__}: {exc}",
                 "网络不通或超时(默认30s，本地大模型加载慢可 set DIAG_LLM_TIMEOUT=120)；确认 llama.cpp/网关已启动且端口正确")
    # Embeddings ping（仅在配置了 MY_AGENT_* 时探测）
    ekey = os.environ.get("MY_AGENT_API_KEY", "").strip()
    ebase = os.environ.get("MY_AGENT_BASE_URL", "").strip().rstrip("/")
    emodel = os.environ.get("MY_AGENT_MODEL", "").strip() or "Qwen/Qwen3-Embedding-4B"
    if not (ekey and ebase):
        skip(g, "Embeddings ping", "MY_AGENT_API_KEY / MY_AGENT_BASE_URL 未设置（embedding 功能不启用）")
    else:
        try:
            t0 = time.perf_counter()
            status, body = _http_post_json(f"{ebase}/embeddings", {"model": emodel, "input": ["诊断测试"]},
                                           {"Authorization": f"Bearer {ekey}"}, timeout=15)
            ms = (time.perf_counter() - t0) * 1000
            dim = 0
            if status == 200 and isinstance(body, dict):
                dim = len(((body.get("data") or [{}])[0]).get("embedding") or [])
            if dim:
                ok(g, "Embeddings ping", f"{emodel} 维度={dim} {ms:.0f}ms")
            else:
                fail(g, "Embeddings ping", f"HTTP {status}: {str(body)[:120]}", "检查 MY_AGENT_MODEL 是否被网关支持；401 则换 key")
        except Exception as exc:
            fail(g, "Embeddings ping", f"{type(exc).__name__}: {exc}", "检查 MY_AGENT_BASE_URL 连通性")
    # Redis（可选）
    if not deps.get("redis"):
        skip(g, "Redis PING", "redis 包未安装（可选，不启用则跳过）", "需要时 pip install redis")
    else:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis as redis_pkg
            redis_pkg.Redis.from_url(url, socket_connect_timeout=3, socket_timeout=3).ping()
            ok(g, "Redis PING", url)
        except Exception as exc:
            warn(g, "Redis PING", f"{url} 不可达: {exc}", "启动 redis-server 或配置 REDIS_URL；不启用亦可运行")
    return llm_ok
# F. 引擎冒烟  # ══════
@guarded("引擎冒烟", "F组整体")
def check_engine_smoke(core_ok: bool, llm_ok: bool) -> None:
    g = "引擎冒烟"; section("F. 引擎冒烟（QueryEngine.arun + 纯 mock 流式）")
    if not core_ok:
        skip(g, "mock 流式帧序", "C 组核心模块导入未通过", "先修复核心导入")
        skip(g, "engine.arun 真实调用", "C 组核心模块导入未通过", "先修复核心导入")
        return
    # 1) 纯 mock 流式检查（无需 LLM）：验证 token→done 帧序
    try:
        from my_agent.core.engine import QueryEngine
        from my_agent.types.session import SessionState
        async def fake_astream(messages, tools, model=None):
            for ch in ("你", "好"):
                yield {"type": "delta", "content": ch}
            yield {"type": "final", "content": "你好", "tool_calls": [], "usage": {"total_tokens": 5}}
        eng = QueryEngine(system_prompt="诊断助手")
        eng.set_async_llm(None, fake_astream)
        async def _collect():
            frames = []
            async for f in eng.arun_stream("测试", session=SessionState.create("诊断助手")):
                frames.append(f)
            return frames
        frames = asyncio.run(_collect())
        tokens = [f["token"] for f in frames if "token" in f]
        done = [f for f in frames if f.get("done")]
        order_ok = bool(done) and frames[-1].get("done") is True and len(done) == 1
        if order_ok and "".join(tokens) == "你好" and done[0].get("stop_reason") == "completed":
            ok(g, "mock 流式帧序", f"{len(tokens)} token 帧 → done 帧，content={done[0].get('content')!r} stop_reason=completed")
        else:
            fail(g, "mock 流式帧序", f"帧序异常: tokens={tokens} done={done[:1]}",
                 "检查 engine.arun_stream 的 delta/final/done 帧生成逻辑")
    except Exception as exc:
        fail(g, "mock 流式帧序", compact_tb(exc, 4), "检查 core/engine.py arun_stream")
    # 2) 真实 LLM 调用（仅在 E 组 LLM ping 通过时）
    if not llm_ok:
        skip(g, "engine.arun 真实调用", "E 组 LLM ping 未通过（会白等超时）", "先修复 LLM 连通性")
        return
    try:
        from my_agent.core.engine import QueryEngine
        from my_agent.llm import AsyncLLMClient
        from my_agent.session_manager import SessionManager
        client = AsyncLLMClient()
        eng = QueryEngine(system_prompt="你是诊断助手，回复必须简短。")
        eng.set_async_llm(client.achat, client.astream)
        _, sess = SessionManager("你是诊断助手，回复必须简短。").get_or_create("diag-engine-smoke")
        async def _run():
            try:
                return await asyncio.wait_for(eng.arun("你好", session=sess), timeout=60)
            finally:
                await client.aclose()
        t0 = time.perf_counter()
        result = asyncio.run(_run())
        ms = (time.perf_counter() - t0) * 1000
        content = (result.get("content") or "").strip()
        sr = result.get("stop_reason")
        if content and sr == "completed":
            ok(g, "engine.arun 真实调用",
               f"{ms:.0f}ms stop_reason={sr} usage={result.get('usage')} 回复 {len(content)} 字: {content[:40]!r}")
        else:
            fail(g, "engine.arun 真实调用", f"{ms:.0f}ms stop_reason={sr} 回复={content[:60]!r}",
                 "stop_reason 非 completed 或回复为空：查 LLM 返回与护栏日志")
    except asyncio.TimeoutError:
        fail(g, "engine.arun 真实调用", "60s 超时", "本地大模型太慢：预热模型或减小 max_tokens")
    except Exception as exc:
        fail(g, "engine.arun 真实调用", compact_tb(exc, 4), "按 traceback 定位 engine/llm 层错误")
# G. HTTP 层（--server 时启用）  # ══════
def validate_prometheus(text: str) -> list:
    """'# TYPE' 行不得含 '{'（标签写进 TYPE 是格式错误）。"""
    return [ln for ln in text.splitlines() if ln.startswith("# TYPE") and "{" in ln]
def _auth_headers() -> dict:
    keys = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]
    return {"X-API-Key": keys[0]} if keys else {}
def _server_get(base: str, path: str, timeout: float = 5.0) -> tuple:
    req = urllib.request.Request(base + path, headers=_auth_headers())
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (resp.status, resp.read().decode("utf-8", "replace"), (time.perf_counter() - t0) * 1000)
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode("utf-8", "replace")[:200], (time.perf_counter() - t0) * 1000)
_HINT_401 = "401 未授权：本地调试可在 .env 设 DEV_AUTH_BYPASS=1，或在 API_KEYS 配置 key 并随 X-API-Key 头发送"
@guarded("HTTP层", "G组整体")
def check_http(server: str) -> None:
    g = "HTTP层"; section(f"G. HTTP 层（{server}）")
    base = server.rstrip("/")
    for path in ("/healthz", "/api/ready", "/api/tools", "/api/conversations"):
        try:
            status, body, ms = _server_get(base, path)
            if status == 200:
                ok(g, f"GET {path}", f"200 {ms:.0f}ms")
            elif path == "/api/ready" and status == 503:
                fail(g, f"GET {path}", f"503 未就绪 {ms:.0f}ms: {body[:120]}", "看响应里哪个 check 为 false（storage/llm/config）")
            elif status == 401:
                fail(g, f"GET {path}", f"401 {ms:.0f}ms", _HINT_401)
            else:
                fail(g, f"GET {path}", f"HTTP {status} {ms:.0f}ms")
        except Exception as exc:
            fail(g, f"GET {path}", f"{type(exc).__name__}: {exc}", "服务器是否已启动？端口是否正确？（python app_prod.py）")
    try:
        status, body, ms = _server_get(base, "/api/metrics")
        bad = validate_prometheus(body) if status == 200 else []
        if status != 200:
            fail(g, "GET /api/metrics", f"HTTP {status}")
        elif bad:
            fail(g, "GET /api/metrics", f"TYPE 行含 '{{': {bad[0][:80]}", "metrics 渲染把标签写进 TYPE 行了")
        else:
            ok(g, "GET /api/metrics", f"200 {ms:.0f}ms，Prometheus 格式合法")
    except Exception as exc:
        fail(g, "GET /api/metrics", f"{type(exc).__name__}: {exc}")
    # POST /api/chat 非流式
    try:
        t0 = time.perf_counter()
        status, body = _http_post_json(base + "/api/chat", {"message": "你好", "stream": False}, _auth_headers(), timeout=90.0)
        ms = (time.perf_counter() - t0) * 1000
        if status == 200 and isinstance(body, dict) and (body.get("reply") or "").strip():
            ok(g, "POST /api/chat (非流式)", f"200 {ms:.0f}ms stop_reason={body.get('stop_reason')}")
        elif status == 401:
            fail(g, "POST /api/chat (非流式)", f"401 {ms:.0f}ms", _HINT_401)
        else:
            fail(g, "POST /api/chat (非流式)", f"HTTP {status} {ms:.0f}ms: {str(body)[:120]}", "查服务器日志（LLM 失败→500，超时→504）")
    except Exception as exc:
        fail(g, "POST /api/chat (非流式)", f"{type(exc).__name__}: {exc}")
    # POST /api/chat SSE：读前 3 帧
    try:
        req = urllib.request.Request(
            base + "/api/chat", data=json.dumps({"message": "你好", "stream": True}).encode(),
            headers={"Content-Type": "application/json", **_auth_headers()}, method="POST")
        frames = []
        with urllib.request.urlopen(req, timeout=90) as resp:
            while len(frames) < 3:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    frames.append(line[5:].strip()[:60])
        if frames:
            ok(g, "POST /api/chat (SSE)", f"收到 {len(frames)} 帧: {frames[0]}")
        else:
            fail(g, "POST /api/chat (SSE)", "未收到任何 data: 帧", "检查 StreamingResponse / 反代是否缓冲了 SSE")
    except Exception as exc:
        fail(g, "POST /api/chat (SSE)", f"{type(exc).__name__}: {exc}")
    # POST /api/intent（无 LLM 的关键词分类）
    try:
        status, body = _http_post_json(base + "/api/intent", {"message": "你好"}, _auth_headers(), timeout=10.0)
        if status == 200 and isinstance(body, dict) and body.get("intent"):
            ok(g, "POST /api/intent", f"200 intent={body['intent']}")
        else:
            fail(g, "POST /api/intent", f"HTTP {status}: {str(body)[:100]}")
    except Exception as exc:
        fail(g, "POST /api/intent", f"{type(exc).__name__}: {exc}")
    # POST /api/sentiment（外围模块，501 视为可接受）
    try:
        status, body = _http_post_json(base + "/api/sentiment", {"text": "很好", "session_id": "diag"}, _auth_headers(), timeout=10.0)
        if status == 200:
            ok(g, "POST /api/sentiment", "200")
        elif status == 501:
            warn(g, "POST /api/sentiment", "501 sentiment 模块未安装（外围功能）", "需要则补齐对应遗留模块依赖")
        else:
            fail(g, "POST /api/sentiment", f"HTTP {status}: {str(body)[:100]}")
    except Exception as exc:
        fail(g, "POST /api/sentiment", f"{type(exc).__name__}: {exc}")
# 汇总  # ══════
def summarize() -> int:
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for status, *_ in RESULTS:
        counts[status] = counts.get(status, 0) + 1
    section("汇总")
    print(f"  {counts['PASS']} pass / {counts['FAIL']} fail / {counts['WARN']} warn / {counts['SKIP']} skip")
    fails = [(g, n, d) for s, g, n, d in RESULTS if s == "FAIL"]
    if fails:
        fails.sort(key=lambda x: _GROUP_ORDER.index(x[0]) if x[0] in _GROUP_ORDER else 99)
        print("\n  建议修复顺序（前 3 项）:")
        for i, (grp, name, detail) in enumerate(fails[:3], 1):
            print(f"    {i}. [{grp}] {name} — {detail[:100]}")
    else:
        print("  全部关键检查通过，可以启动服务器: python app_prod.py")
    return min(counts["FAIL"], 250)
def main() -> int:
    parser = argparse.ArgumentParser(description="simple-agent 一键诊断")
    parser.add_argument("--server", default="", help="已运行服务器地址，如 http://127.0.0.1:8000（启用 G 组在线检查）")
    args = parser.parse_args()
    print("SimpleAgent (P6) — 环境诊断")
    print(f"项目根目录: {ROOT}")
    check_env()
    deps = check_deps() or {}
    core_ok = bool(check_imports(deps))
    check_data(core_ok)
    llm_ok = bool(check_services(deps))
    check_engine_smoke(core_ok, llm_ok)
    if args.server:
        check_http(args.server)
    else:
        section("G. HTTP 层")
        skip("HTTP层", "在线服务检查", "未提供 --server 参数",
             "服务器启动后运行: python scripts\\diagnose.py --server http://127.0.0.1:8000")
    return summarize()
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断退出")
        sys.exit(130)
    except Exception as exc:  # 最外层兜底：诊断脚本本身绝不崩溃
        print(f"[FAIL] 诊断脚本内部错误: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
