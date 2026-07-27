# -*- coding: utf-8 -*-
"""
P6 纯 stdlib 单测 (无 fastapi/httpx/psutil/tiktoken 依赖; LLM 全 mock)。

运行:
    cd simple-agent && python3 -m unittest tests.test_p6_pure -v
"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_agent.session_manager import SessionManager  # noqa: E402
from my_agent.memory.sqlite_store import SqliteConversationStore  # noqa: E402
from my_agent.core.engine import QueryEngine, QueryContext, _Guardrails  # noqa: E402
from my_agent.core.context_assembler import (  # noqa: E402
    estimate_tokens, fit_messages_to_budget)
from my_agent.cost_tracker import CostTracker  # noqa: E402
from my_agent.gateway import ModelGateway, ModelRoute  # noqa: E402
from my_agent.types.message import Message  # noqa: E402

SYS = "You are a test agent."


def _tmp_db(testcase):
    d = tempfile.mkdtemp(prefix="p6test_")
    path = os.path.join(d, "test.db")
    return path


def _run(coro):
    return asyncio.run(coro)


class FakeAsyncLLM:
    """脚本化 mock: scripts 是 (content, tool_calls) 列表,依次返回。"""

    def __init__(self, scripts, usage_tokens=10):
        self.scripts = list(scripts)
        self.calls = 0
        self.usage_tokens = usage_tokens

    async def achat(self, messages, tools, model=None):
        content, tool_calls = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        usage = {"prompt_tokens": self.usage_tokens,
                 "completion_tokens": self.usage_tokens,
                 "total_tokens": self.usage_tokens * 2}
        return content, tool_calls, usage

    async def astream(self, messages, tools, model=None):
        content, tool_calls = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        # 每 3 字符一个 delta,模拟真流式增量
        for i in range(0, len(content), 3):
            yield {"type": "delta", "content": content[i:i + 3]}
        yield {"type": "final", "content": content, "tool_calls": tool_calls,
               "usage": {"prompt_tokens": self.usage_tokens,
                         "completion_tokens": self.usage_tokens,
                         "total_tokens": self.usage_tokens * 2}}


def make_engine(llm, with_tool=None):
    eng = QueryEngine(system_prompt=SYS, context_window=32768)
    eng.set_async_llm(llm.achat, llm.astream)
    if with_tool:
        name, fn = with_tool
        eng.tool_registry.add(
            name=name, handler=fn, description="test tool",
            parameters={"type": "object", "properties": {}})
    return eng


class TestSessionIsolation(unittest.TestCase):
    def test_two_sessions_are_invisible_to_each_other(self):
        sm = SessionManager(SYS, store=None, max_sessions=10)
        sid_a, sess_a = sm.get_or_create("a")
        sid_b, sess_b = sm.get_or_create("b")
        sess_a.append(Message.user("secret-of-A"))
        contents_b = [m.content for m in sess_b.messages]
        self.assertNotIn("secret-of-A", contents_b)
        # 引擎路径: 两个 session 分别 arun,历史互不可见
        llm = FakeAsyncLLM([("ok", [])])
        eng = make_engine(llm)
        _run(eng.arun("hello-A", session=sess_a))
        texts_b = " ".join(m.content or "" for m in sess_b.messages)
        self.assertNotIn("hello-A", texts_b)

    def test_missing_session_id_generates_one(self):
        sm = SessionManager(SYS, store=None)
        sid, _ = sm.get_or_create(None)
        self.assertTrue(sid.startswith("sess-"))


class TestLRUEvictionPersistence(unittest.TestCase):
    def test_lru_evicts_and_persists_and_restores(self):
        db = _tmp_db(self)
        store = SqliteConversationStore(db)
        sm = SessionManager(SYS, store=store, max_sessions=2)
        sid1, s1 = sm.get_or_create("s1")
        s1.append(Message.user("msg-in-s1"))
        sm.get_or_create("s2")
        sm.get_or_create("s3")  # 逐出 s1 (最久未用) 并持久化
        self.assertEqual(sm.active_count, 2)
        self.assertIsNone(sm._sessions.get("s1"))
        rows = store.load_session_messages("s1")
        self.assertTrue(any(r["content"] == "msg-in-s1" for r in rows))
        # get_or_create 恢复被逐出的会话
        _, s1_again = sm.get_or_create("s1")
        self.assertTrue(any((m.content or "") == "msg-in-s1"
                            for m in s1_again.messages))
        store.close()

    def test_restore_recent_on_startup(self):
        db = _tmp_db(self)
        store = SqliteConversationStore(db)
        sm = SessionManager(SYS, store=store, max_sessions=10)
        _, s = sm.get_or_create("boot")
        s.append(Message.user("persisted-line"))
        sm.persist_all()
        # 模拟重启
        sm2 = SessionManager(SYS, store=store, max_sessions=10)
        n = sm2.restore_recent(5)
        self.assertGreaterEqual(n, 1)
        s2 = sm2.get("boot")
        self.assertIsNotNone(s2)
        self.assertTrue(any((m.content or "") == "persisted-line"
                            for m in s2.messages))
        store.close()


class TestSqliteStore(unittest.TestCase):
    def test_role_tool_is_writable(self):
        db = _tmp_db(self)
        store = SqliteConversationStore(db)
        mid = store.add_message("s", "tool", "tool result content")
        self.assertGreater(mid, 0)
        conv = store.get_conversation("s")
        self.assertEqual(conv[0]["role"], "tool")
        with self.assertRaises(ValueError):
            store.add_message("s", "banana", "x")
        store.close()

    def test_ordering_by_autoincrement_id(self):
        db = _tmp_db(self)
        store = SqliteConversationStore(db)
        for i in range(5):
            store.add_message("s", "user", f"m{i}")
        conv = store.get_conversation("s", limit=3)
        self.assertEqual([r["content"] for r in conv], ["m2", "m3", "m4"])
        store.close()


class TestGuardrails(unittest.TestCase):
    def test_same_error_fingerprint_circuit_breaker(self):
        g = _Guardrails()
        err = "工具执行失败 [calc]:ZeroDivisionError: division by zero"
        g.record_tool_result("calc", err, True)
        self.assertFalse(g.tripped)
        g.record_tool_result("calc", err.replace("zero", "0"), True)  # 同指纹
        self.assertTrue(g.tripped)
        self.assertEqual(g.stop_reason, "repeated_tool_error")

    def test_different_errors_do_not_trip(self):
        g = _Guardrails()
        g.record_tool_result("calc", "工具执行失败 [calc]:ValueError: x", True)
        g.record_tool_result("calc", "工具执行失败 [calc]:TypeError: y", True)
        self.assertFalse(g.tripped)

    def test_fingerprint_normalization(self):
        fp1 = _Guardrails.error_fingerprint(
            "t", "工具执行失败 [t]:ValueError: bad 123")
        fp2 = _Guardrails.error_fingerprint(
            "t", "工具执行失败 [t]:ValueError: bad 456")
        self.assertEqual(fp1, fp2)

    def test_no_progress_detection(self):
        g = _Guardrails()
        text = "我将再次尝试调用工具来解决这个问题," * 5
        g.record_assistant_output(text)
        g.record_assistant_output(text)
        g.record_assistant_output(text)
        self.assertTrue(g.tripped)
        self.assertEqual(g.stop_reason, "no_progress")

    def test_budget_guardrail(self):
        g = _Guardrails(max_tokens_budget=100)
        g.record_usage({"total_tokens": 60})
        self.assertFalse(g.tripped)
        g.record_usage({"total_tokens": 60})
        self.assertTrue(g.tripped)
        self.assertEqual(g.stop_reason, "budget_exceeded")


class TestEngineGuardrailsEndToEnd(unittest.TestCase):
    def test_repeated_tool_error_stops_loop(self):
        def bad_tool(params):
            raise ValueError("always fails")
        tc = [{"id": "1", "name": "bad", "arguments": {}}]
        llm = FakeAsyncLLM([("try 1", tc), ("try 2", tc), ("try 3", tc)])
        eng = make_engine(llm, with_tool=("bad", bad_tool))
        result = _run(eng.arun("go", max_tool_calls=8))
        self.assertEqual(result["stop_reason"], "repeated_tool_error")
        self.assertLessEqual(llm.calls, 3)

    def test_budget_stops_loop(self):
        def ok_tool(params):
            return "fine"
        tc = [{"id": "1", "name": "ok", "arguments": {}}]
        llm = FakeAsyncLLM([("looping", tc)] * 10, usage_tokens=50)
        eng = make_engine(llm, with_tool=("ok", ok_tool))
        ctx = QueryContext(max_tokens_budget=150)
        result = _run(eng.arun("go", ctx=ctx, max_tool_calls=10))
        self.assertEqual(result["stop_reason"], "budget_exceeded")

    def test_completed_when_no_tool_calls_and_content(self):
        llm = FakeAsyncLLM([("final answer", [])])
        eng = make_engine(llm)
        result = _run(eng.arun("hi"))
        self.assertEqual(result["stop_reason"], "completed")
        self.assertEqual(result["content"], "final answer")

    def test_max_tool_calls(self):
        def ok_tool(params):
            return str(os.urandom(8))  # 每次不同,避免触发其他护栏
        llm = FakeAsyncLLM(
            [(f"round-{i}-" + "x" * i, [{"id": str(i), "name": "ok",
                                         "arguments": {"i": i}}])
             for i in range(20)])
        eng = make_engine(llm, with_tool=("ok", ok_tool))
        result = _run(eng.arun("go", max_tool_calls=3))
        self.assertEqual(result["stop_reason"], "max_tool_calls")


class TestTrueStreaming(unittest.TestCase):
    def test_stream_order_tokens_progress_done(self):
        def echo_tool(params):
            return "tool-output"
        llm = FakeAsyncLLM([
            ("calling tool", [{"id": "1", "name": "echo", "arguments": {}}]),
            ("all done here", []),
        ])
        eng = make_engine(llm, with_tool=("echo", echo_tool))

        async def collect():
            frames = []
            async for f in eng.arun_stream("go", session_id="sid-1"):
                frames.append(f)
            return frames

        frames = _run(collect())
        kinds = []
        for f in frames:
            if "token" in f:
                kinds.append("token")
            elif "progress" in f:
                kinds.append("progress")
            elif f.get("done"):
                kinds.append("done")
        # 第一轮增量 token → progress → 第二轮增量 token → done
        self.assertIn("progress", kinds)
        self.assertEqual(kinds[-1], "done")
        first_progress = kinds.index("progress")
        self.assertTrue(all(k == "token" for k in kinds[:first_progress]))
        self.assertTrue(any(k == "token" for k in kinds[first_progress + 1:-1]))
        # 增量拼接 == 各轮 content
        tokens = [f["token"] for f in frames if "token" in f]
        self.assertEqual("".join(tokens), "calling tool" + "all done here")
        done = frames[-1]
        self.assertEqual(done["session_id"], "sid-1")
        self.assertEqual(done["stop_reason"], "completed")
        self.assertEqual(done["content"], "all done here")
        self.assertGreater(done["usage"]["total_tokens"], 0)
        prog = [f["progress"] for f in frames if "progress" in f]
        self.assertEqual(prog, ["tool:echo"])


class TestCostTracker(unittest.TestCase):
    def test_default_prices_nonzero_and_recorded(self):
        ct = CostTracker()
        self.assertTrue(ct.price_version)
        rec = ct.record_usage(model="gpt-4o-mini", input_tokens=1_000_000,
                              output_tokens=1_000_000)
        self.assertGreater(rec.total_cost, 0.0)
        self.assertAlmostEqual(rec.input_cost, 0.15, places=6)
        self.assertAlmostEqual(rec.output_cost, 0.60, places=6)

    def test_llm_end_hook_records(self):
        ct = CostTracker()
        hook = ct.make_llm_end_hook(default_model="gpt-4o-mini")
        hook(None, data={"usage": {"prompt_tokens": 100, "completion_tokens": 50},
                         "model": "gpt-4o-mini", "context": None})
        self.assertEqual(len(ct.records), 1)
        self.assertGreater(ct.records[0].total_cost, 0.0)

    def test_sqlite_flush(self):
        db = _tmp_db(self)
        ct = CostTracker(db_path=db, flush_every=1)
        ct.record_usage(model="gpt-4o", input_tokens=10, output_tokens=10)
        import sqlite3
        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM cost_records").fetchone()[0]
        conn.close()
        self.assertEqual(n, 1)


class TestAssembler(unittest.TestCase):
    def test_order_system_first_history_old_to_new_current_last(self):
        msgs = [Message.system(SYS)]
        for i in range(6):
            msgs.append(Message.user(f"q{i}"))
            msgs.append(Message.assistant(f"a{i}"))
        msgs.append(Message.user("current-question"))
        fitted = fit_messages_to_budget(msgs, context_window=32768)
        self.assertEqual(fitted[0].role.value, "system")
        self.assertEqual(fitted[-1].content, "current-question")
        non_sys = [m.content for m in fitted if m.role.value != "system"]
        self.assertEqual(non_sys[:2], ["q0", "a0"])  # 旧→新

    def test_utilization_cap_drops_oldest_keeps_current(self):
        window = 1000  # budget = 600
        msgs = [Message.system("sys")]
        for i in range(30):
            msgs.append(Message.user("上下文历史填充内容" * 20 + str(i)))
        msgs.append(Message.user("current"))
        fitted = fit_messages_to_budget(msgs, context_window=window)
        self.assertEqual(fitted[-1].content, "current")
        used = sum(estimate_tokens(m.content or "") + 4 for m in fitted)
        self.assertLessEqual(used, int(window * 0.6) + 50)
        self.assertLess(len(fitted), len(msgs))

    def test_no_orphan_tool_message_at_head(self):
        msgs = [Message.system("sys"),
                Message.tool_result("id1", "orphan"),
                Message.user("current")]
        fitted = fit_messages_to_budget(msgs, context_window=32768)
        roles = [m.role.value for m in fitted if m.role.value != "system"]
        self.assertEqual(roles[0], "user")

    def test_estimator_mixed_text(self):
        self.assertEqual(estimate_tokens(""), 0)
        # tiktoken is available; cl100k_base encodes these as:
        cjk = estimate_tokens("你好世界")   # → 5
        self.assertAlmostEqual(cjk, 5, delta=1)
        words = estimate_tokens("hello world")  # → 2
        self.assertAlmostEqual(words, 2, delta=1)


class TestGateway(unittest.TestCase):
    def _gw(self):
        gw = ModelGateway()
        gw.add_route(ModelRoute(name="big", endpoint="e", provider="p",
                                context_window=128000, max_tokens=8000, priority=0))
        gw.add_route(ModelRoute(name="small", endpoint="e", provider="p",
                                context_window=8000, max_tokens=2000, priority=1))
        return gw

    def test_select_model_respects_reserved_output(self):
        gw = self._gw()
        # 7000 needed: small 的 8000 - 2000 = 6000 不够 → 选 big
        self.assertEqual(gw.select_model(needed_tokens=7000,
                                         preferred_models=["small", "big"]), "big")
        # 5000 needed: small 可用
        self.assertEqual(gw.select_model(needed_tokens=5000,
                                         preferred_models=["small", "big"]), "small")

    def test_select_model_budget_checked_once(self):
        gw = self._gw()
        gw.create_budget("b1", max_tokens=100)
        gw.budgets["b1"].consume(100)
        self.assertIsNone(gw.select_model(budget_id="b1", needed_tokens=10))

    def test_fallback_chain_starts_after_primary(self):
        gw = self._gw()
        gw.add_route(ModelRoute(name="mid", endpoint="e", provider="p",
                                context_window=64000, max_tokens=4000, priority=1))
        chain = gw.get_fallback_chain("mid")
        self.assertNotIn("big", chain)  # priority 0 < primary 的 1,不回退到更高优先级
        self.assertIn("small", chain)


if __name__ == "__main__":
    unittest.main(verbosity=2)
