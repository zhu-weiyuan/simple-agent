# -*- coding: utf-8 -*-
"""
tests/test_routing_budget_pure.py — 纯 stdlib 单测 (无三方依赖, LLM 全 mock)

覆盖 "多模型路由 + 租户预算 + fallback" 接进 QueryEngine 请求链路后的行为:
- router 按 scene / tenant 选对模型
- **未注入 router/gateway 时行为完全不变** (回归保护 — 渐进增强的底线)
- 预算充足正常放行
- 预算超限 degrade 到更便宜的 tier 模型
- degrade 无可降级目标时 reject (BudgetExceededError)
- policy=reject / policy=warn 的分支
- fallback 链: 主模型失败时依次尝试备用模型
- LLM_END 记账后真实 usage 回写扣减预算
- 租户隔离: A 租户超限不影响 B 租户

运行:
    PYTHONPATH=src python -m unittest tests.test_routing_budget_pure -v
或直接:
    python tests/test_routing_budget_pure.py
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from my_agent.core.engine import QueryContext, QueryEngine  # noqa: E402
from my_agent.core.hooks import HookPoint, HookRegistry  # noqa: E402
from my_agent.gateway import (  # noqa: E402
    BudgetExceededError, BudgetPolicy, BudgetStatus, ModelGateway, ModelRoute)
from my_agent.router import RouteRule, RoutingPriority, RuleBasedRouter  # noqa: E402
from my_agent.cost_tracker import CostTracker  # noqa: E402


# ── helpers ─────────────────────────────────────────────────

PREMIUM = "premium-model"
CHEAP = "cheap-model"
CODE = "code-model"


def make_gateway(with_cheap=True, with_premium=True):
    gw = ModelGateway()
    if with_premium:
        gw.add_route(ModelRoute(
            name=PREMIUM, endpoint="http://x", provider="test",
            context_window=32768, max_tokens=4096, priority=0,
            cost_per_1m_input=10.0, cost_per_1m_output=30.0))
    if with_cheap:
        gw.add_route(ModelRoute(
            name=CHEAP, endpoint="http://x", provider="test",
            context_window=32768, max_tokens=4096, priority=1,
            cost_per_1m_input=0.1, cost_per_1m_output=0.3))
    return gw


def make_router(gateway=None):
    rt = RuleBasedRouter(gateway=gateway)
    rt.add_rule(RouteRule(
        name="coding", model=CODE, priority=RoutingPriority.HIGH,
        scene_patterns=["^code$", "^programming$"]))
    rt.add_rule(RouteRule(
        name="vip-tenant", model=PREMIUM, priority=RoutingPriority.MEDIUM,
        tenant_ids=["vip"]))
    rt.set_default_model(CHEAP)
    return rt


class FakeLLM:
    """mock async LLM: 记录每次收到的 model, 可为特定模型注入异常。"""

    def __init__(self, fail_models=(), usage=None, content="ok"):
        self.fail_models = set(fail_models)
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5,
                               "total_tokens": 15}
        self.content = content
        self.calls = []

    async def achat(self, messages, tools, model=None):
        self.calls.append(model)
        if model in self.fail_models:
            raise TimeoutError(f"simulated timeout on {model}")
        return self.content, [], dict(self.usage)


def build_engine(llm, **kwargs):
    hooks = kwargs.pop("hooks", None) or HookRegistry()
    engine = QueryEngine(system_prompt="sys", hooks=hooks, **kwargs)
    engine.set_async_llm(llm.achat)
    return engine


def run(coro):
    return asyncio.run(coro)


# ── router ──────────────────────────────────────────────────

class TestRouterSelection(unittest.TestCase):

    def test_router_selects_model_by_scene(self):
        engine = build_engine(FakeLLM(), router=make_router())
        self.assertEqual(
            engine._select_model(QueryContext(scene="code")), CODE)

    def test_router_selects_model_by_tenant(self):
        engine = build_engine(FakeLLM(), router=make_router())
        self.assertEqual(
            engine._select_model(QueryContext(tenant_id="vip")), PREMIUM)

    def test_router_falls_back_to_default_model(self):
        engine = build_engine(FakeLLM(), router=make_router())
        self.assertEqual(
            engine._select_model(QueryContext(scene="smalltalk")), CHEAP)

    def test_routed_model_is_passed_to_llm(self):
        llm = FakeLLM()
        engine = build_engine(llm, router=make_router())
        out = run(engine.arun("hi", ctx=QueryContext(scene="code")))
        self.assertEqual(llm.calls, [CODE])
        self.assertEqual(out["model"], CODE)

    def test_router_exception_degrades_to_none(self):
        class Boom:
            def route(self, *a, **k):
                raise RuntimeError("router down")
        engine = build_engine(FakeLLM(), router=Boom())
        self.assertIsNone(engine._select_model(QueryContext(scene="code")))


# ── 回归保护: 不注入时行为完全不变 ─────────────────────────────

class TestNoInjectionUnchanged(unittest.TestCase):

    def test_select_model_returns_none_without_router(self):
        engine = build_engine(FakeLLM())
        self.assertIsNone(engine._select_model(QueryContext(scene="code")))

    def test_llm_receives_model_none(self):
        llm = FakeLLM()
        engine = build_engine(llm)
        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="whoever")))
        self.assertEqual(llm.calls, [None])
        self.assertEqual(out["content"], "ok")
        self.assertEqual(out["stop_reason"], "completed")

    def test_no_budget_hook_registered_without_gateway(self):
        engine = build_engine(FakeLLM())
        self.assertEqual(engine.hooks.list_handlers(HookPoint.LLM_END), [])
        self.assertIsNone(engine.gateway)

    def test_budget_policy_is_noop_without_gateway(self):
        engine = build_engine(FakeLLM())
        model, note = engine._apply_budget_policy(
            QueryContext(tenant_id="anyone"), None, 10 ** 9)
        self.assertIsNone(model)
        self.assertIsNone(note)

    def test_single_candidate_without_gateway(self):
        engine = build_engine(FakeLLM())
        self.assertEqual(engine._model_candidates(PREMIUM), [PREMIUM])


# ── 预算闸门 ─────────────────────────────────────────────────

class TestBudgetGate(unittest.TestCase):

    def test_sufficient_budget_passes_through(self):
        gw = make_gateway()
        gw.create_budget("acme", 1_000_000)
        llm = FakeLLM()
        engine = build_engine(llm, router=make_router(gw), gateway=gw)
        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="vip")))
        self.assertEqual(llm.calls, [PREMIUM])
        self.assertIsNone(out["fallback_reason"])

    def test_exceeded_budget_degrades_premium_to_cheap(self):
        gw = make_gateway()
        gw.create_budget("vip", 100).consume_force(100)
        llm = FakeLLM()
        engine = build_engine(llm, router=make_router(gw), gateway=gw,
                              budget_policy=BudgetPolicy.DEGRADE)
        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="vip")))
        self.assertEqual(llm.calls[0], CHEAP, "premium 应被降级为 cheap")
        self.assertIn("budget_degrade", out["fallback_reason"] or "")

    def test_degrade_impossible_rejects(self):
        # 选中的已经是最便宜的一档 → 无处可降 → reject
        gw = make_gateway()
        gw.create_budget("acme", 100).consume_force(100)
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(CHEAP)
        llm = FakeLLM()
        engine = build_engine(llm, router=router, gateway=gw,
                              budget_policy=BudgetPolicy.DEGRADE)
        self.assertEqual(gw.check_budget("acme", 1000)[0], BudgetStatus.EXCEEDED)
        with self.assertRaises(BudgetExceededError) as cm:
            run(engine.arun("hi", ctx=QueryContext(tenant_id="acme")))
        self.assertEqual(cm.exception.budget_id, "acme")
        self.assertEqual(llm.calls, [], "拒绝时不应真的调用 LLM")

    def test_reject_policy_raises_immediately(self):
        gw = make_gateway()
        gw.create_budget("vip", 100).consume_force(100)
        llm = FakeLLM()
        engine = build_engine(llm, router=make_router(gw), gateway=gw,
                              budget_policy=BudgetPolicy.REJECT)
        with self.assertRaises(BudgetExceededError):
            run(engine.arun("hi", ctx=QueryContext(tenant_id="vip")))
        self.assertEqual(llm.calls, [], "reject 策略下即便有更便宜的档也不调用")

    def test_warn_policy_passes_through_unchanged(self):
        gw = make_gateway()
        gw.create_budget("vip", 100).consume_force(100)
        llm = FakeLLM()
        engine = build_engine(llm, router=make_router(gw), gateway=gw,
                              budget_policy=BudgetPolicy.WARN)
        run(engine.arun("hi", ctx=QueryContext(tenant_id="vip")))
        self.assertEqual(llm.calls[0], PREMIUM, "warn 不拦截也不降级")

    def test_policy_coerce_from_string_and_garbage(self):
        self.assertEqual(BudgetPolicy.coerce("reject"), BudgetPolicy.REJECT)
        self.assertEqual(BudgetPolicy.coerce("WARN"), BudgetPolicy.WARN)
        self.assertEqual(BudgetPolicy.coerce(None), BudgetPolicy.DEGRADE)
        self.assertEqual(BudgetPolicy.coerce("nonsense"), BudgetPolicy.DEGRADE)

    def test_tenantless_context_uses_default_budget(self):
        gw = make_gateway()
        gw.create_budget("default", 100).consume_force(100)
        llm = FakeLLM()
        engine = build_engine(llm, gateway=gw,
                              budget_policy=BudgetPolicy.REJECT)
        with self.assertRaises(BudgetExceededError) as cm:
            run(engine.arun("hi", ctx=QueryContext()))
        self.assertEqual(cm.exception.budget_id, "default")


# ── fallback 链 ──────────────────────────────────────────────

class TestFallbackChain(unittest.TestCase):

    def test_fallback_used_when_primary_fails(self):
        gw = make_gateway()
        llm = FakeLLM(fail_models=[PREMIUM])
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(PREMIUM)
        engine = build_engine(llm, router=router, gateway=gw)
        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="acme")))
        self.assertEqual(llm.calls, [PREMIUM, CHEAP])
        self.assertEqual(out["model"], CHEAP)
        self.assertEqual(out["stop_reason"], "completed")
        self.assertIn("TimeoutError", out["fallback_reason"] or "")

    def test_all_models_failing_reports_llm_error(self):
        gw = make_gateway()
        llm = FakeLLM(fail_models=[PREMIUM, CHEAP])
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(PREMIUM)
        engine = build_engine(llm, router=router, gateway=gw)
        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="acme")))
        self.assertEqual(llm.calls, [PREMIUM, CHEAP])
        self.assertEqual(out["stop_reason"], "llm_error")
        self.assertIn("已尝试 2 个模型", out["stop_detail"])

    def test_candidates_exclude_unhealthy_routes(self):
        gw = make_gateway()
        gw.mark_unhealthy(CHEAP)
        engine = build_engine(FakeLLM(), gateway=gw)
        self.assertEqual(engine._model_candidates(PREMIUM), [PREMIUM])

    def test_stream_path_still_works_and_reports_model(self):
        gw = make_gateway()
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(CHEAP)
        seen = []

        async def astream(messages, tools, model=None):
            seen.append(model)
            yield {"type": "delta", "content": "he"}
            yield {"type": "delta", "content": "llo"}
            yield {"type": "final", "content": "hello", "tool_calls": [],
                   "usage": {"total_tokens": 12}}

        budget = gw.create_budget("acme", 1_000_000)
        engine = QueryEngine(system_prompt="sys", router=router, gateway=gw)
        engine.set_async_llm(None, astream)

        async def collect():
            frames = []
            async for f in engine.arun_stream(
                    "hi", ctx=QueryContext(tenant_id="acme")):
                frames.append(f)
            return frames

        frames = run(collect())
        self.assertEqual([f["token"] for f in frames if "token" in f],
                         ["he", "llo"])
        done = frames[-1]
        self.assertTrue(done["done"])
        self.assertEqual(done["model"], CHEAP)
        self.assertEqual(seen, [CHEAP])
        self.assertEqual(budget.used_tokens, 12)

    def test_no_fallback_without_gateway(self):
        llm = FakeLLM(fail_models=[None])
        engine = build_engine(llm)
        out = run(engine.arun("hi"))
        self.assertEqual(llm.calls, [None])
        self.assertEqual(out["stop_reason"], "llm_error")


# ── LLM_END 记账 → 预算 reconcile ─────────────────────────────

class TestUsageReconcile(unittest.TestCase):

    def test_real_usage_decrements_budget(self):
        gw = make_gateway()
        budget = gw.create_budget("acme", 1_000_000)
        llm = FakeLLM(usage={"prompt_tokens": 100, "completion_tokens": 50,
                             "total_tokens": 150})
        engine = build_engine(llm, gateway=gw)
        self.assertEqual(budget.used_tokens, 0)
        run(engine.arun("hi", ctx=QueryContext(tenant_id="acme")))
        self.assertEqual(budget.used_tokens, 150)
        self.assertEqual(budget.remaining, 1_000_000 - 150)

    def test_reconcile_accumulates_across_requests(self):
        gw = make_gateway()
        budget = gw.create_budget("acme", 1_000_000)
        llm = FakeLLM(usage={"total_tokens": 40})
        engine = build_engine(llm, gateway=gw)
        for _ in range(3):
            run(engine.arun("hi", ctx=QueryContext(tenant_id="acme")))
        self.assertEqual(budget.used_tokens, 120)

    def test_reconcile_can_overshoot_and_flip_status(self):
        gw = make_gateway()
        budget = gw.create_budget("acme", 100)
        llm = FakeLLM(usage={"total_tokens": 400})
        engine = build_engine(llm, gateway=gw, budget_policy=BudgetPolicy.WARN)
        run(engine.arun("hi", ctx=QueryContext(tenant_id="acme")))
        self.assertEqual(budget.used_tokens, 400)
        self.assertEqual(budget.status, BudgetStatus.EXCEEDED)

    def test_cost_hook_and_reconcile_coexist(self):
        """cost_tracker 记账 + 预算 reconcile 同挂 LLM_END, 互不干扰。"""
        gw = make_gateway()
        budget = gw.create_budget("acme", 1_000_000)
        tracker = CostTracker(load_default_prices=False)
        tracker.set_model_cost(CHEAP, 1.0, 2.0)
        hooks = HookRegistry()
        hooks.register(HookPoint.LLM_END,
                       tracker.make_llm_end_hook(default_model="unknown"))
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(CHEAP)
        llm = FakeLLM(usage={"prompt_tokens": 1_000_000,
                             "completion_tokens": 1_000_000,
                             "total_tokens": 2_000_000})
        engine = build_engine(llm, router=router, gateway=gw, hooks=hooks,
                              budget_policy=BudgetPolicy.WARN)
        run(engine.arun("hi", ctx=QueryContext(tenant_id="acme", scene="s")))
        self.assertEqual(budget.used_tokens, 2_000_000)
        self.assertEqual(len(tracker.records), 1)
        rec = tracker.records[0]
        self.assertEqual(rec.tenant_id, "acme")
        self.assertEqual(rec.model, CHEAP, "记账用的是实际调用的模型")
        self.assertAlmostEqual(rec.total_cost, 3.0, places=6)


# ── 租户隔离 ─────────────────────────────────────────────────

class TestTenantIsolation(unittest.TestCase):

    def test_exhausted_tenant_does_not_affect_other(self):
        gw = make_gateway()
        gw.create_budget("tenant-a", 100).consume_force(100)
        gw.create_budget("tenant-b", 1_000_000)
        llm = FakeLLM()
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(PREMIUM)
        engine = build_engine(llm, router=router, gateway=gw,
                              budget_policy=BudgetPolicy.REJECT)

        with self.assertRaises(BudgetExceededError):
            run(engine.arun("hi", ctx=QueryContext(tenant_id="tenant-a")))
        self.assertEqual(llm.calls, [])

        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="tenant-b")))
        self.assertEqual(llm.calls, [PREMIUM])
        self.assertEqual(out["stop_reason"], "completed")

    def test_usage_is_charged_to_the_right_tenant(self):
        gw = make_gateway()
        a = gw.create_budget("tenant-a", 1_000_000)
        b = gw.create_budget("tenant-b", 1_000_000)
        llm = FakeLLM(usage={"total_tokens": 70})
        engine = build_engine(llm, gateway=gw)
        run(engine.arun("hi", ctx=QueryContext(tenant_id="tenant-a")))
        self.assertEqual(a.used_tokens, 70)
        self.assertEqual(b.used_tokens, 0)

    def test_tenant_without_budget_is_unlimited(self):
        gw = make_gateway()
        gw.create_budget("tenant-a", 100).consume_force(100)
        llm = FakeLLM()
        router = RuleBasedRouter(gateway=gw)
        router.set_default_model(PREMIUM)
        engine = build_engine(llm, router=router, gateway=gw,
                              budget_policy=BudgetPolicy.REJECT)
        out = run(engine.arun("hi", ctx=QueryContext(tenant_id="no-budget")))
        self.assertEqual(llm.calls, [PREMIUM])
        self.assertEqual(out["stop_reason"], "completed")


# ── gateway 辅助能力 ─────────────────────────────────────────

class TestGatewayHelpers(unittest.TestCase):

    def test_cheaper_routes_ordering(self):
        gw = make_gateway()
        gw.add_route(ModelRoute(
            name="mid", endpoint="http://x", provider="test",
            context_window=32768, max_tokens=4096, priority=2,
            cost_per_1m_input=1.0, cost_per_1m_output=2.0))
        self.assertEqual(gw.cheaper_routes(PREMIUM), [CHEAP, "mid"])
        self.assertEqual(gw.cheaper_routes(CHEAP), [])

    def test_cheaper_routes_respects_context_window(self):
        gw = make_gateway()
        self.assertEqual(gw.cheaper_routes(PREMIUM, needed_tokens=10 ** 6), [])

    def test_budget_snapshot_shape(self):
        gw = make_gateway()
        gw.create_budget("acme", 1000).consume_force(900)
        snap = gw.budget_snapshot()["acme"]
        self.assertEqual(snap["used_tokens"], 900)
        self.assertEqual(snap["remaining_tokens"], 100)
        self.assertEqual(snap["status"], "warning")

    def test_reconcile_usage_on_unknown_budget_is_noop(self):
        gw = make_gateway()
        self.assertIsNone(gw.reconcile_usage("nope", 100))


if __name__ == "__main__":
    unittest.main(verbosity=2)
