# 多模型路由 + 租户预算 + Fallback

本文档说明 `router.py` / `gateway.py` / `cost_tracker.py` 三个模块是**如何接进
请求链路**的，以及需要哪些 env 才能启用。

## 核心约定：渐进增强，默认零改变

**不配置任何 env 时，`router` 与 `gateway` 都是 `None`，QueryEngine 的行为与接线前
完全一致**：不选模型（`model=None`，由 LLM 客户端用自己的默认模型）、不查预算、
不做 fallback、不注册预算 reconcile hook。所有能力都是"配了才生效"。

## 调用链路

```
HTTP 请求
  → 中间件解析 user_id + tenant_id（JWT claims tenant_id/tenant/org_id > X-Tenant-Id 头 > "default"）
  → QueryContext(tenant_id, scene, user_id, request_id)
  → QueryEngine._select_model(ctx)：RuleBasedRouter 按 scene / tenant_id / user_tags 匹配规则选模型
  → QueryEngine._apply_budget_policy()：ModelGateway.check_budget(budget_id=tenant_id, needed_tokens=估算值)
        · 充足        → 原样放行
        · 超限 + warn   → 只告警，放行
        · 超限 + degrade → 换 gateway 里更便宜的 tier（降不了 → reject）
        · 超限 + reject  → 抛 BudgetExceededError → HTTP 402
  → LLM 调用；失败（异常/超时）→ 沿 gateway.get_fallback_chain() 依次尝试备用模型，全失败才报 llm_error
  → LLM_END hook：① cost_tracker 按**实际使用的模型**记账（tenant/scene/user 归因）
                   ② gateway.reconcile_usage() 用真实 usage 回写扣减租户 token 预算
  → 响应带回 model / fallback_reason / tenant_id
```

预算是 **estimate → reconcile 的简化版**：事前只做只读估算闸门（不预扣，避免失败请求
漏还额度），事后按真实 usage 如实扣减（`consume_force` 允许超额如实记账，让 status
如实翻成 EXCEEDED）。

## 预算超限策略

| BUDGET_POLICY | 行为 |
|---|---|
| `degrade`（默认） | 降级到 gateway 中更便宜的 tier 模型继续服务；无更便宜的档时 reject |
| `reject` | 直接拒绝，抛 `BudgetExceededError`（HTTP 402），不调用 LLM |
| `warn` | 只打告警日志，不拦截 |

未知取值回落到 `degrade`。

## Env 配置

| 变量 | 作用 | 不配时 |
|---|---|---|
| `MODEL_ROUTES_JSON` | gateway 的模型路由表（含单价、优先级、上下文窗口） | 不建 gateway |
| `TENANT_TOKEN_BUDGETS_JSON` | 各租户的 **token** 预算（gateway 闸门用） | 该租户不限量 |
| `TENANT_BUDGETS_JSON` | 各租户的 **月度成本** 预算 USD（cost_tracker 告警用） | 无成本告警 |
| `ROUTING_RULES_JSON` | 路由规则表 | 不建 router |
| `DEFAULT_ROUTE_MODEL` | router 兜底模型（单独配也会启用 router） | 无兜底 |
| `BUDGET_POLICY` | `degrade` / `reject` / `warn` | `degrade` |
| `BUDGET_ALERT_THRESHOLD` | 告警阈值（占比） | `0.8` |

### 示例

```bash
# 1) 模型路由表：priority 小 = 优先；单价用于"哪个更便宜"的降级排序
export MODEL_ROUTES_JSON='[
  {"name":"gpt-4o",      "provider":"openai","priority":0,"context_window":128000,"max_tokens":4096,"cost_per_1m_input":2.50,"cost_per_1m_output":10.00},
  {"name":"gpt-4o-mini", "provider":"openai","priority":1,"context_window":128000,"max_tokens":4096,"cost_per_1m_input":0.15,"cost_per_1m_output":0.60},
  {"name":"qwen-turbo",  "provider":"qwen",  "priority":2,"context_window":32768, "max_tokens":2048,"cost_per_1m_input":0.05,"cost_per_1m_output":0.20}
]'

# 2) 路由规则：scene 正则 / 租户 / 用户标签 三选一或组合；priority: LOW|MEDIUM|HIGH|CRITICAL
export ROUTING_RULES_JSON='[
  {"name":"coding","model":"gpt-4o","priority":"HIGH","scene_patterns":["^code$","^programming$"]},
  {"name":"vip",   "model":"gpt-4o","priority":"MEDIUM","tenant_ids":["vip"]},
  {"name":"bulk",  "model":"qwen-turbo","priority":"LOW","scene_patterns":["^batch$"]}
]'
export DEFAULT_ROUTE_MODEL=gpt-4o-mini

# 3) 租户 token 预算（闸门）与月度成本预算 USD（告警）
export TENANT_TOKEN_BUDGETS_JSON='{"default": 2000000, "vip": 50000000}'
export TENANT_BUDGETS_JSON='{"default": 100.0, "vip": 1000.0}'

export BUDGET_POLICY=degrade
export BUDGET_ALERT_THRESHOLD=0.8
```

请求侧带上租户（JWT 里有 `tenant_id` claim 时可省略这个头）：

```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
     -H 'X-Tenant-Id: vip' \
     -d '{"message":"写个快排","scene":"code"}'
# → {"reply":"...","model":"gpt-4o","tenant_id":"vip","fallback_reason":null,...}
```

## 只读端点

- `GET /api/budgets` — 各租户的 token 预算（used/remaining/percent/status）、月度成本
  预算与用量、当前告警列表、生效的 policy。未配置任何预算时返回 `enabled=false` 的
  空视图（端点始终存在，面板可固定接线）。
- `GET /api/routing` — 生效的路由规则与模型路由表（含健康状态、单价），用于排障/演示。
- `GET /api/costs` — 既有的成本汇总（按模型）。

后台任务每 30s 巡检一次 `cost_tracker.check_budget_alerts()`，越过阈值的租户打
`[BUDGET ALERT]` 日志。

## 代码接线点

- `src/my_agent/core/engine.py`：`QueryEngine(router=..., gateway=..., budget_policy=...)`
  依赖注入；`_select_model` / `_apply_budget_policy` / `_model_candidates` /
  `_budget_reconcile_hook`。
- `src/my_agent/gateway.py`：`BudgetPolicy`、`BudgetExceededError`、
  `TokenBudget.consume_force`、`ModelGateway.cheaper_routes / reconcile_usage /
  budget_snapshot`。
- `app_prod.py`：`_build_gateway()` / `_build_router()` / `_apply_tenant_cost_budgets()`
  / `_resolve_tenant_id()` / `/api/budgets` / `/api/routing`。

## 测试

```bash
python tests/test_routing_budget_pure.py     # 纯 stdlib，LLM 全 mock
python tests/test_user_mem_sa_pure.py        # 既有回归
```
