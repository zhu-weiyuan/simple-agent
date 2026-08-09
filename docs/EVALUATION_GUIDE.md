# 测试与评测导航

项目里同时存在代码回归测试和 Agent 评测，两者目标不同。

## 目录分工

- 根目录 test_*.py：早期跨模块回归，覆盖增强模块、图引擎、API、SQLite 和安全。
- test/：认证、健康检查、日志、指标、优雅退出。
- tests/：gateway、router、预算、持久化、上下文、压缩和用户记忆。
- tests/eval_harness.py：Golden set、trial、记录、评分和报告导出。
- tests/data/sample_golden_set.json：可重复的黄金数据集样例。
- benchmarks/compare.py：冷启动、内存、工具延迟的实验脚本。
- scripts/：诊断、登录、接口验证和数据检查。

## 推荐运行顺序

1. uv run pytest -m "not integration" -q
2. uv run pytest tests/test_p6_pure.py tests/test_context_assembler.py -q
3. uv run pytest tests/test_router.py tests/test_routing_budget_pure.py tests/test_cost_tracker.py -q
4. uv run pytest tests/test_persistence.py tests/test_eval_persistence.py tests/test_task_persistence.py test_sqlite_store.py -q
5. 服务启动后再运行 python scripts/diagnose.py 和 python scripts/verify_endpoints.py。

## 评测 harness 阅读顺序

Trial 是一次输入，Task 组织多个 trial，GoldenSet 管理数据集，EvalRecord 保存模型/prompt/数据集版本和分数，EvaluationRunner 执行 agent 与 exact match 或 LLM judge；最后看 tests/data/sample_golden_set.json 和 src/my_agent/eval_persistence.py。

评测必须记录 prompt_version、model_id、dataset_version。数据库、日志和 checkpoint 是运行产物，不是源码。
