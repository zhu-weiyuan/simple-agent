# my_agent

SimpleAgent Python 包总入口。__init__.py 暴露 SimpleAgent、Agent 类型、多 Agent、A2A 和结构化工具；agent.py 负责组装组件。

运行核心：agent.py、core/、llm.py、resilience.py。
工具协议：tools/、mcp_client.py、a2a.py、multiagent.py。
记忆任务：memory/、session_manager.py、user_memory.py、async_task.py。
生产能力：auth.py、security/、observability.py、metrics.py。
路由成本：gateway.py、router.py、routing.py、cost_tracker.py。
评测提示：eval_persistence.py、prompt_registry.py、prompt_store.py。

顶层模块和子目录有历史兼容关系，例如 context_assembler.py 与 core/context_assembler.py；先看 import 和测试，不要按同名假设它们完全相同。
