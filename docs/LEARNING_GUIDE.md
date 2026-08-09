# SimpleAgent 源码学习路线

先理解一条完整请求链，再按模块扩展，不建议一开始通读所有文件。

## 一、主链路

浏览器 /api/chat → app_prod.py → SimpleAgent → QueryEngine → ToolRegistry + LLM → 工具执行（可选）→ Memory/SQLite → SSE 或 JSON。

第一遍建议按这个顺序：

1. src/my_agent/types/message.py：Message、Role，理解消息和角色。
2. src/my_agent/types/session.py：SessionConfig、SessionState，理解会话。
3. src/my_agent/types/tool.py：工具描述、调用和结果类型。
4. src/my_agent/tools/base.py：BaseTool，理解工具契约。
5. src/my_agent/tools/registry.py：ToolRegistry.add、all_schemas、get_handler。
6. src/my_agent/core/engine.py：QueryEngine.arun、arun_stream 和工具循环。
7. src/my_agent/agent.py：SimpleAgent.run、run_stream、invoke，理解组件组装。
8. app_prod.py：FastAPI 路由、认证、健康检查、SSE 和文件工具注册。

## 二、追踪文件工具

以“请列出当前目录文件”为例：app_prod.py 接收请求；QueryEngine 把工具 schema 交给 LLM；src/my_agent/tools/builtins/file.py 提供 ListFilesTool；ToolRegistry 找到 execute handler；工具结果写回消息后再次请求 LLM；web/index.html 渲染 SSE 或 JSON。

重点看：file.py、registry.py、core/engine.py、app_prod.py、web/index.html。

## 三、记忆与持久化

- src/my_agent/session_manager.py：会话生命周期和消息追加。
- src/my_agent/memory/store.py：存储抽象。
- src/my_agent/memory/sqlite_store.py：SQLite schema 和持久化。
- src/my_agent/memory/retrieval.py：记忆检索。
- src/my_agent/user_memory.py：用户长期记忆。
- conversations.db：本地运行数据，不是源码，不要提交。

## 四、生产能力

auth.py 负责 API key/JWT；security/prompt_guard.py 防提示注入；security/pii_redactor.py 做 PII 脱敏；observability.py 和 metrics.py 负责指标；resilience.py 负责重试和超时；gateway.py、router.py、cost_tracker.py 负责模型路由、fallback、token 和费用。

## 五、增强和协议

最后再看 core/context_assembler.py、core/token_budget.py、enhanced/、graph/、a2a.py、multiagent.py、mcp_client.py。

每读一个模块都回答三个问题：输入是什么？输出流向哪里？出错、超时或权限不足时如何返回？
