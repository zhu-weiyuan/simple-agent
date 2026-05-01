# My Agent — 参考 Claude Code / OpenClaw 架构设计

## 设计原则（来自 CC）

1. **QueryEngine** — 核心循环封装在类里，不是裸函数
2. **Bridge 层** — Agent 与外部环境的适配层（工具注册、权限控制、sandbox 隔离）
3. **Session State** — 消息历史 + 压缩摘要作为一等公民
4. **Feature Gating** — 功能通过随机词对开关控制，可以渐进式启用
5. **Hook System** — 主循环的关键节点有钩子，方便扩展
6. **Permission Model** — 工具调用分三级权限（alwaysAllow / ask / deny）

## 重构目标

从"一堆功能堆在一起" → "清晰分层、可扩展的 Agent 框架"

### 新架构

```
my_agent/
├── core/                  # 核心引擎
│   ├── engine.py          # QueryEngine — 主循环封装
│   ├── session.py         # SessionState — 消息管理 + 压缩
│   └── hooks.py           # HookSystem — 扩展钩子
├── bridge/                # 桥接层（参考 CC bridge）
│   ├── base.py            # Bridge 基类
│   ├── permissions.py     # 工具权限模型
│   └── tool_registry.py   # 动态工具注册
├── tools/                 # 工具实现
│   ├── __init__.py        # 自动注册
│   ├── shell.py           # PowerShell 执行
│   ├── filesystem.py      # 文件操作
│   ├── utility.py         # 时间、计算器
│   └── mcp.py             # MCP 协议支持
├── memory/                # 记忆系统
│   ├── store.py           # JSON facts 持久化
│   ├── recall.py          # 语义检索 + 关键词匹配
│   └── summarizer.py      # LLM 驱动的历史压缩摘要
├── llm/                   # LLM 抽象层
│   ├── client.py          # OpenAI-compatible API 客户端
│   ├── streaming.py       # 流式输出支持
│   └── formatting.py      # 消息格式化转换
├── cli/                   # CLI 入口
│   ├── main.py            # REPL + 命令行
│   └── config.py          # 配置加载
└── config.py              # 全局配置
```

### 关键改进

1. **QueryEngine 模式** — 主循环不再是 `_run_with_tools` 裸方法，而是完整的 QueryEngine 类
2. **Bridge 层** — 工具通过 Bridge 调用，支持权限检查和 sandbox
3. **Hook System** — `on_query_start`, `on_tool_call`, `on_response`, `on_error` 等钩子
4. **Session State** — 独立的消息管理，支持压缩、恢复、序列化
5. **Streaming** — 原生流式输出（参考 OpenClaw streaming）
6. **配置驱动** — 通过 pyproject.toml / .env 配置所有参数
7. **插件化工具** — 工具通过 registry 注册，无需硬编码
