# SimpleAgent

一个从零实现的轻量级 AI Agent 框架，参考 Claude Code 和 OpenClaw 架构设计，用于深入理解 Agent 系统的核心原理。

## ✨ 特性

- **端到端增强推理流水线**：Router → Retrieval → Persona → Generation → Detection → Citation → Output
- **分层架构**：types → core (engine + hooks) → tools → memory → bridge → agent
- **6 大增强模块**：
  - 🔀 **查询路由** — 查询复杂度分类 + 动态策略路由
  - 🧠 **Persona 记忆** — 六大认知域结构化记忆提取
  - 🔗 **确定性引用** — 每个陈述可追溯到来源
  - 🛡️ **实时幻觉检测** — 生成时即时事实校验
  - 📚 **多索引混合检索** — Vector + Keyword + Graph 三模检索
  - 💬 **流式输出** — 实时增量响应
- **图状态编排引擎**（Graph）：节点有向图，支持状态传递、条件分支、循环
- **Hook 系统**：pre/post 扩展点，支持自定义中间件
- **Bridge 层**：权限控制 + 安全沙箱
- **MCP 协议集成**：原生支持 Model Context Protocol
- **请求延迟监控**：每个 API 响应自动附加 `X-Response-Time` 头部（毫秒级）
- **Agent Card 健壮性**：优先使用 `agent.card()` 方法，回退到直接属性读取

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────┐
│                    SimpleAgent                       │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│  types   │  core    │  tools   │ memory   │ bridge  │
│          │          │          │          │         │
│ Message  │ Engine   │ Registry │ Store    │ Base    │
│ Session  │ Hooks    │ Builtins │ Retrieval│ Perms   │
│ Tool     │          │ Calculator│          │         │
│          │          │ Time     │          │         │
├──────────┴──────────┴──────────┴──────────┴─────────┤
│              Enhanced Pipeline                       │
├─────────────────────────────────────────────────────┤
│  Router → Retrieval → Persona → Gen → Detect → Cit  │
└─────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/zhu-weiyuan/simple-agent.git
cd simple-agent
pip install -e .
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API 配置
```

### 运行

```bash
my-agent              # CLI 模式
python app.py         # Web 模式 (默认端口 8000)
```

## 📁 项目结构

```
simple-agent/
├── src/my_agent/
│   ├── types/           # 类型定义（Message, Session, Tool）
│   ├── core/            # 核心引擎 + Hook 系统
│   ├── tools/           # 工具注册表 + 内置工具
│   │   └── builtins/    # calculator, time, file, shell
│   ├── memory/          # 记忆存储 + 检索
│   ├── bridge/          # 桥接层 + 权限控制
│   ├── enhanced/        # 增强模块（6个）
│   ├── graph/           # 图状态编排引擎
│   ├── llm/             # LLM 接口适配
│   ├── agent.py         # Agent 主类
│   ├── routing.py       # 路由逻辑
│   └── mcp_client.py    # MCP 客户端
├── static/              # Web UI
├── tests/               # 测试文件
├── pyproject.toml
└── README.md
```

## 🌐 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | `{"message": "...", "stream": false}` → 对话回复 |
| `/api/chat` (stream) | POST | `{"message": "...", "stream": true}` → SSE 流式输出 |
| `/api/health` | GET | 健康检查（含系统指标、LLM连接状态）|
| `/api/metrics` | GET | Prometheus-style 文本指标 |
| `/api/tools` | GET | 列出所有已注册工具 |
| `/api/memory/stats` | GET | 记忆存储统计信息 |
| `/api/card` | GET | Agent Card (A2A协议兼容的元数据) |
| `/api/conversations` | GET | 列出最近的对话会话（session管理）|
| `/api/conversations/{id}` | DELETE | 删除指定会话 |
| `/api/analytics` | GET | 对话分析统计（会话数、消息量、平均长度）|

## 🧪 测试

```bash
python test_pipeline.py     # 流水线集成测试
python test_enhanced.py     # 增强模块测试
python test_graph_engine.py # 图引擎测试
```

## 📖 设计参考

- **Claude Code** — 迭代式主循环、Hook 系统、Feature Gating、权限三防线
- **OpenClaw** — 子代理系统、Cron 调度、心跳检查
- **LangGraph** — 图状态编排灵感
- **AgentScope** — MCP 集成模式

## 📝 License

MIT
