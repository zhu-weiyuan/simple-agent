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
- **Web UI**：暗色主题、意图分析面板、快捷操作、流式渲染、Markdown 支持
- **Hook 系统**：pre/post 扩展点，支持自定义中间件
- **Bridge 层**：权限控制 + 安全沙箱
- **MCP 协议集成**：原生支持 Model Context Protocol
- **请求延迟监控**：每个 API 响应自动附加 `X-Response-Time` 头部（毫秒级）
- **Agent Card 健壮性**：优先使用 `agent.card()` 方法，回退到直接属性读取
- **滑动窗口限流**：每个 API 端点自动限流，可配置请求数和窗口大小
- **系统健康检查**：CPU/内存/磁盘使用率、LLM 连通性、请求统计
- **A2A 协议**：Agent-to-Agent 互操作，支持 Agent Card 和任务状态

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
│   ├── types/           # 类型定义（Message, Session, Tool, Agent）
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
├── web/                 # Web UI (暗色主题)
├── examples/            # 示例应用
│   └── code_review/     # AI 代码审查助手
├── app.py               # Web 服务器 (FastAPI)
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
| `/api/intent` | POST | `{"message": "..."}` → 意图分类 + 置信度 + 建议回复 |

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

## ?? ?????????

???????????/??????????????????????????????????????

- [??????](docs/LEARNING_GUIDE.md)
- [???????](docs/EVALUATION_GUIDE.md)
- [?????](docs/README.md)
- [??????](src/my_agent/README.md)
- [??????](tests/README.md)
- [??????](docs/operations/README.md)

### ???????

- ?????`app_prod.py`???????? `/api/health` ??????
- ?? shim?`app.prod.py`?
- ?????`app.py`??????????????????????????
- ???????`src/my_agent/tools/builtins/file.py`???????????? `/api/tools` ???

## 📚 源码学习与目录导航

项目文件按“源码、测试/评测、运行脚本、部署文档、运行数据”理解，不建议把数据库和日志当作源码阅读。

- 源码学习路线：docs/LEARNING_GUIDE.md
- 测试与评测导航：docs/EVALUATION_GUIDE.md
- 文档总目录：docs/README.md
- 源码目录说明：src/my_agent/README.md
- 测试目录说明：tests/README.md
- 生产运维入口：docs/operations/README.md

### 当前入口和版本

- 生产服务：app_prod.py，当前生产版本以 /api/health 返回值为准。
- 兼容 shim：app.prod.py。
- 旧版入口：app.py，保留用于历史回归；若浏览器版本不对，先确认端口进程。
- 内置文件工具：src/my_agent/tools/builtins/file.py，由生产入口注册后可通过 /api/tools 查看。

## Production start (P6)

规范入口为 `app_prod.py`(`app.prod.py` 为兼容 shim):

```bash
# 单 worker(默认)
uvicorn app_prod:app --host 0.0.0.0 --port 8000

# 多 worker(内存限流/会话为进程内状态,多 worker 时需外部
# session 亲和或共享存储,见遗留风险)
WORKERS=2 python app_prod.py
```

关键环境变量:`OPENAI_API_KEY`、`OPENAI_BASE_URL`(**须含 /v1**)、
`OPENAI_MODEL`、`REQUEST_TIMEOUT_SECONDS`(默认 60)、`WORKERS`(默认 1)、
`MAX_SESSIONS`(LRU 上限,默认 500)、`CONVERSATIONS_DB`、
`RATE_LIMIT_REQUESTS`/`RATE_LIMIT_WINDOW`、`API_KEYS`、`SHUTDOWN_TIMEOUT_SECONDS`。

探针:`/healthz`(liveness)、`/api/ready`(readiness)、`/api/health`(详情)、
`/api/metrics`(Prometheus)。
