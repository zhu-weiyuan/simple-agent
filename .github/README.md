# SimpleAgent — 项目文档入口与架构总览

> **生产级轻量 AI Agent 框架** | Python 3.10+ | FastAPI | A2A Protocol | MCP | SQLite-WAL

---

## 🎯 项目定位

| 维度 | 说明 |
|------|------|
| **核心价值** | 从零实现的、生产可用的 Agent 运行时，融合 strands-agents / A2A / LangGraph / AgentScope 最佳实践 |
| **适用场景** | 企业级 Agent 平台、多 Agent 协作系统、需审计/可控成本的 LLM 应用 |
| **代码规模** | ~12k 行核心代码（src/my_agent/），单人主导，全栈自研 |
| **交付物** | 核心库 + FastAPI 服务 + Web UI + 评测体系 + 文档 |

---

## 🏗️ 核心架构（分层设计）

```
┌─────────────────────────────────────────────────────────────┐
│                        SimpleAgent                           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│    types     │    core      │    tools     │    memory      │
│  (契约层)    │  (引擎层)    │  (工具层)    │  (记忆层)      │
│ Message      │ QueryEngine  │ ToolRegistry │ MemoryStore    │
│ Session      │ HookRegistry │ Builtins     │ Retrieval      │
│ ToolDef      │ ContextAsm.  │ AgentAsTool  │ SQLite-WAL     │
│ AgentCard    │              │ MCP Client   │                │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                    bridge (桥接/权限)                         │
├─────────────────────────────────────────────────────────────┤
│                  Enhanced Pipeline (7 阶段)                 │
│  Router → Retrieval → Persona → Generation → Detect → Cite  │
├─────────────────────────────────────────────────────────────┤
│              Multi-Agent Orchestration                      │
│  Supervisor / Chain / Parallel / Agent-as-Tool              │
├─────────────────────────────────────────────────────────────┤
│                    A2A Protocol                             │
│  Server / Client / TaskStore / Hub / Web Console            │
└─────────────────────────────────────────────────────────────┘
```

### 关键模块职责

| 模块 | 文件 | 核心能力 |
|------|------|----------|
| **Agent Runtime** | `src/my_agent/agent.py` | SimpleAgent 主类、AgentBase Protocol、A2A Card、工具注册 |
| **QueryEngine** | `src/my_agent/core/engine.py` | 异步主循环、四层护栏、上下文组装、Token 预算、Model Routing |
| **Tool System** | `src/my_agent/tools/` | 声明式注册、15+ 内置工具、Agent-as-Tool、MCP 动态扩展 |
| **Memory** | `src/my_agent/memory/` | 分层存储、向量/关键词/图检索、Persona 记忆 |
| **Enhanced** | `src/my_agent/enhanced/` | 7 阶段增强流水线、幻觉检测、确定性引用、查询路由 |
| **A2A** | `src/my_agent/a2a.py` | 任务状态机、SQLite 持久化、指纹幂等、真 async、Web Console |
| **Graph** | `src/my_agent/graph/` | 状态图编排、条件分支、循环、检查点 |
| **Security** | `src/my_agent/security/` | PII 脱敏、Prompt Guard、权限策略 |

---

## 🚀 快速开始

```bash
# 1. 克隆 & 安装
git clone https://github.com/zhu-weiyuan/simple-agent.git
cd simple-agent
pip install -e .

# 2. 配置 LLM（需 OpenAI 兼容端点：Ollama/LM Studio/vLLM/云 API）
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL

# 3. 运行
my-agent "列出当前目录文件"          # CLI 模式
uvicorn app_prod:app --port 8000    # Web 模式
# 打开 http://localhost:8000 (聊天) / http://localhost:8000/a2a.html (A2A 控制台)
```

---

## 🌐 核心 API 一览

| 领域 | 端点 | 说明 |
|------|------|------|
| **对话** | `POST /api/chat` | 非流式/流式 SSE 对话 |
| **A2A** | `POST /a2a/messages` | 提交任务、`GET /a2a/tasks/{id}` 查询、`POST /a2a/tasks/{id}/cancel` 取消 |
| **工具** | `GET /api/tools` | 已注册工具清单 |
| **记忆** | `GET /api/memory/stats` | 记忆存储统计 |
| **健康** | `GET /healthz /api/ready /api/health` | K8s 探针 / 就绪 / 详情 |
| **指标** | `GET /api/metrics` | Prometheus 文本格式 |

---

## 📚 文档导航

| 文档 | 受众 | 入口 |
|------|------|------|
| **学习指南** | 想深入理解 Agent 架构的开发者 | `docs/LEARNING_GUIDE.md` |
| **评测指南** | QA / 研究员跑基准测试 | `docs/EVALUATION_GUIDE.md` |
| **运维手册** | SRE 部署生产环境 | `docs/operations/README.md` |
| **源码地图** | 贡献者阅读代码 | `src/my_agent/README.md` |
| **架构决策** | 架构师评审设计 | `docs/architecture/` |
| **CI/CD 详情** | 维护者改流水线 | `docs/operations/CI_CD.md` |

---

## 🧪 测试与评测

```bash
# 单元/集成测试
pytest tests/ -x -q

# 评测套件（离线 + 在线）
python evals/run_eval.py
python evals/run_live_eval.py
```

**评测维度**：工具调用、结构化输出、技能路由、安全、性能、E2E、**Reliability-50 压测**

---

## 🔧 开发规范

```bash
# 格式化 & Lint
ruff check --fix && ruff format

# 类型检查
mypy src/

# Pre-commit（已配置 ruff + mypy + 基础钩子）
pre-commit install
```

---

## 📦 目录结构速览

```
simple-agent/
├── .github/              # GitHub 自动化（CI/Release/Dependabot/模板）
├── src/my_agent/         # 核心框架源码（见上架构图）
├── web/                  # 静态 UI：聊天/仪表盘/A2A 控制台
├── app_prod.py           # 生产入口 (FastAPI + Uvicorn)
├── examples/             # 示例应用（代码审查助手等）
├── evals/                # 评测 Harness + 黄金数据集
├── tests/                # 单元 & 集成测试
├── docs/                 # 架构/运维/学习文档
├── benchmarks/           # 性能基准脚本
├── scripts/              # 运维/发布脚本
├── runtime/              # 运行时数据（SQLite/Artifacts/Jobs/Events）
└── pyproject.toml        # 依赖 & 构建配置
```

---

## 🔑 关键技术亮点（面试/选型必看）

| 亮点 | 技术实现 | 业务价值 |
|------|----------|----------|
| **四层护栏** | 最大工具数/同错熔断/无进展检测/Token 预算 | 防无限循环、成本失控、幻觉累积 |
| **A2A 生产级** | 任务状态机/指纹幂等/O(1)查询/真 async 取消/SQLite 恢复 | Agent 间协作像调 HTTP 一样可靠 |
| **Token 预算两阶段** | 请求前 estimate 闸门 + 实时 usage reconcile | 租户成本收敛 ±5% |
| **三大持久化服务** | Job Manager / Artifact Store / Session Events | 可观测、审计、合规、断点恢复 |
| **Resilience 协议** | 错误分类/指数退避熔断/三级权限 | 工具调用自愈率 >95% |
| **增强推理流水线** | 7 阶段：路由→检索→Persona→生成→检测→引用 | 复杂查询准确率显著提升 |
| **评测门禁** | 9 大黄金集 + Reliability-50 / CI 回归 | 核心指标回归率 0 |

---

## 🤝 贡献指南

1. Fork → Clone → `pip install -e ".[dev]"`
2. `pre-commit install`
3. 改代码 → `pytest tests/ -x` 全绿
4. 提 PR（按模板填：动机/测试/回滚方案）
5. 等 CI 绿灯 + Codeowner 审批

---

## 📄 许可证

MIT License — 详见 [LICENSE](../LICENSE)

---

## 🙏 致谢

- [strands-agents](https://github.com/strands-agents) — AgentBase 协议设计
- [A2A Protocol](https://github.com/google/A2A) — Agent 互操作标准
- [LangGraph](https://github.com/langchain-ai/langgraph) — 图状态管理模式
- [AgentScope](https://github.com/modelscope/agentscope) — 生产运行时模式