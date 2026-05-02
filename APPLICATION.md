# SimpleAgent v2.0 — 申请材料

---

## Q4: 请描述你使用 Agent 或 AI 驱动构建的具体成果

### 项目名称：SimpleAgent — 从零自建的轻量级 AI Agent 框架

**GitHub**: https://github.com/zhu-weiyuan/simple-agent
**代码规模**: 67 个 Python 文件，7200+ 行代码

---

### 1. 解决的核心痛点

现有 Agent 框架（LangChain、AutoGen 等）存在三个问题：

- **过重**：引入大量不必要的依赖，一个简单对话需要安装十几个包
- **黑盒**：核心循环、工具调用、记忆管理等关键逻辑被封装在框架内部，不利于学习和定制
- **耦合度高**：想要替换 LLM 后端或添加新的编排模式，往往需要重写整个 pipeline

SimpleAgent 的目标是：**用最少的代码实现一个功能完整的 Agent 系统**，同时保持架构清晰、易于理解和扩展。适合学习 Agent 原理、快速原型开发、以及嵌入到现有项目中作为轻量级 AI 能力层。

---

### 2. 核心逻辑流

SimpleAgent v2.0 采用分层架构，包含以下关键模块：

#### （1）端到端增强推理流水线（长链推理）

```
用户输入 → QueryRouter(复杂度分类) → Retrieval(多索引混合检索)
       → Persona(用户画像匹配) → LLM Generation(核心循环)
       → Hallucination Detection(实时幻觉检测) → Citation(确定性引用)
       → 最终输出 + Memory Auto-Capture(自动记忆沉淀)
```

- **查询路由**：将输入分为 tier_1（简单问答）、tier_2（需工具调用）、tier_3（需多步推理），不同层级采用不同策略
- **混合检索**：Vector + Keyword + Graph 三模索引，支持领域知识增强
- **幻觉检测**：生成后即时校验事实性声明，标记可疑内容
- **确定性引用**：每个陈述可追溯到具体来源

#### （2）多 Agent 协作（Multi-Agent Orchestration）

```
SupervisorAgent (调度器)
├── AgentRole("researcher") → 负责搜索和分析
├── AgentRole("writer")     → 负责内容生成
└── AgentRole("coder")      → 负责代码任务
```

支持三种协作模式：
- **Supervisor（调度）**：根据用户意图关键词自动路由到最合适的子 Agent
- **Chain（链式）**：A → B → C 顺序执行，前一个的输出作为下一个的输入
- **Parallel（并行）**：多个 Agent 同时处理不同维度，结果合并

#### （3）Agent-as-Tool（Agent 互操作）

任意 Agent 可以被包装为工具，供其他 Agent 调用。例如：

```python
researcher = SimpleAgent(name="researcher", description="搜索信息")
writer = SimpleAgent(name="writer", description="撰写内容")

# 将 researcher 作为 writer 的工具
writer.add_tool(researcher.as_tool())
writer.run("写一篇关于AI的文章")  # writer 会自动调用 researcher 获取素材
```

#### （4）A2A 协议（Agent-to-Agent Interoperability）

实现了 A2A 协议的 Agent Card、消息格式和任务状态管理，支持跨系统的 Agent 发现和互操作：

- **Agent Card**：标准化元数据描述（名称、能力、工具列表、版本）
- **TaskStatus**：任务生命周期管理（submitted → working → completed / failed）
- **A2AClient / A2AServer**：远程 Agent 调用的客户端/服务端实现

#### （5）结构化输出

支持通过 JSON Schema 约束 LLM 输出格式，自动提取、验证和重试：

```python
structured = StructuredOutputTool(schema=WeatherInfo)
result = structured.extract(llm_response)  # → WeatherInfo(city="北京", temp=25.0)
```

---

### 3. 技术亮点

| 特性 | 实现方式 | 参考框架 |
|------|---------|---------|
| AgentBase Protocol | Python Protocol + runtime_checkable | strands-agents |
| 图状态编排 | 有向图节点 + 条件分支 + 循环 | LangGraph |
| MCP 集成 | SSE/stdio 双通道客户端 | Model Context Protocol |
| Hook 系统 | pre/post 扩展点，支持中间件 | Claude Code |
| Bridge 层 | 权限三防线（alwaysAllow/ask/deny） | OpenClaw |

---

### 4. 实际使用场景

- **本地开发助手**：连接本地 llama.cpp 服务，零成本运行 Agent
- **教学演示**：代码量小、架构清晰，适合讲解 Agent 系统原理
- **嵌入式 AI 能力层**：作为现有 Python 项目的 AI 模块，不引入重依赖
- **多 Agent 编排实验平台**：快速验证 Supervisor/Chain/Parallel 等协作模式

---

## Q5: 使用证明与影响力证明

### 提交材料清单

1. **GitHub 项目链接**：https://github.com/zhu-weiyuan/simple-agent
2. **终端运行日志截图**：demo.py 输出（见下方）
3. **测试通过截图**：25/25 单元测试全部通过
4. **本地 LLM 推理截图**：连接 llama.cpp 的实时对话

### Demo 输出摘要

```
SimpleAgent v2.0 — Demo
============================================================
[1] Basic Chat          ✅ Reply (455 chars)
[2] Tool Use (time)     ✅ 2026年5月2日，下午4点29分
[3] Calculator          ✅ 123 × 456 = 56,088
[4] Agent Card (A2A)    ✅ SimpleAgent v2.0.0, streaming=True
[5] StructuredOutput    ✅ city=Beijing, temp=25
[6] Agent-as-Tool       ✅ Tool name: research
[7] Supervisor Routing  ✅ "搜索AI最新发展" → researcher
============================================================
All demos completed successfully!
```

### 测试报告

```
Ran 25 tests in 0.118s — OK
- AgentBase Protocol:    4/4 ✅
- A2A Protocol:          3/3 ✅
- StructuredOutput:      5/5 ✅
- Agent-as-Tool:         3/3 ✅
- Multi-Agent:           6/6 ✅
- SimpleAgent v2 API:    4/4 ✅
```
