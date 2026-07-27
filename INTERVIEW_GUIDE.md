# SimpleAgent 面试讲解手册

## 1. 项目定位

SimpleAgent 是一个从零实现的轻量级 AI Agent 框架，重点不是做复杂业务，而是为了把 **Agent Loop、Tool Calling、Session/Memory、Hook、Graph、MCP、A2A、权限控制、可观测** 这些底层工程机制跑通。

## 2. 一分钟自我讲解版

我做这个项目的目的，是避免自己只停留在“会调大模型 API”。所以我从零实现了一套轻量 Agent 框架：核心有 QueryEngine 工具调用循环、ToolRegistry、SessionState 历史压缩、Hook 扩展点、Graph Workflow，以及 MCP/A2A 接入。这个项目让我对 Agent 怎么从消息流、工具执行、上下文管理一路跑到最终输出，有了比较底层的理解。

## 3. 核心链路怎么讲

```text
User Message
   ↓
SessionState
   ↓
QueryEngine
   ↓
LLM decides tool call?
   ├─ no  → final answer
   └─ yes → ToolRegistry.execute()
                ↓
             ToolResult
                ↓
            append to context
                ↓
            continue loop
```

## 4. 你做了什么

### QueryEngine
- 实现 Agent 的主循环：LLM → Tool Call → Tool Result → Continue / Final Answer。
- 限制最大工具调用轮数，防止死循环和 token 失控。

### ToolRegistry
- 统一注册和执行工具。
- 对模型暴露 schema，对运行时暴露 handler。
- 这样引擎和工具实现解耦。

### SessionState
- 对历史对话做管理。
- 达到 max_turns / max_tokens 时触发 compact，压缩长上下文，降低成本。

### Hook
- 在 LLM 调用前后、工具执行前后插入扩展逻辑。
- 类似中间件机制，方便做日志、审计、缓存、策略控制。

### Graph / MCP / A2A
- Graph 负责工作流编排和条件分支。
- MCP 用来理解工具/资源协议化接入。
- A2A 用来验证 Agent 之间的互操作思路。

## 5. 高频追问与回答

### Q1：Agent 和 Chatbot 最大区别是什么？
Chatbot 更像输入到输出的单轮生成；Agent 有外部行动能力，会决定要不要调用工具、怎么根据观察结果继续决策，并带有状态和停止条件。

### Q2：为什么不把工具直接写死在代码里？
因为工具 schema、权限、异常处理、可观测都会散掉。ToolRegistry 把它们统一起来，QueryEngine 只关心调用协议，不关心工具实现细节。

### Q3：Session 压缩的意义是什么？
长对话上下文会越来越长，成本和延迟都会上升，甚至会把关键消息淹没。压缩不是为了省一点 token，而是为了让上下文更稳定、更可控。

### Q4：Hook 系统价值是什么？
Hook 让框架可插拔。你可以在不改引擎主逻辑的情况下，加日志、审计、缓存、权限检查、实验逻辑，这就是框架和脚本的区别。

### Q5：MCP 和 Tool Calling 有什么关系？
Tool Calling 是模型输出“我要调用某个工具”的接口机制；MCP 更像工具/资源的协议层，解决怎么发现工具、怎么标准化接入。

## 6. 你要背住的关键词
- Agent Loop
- Tool Calling
- ToolRegistry
- SessionState / Context Compression
- Hook / Middleware
- Graph Workflow
- MCP
- A2A
- Permission Control
- Observability
