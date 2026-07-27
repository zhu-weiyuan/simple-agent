# SimpleAgent 系统设计说明

## 1. 设计目标

SimpleAgent 的目标不是替代成熟框架，而是用最小可理解实现，把 AI Agent 框架的核心机制拆开讲清楚：

- 消息模型
- Agent Loop
- 工具调用
- 记忆管理
- Hook 扩展
- 图编排
- 权限控制
- 协议互操作
- 可观测

## 2. 核心链路

```text
Client / CLI / Web
    ↓
API Layer
    ↓
SessionState
    ↓
QueryEngine
    ↓
LLM Adapter
    ↓
Tool Call? ── yes ──> ToolRegistry ──> ToolResult
    │                                      ↓
    no <──────────── append observation ───┘
    ↓
Final Answer
```

## 3. 模块说明

### 3.1 Message / ToolCall 类型系统

负责把内部 dataclass 和 OpenAI 风格消息互转。
面试重点：
- 为什么要有统一消息抽象？
- 为什么不能直接传 dict？
- 如何处理连续 assistant 消息兼容问题？

### 3.2 QueryEngine

Agent 的主循环，负责：
1. 构造上下文
2. 调用 LLM
3. 解析 tool call
4. 调用工具
5. 写回 tool result
6. 判断继续还是终止

关键风险：
- 工具调用死循环
- 工具异常未处理
- token 成本失控

对应设计：
- max tool calls
- 统一异常封装
- session compact

### 3.3 ToolRegistry

统一管理工具注册、schema 暴露和执行。

面试回答：
> ToolRegistry 的作用是把工具发现、schema、权限、执行分发从 Agent 主循环里解耦出来。否则每加一个工具都要改 QueryEngine，会破坏框架扩展性。

### 3.4 Memory / Session

- 短期记忆：当前会话上下文
- 长期记忆：facts / lessons / sqlite store
- 压缩策略：按 max_turns / max_tokens 触发

### 3.5 Hook

Hook 类似中间件，可以在 LLM 调用前后、工具调用前后插入逻辑。

可扩展场景：
- 日志
- 审计
- 缓存
- 权限检查
- 实验开关

### 3.6 Graph

用于把线性 Agent Loop 扩展成状态图编排。

适合场景：
- 多步骤任务
- 条件分支
- 循环重试
- 节点级 checkpoint

### 3.7 MCP / A2A

- MCP：工具、资源、上下文服务的协议化接入
- A2A：Agent 之间互操作，暴露 Agent Card 和任务状态

## 4. 工程化能力

- FastAPI API 层
- API Key 鉴权
- 滑动窗口限流
- Prometheus-style metrics
- 健康检查
- 安全扫描
- SSE 流式输出

## 5. 设计取舍

### 为什么不用 LangChain 直接写？
为了理解底层机制。成熟框架隐藏了很多细节，比如消息格式、工具调用循环、上下文压缩、hook 插入点。自己实现一遍更适合面试讲原理。

### 为什么要保留轻量？
个人项目不追求所有能力生产级，而是把关键路径跑通。生产环境还需要更完整的分布式 trace、权限体系、sandbox、队列和任务恢复。

### 为什么要有 Bridge / Permission？
工具调用天然有风险。模型不能直接拥有文件、shell、网络等危险能力，需要通过 Bridge 做权限边界。

## 6. 面试最佳表述

> SimpleAgent 是我从零实现的轻量 Agent 框架，用于理解 Agent 底层工程机制。核心是 QueryEngine 工具调用循环、ToolRegistry、SessionState、Hook、Graph Workflow，以及 MCP/A2A 的协议化接入。它不是成熟商用平台，但覆盖了 Agent 框架最关键的设计点。
