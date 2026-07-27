# SimpleAgent 架构说明

## 1. 项目定位

SimpleAgent 是一个从零实现的轻量级 AI Agent 框架，目标不是堆功能，而是把 Agent 的核心工程机制跑通并讲清楚：

- Agent Loop
- Tool Calling
- Session/Memory
- Hook / Middleware
- Graph Workflow
- MCP / A2A
- 权限控制与可观测

## 2. 总体架构

```text
User Input
   ↓
SessionState
   ↓
QueryEngine
   ↓
LLM decides tool call?
   ├─ no  → Final Answer
   └─ yes → ToolRegistry.execute()
                ↓
             ToolResult
                ↓
           append context
                ↓
            continue loop
```

## 3. 分层设计

### types
定义 Message、ToolCall、Session、AgentCard 等核心数据结构。

### core
- `engine.py`: QueryEngine 主循环
- `hooks.py`: pre/post Hook 扩展点

### tools
- `registry.py`: 工具注册、schema 暴露、执行分发
- `builtins/`: calculator / file / shell / time 等内置工具

### memory
- `store.py` / `sqlite_store.py`: 记忆存储
- `retrieval.py`: 记忆召回

### bridge
- 权限控制
- 外部环境桥接

### enhanced
增强模块：
- Query Router
- Persona Memory
- Hallucination Detector
- Deterministic Citation
- Multi-Index Retrieval
- Streaming Output

### graph
图状态编排与条件边执行。

## 4. 关键设计点

### 4.1 QueryEngine
负责驱动 Agent 主循环：
1. 读取当前消息上下文
2. 调模型
3. 解析是否需要工具调用
4. 执行工具
5. 把工具结果回填上下文
6. 再次进入决策，直到最终回答或触达上限

### 4.2 ToolRegistry
统一管理工具的三个问题：
- 给模型看什么 schema
- 运行时怎么拿到 handler
- 怎么统一处理调用和异常

### 4.3 Session 压缩
对长对话触发 compact，避免 token 成本失控和关键上下文被淹没。

### 4.4 Hook 扩展点
通过 Hook 在不改主循环的前提下插入日志、缓存、审计、A/B 实验逻辑。

### 4.5 MCP / A2A
MCP 用于理解工具/资源协议接入；A2A 用于验证 Agent 间互操作思路。

## 5. 面试里怎么强调

### 不要夸大成“成熟商用平台”
更合适的表达是：

> 这是我为了理解 Agent 底层机制而实现的一套轻量框架，重点在 Agent Loop、Tool、Memory、Hook、Graph、MCP/A2A 和工程化边界控制。

### 要强调“理解”和“取舍”
比如：
- 为什么要限制最大工具调用轮数
- 为什么要做 Session 压缩
- 为什么工具不能散落在业务代码里
- 为什么 Hook 系统能提升框架可扩展性
