# SimpleAgent

SimpleAgent 是一个用 Python 编写的 AI Agent 应用，提供对话、工具调用、会话管理和后台任务接口。项目使用 FastAPI 提供 Web API，并支持接入 OpenAI 兼容的模型服务。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能概览

### 对话与模型接入

- `/api/chat` 支持普通 JSON 响应和 SSE 流式响应
- 支持会话上下文、会话事件和请求幂等处理
- 可配置 OpenAI 兼容的模型地址、模型名称和 API Key
- 处理模型请求超时、临时错误和上游错误详情
- 支持工具调用、结构化响应和多步骤对话

### 工具与任务

- 工具注册、参数校验和权限控制
- 文件、搜索、任务、产物等内置工具
- 后台任务提交、状态查询、取消和输出读取
- 多步骤任务的状态保存、恢复和事件记录
- 支持多个 Agent 之间的串行、并行和委派调用

### 数据与运行状态

- SQLite 保存会话、任务和事件数据
- 任务和会话支持幂等键，避免重复执行
- 支持上下文整理与长会话摘要
- 提供健康检查、就绪检查和 Prometheus 指标
- 提供 A2A 任务接口及对应的 Web 页面

## 开始使用

### 环境要求

- Python 3.10 或更高版本
- 可访问的 OpenAI 兼容模型服务

### 安装

```bash
git clone https://github.com/zhu-weiyuan/simple-agent.git
cd simple-agent
pip install -e .
```

### 配置模型

复制环境变量示例文件：

```bash
cp .env.example .env
```

在 `.env` 中填写模型配置：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b
```

请根据所使用的模型服务调整 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。Ollama、LM Studio、vLLM 以及其他提供 OpenAI 兼容接口的服务都可以使用相同的配置方式。

### 启动服务

```bash
uvicorn app_prod:app --host 0.0.0.0 --port 8000
```

服务启动后：

- `http://localhost:8000`：聊天页面
- `http://localhost:8000/docs`：OpenAPI 接口文档
- `http://localhost:8000/dashboard`：运行状态页面
- `http://localhost:8000/a2a.html`：任务接口页面

命令行模式：

```bash
my-agent "列出当前目录中的文件"
```

## 常用接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/chat` | POST | 发送对话请求，可启用流式响应 |
| `/api/sessions` | GET | 获取会话列表 |
| `/api/session/{session_id}` | GET | 获取单个会话 |
| `/api/sessions/{session_id}/events` | GET | 获取会话事件 |
| `/api/jobs` | GET | 获取后台任务列表 |
| `/api/jobs/{job_id}` | GET | 获取任务状态 |
| `/api/jobs/{job_id}/cancel` | POST | 取消后台任务 |
| `/api/artifacts/{artifact_id}` | GET | 获取任务产物信息 |
| `/api/tools` | GET | 获取可用工具 |
| `/api/card` | GET | 获取 Agent 信息 |
| `/api/health` | GET | 获取服务健康状态 |
| `/api/ready` | GET | 就绪检查 |
| `/healthz` | GET | 存活检查 |
| `/api/metrics` | GET | 获取 Prometheus 指标 |

发送普通对话请求：

```bash
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"介绍一下当前项目"}'
```

发送流式对话请求：

```bash
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"列出当前目录中的文件","stream":true}'
```

提交 A2A 任务：

```bash
curl http://localhost:8000/a2a/messages \
  -H "Content-Type: application/json" \
  -d '{"task_id":"example-1","content":"执行一个简单任务"}'
```

## 项目结构

```text
simple-agent/
├── app_prod.py          # FastAPI 应用入口
├── src/my_agent/        # Agent 核心代码
│   ├── core/            # 对话、上下文和执行逻辑
│   ├── llm/             # 模型客户端和错误处理
│   ├── tools/           # 工具注册、校验和内置工具
│   ├── memory/          # 记忆、检索和持久化
│   ├── security/        # 输入检查和权限处理
│   ├── graph/           # 任务编排和状态图
│   ├── jobs.py          # 后台任务管理
│   └── a2a.py           # A2A 任务接口
├── web/                 # 聊天、仪表盘和任务页面
├── tests/               # 单元测试和集成测试
├── examples/            # 示例配置和示例代码
├── docs/                # 项目文档
├── evals/               # 评测脚本和数据
└── runtime/             # 本地运行时数据
```

## 测试与开发

```bash
# 运行全部测试
pytest tests/

# 运行一组核心回归测试
pytest tests/test_llm_template_errors.py tests/test_chat_idempotency.py \
       tests/test_structured_chat_contract.py tests/test_a2a_enhanced.py

# 代码检查和格式化
ruff check --fix
ruff format

# 类型检查
mypy src/
```

评测脚本位于 `evals/`，评测数据位于 `evals/datasets/`。涉及模型服务的评测需要先配置对应的环境变量。

## 许可证

本项目使用 [MIT License](LICENSE)。
