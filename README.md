# SimpleAgent

SimpleAgent 是一个基于 Python 的 AI Agent 服务，提供对话、工具调用、会话管理、流式输出和异步任务接口。项目使用 FastAPI 提供 Web API，并可连接 OpenAI 兼容的模型服务，例如 Ollama、LM Studio、vLLM 或云端模型 API。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能

- 对话接口，支持普通 JSON 响应和 Server-Sent Events 流式响应
- 工具注册、参数校验和权限策略
- 会话持久化、会话事件记录和幂等请求处理
- 基于状态机的多步骤任务执行与恢复
- 长对话上下文整理和摘要压缩
- 后台任务提交、查询、取消和产物管理
- 多智能体的串行、并行和委派调用
- OpenAI 兼容 LLM 接口，包含重试、超时和错误诊断
- A2A 任务接口、健康检查和 Prometheus 指标

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- 一个 OpenAI 兼容的模型服务

### 安装

```bash
git clone https://github.com/zhu-weiyuan/simple-agent.git
cd simple-agent
pip install -e .
```

### 配置

```bash
cp .env.example .env
```

在 `.env` 中配置模型服务：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b
```

`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 请按实际模型服务调整。

### 运行

```bash
# 命令行
my-agent "列出当前目录中的文件"

# Web API
uvicorn app_prod:app --host 0.0.0.0 --port 8000
```

启动后可访问：

- `http://localhost:8000`：聊天页面
- `http://localhost:8000/docs`：OpenAPI 文档
- `http://localhost:8000/a2a.html`：A2A 控制台

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | 对话请求，支持流式和非流式输出 |
| `/api/conversations` | GET | 查询和管理会话 |
| `/api/tools` | GET | 查询已注册工具 |
| `/api/card` | GET | 获取 Agent 元信息 |
| `/api/health` | GET | 服务健康状态 |
| `/api/ready` | GET | 就绪检查 |
| `/healthz` | GET | 存活检查 |
| `/api/metrics` | GET | Prometheus 指标 |
| `/a2a/messages` | POST | 提交 A2A 任务 |
| `/a2a/tasks/{id}` | GET | 查询 A2A 任务状态 |
| `/a2a/tasks/{id}/cancel` | POST | 取消 A2A 任务 |

普通对话请求示例：

```bash
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"介绍一下当前项目"}'
```

流式请求示例：

```bash
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"列出当前目录中的文件","stream":true}'
```

## 项目结构

```text
simple-agent/
├── app_prod.py              # FastAPI 应用入口
├── src/my_agent/
│   ├── core/                # 对话与上下文处理
│   ├── llm/                 # OpenAI 兼容模型客户端
│   ├── tools/               # 工具系统
│   ├── memory/              # 记忆与持久化
│   ├── security/            # 输入和权限安全处理
│   ├── graph/               # 任务编排
│   └── a2a.py               # A2A 任务协议
├── web/                     # Web 页面资源
├── tests/                   # 测试
├── examples/                # 示例
└── docs/                    # 文档
```

## 测试与开发

```bash
# 运行全部测试
pytest tests/

# 代码检查和格式化
ruff check --fix
ruff format

# 类型检查
mypy src/
```

## 许可证

本项目使用 [MIT License](LICENSE)。

## 致谢

本项目在设计和实现中参考了开源社区中关于 Agent、任务编排和模型服务接入的实践。感谢相关开源项目及其贡献者。