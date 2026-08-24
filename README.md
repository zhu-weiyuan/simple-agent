# SimpleAgent

SimpleAgent 是一个用 Python 编写的 AI Agent 应用。它提供聊天服务、工具调用、会话与任务管理，并可接入 OpenAI 兼容的模型服务。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能

- 对话接口，支持普通响应和 SSE 流式响应
- 工具调用与参数校验
- 会话、任务和产物管理
- 多步骤任务执行、恢复与取消
- 上下文整理与会话摘要
- 多智能体的串行、并行和委派调用
- OpenAI 兼容的模型连接配置
- A2A 任务接口、健康检查和 Prometheus 指标

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

请根据所使用的模型服务调整 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

### 启动服务

```bash
uvicorn app_prod:app --host 0.0.0.0 --port 8000
```

服务启动后：

- `http://localhost:8000`：聊天页面
- `http://localhost:8000/docs`：接口文档
- `http://localhost:8000/dashboard`：运行状态页面

命令行模式：

```bash
my-agent "列出当前目录中的文件"
```

## 常用接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/chat` | POST | 发送对话请求，可启用流式响应 |
| `/api/sessions` | GET | 获取会话列表 |
| `/api/jobs` | GET | 获取后台任务列表 |
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

## 项目结构

```text
simple-agent/
├── app_prod.py          # FastAPI 应用入口
├── src/my_agent/        # Agent 核心代码
│   ├── core/            # 对话和上下文处理
│   ├── llm/             # 模型客户端
│   ├── tools/           # 工具系统
│   ├── memory/          # 记忆和持久化
│   ├── security/        # 安全处理
│   ├── graph/           # 任务编排
│   └── a2a.py           # A2A 任务接口
├── web/                 # Web 页面
├── tests/               # 测试
├── examples/            # 示例
├── docs/                # 文档
└── evals/               # 评测脚本和数据
```

## 开发

```bash
# 运行测试
pytest tests/

# 代码检查和格式化
ruff check --fix
ruff format

# 类型检查
mypy src/
```

## 许可证

本项目使用 [MIT License](LICENSE)。
