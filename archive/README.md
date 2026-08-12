# 归档区

这里保存的是为了追溯而保留、但**不再参与当前服务启动、测试或运行时状态**的文件。归档只做移动，不删除内容。

## 分类

- `legacy-root-files/`：已停用的根目录启动器、过期 API 清单和历史会话快照。
  - `start.bat` 原来尝试启动不存在的 `app:app`，已由根目录 `start_simple_agent.bat` 取代。
  - 生产 Python 入口仍是 `app_prod.py`；`app.prod.py` 仍保留为兼容 shim，故没有移动。
- `runtime-artifacts/reliability50-attempts/`：Reliability-50 的中间/失败/调试运行产物。不要把这些报告当最终结论。
- `runtime-artifacts/browser-debug/`：浏览器调试文本、DOM 快照和临时检查输出。
- `runtime-artifacts/legacy-evaluation/`：早期小规模评测产物。
- `runtime-artifacts/server-logs/`：历史服务日志。当前运行时数据库仍留在 `runtime/`，未被移动。

> `enhanced_state.json` 是当前 Agent 的可选持久化状态，`src/my_agent/agent.py` 仍使用它，故不在归档区。

## 保留的正式评测证据

最终一次可复核的 Reliability-50 受控在线评测报告、逐条轨迹和浏览器截图在：

```text
evals/reports/reliability50/controlled-online-run/
```

该目录的 README 说明了结论范围和不能据此声称“所有真实输入永远 100%”的原因。
