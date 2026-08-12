# 运行产物归档

此目录下的内容均为诊断、浏览器检查或评测过程产物，不能作为当前服务的运行状态。

- `reliability50-attempts/`：中间运行、失败尝试和相关隔离服务输出。
- `browser-debug/`：用于排查页面乱码和浏览器问题的临时文本/快照。
- `legacy-evaluation/`：早期评测运行。
- `server-logs/`：历史日志。

当前服务使用的可变数据库（例如 `runtime/tasks.db`、`runtime/prompts.db`）仍在 `runtime/`，因此未移动。
