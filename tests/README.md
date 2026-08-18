# tests

这里放面向源码和服务行为的自动化回归测试，由 pytest 运行。新测试一律放在本目录，评测运行器和数据集不要混入这里。

## 主要分区

- `test_*.py`：模块、接口、持久化、韧性、安全和生命周期回归测试。
- `test/`：历史兼容回归目录；除非维护旧行为，新测试不要放入这里。
- `../evals/`：独立的 Agent 评测 harness、数据集、评测运行器和报告；测试只验证其行为，不承载评测数据。

推荐入口：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

运行评测数据集格式校验：

```powershell
.\.venv\Scripts\python.exe evals\run_eval.py --validate
```