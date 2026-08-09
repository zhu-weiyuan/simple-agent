# 文档导航

这里集中放学习、架构、运维和评测入口。源码仍保留在原来的包结构中，避免为了整理破坏导入路径。

- LEARNING_GUIDE.md：源码学习路线。
- EVALUATION_GUIDE.md：测试、评测 harness、黄金数据集和 benchmark。
- architecture/：架构说明索引。
- development/：开发、面试和框架 QA 文档索引。
- operations/：部署、健康检查、预算和监控文档索引。
- history/：Phase/PM 历史记录索引。

当前生产入口：app_prod.py；兼容 shim：app.prod.py；旧版入口：app.py。实际实现以 app_prod.py、src/my_agent/ 和测试结果为准。
