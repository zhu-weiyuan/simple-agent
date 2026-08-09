# 运维文档

DEPLOYMENT.md 讲部署，HEALTHCHECK.md 讲健康检查，ROUTING_BUDGET.md 讲多模型路由和预算，monitoring/ 放 Prometheus/Grafana 配置，scripts/diagnose.py 用于本地诊断。

当前生产入口是 app_prod.py；遇到版本或离线问题，先确认 8000 端口加载的是哪个入口。
