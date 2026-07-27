# -*- coding: utf-8 -*-
"""
my_agent.metrics — P6: 与 observability.MetricsCollector 合并为一套。

原本此模块有一个独立的 MetricsCollector(record_request/get_metrics->str),
与 observability 的 Prometheus 收集器重复。现统一保留 observability 那套接口,
此模块仅作兼容 shim:

- ``MetricsCollector`` / ``metrics`` 直接指向 observability 的实现与全局单例
- ``record_request(endpoint, status_code, duration_ms)`` 便捷函数映射为
  counter + histogram(合法 Prometheus 文本由 observability 输出)
- ``get_prometheus_text()`` 输出合法 Prometheus 文本
"""

from .observability import MetricsCollector, get_metrics

# Global metrics instance (shared with observability)
metrics = get_metrics()


def record_request(endpoint: str, status_code: int, duration_ms: float,
                   method: str = "") -> None:
    """Record an HTTP request into the unified collector."""
    labels = {"endpoint": endpoint, "status": str(status_code)}
    if method:
        labels["method"] = method
    metrics.increment_counter("http_requests_total", labels=labels)
    if status_code >= 400:
        metrics.increment_counter("http_errors_total", labels={"endpoint": endpoint})
    metrics.observe_histogram("http_request_duration_ms", duration_ms,
                              labels={"endpoint": endpoint})


def get_prometheus_text() -> str:
    """合法 Prometheus 文本 (统一走 observability 实现)。"""
    return metrics.get_prometheus_text()


__all__ = ["MetricsCollector", "metrics", "record_request", "get_prometheus_text"]
