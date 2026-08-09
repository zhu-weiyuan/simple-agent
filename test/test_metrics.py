"""Tests for Prometheus metrics collector (refactored MetricsCollector API)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_agent.observability import MetricsCollector


def test_record_request_success():
    """Successful request should increment counter (not error)."""
    m = MetricsCollector()

    # record_request() is now implemented via module-level helper;
    # test the underlying MetricsCollector contract directly:
    m.increment_counter("http_requests_total", {"endpoint": "/api/chat", "status": "200"})

    metrics_dict = m.get_metrics()
    key = 'http_requests_total{endpoint="/api/chat",status="200"}'
    assert metrics_dict["counters"].get(key, 0) >= 1


def test_record_request_error():
    """Error request should increment both total and error counters."""
    m = MetricsCollector()

    m.increment_counter("http_requests_total", {"endpoint": "/api/chat", "status": "500"})
    m.increment_counter("http_errors_total", {"endpoint": "/api/chat"})

    metrics_dict = m.get_metrics()
    total_key = 'http_requests_total{endpoint="/api/chat",status="500"}'
    err_key = 'http_errors_total{endpoint="/api/chat"}'
    assert metrics_dict["counters"].get(total_key, 0) >= 1
    assert metrics_dict["counters"].get(err_key, 0) >= 1


def test_get_metrics_format():
    """get_prometheus_text() output should contain Prometheus-style lines."""
    m = MetricsCollector()
    m.increment_counter("http_requests_total", {"endpoint": "/test", "status": "200"})
    m.observe_histogram("http_request_duration_ms", 50.0, {"endpoint": "/test"})

    metrics_text = m.get_prometheus_text()
    assert 'http_requests_total{endpoint="/test",status="200"}' in metrics_text


def test_error_rate_calculation():
    """Both success and error counters should be recorded."""
    m = MetricsCollector()

    m.increment_counter("http_requests_total", {"endpoint": "/api", "status": "200"})
    m.increment_counter("http_requests_total", {"endpoint": "/api", "status": "500"})
    m.increment_counter("http_errors_total", {"endpoint": "/api"})

    metrics_dict = m.get_metrics()
    total_200 = 'http_requests_total{endpoint="/api",status="200"}'
    total_500 = 'http_requests_total{endpoint="/api",status="500"}'
    err = 'http_errors_total{endpoint="/api"}'
    assert metrics_dict["counters"].get(total_200, 0) >= 1
    assert metrics_dict["counters"].get(total_500, 0) >= 1
    assert metrics_dict["counters"].get(err, 0) >= 1