"""Tests for Prometheus metrics collector."""

import sys
sys.path.insert(0, r"C:\Users\Administrator\.openclaw\workspace\simple-agent\src")

from my_agent.metrics import MetricsCollector


def test_record_request_success():
    """Successful request should increment counter and record time."""
    m = MetricsCollector()
    m.record_request("/api/chat", 200, 150.0)
    
    assert m.request_counts["/api/chat"] == 1
    assert m.error_counts.get("/api/chat", 0) == 0


def test_record_request_error():
    """Error request should increment error counter."""
    m = MetricsCollector()
    m.record_request("/api/chat", 500, 200.0)
    
    assert m.request_counts["/api/chat"] == 1
    assert m.error_counts["/api/chat"] == 1


def test_get_metrics_format():
    """Metrics output should contain Prometheus-style lines."""
    m = MetricsCollector()
    m.record_request("/test", 200, 50.0)
    
    metrics_text = m.get_metrics()
    assert "http_requests_total{endpoint=\"/test\"}" in metrics_text
    assert "process_uptime_seconds" in metrics_text


def test_error_rate_calculation():
    """Error rate should be correctly calculated."""
    m = MetricsCollector()
    m.record_request("/api", 200, 10.0)
    m.record_request("/api", 500, 20.0)
    
    metrics_text = m.get_metrics()
    assert "http_error_rate{endpoint=\"/api\"}" in metrics_text
