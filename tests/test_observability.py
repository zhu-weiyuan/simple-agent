# -*- coding: utf-8 -*-
"""Tests for observability module (MetricsCollector + AlertService)."""

import pytest
import time
from my_agent.observability import MetricsCollector, AlertService


class TestMetricsCollector:
    """Test Prometheus-style metrics collection."""

    def setup_method(self):
        self.mc = MetricsCollector()

    def test_increment_counter(self):
        self.mc.increment_counter("requests")
        self.mc.increment_counter("requests")
        metrics = self.mc.get_metrics()
        assert metrics["counters"]["requests"] == 2

    def test_counter_with_labels(self):
        self.mc.increment_counter("requests", {"endpoint": "/api/chat"})
        self.mc.increment_counter("requests", {"endpoint": "/api/health"})
        metrics = self.mc.get_metrics()
        key_chat = 'requests{endpoint="/api/chat"}'
        key_health = 'requests{endpoint="/api/health"}'
        assert metrics["counters"][key_chat] == 1
        assert metrics["counters"][key_health] == 1

    def test_observe_histogram(self):
        self.mc.observe_histogram("latency", 50.0)
        self.mc.observe_histogram("latency", 200.0)
        metrics = self.mc.get_metrics()
        hist = metrics["histograms"]["latency"]
        assert hist["count"] == 2
        assert hist["avg_ms"] == 125.0

    def test_histogram_buckets(self):
        self.mc.observe_histogram("latency", 50.0)
        # 50ms should be in buckets >= 50
        metrics = self.mc.get_metrics()
        hist = metrics["histograms"]["latency"]
        # bucket 50 should have count 1
        assert int(hist["buckets"].get("50", 0)) >= 1

    def test_set_gauge(self):
        self.mc.set_gauge("active_connections", 42.0)
        metrics = self.mc.get_metrics()
        assert metrics["gauges"]["active_connections"] == 42.0

    def test_prometheus_text_format(self):
        self.mc.increment_counter("test_counter")
        self.mc.set_gauge("test_gauge", 1.5)
        text = self.mc.get_prometheus_text()
        assert "# TYPE test_counter counter" in text
        assert "test_counter 1" in text
        assert "# TYPE test_gauge gauge" in text
        assert "test_gauge 1.5" in text


class TestAlertService:
    """Test threshold-based alerting."""

    def setup_method(self):
        self.mc = MetricsCollector()
        self.asvc = AlertService(self.mc)

    def test_add_rule(self):
        self.asvc.add_rule("high_lat", "latency", threshold=1000)
        assert len(self.asvc._rules) == 1

    def test_no_alert_below_threshold(self):
        self.asvc.add_rule("high_lat", "request_latency", threshold=5000)
        self.mc.observe_histogram("request_latency", 100.0)
        triggered = self.asvc.check_and_alert()
        assert len(triggered) == 0

    def test_alert_above_threshold(self):
        self.asvc.add_rule("high_lat", "request_latency", threshold=50)
        self.mc.observe_histogram("request_latency", 1000.0)
        triggered = self.asvc.check_and_alert()
        assert "high_lat" in triggered

    def test_cooldown_prevents_duplicate(self):
        self.asvc.add_rule("high_lat", "request_latency",
                           threshold=50, cooldown_seconds=600)
        self.mc.observe_histogram("request_latency", 1000.0)
        self.asvc.check_and_alert()  # First trigger

        triggered = self.asvc.check_and_alert()  # Should be in cooldown
        assert len(triggered) == 0

    def test_callback_on_alert(self):
        alerts_received = []
        self.asvc.add_rule("test_alert", "latency", threshold=50)
        self.asvc.on_alert("test_alert", lambda n, a, t: alerts_received.append(n))
        self.mc.observe_histogram("latency", 1000.0)
        self.asvc.check_and_alert()
        assert "test_alert" in alerts_received


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
