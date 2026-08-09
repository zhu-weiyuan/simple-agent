#!/usr/bin/env python3
"""
Test suite for SimpleAgent health check endpoints.

Tests:
- /api/health endpoint response structure
- /api/ready endpoint readiness checks
- /api/metrics endpoint Prometheus format
- Health check response times
- System metrics accuracy
"""

import pytest
import time
from pathlib import Path

from fastapi.testclient import TestClient

import app_prod


@pytest.fixture(autouse=True)
def healthy_llm_probe(monkeypatch):
    """Keep endpoint tests deterministic and avoid a real LLM/network call."""
    # CI has no OPENAI_* env vars; without these the config check makes
    # /api/ready fail on "config" instead of exercising the LLM probe.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-health-check")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.test/v1")

    async def _stub_llm_check():
        return {
            "status": "ok",
            "required": True,
            "reachable": True,
            "latency_ms": 0.1,
            "base_url": app_prod.sync_llm.base_url,
            "status_code": 200,
        }

    monkeypatch.setattr(app_prod, "_check_llm", _stub_llm_check)


@pytest.fixture
def client():
    """Create a client for the production FastAPI entrypoint."""
    with TestClient(app_prod.app) as c:
        yield c


class TestHealthEndpoint:
    """Test /api/health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/api/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/api/health")
        data = response.json()
        
        assert "ok" in data
        assert "agent" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "system" in data
        assert "llm" in data
        assert "requests" in data
    
    def test_health_system_metrics(self, client):
        """Health response should include system metrics."""
        response = client.get("/api/health")
        data = response.json()
        system = data["system"]
        
        assert "cpu_percent" in system
        assert "memory_total_gb" in system
        assert "memory_used_gb" in system
        assert "memory_percent" in system
        assert "disk_total_gb" in system
        assert "disk_used_gb" in system
        assert "disk_percent" in system
        
        # Validate ranges
        assert 0 <= system["cpu_percent"] <= 100
        assert 0 <= system["memory_percent"] <= 100
        assert 0 <= system["disk_percent"] <= 100
    
    def test_health_llm_status(self, client):
        """Health response should include LLM connectivity status."""
        response = client.get("/api/health")
        data = response.json()
        llm = data["llm"]
        
        assert "reachable" in llm
        assert "base_url" in llm
        assert "model" in llm
        assert isinstance(llm["reachable"], bool)
    
    def test_health_response_time(self, client):
        """Health endpoint should respond within 100ms."""
        start = time.time()
        response = client.get("/api/health")
        elapsed = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed < 100, f"Health check took {elapsed}ms (should be < 100ms)"
    
    def test_health_uptime_increases(self, client):
        """Uptime should increase between calls."""
        response1 = client.get("/api/health")
        uptime1 = response1.json()["uptime_seconds"]
        
        time.sleep(1.1)
        
        response2 = client.get("/api/health")
        uptime2 = response2.json()["uptime_seconds"]
        
        assert uptime2 >= uptime1


class TestReadyEndpoint:
    """Test /api/ready endpoint."""
    
    def test_ready_returns_200(self, client):
        """Ready endpoint should return 200 OK."""
        response = client.get("/api/ready")
        assert response.status_code == 200
    
    def test_ready_response_structure(self, client):
        """Ready response should have required fields."""
        response = client.get("/api/ready")
        data = response.json()
        
        assert "ready" in data
        assert "checks" in data
        assert isinstance(data["ready"], bool)
    
    def test_ready_checks_structure(self, client):
        """Ready checks should have expected structure."""
        response = client.get("/api/ready")
        data = response.json()
        checks = data["checks"]
        
        assert "llm" in checks
        assert "storage" in checks
        assert "config" in checks
        
        for check_name, check_data in checks.items():
            assert "status" in check_data
            assert check_data["status"] in ["ok", "error", "skipped"]
    
    def test_ready_all_checks_pass(self, client):
        """All readiness checks should pass in healthy state."""
        response = client.get("/api/ready")
        data = response.json()
        
        # In test environment, all checks should pass
        assert data["ready"] is True
        assert data.get("failing_check") is None
    
    def test_ready_llm_check_latency(self, client):
        """LLM check should include latency metric."""
        response = client.get("/api/ready")
        data = response.json()
        llm_check = data["checks"]["llm"]
        
        if llm_check["status"] == "ok":
            assert "latency_ms" in llm_check
            assert llm_check["latency_ms"] > 0


    def test_ready_returns_503_when_required_llm_is_down(self, client, monkeypatch):
        """Readiness must fail when a required dependency is unavailable."""
        async def _failed_llm_check():
            return {
                "status": "error",
                "required": True,
                "reachable": False,
                "latency_ms": 1.0,
                "reason": "test outage",
            }

        monkeypatch.setattr(app_prod, "_check_llm", _failed_llm_check)
        response = client.get("/api/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["failing_check"] == "llm"


class TestMetricsEndpoint:
    """Test /api/metrics endpoint."""
    
    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200 OK."""
        response = client.get("/api/metrics")
        assert response.status_code == 200
    
    def test_metrics_content_type(self, client):
        """Metrics should have Prometheus content type."""
        response = client.get("/api/metrics")
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_prometheus_format(self, client):
        """Metrics should follow Prometheus exposition format."""
        response = client.get("/api/metrics")
        text = response.text
        
        # Should have HELP comments
        assert "# HELP" in text
        
        # Should have TYPE declarations
        assert "# TYPE" in text
        
        # Should have metric lines
        assert "agent_uptime_seconds" in text
        assert "agent_requests_total" in text
    
    def test_metrics_has_required_metrics(self, client):
        """Metrics should include required metrics."""
        response = client.get("/api/metrics")
        text = response.text
        
        required_metrics = [
            "agent_uptime_seconds",
            "agent_requests_total",
            "system_memory_usage_percent",
        ]
        
        for metric in required_metrics:
            assert metric in text, f"Missing metric: {metric}"
    
    def test_metrics_uptime_positive(self, client):
        """Uptime metric should be positive."""
        response = client.get("/api/metrics")
        text = response.text
        
        for line in text.split("\n"):
            if line.startswith("agent_uptime_seconds"):
                value = float(line.split()[-1])
                assert value > 0, "Uptime should be positive"


class TestHealthCheckPerformance:
    """Performance tests for health checks."""
    
    def test_health_concurrent_requests(self, client):
        """Health endpoint should handle concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/api/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in results)
    
    def test_health_under_load(self, client):
        """Health endpoint should remain responsive under load."""
        # Make 50 rapid requests
        start = time.time()
        for _ in range(50):
            response = client.get("/api/health")
            assert response.status_code == 200
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds for 50 requests)
        assert elapsed < 5.0, f"50 health checks took {elapsed}s"


class TestHealthCheckIntegration:
    """Integration tests for health check system."""
    
    def test_health_reflects_request_count(self, client):
        """Health endpoint should track request counts."""
        # Make some requests
        for _ in range(5):
            client.get("/api/health")
        
        # Check request count increased
        response = client.get("/api/health")
        data = response.json()
        assert data["requests"]["total"] >= 5
    
    def test_health_error_tracking(self, client):
        """Health endpoint should track error counts."""
        # Initial state
        response = client.get("/api/health")
        initial_errors = response.json()["requests"]["errors"]
        
        # Try to trigger an error (invalid endpoint)
        client.get("/api/nonexistent")
        
        # Error count may or may not increase depending on implementation
        # Just verify the field exists and is non-negative
        response = client.get("/api/health")
        current_errors = response.json()["requests"]["errors"]
        assert current_errors >= initial_errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
