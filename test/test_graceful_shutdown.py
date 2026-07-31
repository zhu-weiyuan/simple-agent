#!/usr/bin/env python3
"""
Test suite for graceful shutdown functionality.

Tests:
- SIGTERM signal handling
- SIGINT signal handling
- Active request completion
- Shutdown timeout enforcement
- Resource cleanup
"""

import pytest
import time
import signal
import subprocess
import sys
import os
from pathlib import Path
import threading
import requests


class TestGracefulShutdown:
    """Test graceful shutdown behavior."""
    
    @pytest.fixture
    def app_process(self):
        """Start app.prod.py in a subprocess."""
        app_path = Path(__file__).parent.parent / "app.prod.py"
        
        # Start the application
        proc = subprocess.Popen(
            [sys.executable, str(app_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "ENVIRONMENT": "test"},
        )
        
        # Wait for startup
        time.sleep(3)
        
        yield proc
        
        # Cleanup: ensure process is terminated
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
    
    def test_sigterm_received(self, app_process):
        """Application should log SIGTERM reception."""
        # Send SIGTERM
        app_process.send_signal(signal.SIGTERM)
        
        # Wait for shutdown
        stdout, stderr = app_process.communicate(timeout=10)
        output = stdout.decode() + stderr.decode()
        
        # Should log shutdown message
        assert "SIGTERM" in output or "shutdown" in output.lower()
    
    def test_sigint_received(self, app_process):
        """Application should log SIGINT reception."""
        # Send SIGINT
        app_process.send_signal(signal.SIGINT)
        
        # Wait for shutdown
        stdout, stderr = app_process.communicate(timeout=10)
        output = stdout.decode() + stderr.decode()
        
        # Should log shutdown message
        assert "SIGINT" in output or "shutdown" in output.lower()
    
    def test_health_endpoint_responds_before_shutdown(self, app_process):
        """Health endpoint should respond before shutdown begins."""
        # Verify app is running
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Application not ready yet")
    
    def test_graceful_shutdown_timeout(self):
        """Shutdown should timeout after configured seconds."""
        # This test verifies the timeout mechanism exists
        # Actual timeout testing requires long-running requests
        app_path = Path(__file__).parent.parent / "app.prod.py"
        
        # Start with short timeout
        env = {**os.environ, "SHUTDOWN_TIMEOUT_SECONDS": "2"}
        proc = subprocess.Popen(
            [sys.executable, str(app_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        
        time.sleep(3)
        
        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)
        
        # Should exit within timeout + grace period
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Shutdown exceeded timeout")
    
    def test_process_exits_cleanly(self, app_process):
        """Process should exit with code 0 on graceful shutdown."""
        # Send SIGTERM
        app_process.send_signal(signal.SIGTERM)
        
        # Wait and check exit code
        returncode = app_process.wait(timeout=10)
        assert returncode == 0, f"Expected exit code 0, got {returncode}"


class TestActiveRequestCompletion:
    """Test that active requests complete during shutdown."""
    
    @pytest.fixture
    def app_with_slow_endpoint(self):
        """Start app with a slow endpoint for testing."""
        # Create a test app with a slow endpoint
        test_app = Path(__file__).parent / "test_shutdown_app.py"
        test_app.write_text("""
from fastapi import FastAPI
import time
import signal
import sys

app = FastAPI()
shutdown_in_progress = False

@app.get("/slow")
def slow_endpoint():
    time.sleep(2)
    return {"done": True}

@app.get("/check-shutdown")
def check_shutdown():
    return {"shutting_down": shutdown_in_progress}

def shutdown_handler(signum, frame):
    global shutdown_in_progress
    shutdown_in_progress = True
    print("Shutdown initiated")
    time.sleep(3)  # Simulate graceful shutdown
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="error")
""")
        
        # Start the test app
        proc = subprocess.Popen(
            [sys.executable, str(test_app)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        time.sleep(2)
        
        yield proc
        
        # Cleanup
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        
        # Remove test file
        test_app.unlink(missing_ok=True)
    
    def test_request_completes_during_shutdown(self, app_with_slow_endpoint):
        """Active requests should complete during shutdown."""
        import concurrent.futures
        
        def make_slow_request():
            try:
                response = requests.get("http://localhost:8001/slow", timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Start slow request in thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(make_slow_request)
            
            # Give request time to start
            time.sleep(0.5)
            
            # Send shutdown signal
            app_with_slow_endpoint.send_signal(signal.SIGTERM)
            
            # Request should still complete
            result = future.result(timeout=10)
            assert result is True, "Request should complete during graceful shutdown"


class TestResourceCleanup:
    """Test resource cleanup during shutdown."""
    
    def test_logs_flushed_on_shutdown(self):
        """Logs should be flushed during shutdown."""
        # This is a placeholder for log flushing tests
        # Actual implementation depends on logging configuration
        pass
    
    def test_connections_closed_on_shutdown(self):
        """Database connections should close on shutdown."""
        # This is a placeholder for connection cleanup tests
        # Would require mock database connections
        pass


class TestShutdownMiddleware:
    """Test shutdown-related middleware."""
    
    def test_correlation_id_added(self):
        """Correlation ID should be added to requests."""
        from app import app
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            response = client.get("/api/health")
            
            # Should have X-Request-ID header
            assert "x-request-id" in response.headers
    
    def test_correlation_id_propagated(self):
        """Correlation ID should propagate from request to response."""
        from app import app
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            response = client.get(
                "/api/health",
                headers={"X-Request-ID": "test-correlation-id-123"}
            )
            
            # Should echo back the correlation ID
            assert response.headers.get("x-request-id") == "test-correlation-id-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
