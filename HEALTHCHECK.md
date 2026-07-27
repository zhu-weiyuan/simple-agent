# SimpleAgent Health Check & Monitoring Guide

## Overview

This document describes the health check endpoints, monitoring strategy, and alerting configuration for SimpleAgent in production.

## Health Check Endpoints

### 1. `/api/health` - Basic Health Status

**Purpose**: Quick liveness check for load balancers and orchestrators.

**Method**: GET

**Response Time**: < 100ms

**Success Response** (200 OK):
```json
{
  "ok": true,
  "agent": "SimpleAgent",
  "version": "2.0",
  "uptime_seconds": 3600,
  "platform": "Windows 10",
  "python": "3.11.9",
  "system": {
    "cpu_percent": 25.5,
    "memory_total_gb": 16.0,
    "memory_used_gb": 7.2,
    "memory_percent": 45.0,
    "disk_total_gb": 500,
    "disk_used_gb": 300,
    "disk_percent": 60.0
  },
  "llm": {
    "reachable": true,
    "base_url": "http://localhost:8080",
    "model": "qwen3.5"
  },
  "requests": {
    "total": 1523,
    "errors": 12
  }
}
```

**Failure Response** (503 Service Unavailable):
```json
{
  "ok": false,
  "error": "LLM service unreachable"
}
```

**Implementation**:
```python
@app.get("/api/health")
async def health():
    """Enhanced health check with system metrics."""
    # System metrics via psutil
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # LLM connectivity check
    llm_reachable = check_llm_connectivity()
    
    return {
        "ok": True,
        "agent": agent.name,
        "version": agent.version,
        "uptime_seconds": int(time.time() - psutil.boot_time()),
        "system": {...},
        "llm": {"reachable": llm_reachable, ...},
        "requests": _request_counter,
    }
```

### 2. `/api/ready` - Readiness Probe

**Purpose**: Determine if the service is ready to accept production traffic.

**Checks**:
- ✅ LLM API connectivity
- ✅ Redis connection
- ✅ Database/file system write access
- ✅ Configuration validation complete

**Method**: GET

**Success Response** (200 OK):
```json
{
  "ready": true,
  "checks": {
    "llm": {"status": "ok", "latency_ms": 45},
    "redis": {"status": "ok", "latency_ms": 2},
    "storage": {"status": "ok", "writable": true},
    "config": {"status": "ok", "validated": true}
  }
}
```

**Failure Response** (503 Service Unavailable):
```json
{
  "ready": false,
  "checks": {
    "llm": {"status": "error", "message": "Connection timeout"},
    "redis": {"status": "ok", "latency_ms": 2},
    "storage": {"status": "ok", "writable": true},
    "config": {"status": "ok", "validated": true}
  },
  "failing_check": "llm"
}
```

**Implementation**:
```python
@app.get("/api/ready")
async def readiness():
    """Readiness probe for Kubernetes/orchestrators."""
    checks = {}
    
    # LLM check
    try:
        start = time.time()
        llm_ok = check_llm_connectivity(timeout=3)
        checks["llm"] = {
            "status": "ok" if llm_ok else "error",
            "latency_ms": round((time.time() - start) * 1000, 2)
        }
    except Exception as e:
        checks["llm"] = {"status": "error", "message": str(e)}
    
    # Redis check
    try:
        start = time.time()
        redis_ok = redis_client.ping()
        checks["redis"] = {
            "status": "ok" if redis_ok else "error",
            "latency_ms": round((time.time() - start) * 1000, 2)
        }
    except Exception as e:
        checks["redis"] = {"status": "error", "message": str(e)}
    
    # Storage check
    try:
        test_file = Path("memory/.health_check")
        test_file.write_text("test")
        test_file.unlink()
        checks["storage"] = {"status": "ok", "writable": True}
    except Exception as e:
        checks["storage"] = {"status": "error", "message": str(e)}
    
    all_ok = all(c["status"] == "ok" for c in checks.values())
    
    return {
        "ready": all_ok,
        "checks": checks,
        "failing_check": next((k for k, v in checks.items() if v["status"] != "ok"), None)
    }
```

### 3. `/api/metrics` - Prometheus Metrics

**Purpose**: Expose metrics for Prometheus scraping.

**Method**: GET

**Content-Type**: `text/plain; version=0.0.4`

**Sample Output**:
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/api/chat",status="success"} 1511
http_requests_total{endpoint="/api/chat",status="error"} 12
http_requests_total{endpoint="/api/health",status="success"} 450

# HELP http_request_duration_ms Request duration histogram
# TYPE http_request_duration_ms histogram
http_request_duration_ms_bucket{endpoint="/api/chat",le="100"} 234
http_request_duration_ms_bucket{endpoint="/api/chat",le="500"} 890
http_request_duration_ms_bucket{endpoint="/api/chat",le="1000"} 1234
http_request_duration_ms_bucket{endpoint="/api/chat",le="2000"} 1456
http_request_duration_ms_bucket{endpoint="/api/chat",le="+Inf"} 1523
http_request_duration_ms_sum{endpoint="/api/chat"} 456789.23
http_request_duration_ms_count{endpoint="/api/chat"} 1523

# HELP agent_uptime_seconds Agent uptime
# TYPE agent_uptime_seconds gauge
agent_uptime_seconds 3600

# HELP system_memory_usage_percent Memory usage
# TYPE system_memory_usage_percent gauge
system_memory_usage_percent 45.2

# HELP llm_request_latency_ms LLM request latency
# TYPE llm_request_latency_ms histogram
llm_request_latency_ms_bucket{le="500"} 1200
llm_request_latency_ms_bucket{le="1000"} 1450
llm_request_latency_ms_bucket{le="2000"} 1500
llm_request_latency_ms_bucket{le="+Inf"} 1523
llm_request_latency_ms_sum 1234567.89
llm_request_latency_ms_count 1523

# HELP cache_hit_ratio Redis cache hit ratio
# TYPE cache_hit_ratio gauge
cache_hit_ratio 0.75

# HELP active_sessions_count Active user sessions
# TYPE active_sessions_count gauge
active_sessions_count 42
```

**Key Metrics**:

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by endpoint and status |
| `http_request_duration_ms` | Histogram | Request latency distribution |
| `http_errors_total` | Counter | Total errors by type |
| `agent_uptime_seconds` | Gauge | Service uptime |
| `system_memory_usage_percent` | Gauge | System memory usage |
| `llm_request_latency_ms` | Histogram | LLM call latency |
| `cache_hit_ratio` | Gauge | Redis cache hit ratio |
| `active_sessions_count` | Gauge | Current active sessions |
| `rate_limit_events_total` | Counter | Rate limit triggers |

## Monitoring Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  SimpleAgent    │────▶│  Prometheus  │────▶│   Grafana    │
│  /api/metrics   │     │   (scrape)   │     │  (dashboard) │
└─────────────────┘     └──────────────┘     └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Alertmanager│
                       └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Slack/Email │
                       └──────────────┘
```

## Prometheus Configuration

### `prometheus.yml`
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'simple-agent'
    static_configs:
      - targets: ['simple-agent:8000']
    metrics_path: '/api/metrics'
    scrape_timeout: 10s
    
  - job_name: 'llm-service'
    static_configs:
      - targets: ['host.docker.internal:8080']
    metrics_path: '/metrics'
    scrape_timeout: 5s
```

## Alerting Rules

### `alerting_rules.yml`
```yaml
groups:
  - name: simple-agent-alerts
    rules:
      # High Error Rate
      - alert: SimpleAgentHighErrorRate
        expr: |
          sum(rate(http_requests_total{status="error"}[5m])) 
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on SimpleAgent"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
          
      # High Latency (P95)
      - alert: SimpleAgentHighLatency
        expr: |
          histogram_quantile(0.95, 
            sum(rate(http_request_duration_ms_bucket[5m])) by (le, endpoint)
          ) > 2000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on SimpleAgent"
          description: "P95 latency is {{ $value }}ms for endpoint {{ $labels.endpoint }}"
          
      # LLM Unreachable
      - alert: SimpleAgentLLMUnreachable
        expr: |
          probe_success{job="llm-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM service is unreachable"
          description: "LLM service has been unreachable for more than 1 minute"
          
      # High Memory Usage
      - alert: SimpleAgentHighMemory
        expr: system_memory_usage_percent > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on SimpleAgent"
          description: "Memory usage is {{ $value }}%"
          
      # Service Down
      - alert: SimpleAgentDown
        expr: up{job="simple-agent"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SimpleAgent service is down"
          description: "SimpleAgent has been down for more than 1 minute"
          
      # Rate Limit Triggered Frequently
      - alert: SimpleAgentFrequentRateLimiting
        expr: rate(rate_limit_events_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Frequent rate limiting on SimpleAgent"
          description: "Rate limit triggered {{ $value }} times per second"
```

## Grafana Dashboard Panels

### Recommended Panels:

1. **Request Rate** (Graph)
   - Query: `rate(http_requests_total[5m])`
   - Group by: endpoint

2. **Error Rate** (Graph)
   - Query: `rate(http_requests_total{status="error"}[5m]) / rate(http_requests_total[5m])`
   - Threshold: 5%

3. **Latency Distribution** (Heatmap)
   - Query: `rate(http_request_duration_ms_bucket[5m])`

4. **P95/P99 Latency** (Stat)
   - Query: `histogram_quantile(0.95/0.99, rate(http_request_duration_ms_bucket[5m]))`

5. **LLM Latency** (Graph)
   - Query: `histogram_quantile(0.50/0.95, rate(llm_request_latency_ms_bucket[5m]))`

6. **Cache Hit Ratio** (Gauge)
   - Query: `cache_hit_ratio`
   - Thresholds: < 50% (red), < 75% (yellow), > 75% (green)

7. **Active Sessions** (Stat)
   - Query: `active_sessions_count`

8. **System Resources** (Time Series)
   - CPU: `system_cpu_percent`
   - Memory: `system_memory_usage_percent`
   - Disk: `system_disk_usage_percent`

## Log Aggregation

### Structured Log Format
```json
{
  "timestamp": "2026-07-26T10:30:45.123Z",
  "level": "INFO",
  "logger": "my_agent",
  "message": "Request processed successfully",
  "request_id": "req-abc123",
  "session_id": "sess-xyz789",
  "user_ip": "192.168.1.100",
  "endpoint": "/api/chat",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 245.3,
  "llm_tokens": 128,
  "cache_hit": false,
  "module": "agent",
  "function": "run",
  "line": 142
}
```

### ELK Stack Configuration

**Filebeat**:
```yaml
filebeat.inputs:
  - type: container
    paths:
      - /var/lib/docker/containers/*/*.log
    processors:
      - add_kubernetes_metadata:
          host: ${NODE_NAME}
          matchers:
            - logs_path:
                logs_path: "/var/lib/docker/containers/"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "simple-agent-%{+yyyy.MM.dd}"
```

**Kibana Dashboard**:
- Request volume over time
- Error distribution by type
- Slow queries (> 2s)
- Session activity heatmap
- Geographic distribution (if available)

## Circuit Breaker Metrics

Monitor circuit breaker state for LLM calls:

```
# HELP circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half-open)
# TYPE circuit_breaker_state gauge
circuit_breaker_state{service="llm"} 0

# HELP circuit_breaker_failures_total Circuit breaker failure count
# TYPE circuit_breaker_failures_total counter
circuit_breaker_failures_total{service="llm"} 3

# HELP circuit_breaker_successes_total Circuit breaker success count
# TYPE circuit_breaker_successes_total counter
circuit_breaker_successes_total{service="llm"} 1520
```

## Performance Benchmarks

### Target Metrics

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| P50 Latency | < 500ms | > 800ms | > 1500ms |
| P95 Latency | < 1500ms | > 2000ms | > 3000ms |
| P99 Latency | < 2500ms | > 3500ms | > 5000ms |
| Error Rate | < 1% | > 3% | > 5% |
| Cache Hit Ratio | > 75% | < 50% | < 25% |
| Uptime | > 99.9% | < 99% | < 95% |

### Load Testing

Run load tests with k6 or Locust:

```python
# locustfile.py
from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def chat(self):
        self.client.post("/api/chat", json={
            "message": "Hello, how are you?",
            "stream": False
        })
```

```bash
locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10
```

## Incident Response

### Runbook: High Error Rate

1. Check `/api/health` endpoint
2. Review recent logs for error patterns
3. Check LLM service status
4. Verify Redis connectivity
5. Check resource utilization (CPU, memory, disk)
6. Review recent deployments/changes
7. Scale horizontally if load-related
8. Enable debug logging if needed

### Runbook: High Latency

1. Check LLM response times
2. Review cache hit ratio
3. Check database query times
4. Analyze slow request logs
5. Review resource constraints
6. Consider scaling or optimization

## Contact & Escalation

- **Level 1**: On-call engineer (PagerDuty)
- **Level 2**: Platform team lead
- **Level 3**: System architect

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-07-26 | SimpleAgent Team | Initial health check documentation |
