# SimpleAgent Production Deployment Guide

## Overview

This guide covers production deployment hardening for SimpleAgent v2.0, implementing JavaGuide "AI Application System Design" best practices.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  SimpleAgent │────▶│    Redis    │
│             │     │   (FastAPI)  │     │  (Cache)    │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  LLM Service │
                   │ (llama.cpp)  │
                   └──────────────┘
```

## Quick Start

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Set environment variables
export API_KEY=your-secret-api-key
export JWT_SECRET=your-jwt-secret-min-32-chars
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=http://your-llm-service:8080

# Start with production profile
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

## Environment Variables

### Required
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | LLM API key | `sk-local` |
| `OPENAI_BASE_URL` | LLM endpoint URL | `http://host.docker.internal:8080` |

### Security
| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API authentication key | (empty) |
| `JWT_SECRET` | JWT signing secret (min 32 chars) | `change-me-in-production` |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `JWT_EXPIRATION_MINUTES` | JWT token validity | `60` |

### Rate Limiting
| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_REQUESTS` | Max requests per window | `30` |
| `RATE_LIMIT_WINDOW` | Window size in seconds | `60` |
| `MAX_CONCURRENT_REQUESTS` | Max concurrent requests | `10` |

### Observability
| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `METRICS_ENABLED` | Enable Prometheus metrics | `true` |
| `TRACING_ENABLED` | Enable distributed tracing | `true` |
| `CORRELATION_ID_HEADER` | Header name for request ID | `X-Request-ID` |

### Resource Management
| Variable | Description | Default |
|----------|-------------|---------|
| `REQUEST_TIMEOUT_SECONDS` | Request timeout | `120` |
| `SHUTDOWN_TIMEOUT_SECONDS` | Graceful shutdown timeout | `30` |

## Health Check Endpoints

### `/api/health` - Basic Health
```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "ok": true,
  "agent": "SimpleAgent",
  "version": "2.0",
  "uptime_seconds": 3600,
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "disk_percent": 60.1
  },
  "llm": {
    "reachable": true,
    "base_url": "http://localhost:8080",
    "model": "qwen3.5"
  }
}
```

### `/api/ready` - Readiness Probe
Checks if the service is ready to accept traffic:
- LLM connectivity
- Redis connection
- Database availability

```bash
curl http://localhost:8000/api/ready
```

### `/api/metrics` - Prometheus Metrics
```bash
curl http://localhost:8000/api/metrics
```

Sample output:
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/api/chat"} 1523
http_requests_total{endpoint="/api/health"} 450

# HELP http_request_duration_ms Request duration in milliseconds
# TYPE http_request_duration_ms summary
http_request_duration_ms{endpoint="/api/chat",quantile="0.50"} 245.32
http_request_duration_ms{endpoint="/api/chat",quantile="0.95"} 892.15
http_request_duration_ms{endpoint="/api/chat",quantile="0.99"} 1523.67

# HELP process_uptime_seconds Process uptime in seconds
# TYPE process_uptime_seconds gauge
process_uptime_seconds 3600.45
```

## Graceful Shutdown

The application handles SIGTERM and SIGINT signals:

1. Stop accepting new requests
2. Wait for in-flight requests to complete (max 30s)
3. Flush logs and metrics
4. Close database connections
5. Exit cleanly

```bash
# Docker sends SIGTERM by default
docker stop simple-agent-app

# Or manually
docker kill --signal=SIGTERM simple-agent-app
```

## Configuration Validation

On startup, the application validates:

- ✅ Environment variables are present
- ✅ API keys are non-empty (if required)
- ✅ LLM endpoint is reachable
- ✅ Redis connection succeeds
- ✅ Required directories are writable

Invalid configuration causes immediate exit with error code 1.

## Structured Logging

Logs are JSON-formatted with correlation IDs:

```json
{
  "timestamp": "2026-07-26T10:30:45.123Z",
  "level": "INFO",
  "logger": "my_agent",
  "message": "Request processed",
  "request_id": "req-abc123",
  "session_id": "sess-xyz789",
  "duration_ms": 245.3,
  "module": "agent",
  "function": "run",
  "line": 142
}
```

### Log Aggregation

Configure Docker logging driver or use Fluentd/Logstash:

```yaml
logging:
  driver: "fluentd"
  options:
    fluentd-address: "localhost:24224"
    tag: "simple-agent"
```

## Scaling Strategies

### Horizontal Scaling
```bash
# Docker Swarm
docker service scale simple-agent=3

# Kubernetes
kubectl scale deployment simple-agent --replicas=3
```

### Load Balancing
Use nginx or HAProxy:

```nginx
upstream simple_agent {
    least_conn;
    server agent1:8000;
    server agent2:8000;
    server agent3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://simple_agent;
        proxy_set_header X-Request-ID $request_id;
    }
}
```

### Session Affinity
For stateful sessions, enable sticky sessions:

```nginx
upstream simple_agent {
    ip_hash;
    server agent1:8000;
    server agent2:8000;
}
```

## Monitoring

### Prometheus Integration

1. Prometheus scrapes `/api/metrics` every 15s
2. Metrics stored for 15 days
3. Access at `http://localhost:9090`

### Grafana Dashboards

Access at `http://localhost:3000` (admin/admin)

Pre-configured dashboards:
- Request rate and latency
- Error rates by endpoint
- LLM response times
- Cache hit/miss ratios
- System resource usage

### Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
  - name: simple-agent
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_ms_bucket[5m])) > 2000
        for: 5m
        annotations:
          summary: "P95 latency exceeds 2s"
          
      - alert: LLMUnreachable
        expr: probe_success{job="llm-health"} == 0
        for: 1m
        annotations:
          summary: "LLM service is unreachable"
```

## Security Hardening

### Network Isolation
- Redis exposed only on localhost
- Database not exposed externally
- Services communicate via internal network

### Secrets Management
```bash
# Use Docker secrets (Swarm mode)
docker secret create api_key api_key.txt
docker secret create jwt_secret jwt_secret.txt

# Or use environment files
docker-compose --env-file .env.prod up -d
```

### Rate Limiting
- Per-IP rate limiting (30 req/min default)
- Configurable via environment variables
- Returns 429 with Retry-After header

### Input Validation
- Message length limits (4000 chars)
- PII detection and redaction
- Prompt injection detection

## Performance Tuning

### Redis Optimization
```yaml
command: ["redis-server", "--appendonly", "yes", "--maxmemory", "256mb", "--maxmemory-policy", "allkeys-lru"]
```

### Connection Pooling
Configure in application:
```python
REDIS_POOL_SIZE = 10
REDIS_MAX_CONNECTIONS = 20
```

### LLM Caching
Enable response caching for repeated queries:
```bash
export CACHE_ENABLED=true
export CACHE_TTL=3600
```

## Backup and Recovery

### Data Backup
```bash
# Backup Redis data
docker exec simple-agent-redis redis-cli BGSAVE
cp /var/lib/docker/volumes/redis_data/_data/dump.rdb ./backup/redis-$(date +%Y%m%d).rdb

# Backup application data
tar -czf backup/simple-agent-$(date +%Y%m%d).tar.gz ./memory ./checkpoints
```

### Recovery
```bash
# Restore Redis
cp ./backup/redis-20260726.rdb /var/lib/docker/volumes/redis_data/_data/dump.rdb
docker restart simple-agent-redis

# Restore application data
tar -xzf backup/simple-agent-20260726.tar.gz -C ./
```

## Troubleshooting

### Check Service Health
```bash
docker-compose ps
docker logs simple-agent-app
docker exec simple-agent-app curl http://localhost:8000/api/health
```

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
docker-compose up -d
docker logs -f simple-agent-app
```

### Resource Issues
```bash
# Check container resource usage
docker stats simple-agent-app

# Inspect container
docker inspect simple-agent-app
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Deploy
        run: |
          docker-compose -f docker-compose.prod.yml build
          docker-compose -f docker-compose.prod.yml up -d
```

### Health Check in CI
```yaml
- name: Wait for Health
  run: |
    for i in {1..30}; do
      curl -f http://localhost:8000/api/health && break
      sleep 5
    done
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-26 | Initial production deployment guide |
