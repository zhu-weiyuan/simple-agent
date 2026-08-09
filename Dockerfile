# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- dependency layer (cached unless pyproject/uv.lock change) ----
FROM base AS deps
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir "uv==0.5.*" \
    && uv sync --frozen --no-dev --system

# ---- runtime ----
FROM base AS runtime
WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=deps /app/src ./src
COPY pyproject.toml uv.lock ./

# Non-root user
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/runtime /app/checkpoints \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "app_prod.py"]
