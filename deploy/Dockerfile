# SciX MCP HTTP server — hardened container image.
#
# - Python 3.12 slim base
# - Non-root user (uid 1001)
# - No build toolchain in final image (multi-stage)
# - HF models mounted read-only from host cache (not baked in)
# - DB connection to host via host-gateway
#
# Build:   docker build -f deploy/Dockerfile -t scix-mcp:latest .
# Run:     see deploy/docker-compose.yml or deploy/run.sh

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install build deps only in the builder stage so they don't bloat the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed to resolve dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install '.[mcp,search,embed]' uvicorn

# ---- final image ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Runtime-only libs (libpq for psycopg, no compiler).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user.
RUN groupadd --system --gid 1001 scix && \
    useradd  --system --uid 1001 --gid 1001 --home-dir /home/scix --shell /usr/sbin/nologin scix && \
    mkdir -p /home/scix /app && \
    chown -R scix:scix /home/scix /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=scix:scix src /app/src

WORKDIR /app
USER scix:scix

# The HF cache is mounted read-only at runtime via docker-compose.
# The DB DSN is injected via SCIX_DSN env var.
EXPOSE 8000

# Simple HTTP healthcheck so compose/k8s know when the service is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "scix.mcp_server_http"]
