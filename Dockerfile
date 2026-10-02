# ---------------------------------------------------------------------------
# Chief Agent - production image
#
# Multi-stage: the frontend is built with Node, then both the built assets and
# the Python backend are copied into a slim runtime image.
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

# Minimal system dependencies. tini gives us correct signal handling so the
# container stops cleanly (important for a trading process).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini curl postgresql-client \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY alembic.ini ./
COPY --from=frontend /build/dist ./frontend/dist

# The runtime user must not be root, and the data directory must be writable so
# it can be mounted as a persistent volume.
RUN useradd --create-home --uid 10001 chief \
 && mkdir -p /app/var/cache /app/var/backups /app/research_data \
 && chown -R chief:chief /app

USER chief
VOLUME ["/app/var"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "python scripts/migrate.py --init && exec python -m uvicorn chief_agent.api.app:get_app --factory --host 0.0.0.0 --port 8000"]
