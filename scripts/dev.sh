#!/usr/bin/env bash
# Development mode: FastAPI (reload) + the Vite dev server with HMR.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/backend"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

.venv/bin/python -m uvicorn chief_agent.api.app:get_app --factory --host 0.0.0.0 --port 8000 --reload &
( cd frontend && npm run dev -- --host 0.0.0.0 --port 5173 ) &
wait
