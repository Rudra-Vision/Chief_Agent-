#!/usr/bin/env bash
# Start the Chief Agent API + dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/backend"

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

if [ ! -x .venv/bin/python ]; then
  echo "No virtualenv found. Create one first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "Chief Agent starting on http://${HOST}:${PORT}"
exec .venv/bin/python -m uvicorn chief_agent.api.app:get_app --factory --host "$HOST" --port "$PORT" --log-level info
