#!/usr/bin/env bash
# Container / VPS health probe. Exits non-zero when the API is not HEALTHY.
set -euo pipefail
PORT="${API_PORT:-8000}"
STATUS=$(curl -fsS "http://127.0.0.1:${PORT}/health" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "STOPPED")
echo "health: ${STATUS}"
[ "${STATUS}" = "HEALTHY" ] || [ "${STATUS}" = "DEGRADED" ]
