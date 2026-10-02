#!/usr/bin/env bash
# Compressed database backup with retention. Safe to run while the app is live.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/backend"
exec .venv/bin/python -c "
from chief_agent.monitoring.maintenance import backup_database, list_backups
import json
result = backup_database()
print(json.dumps(result, indent=2))
print('total backups:', len(list_backups()))
"
