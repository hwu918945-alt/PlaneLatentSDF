#!/usr/bin/env bash
# Current shell protocol has no sphere initialization gate.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bash "$TASK_ROOT/scripts/preflight_gpu.sh"
