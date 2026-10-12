#!/usr/bin/env bash
# Supply this site's existing CPU partition/account/time options to sbatch.
set -euo pipefail
: "${TASK_SUITE:?Set the absolute prepared suite directory}"
TASK_PY="${TASK_PY:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_SUITE/runtime"
"$TASK_PY" -B -m plane_auto_occ.paired_suite precache --suite "$TASK_SUITE" --workers "${TASK_CACHE_WORKERS:-3}"
