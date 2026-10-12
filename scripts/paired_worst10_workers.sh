#!/usr/bin/env bash
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
# Four Slurm tasks cover the unchanged 100 scientific task indices. This fits
# the site's max-submit=8 policy, unlike a 100-element Slurm array.
set -uo pipefail
: "${TASK_SUITE:?Set the prepared suite directory}"
TASK_PY="${TASK_PY:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
test -f "$TASK_SUITE/PRECACHE_COMPLETE.json" || exit 1
test -f "$TASK_SUITE/gates/GPU_GATE_COMPLETE.json" || exit 1
cd "$TASK_SUITE/runtime" || exit 1
TASK_START_INDEX=$((SLURM_ARRAY_TASK_ID * 25))
TASK_STOP_INDEX=$((TASK_START_INDEX + 25))
TASK_FAILED_INDICES=()
for ((TASK_INDEX=TASK_START_INDEX; TASK_INDEX<TASK_STOP_INDEX; TASK_INDEX++)); do
  echo "START scientific task $TASK_INDEX / 100"
  "$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --task-index "$TASK_INDEX" --device cuda
  TASK_EXIT_CODE=$?
  if [[ "$TASK_EXIT_CODE" -ne 0 ]]; then
    TASK_FAILED_INDICES+=("$TASK_INDEX")
    echo "FAILED task $TASK_INDEX: exit $TASK_EXIT_CODE; retaining its failure record and proceeding to the next independent task."
  fi
done
"$TASK_PY" -B "$TASK_SUITE/finalize_workers.py" --suite "$TASK_SUITE" --worker "$SLURM_ARRAY_TASK_ID" --failed "${TASK_FAILED_INDICES[@]}"
TASK_FINALIZE_EXIT=$?
if [[ "$TASK_FINALIZE_EXIT" -ne 0 || "${#TASK_FAILED_INDICES[@]}" -ne 0 ]]; then exit 1; fi
