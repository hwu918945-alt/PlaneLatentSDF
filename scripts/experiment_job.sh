#!/usr/bin/env bash
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6-23:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
set -uo pipefail
: "${TASK_EXPERIMENT:?Set the independent experiment directory}"
TASK_CODE="${TASK_CODE:-$TASK_EXPERIMENT/runtime}"
TASK_PY="${TASK_PY:-/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_CODE" || exit 1
TASK_CHILD=""
trap 'if [[ -n "$TASK_CHILD" ]]; then kill -USR1 "$TASK_CHILD" 2>/dev/null || true; fi' USR1 TERM
"$TASK_PY" -B -m plane_auto_occ.run_experiment --experiment "$TASK_EXPERIMENT" --device cuda &
TASK_CHILD=$!
TASK_STATUS=0
wait "$TASK_CHILD" || TASK_STATUS=$?
# A forwarded signal interrupts Bash wait before Python has saved its checkpoint.
if kill -0 "$TASK_CHILD" 2>/dev/null; then
  TASK_STATUS=0
  wait "$TASK_CHILD" || TASK_STATUS=$?
fi
if [[ "$TASK_STATUS" -eq 85 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$TASK_STATUS"
