#!/usr/bin/env bash
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-99%4
set -euo pipefail
: "${TASK_SUITE:?Set the absolute prepared suite directory}"
TASK_PY="${TASK_PY:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_SUITE/runtime"
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --task-index "$SLURM_ARRAY_TASK_ID" --device cuda
