#!/usr/bin/env bash
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
: "${TASK_EXPERIMENT:?Set the independent experiment directory}"
: "${TASK_CODE:?Set the source checkout directory}"
: "${TASK_PY:?Set the existing CUDA Python executable}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_CODE"
"$TASK_PY" -B -m experiments.plane_auto_occ.deployment --experiment "$TASK_EXPERIMENT" --device cuda
