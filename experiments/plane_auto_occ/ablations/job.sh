#!/usr/bin/env bash
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
set -euo pipefail
: "${TASK_EXPERIMENT:?Set the ablation experiment directory}"
TASK_PY="${TASK_PY:-/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_EXPERIMENT/runtime"
"$TASK_PY" -B -m experiments.plane_auto_occ.ablations.deploy --experiment "$TASK_EXPERIMENT" --device cuda
