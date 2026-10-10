#!/usr/bin/env bash
#SBATCH --job-name=PlaneSDF_shell_gate
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shell_gate_%j.out
#SBATCH --error=logs/shell_gate_%j.err
set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_PYTHON="${PYTHON_BIN:-/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$TASK_PYTHON" scripts/gate_shell_deployment.py
