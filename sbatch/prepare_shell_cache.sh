#!/usr/bin/env bash
#SBATCH --job-name=PlaneSDF_shell_cache
#SBATCH --partition=cpudebug
#SBATCH --qos=cpudebug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/shell_cache_%j.out
#SBATCH --error=logs/shell_cache_%j.err
set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_PYTHON="${PYTHON_BIN:-/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p preflight/shell_deployment data/plane_sdf_shell
exec 9>data/plane_sdf_shell/.job.lock
flock -n 9
"$TASK_PYTHON" scripts/prepare_shell_cache_parallel.py --workers "$TASK_WORKERS" --resume
"$TASK_PYTHON" scripts/verify_shell_cache_parallel.py --workers "$TASK_WORKERS"
