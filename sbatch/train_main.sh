#!/usr/bin/env bash
#SBATCH --job-name=PlaneLatentSDF_shell
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/shell_%j.out
#SBATCH --error=logs/shell_%j.err
set -euo pipefail
TASK_REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export PYTHON_BIN="${PYTHON_BIN:-/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python}"
export SHELL_MODE=main
cd "$TASK_REPO"
bash sbatch/train_portable.sh
