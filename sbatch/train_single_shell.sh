#!/usr/bin/env bash
#SBATCH --job-name=PlaneSDF_shell_002
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/shell_single_002_%j.out
#SBATCH --error=logs/shell_single_002_%j.err
set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export PYTHON_BIN="${PYTHON_BIN:-/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python}"
export SHELL_MODE=single_002
bash sbatch/train_portable.sh
