#!/usr/bin/env bash
#SBATCH --job-name=bunny_slice_sweep
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
slice_counts=(16 25 50 75)
exec /gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python scripts/run_group.py --slices "${slice_counts[${SLURM_ARRAY_TASK_ID}]}"
