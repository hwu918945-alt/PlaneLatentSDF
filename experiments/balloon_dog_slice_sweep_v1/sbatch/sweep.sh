#!/usr/bin/env bash
#SBATCH --job-name=balloon_dog_slice_sweep
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec /gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python scripts/run_all.py
