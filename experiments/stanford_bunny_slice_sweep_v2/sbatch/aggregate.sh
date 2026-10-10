#!/usr/bin/env bash
#SBATCH --job-name=bunny_sweep_summary
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec /gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python scripts/aggregate.py
