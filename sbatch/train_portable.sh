#!/usr/bin/env bash
# Submit from the repository root; choose cluster resources on the sbatch command line.
#SBATCH --job-name=PlaneLatentSDF_main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err
set -euo pipefail

TASK_REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_PYTHON="${PYTHON_BIN:-python}"
cd "$TASK_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p runs/main outputs/main provenance/release
exec 9>runs/main/.job.lock
flock -n 9
"$TASK_PYTHON" scripts/verify_release.py --output provenance/release/local_cache_verification.json
"$TASK_PYTHON" verify_core.py --output provenance/release/local_core_verification.json
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA device required"'

stage=training
trap 'export FAILURE_STAGE="$stage"; "$TASK_PYTHON" scripts/record_failure.py' ERR
resume=()
if [[ -f runs/main/last.pt ]]; then resume=(--resume runs/main/last.pt); fi
if [[ ! -f runs/main/TRAINING_COMPLETE.json ]]; then
    "$TASK_PYTHON" train.py --config config.json --data data/plane_sdf --output runs/main --device cuda "${resume[@]}"
fi
test -f runs/main/best.pt
for split in val test; do
    stage="${split}_evaluation"
    "$TASK_PYTHON" evaluate.py --checkpoint runs/main/best.pt --data data/plane_sdf --split "$split" \
        --output "outputs/main/$split" --device cuda --meshes --resolution 160
done
stage=completion
"$TASK_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
reports = {split: json.loads((Path('outputs/main') / split / 'numeric_eval_summary.json').read_text())
           for split in ['val', 'test']}
assert all(r['patient_count'] == 36 and r['phase_file_count'] == 72 for r in reports.values())
assert Path('runs/main/TRAINING_COMPLETE.json').exists()
Path('outputs/main/PIPELINE_COMPLETE.json').write_text(json.dumps({
    'job_id': os.environ.get('SLURM_JOB_ID'), 'training_complete': True,
    'evaluations': reports, 'clinical_quality_passed': False}, indent=2) + '\n')
PY
