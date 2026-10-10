#!/usr/bin/env bash
#SBATCH --job-name=PlaneLatentSDF_shell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/shell_%j.out
#SBATCH --error=logs/shell_%j.err
set -euo pipefail
TASK_REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_PYTHON="${PYTHON_BIN:-python}"
cd "$TASK_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p runs/shell outputs/shell preflight
exec 9>runs/shell/.job.lock
flock -n 9
stage=preflight
trap 'export FAILURE_STAGE="$stage"; "$TASK_PYTHON" scripts/record_failure.py --output runs/shell' ERR
"$TASK_PYTHON" scripts/verify_release.py --hashes-only
"$TASK_PYTHON" scripts/verify_dataset.py
"$TASK_PYTHON" verify_core.py --output preflight/shell_core.json
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA device required"'
# Exercise every loss on training data before the long run, with fresh weights.
"$TASK_PYTHON" scripts/smoke_shell.py --data data/plane_sdf_shell --device cuda --output preflight/shell_cuda.json
stage=training
resume=()
if [[ -f runs/shell/last.pt ]]; then resume=(--resume runs/shell/last.pt); fi
if [[ ! -f runs/shell/TRAINING_COMPLETE.json ]]; then
    "$TASK_PYTHON" train.py --config config.json --data data/plane_sdf_shell --output runs/shell --device cuda "${resume[@]}"
fi
test -f runs/shell/best.pt
for split in val test; do
    stage="${split}_evaluation"
    "$TASK_PYTHON" evaluate.py --checkpoint runs/shell/best.pt --data data/plane_sdf_shell --split "$split" \
        --output "outputs/shell/$split" --device cuda --meshes --resolution 160
done
stage=completion
"$TASK_PYTHON" - <<'PY'
import json,os
from pathlib import Path
reports={split:json.loads((Path('outputs/shell')/split/'numeric_eval_summary.json').read_text()) for split in ['val','test']}
assert all(r['patient_count']==36 and r['phase_file_count']==72 for r in reports.values())
assert Path('runs/shell/TRAINING_COMPLETE.json').exists()
Path('outputs/shell/PIPELINE_COMPLETE.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),training_complete=True,evaluations=reports,clinical_quality_passed=False),indent=2)+'\n')
PY
