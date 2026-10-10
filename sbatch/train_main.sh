#!/bin/bash
#SBATCH --job-name=PlaneLatentSDF_main
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneLatentSDF/logs/main_%j.out
#SBATCH --error=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneLatentSDF/logs/main_%j.err

set -euo pipefail
REPO=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneLatentSDF
PYTHON=/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python
cd "$REPO"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p runs/main outputs/main
exec 9>runs/main/.job.lock
flock -n 9
stage=preflight
trap 'export FAILURE_STAGE="$stage"; "$PYTHON" scripts/record_failure.py' ERR
test -f preflight/GPU_PREFLIGHT_COMPLETE.json
test -f preflight/dataset/dataset_audit.json
sha256sum -c provenance/code_sha256.txt
"$PYTHON" - <<'PY'
import json,torch,os
from pathlib import Path
assert torch.cuda.is_available()
assert json.loads(Path('preflight/GPU_PREFLIGHT_COMPLETE.json').read_text())['status']=='PASS'
assert json.loads(Path('preflight/dataset/dataset_audit.json').read_text())['status']=='PASS'
Path('runs/main/JOB_STARTED.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,test_used_for_selection=False),indent=2))
PY
stage=training
resume=()
if [ -f runs/main/last.pt ]; then resume=(--resume runs/main/last.pt); fi
if [ ! -f runs/main/TRAINING_COMPLETE.json ]; then
    "$PYTHON" train.py --config config.json --data data/plane_sdf --output runs/main --device cuda "${resume[@]}"
fi
stage=validation_evaluation
"$PYTHON" evaluate.py --checkpoint runs/main/best.pt --data data/plane_sdf --split val --output outputs/main/val --device cuda --meshes --resolution 160
stage=test_evaluation
"$PYTHON" evaluate.py --checkpoint runs/main/best.pt --data data/plane_sdf --split test --output outputs/main/test --device cuda --meshes --resolution 160
stage=completion
"$PYTHON" - <<'PY'
import json,os
from pathlib import Path
reports={split:json.loads((Path('outputs/main')/split/'numeric_eval_summary.json').read_text()) for split in ['val','test']}
assert all(r['patient_count']==36 and r['phase_file_count']==72 for r in reports.values())
Path('outputs/main/PIPELINE_COMPLETE.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],training_complete=True,evaluations=reports,clinical_quality_passed=False),indent=2))
PY
