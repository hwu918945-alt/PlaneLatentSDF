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
TASK_MODE="${SHELL_MODE:-main}"
if [[ "$TASK_MODE" == main ]]; then
    TASK_CONFIG=config.json
    TASK_DATA=data/plane_sdf_shell
    TASK_RUN=runs/shell
    TASK_OUTPUT=outputs/shell
    TASK_CHECKPOINT=best.pt
    TASK_SPLITS=(val test)
elif [[ "$TASK_MODE" == single_002 ]]; then
    TASK_CONFIG=configs/shell_single_002.json
    TASK_DATA=preflight/shell_single_002
    TASK_RUN=runs/shell_single_002
    TASK_OUTPUT=outputs/shell_single_002
    TASK_CHECKPOINT=last.pt
    TASK_SPLITS=(train val)
else
    echo "Unsupported SHELL_MODE: $TASK_MODE" >&2
    exit 2
fi
mkdir -p "$TASK_RUN" "$TASK_OUTPUT" preflight/shell_deployment
exec 9>"$TASK_RUN/.job.lock"
flock -n 9
stage=preflight
trap 'export FAILURE_STAGE="$stage"; "$TASK_PYTHON" scripts/record_failure.py --output "$TASK_RUN"' ERR
"$TASK_PYTHON" scripts/verify_release.py --hashes-only
# Continuous-source geometry was audited once during cache preparation. Check
# the frozen inventory here rather than repeating that CPU audit on every GPU.
"$TASK_PYTHON" scripts/check_shell_ready.py
if [[ "$TASK_MODE" == single_002 ]]; then
    "$TASK_PYTHON" scripts/prepare_single_patient.py --data data/plane_sdf_shell --output "$TASK_DATA"
fi
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA device required"'
# Exercise every loss on training data before the long run, with fresh weights.
"$TASK_PYTHON" scripts/smoke_shell.py --data "$TASK_DATA" --config "$TASK_CONFIG" --device cuda \
    --output "preflight/shell_deployment/${TASK_MODE}_cuda_${SLURM_JOB_ID:-local}.json"
export TASK_MODE TASK_CONFIG TASK_DATA TASK_RUN TASK_OUTPUT
"$TASK_PYTHON" - <<'PY'
import hashlib,json,os,subprocess,torch
from pathlib import Path
report=dict(job_id=os.environ.get('SLURM_JOB_ID'),mode=os.environ['TASK_MODE'],
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            config=os.environ['TASK_CONFIG'],config_sha256=hashlib.sha256(Path(os.environ['TASK_CONFIG']).read_bytes()).hexdigest(),
            data=os.environ['TASK_DATA'],dataset_manifest_sha256=hashlib.sha256(Path('data/plane_sdf_shell/DATASET_READY.json').read_bytes()).hexdigest(),
            test_used_for_selection=False)
(Path(os.environ['TASK_RUN'])/'JOB_STARTED.json').write_text(json.dumps(report,indent=2)+'\n')
PY
stage=training
resume=()
if [[ -f "$TASK_RUN/last.pt" ]]; then resume=(--resume "$TASK_RUN/last.pt"); fi
if [[ ! -f "$TASK_RUN/TRAINING_COMPLETE.json" ]]; then
    "$TASK_PYTHON" train.py --config "$TASK_CONFIG" --data "$TASK_DATA" --output "$TASK_RUN" --device cuda "${resume[@]}"
fi
test -f "$TASK_RUN/$TASK_CHECKPOINT"
for split in "${TASK_SPLITS[@]}"; do
    stage="${split}_evaluation"
    "$TASK_PYTHON" evaluate.py --checkpoint "$TASK_RUN/$TASK_CHECKPOINT" --data "$TASK_DATA" --split "$split" \
        --output "$TASK_OUTPUT/$split" --device cuda --meshes --resolution 160
done
stage=completion
"$TASK_PYTHON" - <<'PY'
import json,os
from pathlib import Path
single=os.environ['TASK_MODE']=='single_002'
root=Path(os.environ['TASK_OUTPUT']);run=Path(os.environ['TASK_RUN'])
reports={split:json.loads((root/split/'numeric_eval_summary.json').read_text()) for split in (['train','val'] if single else ['val','test'])}
assert all(r['patient_count']==(1 if single else 36) and r['phase_file_count']==(2 if single else 72) for r in reports.values())
assert (run/'TRAINING_COMPLETE.json').exists()
(root/'PIPELINE_COMPLETE.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),mode=os.environ['TASK_MODE'],training_complete=True,evaluations=reports,clinical_quality_passed=False),indent=2)+'\n')
PY
