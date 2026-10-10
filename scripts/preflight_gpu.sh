#!/bin/bash
set -euo pipefail
REPO=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneLatentSDF
PYTHON=/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python
cd "$REPO"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$PYTHON" - <<'PY'
import torch,json,platform,numpy,scipy,skimage
from pathlib import Path
assert torch.cuda.is_available()
report=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,skimage=skimage.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
Path('preflight/gpu_environment.json').write_text(json.dumps(report,indent=2));print(report)
PY
"$PYTHON" train.py --config configs/smoke_full_losses.json --data preflight/real_smoke --output preflight/real_gpu_run --device cuda
"$PYTHON" train.py --config configs/single_patient_fit.json --data preflight/single_patient --output preflight/single_patient_run --device cuda
"$PYTHON" evaluate.py --checkpoint preflight/single_patient_run/last.pt --data preflight/single_patient --split train --output preflight/single_patient_evaluation --device cuda --meshes --resolution 128
"$PYTHON" - <<'PY'
import json
from pathlib import Path
for name in ['real_gpu_run','single_patient_run']:
 rows=[json.loads(s) for s in (Path('preflight')/name/'metrics.jsonl').read_text().splitlines()]
 assert rows[-1]['nonfinite_count']==0
 assert any(r['ramp']==1. for r in rows)
Path('preflight/GPU_PREFLIGHT_COMPLETE.json').write_text(json.dumps({'status':'PASS','real_all_losses':True,'single_patient_epochs':100,'clinical_quality_established':False},indent=2))
PY
