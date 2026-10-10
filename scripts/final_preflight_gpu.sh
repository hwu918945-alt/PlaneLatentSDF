#!/bin/bash
set -euo pipefail
cd /gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneLatentSDF
PYTHON=/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON" train.py --config configs/smoke_full_losses_initialized.json --data preflight/synthetic --output preflight/synthetic_initialized --device cuda
"$PYTHON" train.py --config configs/smoke_full_losses_initialized.json --data preflight/real_smoke --output preflight/real_initialized --device cuda
"$PYTHON" evaluate.py --checkpoint preflight/paired_initialization_fit/last.pt --data preflight/single_patient --split train --output preflight/paired_evaluation/train --device cuda --meshes --resolution 128
"$PYTHON" evaluate.py --checkpoint preflight/paired_initialization_fit/last.pt --data preflight/single_patient --split val --output preflight/paired_evaluation/val --device cuda --meshes --resolution 128
"$PYTHON" - <<'PY'
import json
from pathlib import Path
fit=json.loads(Path('preflight/paired_evaluation/train/numeric_eval_summary.json').read_text())
val=json.loads(Path('preflight/paired_evaluation/val/numeric_eval_summary.json').read_text())
assert not fit['mesh_failures'] and not val['mesh_failures']
assert fit['metrics']['direct_coverage']==1 and fit['metrics']['loo_coverage']==1
for name in ['synthetic_initialized','real_initialized']:
 rows=[json.loads(s) for s in (Path('preflight')/name/'metrics.jsonl').read_text().splitlines()]
 assert len(rows)==2 and all(r['ramp']==1 and r['nonfinite_count']==0 for r in rows)
 assert all(r['calibration_valid_fraction']>0 for r in rows)
Path('preflight/GPU_PREFLIGHT_COMPLETE.json').write_text(json.dumps(dict(status='PASS',decoder_initialization='geometric_sphere_paired_directions',synthetic_and_real_full_loss_runs=True,single_patient_steps=400,fitting_metrics=fit['metrics'],untrained_validation_patient_metrics=val['metrics'],clinical_quality_established=False),indent=2))
PY
