#!/usr/bin/env bash
set -euo pipefail
TASK_REPO="$(cd -- "$(dirname -- "$0")/.." && pwd)"
TASK_PYTHON="${PYTHON_BIN:-python}"
cd "$TASK_REPO"
mkdir -p preflight
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$TASK_PYTHON" - <<'PY'
import torch,json,platform,numpy,scipy,skimage
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA required for this preflight'
report=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,skimage=skimage.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
Path('preflight/gpu_environment.json').write_text(json.dumps(report,indent=2));print(report)
PY
"$TASK_PYTHON" scripts/verify_release.py --hashes-only
"$TASK_PYTHON" scripts/verify_dataset.py
"$TASK_PYTHON" verify_core.py --output preflight/shell_core.json
"$TASK_PYTHON" scripts/smoke_shell.py --data data/plane_sdf_shell --device cuda --output preflight/shell_cuda.json
