#!/usr/bin/env bash
# Usage: bash scripts/prepare_shell.sh /path/to/LV_Epi_CrossSDF_MeshPrior_MnM2_v11_5static
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
TASK_SOURCE="${1:?Provide the original aligned static repository root}"
TASK_PYTHON="${PYTHON_BIN:-python}"
cd "$TASK_ROOT"
mkdir -p preflight data
# Never replace the manifest for caches that were built from different inputs.
TASK_MANIFEST="$(mktemp data/aligned_sax_shell_manifest.XXXXXX.json)"
trap 'rm -f "$TASK_MANIFEST" "${TASK_MANIFEST%.json}_audit.json"' EXIT
"$TASK_PYTHON" adapt_static_cache.py --source "$TASK_SOURCE" --output "$TASK_MANIFEST"
if [[ -f data/aligned_sax_shell_manifest.json ]]; then
  "$TASK_PYTHON" - "$TASK_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
previous=json.loads(Path('data/aligned_sax_shell_manifest.json').read_text())
new=json.loads(Path(sys.argv[1]).read_text())
if previous!=new:
    raise ValueError('Aligned input changed; use a new data/output directory instead of replacing existing shell inputs')
PY
fi
cp "$TASK_MANIFEST" data/aligned_sax_shell_manifest.json
cp "${TASK_MANIFEST%.json}_audit.json" data/aligned_sax_shell_manifest_audit.json
"$TASK_PYTHON" prepare_cache.py --manifest data/aligned_sax_shell_manifest.json --output data/plane_sdf_shell --resume
"$TASK_PYTHON" scripts/verify_dataset.py
