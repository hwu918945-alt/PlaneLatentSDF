#!/usr/bin/env bash
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
# Submit once per arm: A, B, C, D, FULL. Each job fits ten ED/ES patients serially.
set -uo pipefail
: "${TASK_SUITE:?Set the prepared suite directory}"
: "${TASK_ARM:?Set one of A B C D FULL}"
TASK_PY="${TASK_PY:-python}"
case "$TASK_ARM" in A|B|C|D|FULL) ;; *) echo "Invalid TASK_ARM: $TASK_ARM"; exit 1 ;; esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
test -f "$TASK_SUITE/PRECACHE_COMPLETE.json" || exit 1
test -f "$TASK_SUITE/gates/GPU_GATE_COMPLETE.json" || exit 1
cd "$TASK_SUITE/runtime" || exit 1
TASK_INDICES=$("$TASK_PY" -B - "$TASK_SUITE" "$TASK_ARM" <<'PYTHON'
import json,sys
from pathlib import Path
from plane_auto_occ.paired_suite import verify
root=Path(sys.argv[1]);verify(root)
tasks=[t for t in json.loads((root/'plan.json').read_text())['tasks'] if t['arm']==sys.argv[2]]
assert len(tasks)==20 and len({t['key'] for t in tasks})==20
print(' '.join(str(t['task_index']) for t in tasks))
PYTHON
) || exit 1
TASK_FAILED_INDICES=()
for TASK_INDEX in $TASK_INDICES; do
  echo "START arm $TASK_ARM scientific task $TASK_INDEX / 100"
  "$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --task-index "$TASK_INDEX" --device cuda
  TASK_EXIT_CODE=$?
  if [[ "$TASK_EXIT_CODE" -ne 0 ]]; then
    TASK_FAILED_INDICES+=("$TASK_INDEX")
    echo "FAILED task $TASK_INDEX: exit $TASK_EXIT_CODE; retaining its failure record and proceeding to the next independent task."
  fi
done
"$TASK_PY" -B "$TASK_SUITE/finalize_arms.py" --suite "$TASK_SUITE" --arm "$TASK_ARM" --failed "${TASK_FAILED_INDICES[@]}"
TASK_FINALIZE_EXIT=$?
if [[ "$TASK_FINALIZE_EXIT" -ne 0 || "${#TASK_FAILED_INDICES[@]}" -ne 0 ]]; then exit 1; fi
