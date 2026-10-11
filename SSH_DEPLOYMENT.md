# SSH deployment of the sampling/layer-weight repair

Use the same repository with baseline9b6add39f1efc131ca08ee2876179211458653b7,
new branch `experiment/plane-auto-occupancy-repair`. Run code in an isolated
worktree and keep V1 outputs intact. See README.md and REPAIR_PLAN.md for method.

Create a separate local experiment with cache/train/002_ED.npz copied byte for
byte from the V1 occupancy cache. Build the initial observed0,3,6 area pool as
pools/generation_000.npz. Source/cache lock at the experiment root contains
commit, files mapping relative source paths to SHA256, cache_sha256 and the
initial pool_sha256. Create gates/ and logs/. The deployment validates this lock.

```bash
export TASK_EXPERIMENT=/absolute/path/local_repair_experiment
export TASK_CODE=/absolute/path/local_repair_experiment/runtime
export TASK_PY=/absolute/path/existing_cuda_environment/bin/python
sbatch --job-name=auto_occ_repair_002_ED \
  --output="$TASK_EXPERIMENT/logs/auto_occ_repair_%j.out" \
  --error="$TASK_EXPERIMENT/logs/auto_occ_repair_%j.err" \
  "$TASK_CODE/experiments/plane_auto_occ/single_case_job.sh"
```

The inherited job requests aiaca40/4a40, one GPU, four CPUs,24GB and four hours.
Deployment runs baseline/repair tests and a two-update CUDA cold smoke, then
starts a fresh2000-update fit. Only model initialization is compared to V1;
no V1 trained weights are loaded into the repaired model.

Outputs: outputs/cuda_smoke; outputs/fit/history.jsonl and step_XXXXX/diagnostics;
outputs/fit/sampling_pools/generation_NNN.npz/json for the80-update refreshes;
outputs/final_eval and final_eval_margin20mm; outputs/myocardium.obj/json;
outputs/posthoc_audit.json with per-class metrics, ring check and open-edge locations.
GPU_SMOKE_COMPLETE/TRAIN_COMPLETE/RUN_COMPLETE identify execution stages.
Read geometry and proxy Dice before accepting a result; PIPELINE_FAILED records
execution errors. No early stopping, schedule, best-checkpoint selection or
alteration of hidden-plane supervision is introduced. Cache/checkpoint artifacts
are local only. Publish source and code patches, without real patient pools.
