# SSH single-case deployment

The delivered package is applied as an additive Git patch on `origin/main` in
the isolated `experiment/plane-auto-occupancy` worktree. Existing SDF experiments
and uncommitted main-worktree changes are preserved. New real caches, checkpoints,
logs and runtime reports remain local; push only the new source commits.

Use patient 002 ED, with seven sorted cached SAX planes. Train jointly from a
randomly initialized shared generator/decoder and a Gaussian case code, using
observed indices **0,3,6**. Indices **1,2,4,5** are strict heldout geometry in the
fixed cached alignment system, evaluated only after the formal fit completes.
The cache is derived from a genuine Epi/Endo myocardium-shell input, through the
delivered `convert_cache` without changes to its labels or exclusion bands.

Default configuration is unchanged: 128-dimensional code, width 256, 1024 points
per observed plane, 512/256/256 myocardium/cavity/background allocation,
Direct/LOO/L2 weights 0.5/1/1e-4, Seam disabled, both learning rates 1e-3,
seed 19, **2000 optimizer updates**. This is the deployment document's single-case
budget; a step is one optimizer update, not a full-dataset epoch.

## Added deployment support

`train --milestones` is optional; absent that flag, the original training behavior
is preserved. It saves checkpoints and launches `diagnostics` in a separate
process at exact update counts. Diagnostics have no optimizer and use only the
observed list: Direct/LOO BCE and parameter-group gradients, uniform-plane proxy
Dice, connected-ring topology, unlabelled gaps, latent perturbations and an
uncapped 40-cubed mesh. No latent diversity loss or collapse early stopping is
introduced. All original model, objective, interpolation and sampling code is
unchanged. Independent initial/50/100/200/500/1000/2000 artifacts do not select a
checkpoint; only the fixed final update is used for heldout evaluation.

`deployment` verifies a local source/cache lock, performs the two-step CUDA smoke,
and starts the formal 2000-update fit from scratch in another process. It then
runs the supplied evaluation at resolution 160 and mesh extraction at 128.
Evaluation bounds depend only on observed Epi contours; if heldout GT touches the
5-mm margin, evaluate again with a 20-mm margin and report any remaining contact.
No padding, artificial caps or extrapolation beyond observed axial extent are
used. No zero crossing is explicitly recorded as absent geometry. Execution
completion is distinct from a usable myocardium surface or anatomical accuracy.

## Submission

Set absolute paths to the independent experiment, source worktree and existing
CUDA Python environment. Before submission, write `source_lock.json` at the
experiment root with `commit`, `files` (relative source paths to SHA256), and
`cache_sha256` for `cache/train/002_ED.npz`. Create `logs/` and `gates/` there.

```bash
export TASK_EXPERIMENT=/absolute/path/local_experiment
export TASK_CODE=/absolute/path/local_experiment/runtime
export TASK_PY=/absolute/path/existing_cuda_environment/bin/python
sbatch --job-name=auto_occ_002_ED \
  --output="$TASK_EXPERIMENT/logs/auto_occ_%j.out" \
  --error="$TASK_EXPERIMENT/logs/auto_occ_%j.err" \
  "$TASK_CODE/experiments/plane_auto_occ/single_case_job.sh"
```

Outputs are `outputs/cuda_smoke`, `outputs/fit/step_XXXXX`,
`outputs/final_eval` (or `final_eval_margin20mm`) and `outputs/myocardium.obj`.
`GPU_SMOKE_COMPLETE`, `TRAIN_COMPLETE` and `RUN_COMPLETE` markers identify
execution stages. Read final topology, Dice, latent sensitivity and mesh contacts
before drawing scientific conclusions. Dice uses cached filled-contour proxies,
not clinical raster masks. The two-step smoke is not evidence of convergence.
