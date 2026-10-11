# 002_ED V2 ablation: A case-only / B plane-nearest / C plane-linear

Source is fixed at989601aabc0083388ec88230d3ab0eee117e5e68 on an isolated
`experiment/plane-auto-occupancy-ablation` worktree. All original V2 files are
byte-identical. Only this ablations directory is new. This compatibility
runtime sits inside the independent PlaneAutoOccupancy run directory; its
training code reads local copies, without imports from another repository.

A reuses the exact OccupancyDecoder class and a single128D case code, no G or
plane parameters. A loss is1.5mean(Direct)+1e-4mean(code²), LOO0, Seam0.
Reuse original AreaSampler/build_pool/chunked backward_case,32768points/plane,
4096chunks, once-per-complete-case clip5 and Adam,2000updates,seed19. New Adam
and fresh pool cursors for smoke and formal. If available, only C **step0**
decoder/code initialize A bit-identically; final C initialization is rejected.
A checkpoints have explicit plane_auto_occ_case_only_v1/model_variant metadata.

B uses unchanged trained C weights, case code, native G outputs and nearest
**observed** node, with exact ties toward lower height. C uses native query /
model.field / interpolate. No B training/fine-tuning, no C retraining, no fourth
LOO group. The only new formal training is A.

Same002_ED cache, observed0/3/6, heldout1/2/4/5 only in completed-model scoring.
160²/+5mm, separate+20mm robustness, logit0,128³ native mesh, no extrapolation,
padding, caps, smoothing, repairs or deletion of failed components. Code and
input hashes live in run protocol/input/source locks.

## Numerical evaluation adaptation

For the true oblique SAX axis, native float32 projection and rounded
(1-alpha) coefficients produce2e-8 latent differences at observed heights.
Float32 cancellation yields~6e-5 B/C logit differences, with **zero classification
flips**. The strict1e-5 requirement is retained. All A/B/C evaluations uniformly
promote field arithmetic (s/axis/latent/decoder) to float64 **after the same native
float32 xyz rounding**, using a parameter-free wrapper that calls the original
model.field/interpolate/decoder. G's native float32 outputs are fixed before
promotion. Native C query and geometry/mesh functions and all original source
files remain unchanged. Training always uses original float32.

This precision change is disclosed rather than hiding the raw discrepancy:
validation_checks.json saves raw float32 differences and promoted equality.
C must still reproduce all seven native Dice/topology/area, and exact native
128³ vertex/face/boundary counts. The adapter never changes a threshold or
checkpoint, never replaces nearest with interpolation and never alters the
observed query grid. No results are acceptable unless these gates pass.

```bash
python -m unittest experiments.plane_auto_occ.tests experiments.plane_auto_occ.repair_tests experiments.plane_auto_occ.ablations.test_ablations
python -m experiments.plane_auto_occ.ablations.train_case_only --case CASE --observed 0,3,6 --area-pool POOL --output NEW_A --device cuda --steps 2000 --init-from-c-step0 C_STEP0
python -m experiments.plane_auto_occ.ablations.evaluate_three --case CASE --a-checkpoint NEW_A/last.pt --c-checkpoint C_FINAL --output NEW_COMPARISON --device cuda --reference-metrics NATIVE_METRICS --reference-mesh NATIVE_MESH --reference-audit NATIVE_AUDIT
TASK_EXPERIMENT=/absolute/run sbatch experiments/plane_auto_occ/ablations/job.sh
```

Output: A_train history/effective_config/fit_manifest/final checkpoint;
compare_final/{A_case_only,B_plane_nearest,C_plane_linear}/metrics.json and
myocardium.obj/json; CSV, matched plane/mesh triptychs,15/45mm switch audits,
+20mm robustness, validation checks, computation costs and final REPORT.md.
Use RUN_COMPLETE.json to distinguish completion from submission/gate passing.
Do not push real caches/checkpoints/meshes/logs. Single-case exploratory results
cannot establish cross-patient generalization or a standalone LOO benefit.
