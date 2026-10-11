# Plane Auto-Decoder Occupancy: observed-area sampling repair

Baseline: PlaneLatentSDF Git commit `9b6add39f1efc131ca08ee2876179211458653b7`,
branch `experiment/plane-auto-occupancy`. This is an additive repair branch in
that repository, not a replacement repository. The supplied repair plan is
[REPAIR_PLAN.md](REPAIR_PLAN.md); the supplied objective.py is installed byte for
byte (SHA256 `e627f73aac2ba79f57f844e4d9ed7b586496071bcf7f5762ffb07e26516e5b5e`).

## Method

The same 128D case code, position/phase generator, physical-distance latent
interpolation and width-256 five-Linear/four-Softplus occupancy decoder are used.
Model, inference, evaluation and mesh extraction are unchanged. Cached coordinates,
center_mm, scale_mm and sorted physical plane heights are preserved. No new
regularizer, coordinate feature, architecture, learning-rate schedule or stage is
introduced. Seam remains zero.

For each observed plane, draw 32768 points: myocardium 16384, cavity 8192,
background 8192. Region proposals are uniform continuous 2D coordinates in the
joint **observed-only** Epi bounding box plus 20mm. Polygon membership defines
classes, and distance <=0.05mm to either boundary is excluded. These are labels
on true observed planes only; no off-plane or hidden-plane labels are generated.
The fast segment-midpoint KD-tree boundary search is exact for both exclusion
and 3mm audit thresholds and is tested against exhaustive segment distances.

New schema `plane_auto_occ_area_uniform_v2` pools contain 131072/65536/65536
independent points per plane. Sampling consumes random permutations without
replacement, covers every point in 8 updates, and reshuffles. Every 80 updates
(10 complete covers), rebuild the pool with a deterministic independent new seed.
Near/far is an audit statistic only. No small pool is repeated to fill a quota.
Coordinates, hashes, source contours, observed indices, ROI, seeds and generation
are recorded in each pool NPZ and JSON. All real pools remain local.

The objective version is `per_plane_direct_loo_mix_v2`. Per-plane regional BCE is
0.5 myocardium + 0.25 cavity + 0.25 background. Endpoint loss is Direct; internal
loss is 0.75 Direct + 0.25 LOO, fixed from the first update, using identical
coordinates for the two paths. Total is 1.5 times the mean over all planes plus
1e-4 times mean(case_code squared). Direct/LOO log values are diagnostic means,
not the old additive summands. The old direct_weight/loo_weight config fields
remain solely for the supplied objective's compatibility interface and Direct-only
switch; data_weight=1.5 and loo_mix=0.25 define this run's actual coefficients.
For three planes, endpoint coefficients are 0.5; middle Direct/LOO coefficients
are 0.375/0.125. The previous approximately 7:1 middle-plane weighting is removed.

Each 32768-point plane is divided into eight balanced 4096-point blocks. Recompute
the generator graph for each block and accumulate BCE gradients with factors
4096/32768, 1/N and the fixed mixing coefficients. Backward the unchanged code
regularizer exactly once. Adam, global clip=5 and optimizer.step occur exactly
once per complete case update. Three planes give **98304 independent inputs and
131072 decoder queries per update**, 32 times the V1 point budget. The code matrix,
Adam defaults, shared/code learning rates 1e-3, seed19, log_every50/save_every250,
2000-update budget and fixed-final-checkpoint protocol are preserved.

## Single ED deployment

Select patient002_ED observed indices0,3,6, with1,2,4,5 heldout until training
completes. Keep the V1 occupancy cache for contours and final evaluation; its
old proposal coordinates are never the repaired training sampling source.
The separate new pool has observed planes only. This interface arrangement keeps
inference/evaluate/extract_mesh byte-identical and does not alter normalization.

```bash
python -m experiments.plane_auto_occ.area_pool \
  --case /local/experiment/cache/train/002_ED.npz \
  --observed 0,3,6 --output /local/experiment/pools/generation_000.npz
python -m experiments.plane_auto_occ.train \
  --case /local/experiment/cache/train/002_ED.npz --observed 0,3,6 \
  --area-pool /local/experiment/pools/generation_000.npz \
  --output /local/experiment/outputs/fit --device cuda --steps 2000 \
  --milestones 0,50,100,200,500,1000,2000
```

New-output-only behavior and no implicit resume are preserved. Deployment runs a
cold two-step CUDA smoke, then an independent formal fit with reset network,
code, optimizer and RNG. It compares initialization against the V1 CUDA initial
checkpoint if it is available. No geometry threshold stops or stages training.

Milestone diagnostics only access observed contours, using an isolated sampler
and 4096 points per plane for gradient auditing (distinct from the formal32768).
They preserve the grid/plot/mesh protocol and add true zero-code and three random
code interventions. Another patient's code is unavailable in this single-case
experiment and is explicitly marked as such. These diagnostics do not consume
training permutations or alter the training RNG.

After the fixed2000th update, run native160x160/logit0 evaluation on observed,
strict heldout and unlabelled gaps. Main ROI remains +5mm; always additionally
report +20mm robustness evaluation. Export the native uncapped128-cubed mesh in
aligned world mm. Separate axial and lateral open edges. Post-hoc report all
plane Dice, myocardium recall, cavity/background false positives and ring topology.
The plan's observed Dice>=0.90 and connected-ring requirement is a post-training
engineering check, never a checkpoint selector or training gate.

## Checks and provenance

```bash
python -m experiments.plane_auto_occ.tests
python -m experiments.plane_auto_occ.repair_tests
```

Repair tests cover a thin ring with no far pool, exact boundary bands, no-replacement
eight-update coverage, ten-cover refresh at update81, hidden-contour exclusion,
plane coefficient conservation and full/chunked gradients plus the single Adam
update. See [SSH_DEPLOYMENT.md](SSH_DEPLOYMENT.md) for Slurm.

Pools, checkpoints, logs and real-case outputs are not included in the code push.
Execution markers are not proof of anatomical success or a learned population
prior. This branch changes sampling distribution, point count and layer weights;
any resulting improvement cannot be attributed to one change without ablations.
