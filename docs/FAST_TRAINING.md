# Offline pools and batched complete-case training

2026-10-11: the full-heart preparation script now selects `pipeline_backend=offline_batched_v1`. Existing frozen runs and the legacy single-case/solid entrypoints remain compatible.

## Problem and measured results

The original heart run reached 309 updates in 03:00:54, then failed on a single duplicate float32 point in 310_ED. Its valid history averaged 34.3004 seconds/update; pool construction averaged 32.2772 seconds, or 94.10% of that time. Only the step-0 checkpoint existed. The new run restarts with seed 19 and reuses valid point pools; it cannot recover the uncheckpointed weights.

A40, Torch 2.9.1+cu128, FP32, AMP off, TF32 off:

| Measurement | Seconds | Scope |
|---|---:|---|
| Original decoder backward, 4096-point chunks | 0.2021 | 7 planes, 229376 independent points, 393216 decoder queries |
| Batched decoder backward, 524288 query limit | about 0.091 | Same architecture, objective, coordinates, query count |
| Original geometry classification | about 0.296 | 65536 points, real 310_ED contour |
| Indexed compiled CPU geometry | about 0.014 | All region/exclusion/near labels equal |
| Optional float64 CUDA geometry | about 0.047 | All labels equal; slower than indexed CPU on this example |
| End-to-end joint training pilot | 0.1125/update | 16 phase records, 64 updates, mean 217088 independent points |
| End-to-end frozen-code pilot | 0.0489/update | 200 updates, three observed planes, full point budget |

These are measured benchmark and pilot results, not the final 570-case warmed epoch or a completed scientific training result. Evaluation, visualization, offline preprocessing and checkpoint overhead must be accounted separately. Detailed machine-readable reports live in the run's `analysis/` directory, which is not pushed.

## Contracts preserved

- Shared G/D architecture, one independent 128-D code and independent Adam state per case/phase.
- 32768 independent points per observed plane: 16384 myocardium, 8192 cavity, 8192 background.
- 262144-point pools per plane; exact deterministic visit permutations, eight-visit coverage, 80-visit refresh.
- Original observed-only ROI, class-containing uniform proposal boxes, seed sequence and stored float32 coordinates. Only duplicate coordinates are removed and replaced with additional accepted points from the same region/proposal stream. No raster labels or 3D reference supervision.
- Endpoints Direct; interior 0.75 Direct + 0.25 LOO. Objective 1.5 times the mean over planes plus 1e-4 times mean(code squared), Seam=0.
- One joint clip and one Adam update per complete case. No multi-case Adam batching, AMP, TF32, loss rescaling, point-budget reduction or early stopping.
- Patient split 285/36/36; 570/72/72 phase records. 200 shared epochs. Validation at epochs 100 and 200 fits fresh codes for 2000 updates with G/D frozen and first/middle/last observations. Validation selects the checkpoint; test runs once afterward.

## Implementation

`precache.py` prepares immutable pools outside GPU optimization. Valid legacy NPZ pools are imported without changing coordinates, labels or sample order. New pools use conservative edge indexing in `fast_geometry.py`: y bins discard impossible ray-crossing edges and a spatial index discards segments farther than the reporting band. Remaining segments use float64 exact crossing and point-to-segment equations, without fastmath. Indexing changes execution, not labels.

Pools store world coordinates as float32 NPY, near flags as losslessly packed bits, and fixed regional labels implicitly. File hashes, source hashes, observed indices, seed, generation and budgets are recorded in metadata. `offline_store.py` opens arrays with mmap, caches samplers and prefetches two complete cases on a CPU worker. Missing pools fail explicitly; `fit` never builds geometry. Full hashes are checked during preparation; training checks immutable metadata/size once per pool rather than rehashing huge arrays on every update. The startup-verified source hash is retained in memory. All 570 current-generation samplers and their current-cycle permutations can be retained (about 8.8 GB permutation arrays plus 13.3 GB mmap data pages), within the 48 GB host-memory allocation, avoiding repeated permutation work after LRU eviction. Old generations leave the bounded sampler cache as the next generation replaces them.

`fast_chunked.py` packs a complete case's points and labels for bulk transfer, combines Direct/LOO decoder queries, and reuses one G graph. Decoder backward accumulates gradients into detached plane-latent leaves; a single chain-rule backward then propagates their accumulated gradient through G and the case code. The code regularizer is applied exactly once. Class proportions make the old regional weighted BCE equal a mean over the packed points. All detailed statistics are transferred in bulk at the end, eliminating per-chunk scalar synchronization. Plane positions and axes are uploaded once per fit.

FP32 summation order changes. At the full 7-plane budget, the selected batch has zero displayed loss difference and about 1.96e-5 relative gradient L2 difference from the old implementation; gradient cosine is 1. Tests compare losses, every parameter gradient and one Adam update within FP32 tolerances. Resume within the optimized backend is bit-identical, including code values and optimizer state.

Checkpointing occurs every 250 updates or five minutes, plus epoch boundaries. On exceptions the last successful complete update is saved, including the still-pending case. A test injects a missing pool and verifies resumed weights equal uninterrupted training.

## Cache lifecycle and Slurm

Prepare all three train generations before the full-heart GPU job. The compact arrays require about 39.9 GB decimal for all 4185 train planes. At epoch 80, remove only this run's consumed generation-0 arrays, keep their metadata/hashes, and launch offline validation-cache preparation on three CPU workers. At epoch 100, all validation pools must be ready before code fitting. They are reused at epoch 200. At epoch 160, remove only the consumed train generation 1. After shared training completes, release train generation 2 and prepare all test pools before test code fitting. Validation/test each require approximately 17.2 GB. Old runs and imported source pools are never deleted. Retired generations can be regenerated deterministically for an older checkpoint.

Preparation uses four CPUs and an automatically requeued cpudebug job. Each pool is atomically published, so a one-hour partition limit does not discard completed work. The GPU job has an `afterok` dependency on successful CPU preparation; it is not eligible to start after failed preprocessing.

After preparing run data/config and freezing committed source:

```bash
PYTHONPATH=. python scripts/submit_fast_heart.py --experiment /absolute/run
```

Equivalent stages can be run explicitly from the frozen runtime:

```bash
python -m plane_auto_occ.precache --experiment /absolute/run --split train --workers 4
python -m plane_auto_occ.run_experiment --experiment /absolute/run --device cuda
```

Validation:

```bash
python -m unittest plane_auto_occ.tests plane_auto_occ.repair_tests \
  plane_auto_occ.migration_tests plane_auto_occ.population_tests plane_auto_occ.fast_tests
python -m plane_auto_occ.fast_benchmark --experiment /absolute/run
python -m plane_auto_occ.pipeline_benchmark --experiment /absolute/run
```

Benchmarks write their own new gate outputs; rerunning the pipeline pilot needs a fresh pilot output. It does not serve as scientific training or checkpoint selection.

## Reference implementations

The official [DeepSDF training implementation](https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py) loads preprocessed scene samples, batches points and accumulates chunked gradients before one optimizer update. Its [dataset loader](https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/data.py) exposes disk-backed or in-memory samples. These execution choices motivated offline preprocessing and bulk point handling here; neither its SDF objective nor its sampling distribution was adopted. The official [instant-ngp SDF CUDA implementation](https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_sdf.cu) was inspected as a GPU geometry/sampling reference; its representation and 3D supervision were not adopted. References accessed 2026-10-11.
