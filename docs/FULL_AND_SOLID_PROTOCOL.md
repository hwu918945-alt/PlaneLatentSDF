# Full heart training and 25-section solid fitting

These experiments use the independent PlaneAutoOccupancy package. The shared
G/D architecture, physical interpolation, random initialization, direct/LOO
mixture and complete-case gradient accumulation remain the repaired method.
The original `objective.py` and `model.py` are unchanged.

## Heart

- Patient-disjoint 285 train / 36 validation / 36 test; ED+ES yields 570/72/72 records. Three stress patients remain excluded. Input is the compatible aligned **myocardium shell** cache, with actual Epi and Endo, copied into this repository. Epi-only caches are rejected.
- Initial full training budget: 200 epochs = 114000 Adam updates. Shuffle every epoch; each update uses one case/phase and all its SAX planes. G/D are shared across patients, with independent 128D codes for each case/phase.
- Positive/cavity pool proposals are restricted to their containing Epi/Endo bounding boxes; background proposals retain the common observed ROI+20mm. Accepted distributions remain exactly area-uniform in the same three regions with the same boundary band and budgets. This avoids wasting positive proposals outside small apical sections.
- Independent ParameterList codes ensure inactive latent gradients remain None: neither their values nor Adam moments/step advance when another case is trained. Complete-case global clip5 and one Adam step retain the original objective.
- For validation/test: freeze both G and D, initialize a **new random** code, optimize 2000 updates from first/middle/last SAX only. Evaluate remaining cached planes as strict heldout slices. No training patient's ED or ES can enter new-patient fitting.
- Evaluate all 36 validation patients (72 phases) at epochs100 and200. Select shared checkpoint by patient-mean strict-heldout-slice Dice; equal scores favor the earlier epoch. Evaluate all36 test patients once after selecting the checkpoint. Test cannot select threshold, fitting budget or weights.
- Observed/heldout classification uses uniform160² grids, logit0 and observed-only bbox+20mm. This is aligned contour-proxy evaluation; inherited alignment/normalization is not a fresh blind MRI preprocessing experiment.
- Representative training cases get observed-only PNG/OBJ diagnostics at epochs0/50/100/150/200. Final heldout cases also get uncapped96³ meshes. A failed/random zero crossing is recorded without stopping training.

## Bunny and balloon dog

- Independent jobs; each uses its existing25 real observed sections, all simultaneously, random seed19,4000 optimizer updates. No pretrained heart model, spherical initialization, reference-mesh targets or SDF loss.
- Preserve every closed section ring. Even/odd fill handles disjoint components and holes; crossing rings are rejected. No implicit segment joins separate rings. World benchmark coordinates and scale100 are inherited from the input and are not claimed to be verified real physical object dimensions.
- Each plane still has32768 unique samples per update. The unchanged .5/.25/.25 BCE is represented by half inside and two quarter **outside** buckets; the latter are exchangeable negative samples, not a fabricated endocardium. This equals .5inside+.5outside BCE.
- Inside samples are uniform over filled components using component bounding-box proposals weighted by box area, accepted only within the proposed component and outside the exact0.05mm boundary band. This is rejection sampling with constant density; it remains practical for the tiny bunny ear-tip section. Outside samples are uniform in observed-only bbox+20mm with the same band rejection.
- Capacity262144 points/plane, eight updates for full no-replacement coverage, refresh every80 **visits to that case**, near3mm recorded without near/far balancing. LOO uses23 interior planes; per update819200 independent input points and1572864 decoder queries. Balanced4096 chunks retain one complete-case Adam update.
- PNG contours, PNG mesh preview, OBJ and a checkpoint at steps0/1000/2000/3000/4000. No geometric early stopping.
- Only after update4000 evaluate the frozen121 heldout section contours, using the frozen benchmark XY domain. Reference meshes are copied for provenance and future evaluation but never read for loss; current final metric is heldout section Dice, not a 3D reference-mesh score.

## Runtime, storage and continuation

`run_experiment` verifies locally locked input hashes, runs contract tests and
fresh two-update GPU gates at **formal32768-point budgets**, then initializes a
fresh formal model/code. Heart additionally verifies frozen G/D on a new
validation patient during the gate. Gates are execution checks, not scientific
acceptance. Current-generation pool arrays stay on CPU with a small LRU; large
pool NPZs use uncompressed storage to avoid decompression every population
visit. Older pool NPZs are retired, with audit JSON retained; deterministic
seeds allow exact pool reconstruction from checkpoint case-visit counters.

Checkpoints atomically save model/codes, Adam state, all RNG states, shuffle
order and per-case visits. Resume refuses changed input/config/observation
contracts and trims log updates newer than the committed checkpoint. Signals
save progress and Slurm requeues before walltime; validation/test latent fits
also resume. Evaluation RNG is isolated from training. Only code/config/docs
are tracked; all actual caches, pools, meshes, logs and checkpoints are ignored.

Use `scripts/prepare_requested_runs.py` to copy/convert inputs, run `scripts/freeze_experiment.py` to freeze the
tracked package and plan into `run/runtime`, then submit `scripts/experiment_job.sh`
with `TASK_EXPERIMENT` pointing to that run. The job reads its frozen runtime
and local copies, never the source repository. `TRAIN_COMPLETE.json` and
`RUN_COMPLETE.json` distinguish training completion from final evaluation.
