# A/B/C/D/FULL on 25-slice binary solids

The 2026-10-13 extension uses the existing Stanford bunny and balloon dog
occupancy caches, each with 25 observed sections. Each arm independently fits
both solids from random G/D/z/Adam initialization for 4000 complete updates.
One GPU worker per arm fits bunny followed by balloon dog. All arms share the
same 100 offline pools: 2 solids × 50 generations, refreshed every 80 visits.

The five decoder/loss definitions are unchanged from the integrated heart
ablation: A is BCE/Softplus; B adds uniform-ROI Dice with weight 1; C changes
the last hidden activation to LeakyReLU; D adds h2→h4 residual; FULL repeats D
with the same seed. Whole-plane Dice uses the same exact two-pass gradient.
Seed19, Adam rates0.001, clip5, Direct/LOO mix0.25 and latent regularization
are retained. One update covers all 25 planes, 32768 points per plane.

## Solid geometry adaptation

Binary solids do not have myocardial Epi/Endo labels. Every observed real ring
is retained, with even/odd filling for nested holes and disconnected components.
No segments connect separate rings. Two exchangeable independent real-boundary
banks each contribute 3072 inside/outside pairs; the uniform ROI supplies20480
points. Thus source counts remain3072/3072/3072/3072/20480, with both negative
boundary sources representing actual outside occupancy. Dice still sees only
the independently tagged uniform ROI source4.

Offsets use the same0.25/0.75/2mm scales,0.5/0.3/0.2 quotas and±20% jitter.
For solids, clearance is the first other real boundary hit along the interior
normal ray in the observed section; the offset is capped at0.4×clearance.
All final float32 points are classified by actual even/odd solid geometry,
must lie in the observed ROI, be unique and exceed the0.05mm boundary band.
Direction retries recalculate the applicable interior-ray clearance.

Small polygon spurs can have no admissible bilateral pair at an anchor under
the boundary-band constraint. After same-anchor attempts, solid-only sampling
conditions real-boundary arclength on supported locations, preserves each
rejected unit's scale and requested offset, and records every rejected arc,
clearance and retry round. This adaptation is explicit in the pool contract
and metadata. It is not a claim of identical cardiac Epi/Endo geometry sampling;
the cardiac geometry core and its unsupported-anchor policy remain unchanged.

The training ROI uses the bbox of all real rings from all25 observed slices
plus20mm. Solid packed rings have variable vertex counts; the inference bounds
adapter selects their actual rings rather than indexing25 individual packed
geometry points. Heart bounds and heart default pool contracts are preserved.

## Outputs and heldout protocol

Steps0/1000/2000/3000/4000 save checkpoint copies, observed sections/gaps,
PNG/OBJ and visualization completion markers. Native mesh128³, section grids
160², logit0, no padding/caps/smoothing/repair. Missing initial zero crossing
is recorded and does not stop training.

Only final step4000 reads the existing strict heldout contours and frozen
evaluation domain; those inputs never enter pools, loss or checkpoint selection.
The reference PLY is archived for provenance, with no mesh supervision.
After all five workers finish, summarize observed and strict heldout section
Dice, retaining every failed task. FULL is a same-seed repeat, not a fifth
independent architecture or an additional statistical sample.

The GPU preparation job validates both real full first generations and all
ten CUDA two-update interfaces before generating all remaining generations.
Five training jobs depend only on successful GPU preparation. Submission and
gate completion do not imply scientific completion or reconstruction quality.
