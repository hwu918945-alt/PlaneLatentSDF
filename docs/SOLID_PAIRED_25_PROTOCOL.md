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

## Exact preprocessing acceleration and degenerate-ray repair

The compiled backend preserves the logical pool contract, NumPy proposal RNG,
source quotas, pair ordering, float32 mapping and exclusion band. Read-only
source data, polygon edges and spatial indices are cached. Float64 Numba
segment predicates batch even/odd crossing and band checks without fastmath;
conservative edge indexing removes only impossible candidates. Distances far
outside the search radius use an infinity sentinel, which does not enter the
loss or sampling decisions. Threshold cases revert to GEOS.

Interior-ray intersections are evaluated by compiled segment equations. All
potentially active offset caps, ambiguous endpoints/collinearity and missing
rays are refined by the original GEOS operation. Consequently the stored
capped offsets and coordinates retain the reference rounding. Compact float32
coordinate keys preserve numeric +0/-0 equality, and the final uniqueness
proof is reused instead of sorting the entire point array a second time.

A missing second boundary intersection is now rejected as an individual ray
within the existing same-anchor retries and audited support rejection. It no
longer aborts unrelated valid units in an entire pool. Missing-ray attempts
are recorded; unresolved units retain their scale/requested offset during
supported-anchor resampling.

Both real25-plane first-generation pools were regenerated at full capacity:
all six array SHA values match the published reference byte for byte, covering
13,107,200 points.49,152 real uniform proposals were additionally compared
against GEOS labels and acceptance masks. The originally failed bunny gen18
completed with8 missing-ray attempts audited. On the profiling host its first
plane decreased12.919s→3.508s; the reference number includes profiling overhead.
Full single-worker pools took46.18s(bunny),34.61s(dog),45.57s(repaired bunny gen18).
These do not assert an8× end-to-end speedup from8 workers. Validation artifacts
remain in the local experiment directories and are not pushed with code.
