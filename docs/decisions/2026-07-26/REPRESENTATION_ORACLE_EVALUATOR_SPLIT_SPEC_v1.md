# Representation geometry, estimator, evaluator, and split specification v1

Date: 2026-07-27
Status: **frozen for specification, split-infrastructure, loader, and CPU-oracle implementation; no training or GPU authority**
Owner: Kirubes
Branch: `agent/representation-oracles-20260726`
Required base: `f641af5220ededb22a9ca0555a05250440aed0b8`

This document freezes the intended semantics for the first three steps of the
v2.6 programme amendment. Kirubes authorized the bounded preparation work on
2026-07-27 after clarifying the hybrid within-object protocol. Independent
Fable 5 High review was performed with read-only file inspection and reported
no methodological blocker to the hybrid design. This status authorizes only
the versioned specification, provenance manifests, split generation and
verification, loader/training-mode implementation, and deterministic CPU
oracles. It authorizes no model training, checkpoint-driven scientific
selection, GPU smoke, matrix, rendering, or cluster submission.

## Plain-language scope

This document does **not** change the dataset. In this document:

- **geometry** means writing down which way is right, down, clockwise, larger,
  and nearer in the stored images so that the code cannot silently use the
  opposite sign;
- an **estimator** is the existing calculation that turns each heatmap into one
  `(x, y)` point;
- an **evaluator** is scoring code that decides whether predicted points are
  distinct, on the object, stable, and moving as intended;
- an **oracle** is a deliberately obvious made-up example, such as four
  separated points rotated exactly six degrees or four points collapsed onto
  one location, used to prove that the scoring code says the right thing;
- a **split file** is a small JSON list saying which already-existing image
  pairs may be used for training, checking during training, or final testing.
  It does not edit, copy, rerender, relabel, or replace any image.

The historical Gate 0, Decision 2.3, 324-run reanalysis, and saved Task
20/55/80 evaluations already exist. None of the **new** training or GPU work
proposed here has run on this branch. The new split and oracle work begins only
after the corresponding rules are versioned. If a frozen split rule is
impossible or leaves too little evidence to be useful, the required action is
to report that fact and ask Kirubes—not to change the dataset or quietly
weaken the rule.

## 1. Decision lock

**Decision.** Decide whether one production evaluator and one auditable split
protocol are semantically correct enough to compare representation quality
before resolution or descriptor training.

**Current claim.** The shared affine dynamics result is real, but it does not
establish distinct, active, materially attached, stable keypoints. Gate 0 and
Decision 2.3 closed the readout-repair question, not the representation-quality
question.

**Known evidence.**

- Gate 0 is complete at 6/150 (4.0 percent) correct-dominant-mode high-error
  pairs versus the frozen 50 percent rule. Windowing remains demoted.
- Decision 2.3 A/B/C each passed 3/3 supervised hammer-roll seeds. The fixed
  expectation remains the baseline.
- The 324-run hammer roll sweep recovered a near-6-degree shared affine
  operator throughout, including clean prediction-only configurations.
- Task 55 is the clean baseline, Task 80 is the assisted baseline, and Task 20
  is a collapsed negative-control fixture, never a winner candidate.
- The present training/evaluation code cannot yet consume the requested
  protocol safely: an index-driven run reuses its training dataset as
  validation, pair metadata are not transformation-complete, and existing
  rollout code is roll-specific.

**Critical unknown.** Can the new evaluator reject known semantic failures
(wrong rotation sign, reflection, collapse, sliding, coincidence, inactivity,
off-object points, and split leakage) while accepting exact planted cases
through the same production code path?

**Main failure mode.** A structurally valid implementation passes unit tests
but silently evaluates the wrong transform, leaks frames, rewards collapse, or
uses a diagnostic definition that differs between planted and checkpoint data.

**Constraints.** Existing rendered frames only; no Gate 0 or Decision 2.3
rerun; no GPU work; no new scientific tolerance selected after viewing an
intervention result; no pooled directions, strides, or transform families; no
SEM over correlated frames; no one-number winner score.

**Next gate.** Generate and independently verify the deterministic two-way
split manifests from existing frames, then prove the development and
fixed-final loader modes cannot alias train and held-out data. Only after that
split gate passes may deterministic planted geometry/estimator/metric cases
and the saved Task 20 checkpoint run through the production evaluator. This
preserves the binding v2.6 order: specification, split indices, then oracles.
Any critical mismatch blocks all Task 55/80 training.

## 2. Scope and non-scope

This specification covers:

1. output-coordinate and transformation semantics;
2. exact analytic/perfect-point cases;
3. the finite-grid spatial-expectation estimator path;
4. the production evaluator input/output contract and metric definitions;
5. deterministic oracle acceptance criteria;
6. frame/pair/object split constraints and manifests.

It does not freeze:

- descriptor layer, patch radius, negatives, temperature/margin, or loss
  weight;
- a 64-versus-128 architectural implementation;
- fresh training seeds, budget, or noninferiority margins;
- the numeric coincidence-trigger threshold;
- a yaw/pitch scientific acceptability threshold;
- any Slurm matrix.

Those require later versioned specifications after this gate passes. In
particular, this document does not silently reopen the timeboxed family of
post-softmax fixed-weight shape losses. Any separation arm in a later targeted
intervention matrix requires its own explicit mechanism definition and
synthetic authorization.

## 3. Canonical coordinate and estimator semantics

### 3.1 Output coordinates

- A point is a column vector `p = [x, y]^T`.
- Normalized image coordinates occupy the closed square `[-1, 1]^2`.
- `x = -1` is the leftmost image-sample centre and `x = +1` the rightmost.
- `y = -1` is the top image-sample centre and `y = +1` the bottom.
- Pixel conversion for an `H x W` image is
  `u = (x + 1)(W - 1)/2`, `v = (y + 1)(H - 1)/2`.
- A shared affine operator is `p' = A p + b`. Implementation storage may use
  row batches (`pts @ A.T + b`), but artifact semantics remain column-vector
  semantics.
- The image origin for affine formulas is `[0, 0]^T`, the image centre. A
  transform about another centre `c` has `b = c - A c`.

These statements match the current endpoint-aligned `torch.linspace(-1, 1, W)`
and `torch.linspace(-1, 1, H)` estimator grid. Therefore adjacent estimator
grid spacing is exactly `2/(W-1)` in x and `2/(H-1)` in y. The shorthand
`2/W` or `2/H` must not be used in metrics or metadata.

### 3.2 Fixed spatial expectation

For heatmap logits `L[n,v,u]` and temperature `tau`,

`P = softmax(vec(L)/tau)`,
`K_x = sum(P[u,v] x_u)`, and
`K_y = sum(P[u,v] y_v)`.

The following are provenance fields, not implicit defaults:

- observed input height/width;
- observed heatmap height/width;
- endpoint-grid convention;
- `tau`;
- logit dtype;
- softmax dtype;
- crop and resize operations;
- `align_corners` choice for any sampling path.

The production evaluator must reconstruct these fields from a bound manifest or
checkpoint and fail closed if any are absent or inconsistent. It must not infer
64 or 128 from a command-line label.

### 3.3 Exact estimator fixtures

The estimator oracle uses two non-learned logit constructions:

1. **Grid-centre fixture:** one finite logit at a selected grid cell and
   negative infinity elsewhere. The expectation must equal that cell centre.
2. **Bilinear-mass fixture:** for a point inside one grid cell, distribute
   probability over its four vertices with the analytic bilinear weights and
   use `L = tau * log(P) + C` as logits for any finite constant `C`
   (negative infinity for zero mass). After division by `tau`, the expectation
   must equal the planted point.

Fixtures cover centre, edge, corner, and asymmetric sub-cell points on both
64x64 and 128x128 grids, in float64 and production float32. A separate
isotropic-Gaussian-logit sweep measures finite-grid bias but is descriptive
calibration, not an exactness oracle.

## 4. Transformation semantics and expected operator families

All dataset manifests must name `transform_family`, physical axis or direction,
signed generator, stride, cyclicity, source state, target state, and expected
2-D family. Reverse directions and different strides are separate group
elements and separate pair files.

### 4.1 Full-circle world-Z roll

- Corpus: six objects, 180 frames/object, physical angles
  `0, 2, ..., 358` degrees.
- Primary generator: cyclic `+6` physical degrees, source index `i`, target
  `(i+3) mod 180`.
- In the locked image coordinate system, the existing code-path and rendered
  mask audit define positive dataset `theta` by
  `R_img(delta) = [[cos(delta), -sin(delta)], [sin(delta), cos(delta)]]`.
  Because image y increases downward, this appears clockwise on screen.
- About projected centre `c`, the exact planar family is
  `A = R_img(delta)`, `b = c - A c`.
- A wrong-sign rotation is a critical failure. The evaluator must not report
  the better of `+delta` and `-delta`.
- Intermediate rollout horizons are `k=1..59`; `k=60` is closure and is
  reported separately.

Before this sign is used against learned results, code-path inspection and a
rendered output-level mask/landmark check must independently agree. Failure to
agree blocks the roll oracle rather than permitting sign minimization.

### 4.2 Camera-plane translation

- Physical world-x and world-y directions, each sign, and each stride remain
  separate.
- Exact synthetic image-plane fixtures use `A = I` and a signed normalized
  image displacement `b`.
- For rendered data, world displacement is not substituted directly for `b`.
  A frozen camera calibration maps world displacement to signed image
  displacement. Under the frozen camera used by the existing translation
  corpus, increasing either world-x or world-y is expected to decrease the
  corresponding normalized image coordinate; this sign must also be confirmed
  from rendered mask centroids before learned results are evaluated.
- The evaluator reports `||A-I||_F`, signed bias-component error, orthogonal
  bias leakage, and multi-step bias accumulation. It does not report a rotation
  AUC or cyclic closure.

### 4.3 Uniform scale

- Directions (expansion/contraction) and ratios remain separate.
- For ratio `r` about projected centre `c`, the exact image-plane family is
  `A = r I`, `b = c - r c`.
- The evaluator reports mean scale error, anisotropy
  `|sigma_max(A)-sigma_min(A)|`, off-diagonal/shear residual, centre/bias
  residual, and non-cyclic multi-step error.
- The 31-frame staggered absolute-size holdout is never used for model,
  checkpoint, threshold, or recipe selection.

### 4.4 World-Y yaw and world-X pitch

- Each physical axis, sign, and stride remains separate and non-cyclic.
- The scientific claim is limited to a shared local 2-D affine approximation
  under projection, foreshortening, visibility changes, and self-occlusion.
- No exact 2-D equivariance, 3-D landmark recovery, or finite-arc closure is
  claimed.
- The exact affine control uses an orthographic camera and a constant-depth
  planar point set; it must be recovered to numerical tolerance.
- Separate planted pinhole-camera fixtures project frozen 3-D points with fixed
  intrinsics/extrinsics, apply the requested world-axis rotation about a fixed
  centre, preserve point identities, and explicitly mark visibility. Even a
  plane generally induces a projective homography rather than an exact affine
  map under perspective. The evaluator therefore fits its shared affine map on
  the visible correspondence set and reports residuals by source angle, target
  angle, depth configuration, direction, and stride.
- Pinhole planar and depth-varying cases produce descriptive approximation
  floors. This specification deliberately sets no scientific pass threshold
  for those floors; the measured floors and a later user-approved claim margin
  are required before yaw/pitch training.

## 5. Production evaluator contract

### 5.1 One path for planted and saved cases

The production evaluator accepts an immutable evaluation bundle containing:

- ordered points `[object, seed, frame, channel, xy]`;
- optional logits/probabilities plus full estimator metadata;
- per-frame object masks and mask geometry metadata;
- per-frame physical state and visibility flags;
- a transformation manifest;
- explicit pair index;
- optional learned `A,b` or permission to fit them on the named training
  partition only;
- source commit, file hashes, checkpoint hash, and evaluator-config hash.

Planted cases and saved checkpoint cases use the same public evaluation
function after bundle construction. No metric may have a special
"oracle-only" implementation.

Unknown transform families, missing hashes, mixed directions/strides, duplicate
pair rows, out-of-range indices, inconsistent cyclic flags, or missing
estimator fields are fatal errors. Derived coordinates and metrics must be
finite. Declared estimator fixtures may use negative-infinity logits only when
each heatmap has at least one finite logit; NaN, positive infinity, or an
all-negative-infinity heatmap is always fatal.

### 5.2 Operator metrics

- **Proper-rotation extraction:** compute the SVD polar factor with determinant
  correction so the reported rotation lies in `SO(2)`. If `det(A) <= 0`,
  record a reflection/improper failure before reporting an angle.
- **Operator-angle error:** locked signed circular distance between the proper
  rotation angle and the manifest target; roll primary only.
- **Full-corpus identity-normalized rollout AUC:** for each roll horizon
  `k=1..59`, compute mean model MSE across all cyclic start frames divided by
  mean identity-predictor MSE across the same frames, then take the unweighted
  arithmetic mean across the 59 ratios. This is the historical full-orbit
  representation/dynamics diagnostic and remains required for fixture
  comparability. Under the blocked split it mixes train and holdout frames, so
  it is not held-out generalization evidence and may not select or confirm a
  model.
- **Role-scoped holdout rollout AUC:** validation/test evidence uses only
  horizons for which source and target endpoints remain inside the named
  holdout block. For the frozen 24-frame roll holdout and skip 3, these are
  `k=1..7`. At each horizon, use every valid holdout start, record numerator,
  denominator, and start IDs, and take the unweighted arithmetic mean of the
  seven horizon ratios. A zero/non-finite denominator or empty horizon is
  fatal.
- `k=60` roll closure is a separate full-corpus diagnostic and never enters
  either AUC.
- Variance-normalized AUC and old `k10/k1` may be emitted only as diagnostics
  clearly marked forbidden for selection.

Start-frame distributions are correlated descriptive samples. Store their
median, interquartile range, and full per-start values; do not compute or plot
SEM. Seeds and objects are the replication units.

### 5.3 Representation metrics

All spatial distances below use `[0,1]^2` image coordinates divided by the
per-frame object-mask bounding-box diagonal unless explicitly labelled
normalized `[-1,1]` units.

- **Continuous canonical drift:** per channel, transform each point back by the
  manifest's exact inverse planar transform, subtract that channel's temporal
  canonical mean, and report RMS. Report all channel values plus seed/object
  median and maximum. For yaw/pitch use fitted-reference residual terminology,
  not canonical exactness.
- **Trajectory separation:** report each frame/channel nearest-neighbour
  distance, per-pair median/minimum, and trajectory median nearest-neighbour
  distance twice: once over all channels, and once over channels that pass the
  frozen active/on-object eligibility rule. The all-channel view is mandatory
  for collapse detection so static coincident channels cannot disappear from
  the denominator.
- **Dual-denominator safeguard:** always store the total channel count and the
  eligible active/on-object count. If the eligible count falls below a
  separately preregistered minimum, mark eligible-only representation metrics
  void rather than healthy; the all-channel collapse metrics still apply.
- **Close-pair categories:** retain the audited descriptive definitions:
  close distance `<0.06` object diagonals; persistent if close in at least
  `0.50` of frames; recurrent if at least `0.10` but below `0.50`; transient
  crossing if nonzero with longest cyclic run at most `0.10` of frames; and
  clustered if median distance `<0.12` without an earlier category. These
  values define evaluator output categories only. They are not the
  preregistered coincidence-trigger threshold.
- **Channel-health terms remain distinct:** `motion_inactive` means failing an
  absolute, transform-normalized motion criterion; `inactive_off_object` adds
  failure of the on-object criterion; `heatmap_flat_dead` is the separate
  historical heatmap condition (median entropy greater than
  `log(HW)-0.5` and median maximum probability less than `2/(HW)`). Store
  absolute motion energy, fraction of known object-transform magnitude,
  heatmap peak, entropy, and on-object rate per channel. A purely max-relative
  activity rule is forbidden because a collapsed run can make numerical jitter
  look active. No one channel-health label may silently stand in for another.
- **Peak/channel switching:** compare heatmap modes and descriptor-free
  trajectory continuity under the known transform. Report mode-cell jumps,
  channel-order assignment changes, and durations. Do not infer material
  identity from coordinate smoothness alone.
- **On-object:** sample the correspondingly cropped/resized mask with the same
  coordinate convention as the image. Report per-channel fraction and aggregate
  count; fail closed if image and mask geometry differ.

The metric artifact stores raw per-frame/per-pair values, definitions,
thresholds, sample unit, `n`, aggregation hierarchy, and whether each summary
is descriptive or inferential.

### 5.4 Selection boundary

The primary axes remain separate:

1. roll operator-angle error;
2. role-scoped holdout identity-normalized rollout AUC (`k=1..7` for the
   frozen roll holdout);
3. continuous canonical drift;
4. trajectory separation plus persistent/recurrent duplication;
5. active/on-object safety.

The full-corpus identity-normalized `k=1..59` AUC remains mandatory as a
separately labelled historical/composition diagnostic but is forbidden as
held-out evidence or a confirmation checkpoint selector.

There is no overall scalar winner. Later intervention specifications must
freeze safety floors, noninferiority margins, seed rules, and a deterministic
Pareto/tie-break rule before training. Exact fixtures set only numerical
correctness tolerances. Separately frozen replay tolerances bind historical
checkpoints. Scientific margins may be informed by prespecified Task 55/80
baseline spread in scientific units, but never by Task 20 or by unblinded
intervention outcomes.

### 5.5 Forbidden decision evidence

The following may be retained only as explicitly labelled diagnostics and may
not select a recipe or establish semantic correctness:

- the historical dispersion ranking;
- choosing the better of `+angle` and `-angle`;
- variance-normalized AUC or isolated `k10/k1`;
- file counts, valid JSON, a zero exit code, or checkpoint recovery alone;
- pooling directions, strides, transformation families, objects, or channels
  to hide a failing stratum;
- SEM over correlated frames, horizons, or overlapping windows;
- one scalar score that trades collapse against operator accuracy.

## 6. Deterministic oracle matrix and proposed acceptance rules

### 6.1 Two separate threshold registries

The evaluator keeps two registries that must never be conflated:

1. **numerical correctness tolerances** answer whether code reproduces an exact
   planted answer within floating-point error;
2. **scientific decision thresholds** answer whether a learned representation
   is good enough to continue, reject, or trigger the conditional coincidence
   branch.

Exact fixtures may calibrate registry 1. Historical Tasks 20/55/80 use a
separate frozen replay-regression tolerance and no learned outcome may be
relabeled as an exact answer. Task 55/80 baseline spread may inform Registry 2
only in prespecified scientific units; Task 20 may not. Registry 2 must be
frozen before unblinded intervention outcomes are viewed. No number may appear
in both registries, and no fixture may be used to retune a boundary so its
expected pass, failure, or classification changes.

### 6.2 Numerical tolerance calibration

Oracle tolerances are implementation-correctness tolerances, not scientific
effect thresholds.

Before the first production-oracle invocation, an independent reference
implementation evaluates the exact fixtures in float64 and float32. For each
metric/dtype, freeze:

`T = max(32 * machine_epsilon * S, 4 * E_ref)`,

where `S = max(1, maximum exact expected magnitude)` and `E_ref` is the maximum
absolute reference error over the preregistered calibration fixtures. The
calibration fixture IDs, reference-code hash, environment, raw errors, and
resulting numeric `T` values are committed to the specification in a v1.1
amendment before production-oracle execution. No tolerance may be widened
after a production failure without stopping for user approval.

Provisional ceilings, which the calibrated values may equal or tighten but not
exceed without review, are:

- float64 direct geometry/operator metrics: `1e-10`;
- float64 estimator coordinates: `1e-10` normalized-coordinate units;
- production float32 estimator/metric coordinates: `1e-5`
  normalized-coordinate units;
- exact fractions/counts/categories/hashes/split predicates: exact equality.

### 6.3 Required cases

| Case | Must pass | Deliberate falsifier that must fail |
|---|---|---|
| Exact centred roll | locked-sign angle, bias, rollout, closure, canonical drift | opposite sign |
| Off-centre roll | recovered `b=c-Ac`, zero canonical drift | centre silently forced to zero |
| Proper vs improper map | proper rotation accepted | reflection or `det(A)<=0` |
| Translation x/y | signed bias and linear accumulation | swapped axis or wrong y sign |
| Uniform scale | ratio, isotropy, centre/bias | additive-scale or anisotropic map labelled exact |
| Orthographic planar yaw/pitch | exact affine recovery | cyclic/exact-rotation claim |
| Pinhole planar/depth-varying yaw/pitch | descriptive nonzero floor | zero-floor/exact-equivariance claim |
| Controlled drift/sliding | planted amplitude recovered | drift reported as zero |
| Separated channels | zero persistent/recurrent duplicates | false collapse |
| Coincident channels | persistent duplicate detected | rigid-distance metric alone reports healthy |
| Transient crossing | transient category | persistent duplicate category |
| Active/on-object | expected counts exactly | max-relative jitter marked active |
| Off-object points | expected on-object failure | crop/mask mismatch accepted |
| Peak switch/dead heatmap | switch/dead evidence recovered | coordinate-only path reports healthy |
| Task 20 checkpoint | collapse flag; persistent rate and separation reproduce bound fixture | eligible winner |

For Task 20, the fixture manifest binds the exact saved checkpoint, source
commit, preprocessing, dataset, mask, and expected historical signature. The
checkpoint SHA-256 is
`96433168767659ae9144a35a4f7889c3226b65a7a0c5341197d48232a66fe622`
(persistent duplicate rate `1.0`; trajectory median nearest-neighbour distance
approximately `1.27e-7` object diagonals). Before execution, the v1.1
calibration amendment must freeze a regression tolerance around the recomputed
bound fixture. The historical signature currently comes from an unversioned
reanalysis script, so it is evidence for fixture selection, not executable
authority; the needed metric code must be implemented and versioned in this
branch rather than imported from that script. Task 20 passes the oracle only
when the evaluator flags collapse; it never passes as a representation
candidate.

The saved Task 55 and Task 80 checkpoints are also calibration-only replay
fixtures, bound respectively to SHA-256
`942a32082b6bfe83526253cc2dd39e49792b8260d12fc1361a11bae812992418`
and
`ccb613c87788a229929b1f6ead002d626819e970237d0e0ad5ca75c9318004f3`.
They may verify legacy model reconstruction, expected non-collapse behavior,
and the 64-resolution evaluator path. They are not recipe-held-out
confirmation objects, cannot select a future intervention, and cannot justify
widening a failed tolerance. The future matched Task 55/80 experiments remain
later programme steps and are distinct from these saved fixtures.

### 6.4 Gate verdict

The oracle suite passes only if:

- every positive case is within its frozen tolerance;
- every falsifier produces the expected critical failure;
- Task 20 is classified as collapse by the same evaluator entry point;
- Tasks 55/80 match their separately frozen replay fixtures and are not
  classified as all-channel collapse, without becoming selection winners;
- rerunning deterministic fixtures yields byte-identical canonical JSON after
  excluding explicitly named runtime fields;
- all artifact hashes and provenance fields validate.

Any critical case failure stops. Passing unit tests, valid JSON, exit status
zero, or agreement with an old summary alone is insufficient.

## 7. Dataset split specification

### 7.1 Immutable source corpus

Split generation reads existing frames, masks, metadata, and existing all-pair
indices without modifying them. It writes version-controlled pair-index files
and manifests under:

`docs/decisions/2026-07-26/representation_oracle_splits/`

No image is copied, moved, renamed, or rerendered. The split manifest records
the external dataset root as a provenance source but uses dataset-relative
paths in all pair files.

Before generation, a read-only inventory must bind:

- dataset basename and semantic-lock hash;
- sorted frame/mask/meta relative paths and SHA-256 hashes;
- object identities and per-object counts;
- physical states and generator command/axis;
- every source all-pair-index hash;
- source commit or, for unversioned rendered corpora, an explicit
  `unversioned_source=true` warning plus content manifest.

An incomplete or semantically inconsistent corpus blocks split generation.
In this document, the "four future datasets" are already-rendered yaw, pitch,
scale, and translation corpora that are future only in the experimental
programme: they have not yet been used for operator training. "Future pair
files" means their existing, not-yet-trained pair files.

For roll, the tracked generator copy must not be treated as provenance merely
because it is in Git: the existing recovered-source audit identifies it as an
unfaithful rewrite and identifies the actual rendered corpus with an
unversioned verified generator. The verified generator and dependency snapshot
must be imported as versioned evidence, without modifying the source dataset,
before the roll corpus can pass this inventory gate. The four future datasets
and their generator are likewise untracked inputs and require content-bound
provenance snapshots.

The current inventory also records unresolved contradictions that must be
closed in that snapshot rather than copied into a new manifest:

- the yaw/pitch documented plan/locks state margin `>=0.05` and bbox `<=0.82`,
  while the executed generator uses a two-pixel edge margin plus
  area/base `>=0.60`; observed maximum bboxes are approximately `0.861` for
  yaw and `0.893` for pitch;
- the scale area gate `>=0.015` exists in the generator but is omitted from its
  semantic lock;
- no frozen translation pixel/world calibration, camera field-of-view, or
  intrinsic matrix artifact exists;
- future pair files omit operator identity, and translation uses `direction`
  for axis while rotation/scale use it for forward/reverse;
- inherited roll `source_split` labels and the existing frame-list splits are
  not the new recipe roles or leak-free pair splits;
- the historical handoff says the corrected reanalysis emits percentile bands,
  but the implementation emits mean/std/SEM/min/max and no percentile-band
  artifact was found.

Each discrepancy requires a versioned, evidence-backed disposition. A new
manifest may describe what was actually executed, but it must not rewrite the
historical plan as though the two had agreed. These dispositions are dated
documentation errata, not dataset edits.

### 7.2 Object roles

The same recipe-level roles are frozen for the five existing transform
corpora:

- **development:** `engineers_hammer_vray`;
- **confirmation:** `b03_banana_01_high`, `kettle`;
- **final test:** `dewalt_compact_drill_vray`, `b03_trumpet_vray`,
  `toy_monkey_medium`.

"Development" is forced by history: only the hammer has already driven recipe
and diagnostic decisions. The remaining five objects are locked into a
two-object confirmation tier and a three-object final tier spanning
handled/compact, elongated organic, tool-like, thin structured, and articulated
shapes. The assignment is a design lock, not a claim that these objects are
exchangeable or independent draws from a population.

"Recipe-held-out" means held out from recipe and hyperparameter selection, not
that a pretrained hammer network is evaluated zero-shot. After the recipe is
frozen, each confirmation/final object receives a fresh model trained only on
that object's training pairs. There is no validation loader, early stopping,
or best-checkpoint choice on those objects. The authoritative checkpoint is the
final checkpoint after the hammer-frozen epoch count, and its test pairs are
opened once under a frozen finalizer. Confirmation objects may reject the
recipe but may not tune it silently. If a confirmation outcome changes the
recipe, that object is thereafter development evidence and cannot retain a
confirmation label. Final-test objects remain unopened until the confirmation
decision is frozen.

This step tests recipe transfer to freshly trained object-specific models. It
does **not** test a model-generalization holdout, where one already-trained
network is evaluated on an unseen object, and it authorizes no zero-shot
other-object claim.

A versioned contamination ledger must distinguish representation-recipe
influence from dataset-generation/geometry inspection. Inspecting an object to
set a corpus-wide rendering range is recorded but does not by itself make its
learned representation outcome known. If evidence is found that any
non-hammer learned outcome influenced a recipe, threshold, duration, or
diagnostic choice, stop and revise the role lock before opening new outcomes.

### 7.3 Frame/pair separation

The following leakage assertions are scoped within one
`(transform, stride, direction)` family, because the programme forbids pooling
families. Global frame-disjointness across different stride families is not
claimed and is infeasible on the smaller grids. Cross-family protection comes
from the recipe/object-role and contamination rules. If a later experiment
combines strides, axes, or directions in one model, it requires a new
combined-family split before training.

Every object has exactly two roles:

- development: `train` and `validation`;
- confirmation/final: `train` and `test`.

For every dataset, object, direction, and primary stride:

- train and holdout frame-ID sets are disjoint;
- both endpoints of a pair belong to its named split;
- no frame appears as source or target in both train and holdout;
- the minimum physical/index separation between frames assigned to different
  splits is at least the relevant primary stride;
- pairs crossing a split or guard boundary are dropped, never reassigned;
- forward and reverse pair files preserve the same endpoint partition and
  remain separate;
- no axis, direction, stride, or transform family is pooled.

The phrase "full 180-frame roll dataset" means the immutable primary corpus and
full-circle support remain available. It does not require every frame or every
cyclic edge to be used in every split. Guard frames and boundary-crossing pairs
are expected exclusions and are enumerated.

The primary split geometry is identical across objects. The holdout label is
`validation` for the hammer and `test` for the other five objects. This avoids
changing the state region when moving from development to confirmation.

### 7.4 Frozen split-generator method

The generator is deterministic, pure over its input manifest, and receives no
model outputs. It filters existing primary all-pair files; it does not construct
transitions from flattened frame adjacency. A pair is retained only when both
endpoints belong to the same named block. Guard and boundary-crossing pairs are
enumerated as exclusions.

The frozen primary templates and exact per-object/per-direction counts are:

| Family | Primary group element | Train frames | Holdout frames | Guard frames | Train pairs | Holdout pairs |
|---|---|---:|---:|---:|---:|---:|
| roll | cyclic skip 3, `+6 deg` | `27..176` (150) | `0..23` (24) | `24..26`, `177..179` (6) | 147 | 21 |
| yaw | non-cyclic skip 6, each sign | `0..46`, `74..120` (94) | `53..67` (15) | `47..52`, `68..73` (12) | 82 | 9 |
| pitch | non-cyclic skip 6, each sign | `0..46`, `74..120` (94) | `53..67` (15) | `47..52`, `68..73` (12) | 82 | 9 |
| scale | non-cyclic skip 4, each direction | `0..48`, `72..120` (98) | `53..67` (15) | `49..52`, `68..71` (8) | 90 | 11 |

Roll guards are wrap-aware. The yaw/pitch holdout is the interior
`-7..+7 deg` band, so the primary claim is interpolation to unseen poses, not
endpoint extrapolation. The scale holdout is an interior log-scale band.

Translation is split in two-dimensional grid coordinates, never flattened
indices:

- x-family: train rows `grid_y=0..6`, guard rows `7..8`, holdout rows
  `9..10`, with all eleven x positions retained inside each used row;
- y-family: transpose the rule—train columns `grid_x=0..6`, guard columns
  `7..8`, holdout columns `9..10`, with all eleven y positions retained.

For primary stride 3 this gives, per object/axis/sign, 77 train frames,
22 holdout frames, 22 guard frames, 56 train pairs, and 16 holdout pairs. The
nearest train and holdout context indices differ by 3, exactly the frozen
primary stride. A compact square patch is rejected: at stride 3 it retains too
few holdout edges and discards substantially more usable training pairs.

The generator and an independent verifier must reproduce every count above.
Any mismatch, endpoint overlap, guard membership, wrong axis, or non-identical
regeneration fails. It must not relax a guard, pool a family, change a primary
stride, or repurpose the absolute scale holdout.

### 7.5 Dataset-specific split rules

- **Roll:** cyclic `+6` pair edges use the primary split above with wrap-aware
  guards. Holdout image contents are not opened by training.
- **Yaw/pitch:** positive and negative skip-6 files remain separate but use
  identical frame blocks. Endpoint residuals remain report strata; the frozen
  holdout itself tests interior interpolation.
- **Scale:** expansion and contraction skip-4 files remain separate but use
  identical frame blocks. `eval_abs_holdout/` and its 31 staggered sizes are
  absent from training, validation, checkpoint selection, recipe selection,
  and primary test manifests. It is a later one-shot absolute-size probe.
- **Translation:** x and y directions, signs, and strides are separate.
  Orthogonal row/column bands preserve the full transformed-axis range in both
  train and holdout. A frame may have a different role for x and y because
  those families are never pooled; any pooled x/y model requires a new joint
  partition.

Only the primary group elements above train or select models. Secondary
strides are post-freeze composition/stress diagnostics, not separately tuned
training regimes. They cannot change the recipe, epoch, checkpoint, or primary
verdict. Their pair files require a stride-aware filter and explicit
`diagnostic_only=true` metadata. Any secondary stride whose span is at least
the holdout span has zero holdout pairs and may be reported only as an
in-training-support composition diagnostic. In particular, the 15-frame scale
holdout has zero skip-20 pairs. Such a diagnostic is not held-out evidence. The
resulting claim is about how a primary-step model composes to other step sizes,
not about models trained independently at those sizes.

### 7.6 Split artifacts and pass rule

Each JSON pair file includes schema version, dataset basename/hash, object-role
mapping, split, transform family, physical axis/direction/sign, stride and
units, cyclicity, generator commit/config hash, ordered pairs, and a content
hash of its canonical JSON payload with the content-hash field omitted. A
top-level manifest records counts and hashes for every file plus:

- frame endpoint sets;
- excluded guard frames;
- dropped cross-boundary pairs;
- pair/source/target counts by object/direction/stride/split;
- physical-state coverage;
- object-role lock;
- scale-holdout non-membership proof;
- independent-verifier result.

The split gate passes only when all exact predicates above pass, regeneration
is byte-identical, every relative path resolves against the bound corpus, and
the production loader enforces both modes without fallback:

- development: explicit train plus validation indices, validation-selected
  `best_model.pt`, and no test loader;
- fixed-final: explicit train plus test indices, no validation loader, no
  `best_model.pt`, exact frozen epochs, authoritative `final_model.pt`, and one
  post-training test evaluation.

Invalid arguments and overlapping endpoint sets fail before a run directory is
created. Legacy whole-directory auto-evaluation is forbidden for indexed
split modes because it is not partition-aware.

## 8. Required implementation changes before oracle execution

This requirements list is authorized for implementation under the status at
the top of this specification:

1. add transformation-complete pair metadata and strict schema validation;
2. add distinct train/validation and train/test indexed modes, remove
   train-as-validation fallback, and make fixed-final checkpoint policy
   unreachable from validation or test selection;
3. bind crop, resize, actual heatmap shape, estimator temperature/dtype, source
   commit, and index hashes into checkpoints/evaluation bundles;
4. implement a transformation-aware evaluator rather than deriving all
   semantics from `frame_skip * yaw_step_deg`;
5. correct proper-rotation extraction and lock the target sign;
6. preserve logits/heatmaps for switching and dead-channel metrics;
7. replace correlated-frame SEM outputs with named descriptive summaries;
8. make missing auto-evaluation evidence a fatal gate failure;
9. prove the 64/128 checkpoint reconstruction path before the resolution
   experiment; and
10. add the deterministic planted-case suite and Task 20 fixture binding.

## 9. Statistical reporting

- Frame, horizon, channel, and overlapping start-frame summaries are
  descriptive correlated samples.
- Optimization seeds and objects are the replication units.
- Every plot/table must name its statistic, exact computation, sample unit,
  `n`, and descriptive/inferential status.
- No population inference is made from three seeds.
- Any later hypothesis test must separately preregister its null, direction,
  alpha, independence assumptions, multiplicity handling, effect size, and
  confidence interval.

## 10. Stop conditions

Stop and ask Kirubes before proceeding if:

- the roll sign or any physical axis fails code-path/output-level agreement;
- a dataset semantic lock, frame/mask count, or hash is missing/inconsistent;
- a requested split is infeasible under the frozen leakage/guard rules;
- the production loader cannot keep train and validation distinct;
- a tolerance cannot be calibrated below its provisional ceiling;
- any planted falsifier is accepted;
- Task 20 is not flagged as collapse;
- Fable returns no substantive review or raises an unresolved methodological
  blocker;
- the work would require changing the v2.6 programme order;
- any file would need to be deleted or moved.

## 11. Expected closing artifacts

After later authorized implementation and execution:

- `REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.1.md` or a v1.1 amendment with
  frozen numeric calibration;
- version-controlled split JSON files and `SPLIT_MANIFEST.json`;
- `ORACLE_CASE_MANIFEST.json`;
- per-case raw and summary JSON;
- `TASK20_NEGATIVE_CONTROL.json`;
- `ORACLE_GATE_REPORT.md`;
- exact source commit, dependency/runtime manifest, and artifact hashes.

Passing this gate authorizes only the matched 64-versus-128 specification and
smoke design. It does not authorize a full GPU matrix.
