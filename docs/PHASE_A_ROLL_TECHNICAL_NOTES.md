# Phase A Keypoint Project: Technical Notes and Current Status

Last updated: 2026-06-09

This is the primary technical handoff document for the current Phase A
keypoint project. It combines the model specification, dataset and training
protocol, evaluation definitions, completed hammer sweep, current scientific
problems, and planned validation work.

The older documents remain useful as chronological records:

- `docs/PROJECT_HISTORY.md`: yaw/dense-operator experiments through 2026-03-06.
- `docs/PHASE_A_ROLL_RESULTS_AND_OPEN_QUESTIONS.md`: first full-360 result and
  the action-loss diagnosis.
- `docs/MOE_2026-06-08.md`: concise June 8 research update.

Where those documents describe a planned sweep or retain older promotion
criteria, this document takes precedence.

## 1. Executive Technical Summary

The project tests whether sparse image keypoints can emerge because they make a
known visual transformation simple, predictable, and compositional.

The current controlled setting is a TDW-rendered 360-degree World-Z roll of a
single object. A shared CNN maps each RGB image to 10 two-dimensional
keypoints. A constrained `shared_affine` operator then applies the same learned
2-by-2 affine map to every keypoint:

```text
p_i(t+1) = A p_i(t) + b
```

The current hammer sweep contains 324 hyperparameter configurations, all using
one seed (`seed=42`) and the same 180-frame engineers' hammer sequence.

What is established in this controlled setting:

- The constrained operator reliably learns the correct transformation
  direction.
- All 324 runs learned a positive rotation; 322/324 were within 0.5 degrees of
  the intended 6-degree step.
- Median learned angle was 6.15 degrees.
- Median keypoint-on-object rate was approximately 80%.
- Replacing the learned operator or keypoints with strong ablations causes
  large prediction failures in the leading run.

What is not established:

- The keypoints are not yet consistently distinct.
- Many keypoints slide across the object rather than remaining attached to the
  same object-relative location.
- Every configuration has at least one near-duplicate keypoint pair under the
  current one-pose audit.
- The leading configuration has one dead channel, approximately 22%
  near-duplicate useful keypoints, 40% sliding keypoints, and only 50% clean
  keypoints.
- The leading configuration uses inverse and cycle losses, so its strong
  operator geometry is assisted rather than a pure prediction-only emergence
  result.
- The sweep is one object, one sequence, and one seed. It has no held-out
  sequence-level validation.

The central current result is therefore:

> Learning an accurate shared rotation operator is substantially easier than
> learning a distinct, object-attached, stable keypoint representation.

## 2. Current Scientific Question and Semantic Lock

The Stage 0 question is:

> Under a clean planar object rotation, can heatmap keypoints emerge as
> distinct, object-attached coordinates that support a simple shared,
> compositional transformation operator?

### Must Be True

For a strong scientific success claim, all of the following must be shown:

1. The rendered transformation is the intended TDW World-Z roll.
2. One learned primitive operator represents one fixed `+6 deg` step.
3. The same 2-by-2 rule acts independently on every keypoint.
4. The learned operator is non-trivial and improves on identity.
5. Keypoints are predominantly on the object.
6. Keypoints participate in the motion rather than remaining static.
7. Different channels occupy distinct object locations.
8. Keypoints remain stable in object-relative coordinates rather than sliding.
9. Intermediate repeated applications of the operator remain accurate.
10. Full-orbit closure is good when interpreted jointly with non-triviality and
    representation quality.
11. The result replicates across independent seeds and multiple objects.

### Must Not Happen

- Static keypoints plus an identity-like operator must not count as success.
- Low closed-orbit error alone must not count as success.
- On-object keypoints that overlap or slide must not be called distinct
  landmarks.
- A good operator must not be treated as proof of a good representation.
- Results using inverse/cycle supervision must not be described as
  prediction-only emergence.
- Within-sequence error bars must not be presented as population uncertainty.
- A single-object, single-seed sweep must not support a generalization claim.

### Evidence Required

- Code-path verification of the TDW axis, pair step, operator form, and losses.
- Visual and quantitative verification of object-relative motion.
- Direct measurement of observed keypoint separation, not inference from
  `lambda_disp`.
- Absolute errors over multiple rollout horizons.
- Identity, random-operator, shuffled-keypoint, and random-keypoint ablations.
- Independent-seed and multi-object reruns before publication-level claims.

## 3. Relation to the Earlier Project Stages

The earlier experiments used limited yaw arcs and a dense 20-by-20 operator.
They established that action supervision could break a static-keypoint shortcut
and that a learned dense operator could outperform identity. However, the dense
operator could mix keypoints arbitrarily, and the limited arc did not test a
closed transformation orbit.

The current roll stage changes the question in two important ways:

1. The dataset covers a complete 360-degree orbit with cyclic training pairs.
2. The main operator is `shared_affine`, which cannot mix keypoint identities.

The current results should not be read as a direct continuation of the old yaw
metrics. In particular, action-direction classification and old identity-ratio
thresholds behave differently on a closed orbit with a small 6-degree step.

## 4. Dataset

Dataset root:

```text
/Users/kirubeso.r/Documents/PhD/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2
```

Objects:

```text
engineers_hammer_vray
b03_banana_01_high
kettle
dewalt_compact_drill_vray
toy_monkey_medium
b03_trumpet_vray
```

Dataset specification:

- Renderer: TDW.
- Transformation: World-Z roll.
- Rendering method: TDW re-rendering at every angle, not PIL image rotation.
- Image size: `512 x 512`.
- Frames per object: `180`.
- Angles: `theta_t = 2t deg`, for `t = 0,...,179`.
- Full orbit: `0, 2, 4, ..., 358 deg`.
- RGB frames: `1080`.
- ID frames: `1080`.
- Binary mask frames: `1080`.
- Background RGB: `[166, 166, 166]`.
- Lighting, camera, and object scale are fixed within each sequence.
- Rotation is applied around the object centroid.

The dataset index records:

```text
operator_name = tdw_world_z_roll
theta_step_deg = 2.0
frames_per_object = 180
img_size = 512
```

All six generated object sequences passed the dataset-generation geometry
checks. The completed 324-run sweep currently uses only
`engineers_hammer_vray`.

## 5. Pair Construction

Training uses:

```text
frame_skip = 3
```

Therefore one primitive transition is:

```text
t -> (t + 3) mod 180
```

which is always a `+6 deg` step:

```text
frame 0,   0 deg -> frame 3,   6 deg
frame 1,   2 deg -> frame 4,   8 deg
...
frame 177, 354 deg -> frame 0, 0 deg
frame 178, 356 deg -> frame 1, 2 deg
frame 179, 358 deg -> frame 2, 4 deg
```

Main pair index:

```text
indices/pairs_skip3_cyclic.json
```

Counts:

- 180 forward cyclic pairs per object.
- 1080 forward cyclic pairs across all six objects.
- Three wrap pairs per object.
- The hammer sweep trains on 180 forward samples because `lambda_act=0`.

Other available indices:

```text
indices/pairs_skip1_cyclic.json       # 2 deg
indices/pairs_skip5_cyclic.json       # 10 deg
indices/split_phase_mod6.json         # generated but not used in this sweep
indices/split_arc_holdout_300_360.json # generated but not used in this sweep
```

## 6. Important Data-Split Limitation

For index-driven training, the current code sets:

```text
val_dataset = train_dataset
```

Consequences:

- Training and reported validation use the same 180 cyclic hammer pairs.
- `best_model.pt` is selected by total loss on the same sequence used for
  optimization.
- Evaluation also uses the same 180-frame sequence.
- The current sweep is an optimization and representation audit, not a
  held-out generalization experiment.

The term "validation loss" in the current artifacts means an unshuffled
evaluation pass over the training sequence. It must not be interpreted as
held-out validation.

This is a major current methodological limitation. Future confirmatory runs
need a phase holdout, an arc holdout, a separately rendered sequence, or an
object-level split depending on the claim being tested.

## 7. Model Architecture

### 7.1 Input Processing

- Input: RGB image.
- Sweep input resolution: `512 x 512`.
- No center crop.
- No data augmentation.
- ImageNet channel normalization.
- CNN padding mode: `reflect`.

### 7.2 Keypoint Extractor

The same CNN `K(.)` is applied to both frames in a pair.

For the 512-pixel sweep input:

| Layer | Operation | Channels | Spatial size |
|---|---|---:|---:|
| 1 | 7x7 conv, stride 2, BN, ReLU | 3 -> 32 | 512 -> 256 |
| 2 | 3x3 conv, stride 2, BN, ReLU | 32 -> 64 | 256 -> 128 |
| 3 | 3x3 conv, stride 2, BN, ReLU | 64 -> 128 | 128 -> 64 |
| 4 | 3x3 conv, stride 1, BN, ReLU | 128 -> 128 | 64 -> 64 |
| Head | 1x1 conv | 128 -> 10 | 64 -> 64 |

The head produces 10 raw heatmaps:

```text
H(x) in R^(10 x 64 x 64)
```

Each heatmap is spatially softmax-normalized with temperature `1.0`.
Soft-argmax computes the expected coordinate over a grid normalized to:

```text
x, y in [-1, 1]
```

The result is:

```text
K(x) in R^(10 x 2)
p = flatten(K(x)) in R^20
```

There is no decoder and no image reconstruction objective.

Parameter counts for the Task 80 architecture:

- Keypoint extractor: `246,666`.
- Forward shared-affine operator: `6`.
- Inverse shared-affine operator: `6`.
- Total: `246,678`.

The extractor therefore contains almost all trainable parameters. The
transformation model itself is intentionally minimal.

### 7.3 Forward Pass

For a pair `(x_t, x_t1)`:

```text
p_t      = K(x_t)
p_t1     = K(x_t1)
p_hat_t1 = F(p_t)
```

The extractor is trained jointly with the operator. Keypoints are neither
precomputed nor frozen.

## 8. Operators

### 8.1 Legacy Dense Operator

```text
p(t+1) = W p(t) + b
```

For 10 keypoints:

```text
W in R^(20 x 20)
b in R^20
```

This operator can mix every coordinate of every keypoint. It is useful as a
flexible comparison but does not directly test whether one common rigid-motion
rule acts on all points.

### 8.2 Current Shared-Affine Operator

```text
p_i(t+1) = A p_i(t) + b
```

where:

```text
A in R^(2 x 2)
b in R^2
```

The same `A` and `b` are applied independently to every keypoint. The operator
has six trainable parameters and cannot permute or mix keypoint identities.

### 8.3 Optional Inverse Operator

When `lambda_inv > 0` or `lambda_cycle > 0`, a separate inverse operator is
created:

```text
p_hat_t = F_inv(p_t1)
```

For `shared_affine`, the inverse path also has one learned 2-by-2 matrix and
one 2D bias. It is not computed analytically from the forward operator.

## 9. Losses

The implemented total objective is:

```text
L = L_pred
  + lambda_smooth L_smooth
  + lambda_disp L_disp
  + lambda_ent L_ent
  + lambda_act L_act
  + lambda_loc L_loc
  + lambda_inv L_inv
  + lambda_cycle L_cycle
```

### 9.1 Prediction

```text
L_pred = MSE(F(p_t), p_t1)
```

This is the primary learning objective.

### 9.2 Temporal Smoothness

```text
L_smooth = MSE(p_t1, p_t)
```

This discourages large frame-to-frame changes. It can also favor near-static
keypoints, so it is treated cautiously.

### 9.3 Dispersion

For destination-frame keypoints:

```text
L_disp =
  mean_(i != j) exp(-||p_i - p_j||^2 / sigma^2)
```

with:

```text
sigma = 0.1
```

The implementation normalizes by `N(N-1)`. It is evaluated on `p_t1`. Because
the cyclic dataset uses every frame as a destination, all frames receive this
pressure over an epoch.

Important limitation:

`L_disp` is a soft average repulsion. It can be small even when one local pair
remains duplicated. Therefore `lambda_disp > 0` is not evidence that the
observed keypoints are distinct.

### 9.4 Heatmap Entropy

```text
L_ent = mean_i Entropy(softmax(H_i))
```

It is evaluated on destination-frame heatmaps. Minimizing it encourages sharp
spatial distributions.

Low entropy alone does not guarantee:

- that the peak is on the object,
- that different channels peak at different locations,
- that the channel moves,
- or that the peak tracks the same object part.

### 9.5 Action Classification

```text
delta_k = p_t1 - p_t
logits = Linear(delta_k)
L_act = CrossEntropy(logits, action_label)
```

Labels are:

```text
0 = forward +6 deg
1 = backward -6 deg
```

This loss is disabled in the full-360 sweep:

```text
lambda_act = 0
```

The reason is geometric, not merely empirical. On a complete closed orbit,
forward displacement vectors sweep the full tangent field, and a forward
displacement at one phase can match a backward displacement at the opposite
phase. A single linear classifier that sees only `delta_k`, without starting
phase, cannot globally separate the two classes.

Action classification remains meaningful as a separate limited-arc control.

### 9.6 Localization Loss

The optional implementation samples a differentiable foreground mask produced
by corner-color background subtraction:

```text
L_loc = -log M_t(p_t) - log M_t1(p_t1)
```

It is disabled in the current sweep:

```text
lambda_loc = 0
```

True TDW ID masks are used for evaluation. They are not used as training
supervision in the current headline protocol.

### 9.7 Inverse Prediction

```text
L_inv = MSE(F_inv(p_t1), p_t)
```

This trains a separate inverse operator.

### 9.8 One-Step Cycle Consistency

```text
L_cycle =
  0.5 * (
    MSE(F_inv(F(p_t)), p_t)
    + MSE(F(F_inv(p_t1)), p_t1)
  )
```

This is a one-step forward/inverse cycle. It is not the same as the full-orbit
`F^60` diagnostic.

Inverse and cycle losses improve invertibility and orbit closure, but the
completed sweep shows that they do not solve keypoint distinctness or sliding.

## 10. Training Protocol

Fixed settings for all 324 sweep configurations:

```text
object              = engineers_hammer_vray
operator_type       = shared_affine
num_keypoints       = 10
img_size            = 512
frame_skip          = 3
effective_step      = 6 deg
epochs              = 1000
batch_size          = 16
learning_rate       = 1e-4
weight_decay        = 1e-5
seed                = 42
sigma               = 0.1
padding_mode        = reflect
lambda_act          = 0
lambda_loc          = 0
```

Optimization:

- Optimizer: Adam.
- Learning-rate schedule: cosine annealing from `1e-4` to `1e-6`.
- Training loader: shuffled.
- Checkpoint logging interval: 10 epochs.
- `best_model.pt`: lowest same-sequence total "validation" loss.
- `final_model.pt`: epoch 1000.

For evaluation and reproducibility, use `best_model.pt`.

## 11. Completed Hammer Sweep

### 11.1 Grid

```text
lambda_ent    = [0.01, 0.05, 0.1, 0.5]
lambda_disp   = [0.0, 0.05, 0.1]
lambda_inv    = [0.0, 0.1, 0.5]
lambda_cycle  = [0.0, 0.1, 0.5]
lambda_smooth = [0.0, 0.001, 0.01]
lambda_act    = [0.0]
lambda_loc    = [0.0]
```

Total:

```text
4 * 3 * 3 * 3 * 3 = 324 configurations
```

The dense-operator comparison has not yet been run.

### 11.2 Artifact Completeness

The local archive contains:

- 324 run directories.
- 324 task summaries.
- 324 configs.
- 324 best checkpoints.
- 324 ablation folders.
- 324 compositionality folders.
- 324 rollout folders.
- 324 visualization folders.
- 1620 visualization PNGs.
- 322 training histories.
- 322 final checkpoints.

Tasks 251 and 252 are missing `history.json` and `final_model.pt` because their
original jobs stopped after valid best checkpoints were written. Their
evaluation products were regenerated from `best_model.pt`.

## 12. Evaluation Metrics

### 12.1 Localization

TDW masks provide:

- `on_object_pct`: fraction of all keypoint observations inside the object
  mask.
- Per-keypoint on-object rate.
- Mean distance to object mask.
- Border occupancy.
- Out-of-bounds rate.

The current per-channel on-object threshold is:

```text
per_kp_on_object_pct >= 0.5
```

This is a foreground-membership criterion, not semantic landmark accuracy.

### 12.2 Participation

For keypoint `i`:

```text
E_i = mean_t ||p_i(t+1) - p_i(t)||^2
```

A channel is active when:

```text
E_i > 0.1 * max_j E_j
```

Reported:

- `active_kp_frac`.
- `active_count`.
- `top1_energy_frac`.

Activity is not semantic quality. A jittering or sliding point can be active.

### 12.3 Motion/Object Partition

Each channel is assigned to exactly one bucket:

```text
active_on_object
active_off_object
static_on_object
dead_off_object
```

The fractions should sum to one:

```text
motion_object_partition_sum = 1
```

### 12.4 Canonical Object-Relative Stability

For real extracted keypoints:

```text
p_i(t) = K(x_t)_i
theta_t = frame metadata angle
c_i(t) = R(-theta_t) p_i(t)
```

Per-keypoint drift:

```text
mu_i = mean_t c_i(t)
drift_i = sqrt(mean_t ||c_i(t) - mu_i||^2)
```

The sign is locked to `-theta`. It was verified independently by unrotating TDW
masks and measuring mask IoU. `+theta` is retained only as an audit.

The current clean-channel definition requires:

```text
active
AND per_kp_on_object_pct >= 0.5
AND canonical_rms <= 0.20
```

Reported:

- `clean_kp_frac`.
- `sliding_on_object_frac`.
- Mean, median, and maximum canonical RMS.

### 12.5 Operator Geometry

Reported:

- Closest rotation angle.
- Error to analytic `R(+6 deg)` and `R(-6 deg)`.
- Singular values.
- Determinant.
- Spectral radius.
- Orthogonality error.
- Bias norm.
- Inverse condition number when an inverse operator exists.

### 12.6 Compositionality

Two related evaluation paths currently exist:

1. Non-wrapped start-frame evaluation used for `k1` through `k10` summary
   metrics. At `k=1`, `n=177`; at `k=10`, `n=150`.
2. Cyclic modular evaluation over all 180 start frames, including the
   full-orbit `k=60` diagnostic.

For the full orbit:

```text
60 * 6 deg = 360 deg
F^60(p_t) should return to p_t
```

Closed-orbit error is not sufficient by itself. Identity and static keypoints
also close a cycle.

### 12.7 Ablations

Prediction error is recomputed after:

- replacing the learned operator with identity,
- replacing it with a seeded random operator,
- shuffling keypoint identities,
- replacing keypoints with random coordinates.

These tests establish whether the learned operator and keypoint arrangement
carry non-trivial predictive structure.

### 12.8 Observed Keypoint Separation

The post-sweep separation audit extracts heatmap peak locations from frame 88
and normalizes distances by the hammer-mask bounding-box diagonal.

Current exploratory thresholds:

```text
median nearest-neighbor distance >= 0.06 object diagonals
near-duplicate keypoint fraction <= 0.25
spatial bounding-box diagonal >= 0.5 object diagonals
```

The duplicate threshold `0.06` approximately matches the model's dispersion
length scale in object-normalized units.

This is currently a one-pose audit. It must be repeated across multiple poses
before being treated as a robust representation measure.

## 13. Sweep Results

### 13.1 Global Operator and Localization Result

Across all 324 configurations:

- 324/324 learned the positive rotation direction.
- 322/324 learned an angle within 0.5 degrees of the intended 6-degree step.
- Median angle: `6.15 deg`.
- Median on-object localization: `0.802`.
- Active-keypoint fraction ranged from `0.8` to `0.9`.

This is strong evidence that the shared-affine architecture can recover the
controlled transformation law.

It is not evidence that the keypoints form a good coordinate representation.

### 13.2 Original Sweep Ranking

The original registered screening logic used:

Mandatory:

```text
on_object_pct > 0.5
active_kp_frac > 0.3
active_on_object_frac > 0.5
```

Soft:

```text
k10/k1 < 5
identity_ratio > 1
closed_orbit_mse < 0.05
```

Outcome:

```text
full tier = 19
soft tier = 224
none tier = 81
mandatory localization/participation pass = 324
```

Critical post-hoc finding:

All 19 full-tier runs failed the representation audit. They had:

```text
clean_kp_frac = 0
sliding_on_object_frac = 0.9
one dead keypoint
```

The top original task, Task 189, clustered most useful keypoints together:

```text
median nearest-neighbor separation = 0
near-duplicate keypoint fraction = 0.889
clean keypoint fraction = 0
sliding fraction = 0.9
```

Task 189 is therefore a dynamics-metric winner but a representation failure.

### 13.3 Why `k10/k1 < 5` Is Being Retired

The ratio can improve because the denominator becomes worse, not because
long-horizon prediction becomes good.

Task 189 illustrates the problem:

```text
k10/k1 = 4.81
k1 error = 0.00500
```

Task 80 has a worse ratio:

```text
k10/k1 = 13.68
k1 error = 0.00203
```

The ratio favors Task 189 partly because its one-step error is already larger.
It should not remain a primary scientific gate.

The replacement is to report absolute errors at several horizons, such as
`k=1, 5, 10, 20, 30, 60`, together with identity and other baselines.
Thresholds must be calibrated before confirmatory runs rather than chosen after
seeing the outcomes.

### 13.4 Dispersion-Aware Representation Audit

The stricter exploratory representation screen requires:

```text
active_on_object_frac >= 0.8
clean_kp_frac >= 0.3
sliding_on_object_frac <= 0.5
dead_off_object_frac <= 0.1
median_nn_objdiag >= 0.06
near_duplicate_kp_frac <= 0.25
spatial_bbox_diag_objdiag >= 0.5
```

Outcome:

```text
fully qualified = 0
near-qualified = 8
representation-only = 0
failed representation gate = 316
runs with zero near-duplicate keypoints = 0
```

"Near-qualified" means passing all current representation gates and at least
two of the three original dynamics gates. Every near-qualified run fails the
deprecated `k10/k1 < 5` criterion.

Because that ratio is no longer considered a valid primary gate, the
near-qualified label is informative but not a final scientific category.

### 13.5 Current Candidate Configurations

| Task | ent | disp | inv | cycle | smooth | clean | sliding | duplicate frac | k10/k1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.01 | 0.10 | 0.50 | 0.50 | 0.001 | 0.50 | 0.40 | 0.222 | 13.68 |
| 70 | 0.01 | 0.10 | 0.50 | 0.10 | 0.000 | 0.40 | 0.40 | 0.250 | 13.36 |
| 65 | 0.01 | 0.10 | 0.00 | 0.10 | 0.001 | 0.30 | 0.50 | 0.250 | 10.98 |

Interpretation:

- Task 80 is the best current overall balance.
- Task 70 has slightly stronger identity/non-triviality behavior but weaker
  clean-keypoint quality.
- Task 65 has the strongest nearest-neighbor separation and least-bad ratio
  among these candidates, but weaker overall representation/operator scores.

These are screening candidates, not confirmed models.

### 13.6 Task 80 Detailed Result

Run:

```text
phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_pid3289780
```

Configuration:

```text
lambda_ent    = 0.01
lambda_disp   = 0.1
lambda_inv    = 0.5
lambda_cycle  = 0.5
lambda_smooth = 0.001
lambda_act    = 0
lambda_loc    = 0
```

Prediction and composition:

```text
one-step baseline MSE = 0.002029
k10 MSE              = 0.027755
k10/k1               = 13.678
closed-orbit k60 MSE = 0.009349
```

Forward/inverse behavior:

```text
forward one-step MSE = 0.002029
inverse one-step MSE = 0.002181
validation cycle loss = 3.89e-6
inverse condition number = 1.006
```

Operator geometry:

```text
closest rotation angle = 5.866 deg
singular values = [0.993, 1.000]
spectral radius = 0.9965
orthogonality error = 0.0423
```

Localization and representation:

```text
on_object_pct = 0.821
active_kp_frac = 0.9
active_on_object_frac = 0.9
dead_off_object_frac = 0.1
clean_kp_frac = 0.5
sliding_on_object_frac = 0.4
canonical mean RMS = 0.191
median nearest-neighbor distance = 0.080 object diagonals
near-duplicate keypoint fraction = 0.222
spatial coverage = 0.925 object diagonals
```

Ablations, expressed as error relative to the learned baseline:

```text
identity operator = 1.90x
random operator = 84.3x
shuffled keypoints = 105.4x
random keypoints = 256.1x
```

Task 80's positive result:

- The learned state/operator pair is highly non-trivial.
- The operator is close to a well-conditioned rigid rotation.
- Most channels are active and on the object.
- The representation is much better distributed than the original
  dynamics-ranked winners.

Task 80's failures:

- One channel is dead and off-object.
- One near-duplicate pair remains among the nine useful channels.
- Four of ten channels are active and on-object but exceed the canonical-drift
  threshold.
- Only five of ten channels are currently classified as clean.
- Separation was measured at only one pose.
- Inverse and cycle losses assist the operator geometry.
- All results are from one seed on the training sequence.

Task 80 must therefore be described as the provisional leading configuration,
not as a solved keypoint representation.

### 13.7 Unassisted Configurations

The sweep contains 36 configurations with:

```text
lambda_inv = 0
lambda_cycle = 0
```

None passes the current full representation screen.

The strongest unassisted representation candidates still show substantial
sliding or duplicate channels. For example, Task 57 has:

```text
lambda_ent = 0.01
lambda_disp = 0.1
lambda_smooth = 0.01
clean_kp_frac = 0.3
sliding_on_object_frac = 0.6
near_duplicate_kp_frac = 0.222
learned angle = 5.83 deg
```

Thus the clean prediction-driven emergence claim remains unresolved even
though the shared rotation itself emerges.

## 14. Hyperparameter Findings

These are descriptive associations from one object and one seed.

### Entropy

- `lambda_ent=0.01` produced all current representation-screen candidates.
- Higher entropy penalties often produced sharp but clustered or redundant
  channels.
- Median clean fraction was zero for `lambda_ent` values 0.05, 0.1, and 0.5.
- Sharpness is therefore not equivalent to distinctness or stability.

### Dispersion

- Every run passing the current representation screen used
  `lambda_disp=0.1`.
- Dispersion improves broad spatial coverage.
- It does not eliminate local duplicates because the implemented objective is
  an average soft repulsion.

### Inverse and Cycle

- Larger inverse/cycle weights generally improve conditioning,
  orthogonality, and full-orbit closure.
- They do not substantially improve keypoint distinctness.
- They do not substantially solve intermediate-horizon error.
- Strong results with these losses are assisted operator-learning results.

### Smoothness

- `lambda_smooth` has little measurable effect across the tested values.
- It does not solve sliding or duplicate channels.
- Larger values retain the theoretical risk of favoring static keypoints.

### Localization

- On-object localization is relatively stable across the grid, with a median
  around 0.80.
- Localization is no longer the only or main bottleneck.
- The harder problem is obtaining distinct and object-relative stable
  coordinates after the keypoints are already on the object.

## 15. Current Problems and Failure Modes

### 15.1 Representation Degeneracy

The model can learn the correct shared rotation while several channels occupy
the same local region. This means operator accuracy does not identify a unique
or informative coordinate chart.

### 15.2 Object-Relative Sliding

Many channels remain on the foreground but move between object regions after
ground-truth de-rotation. Foreground localization alone therefore overstates
representation quality.

### 15.3 Dead Heatmap Channel

The leading run retains one inactive off-object channel. Low entropy weight
does not guarantee all channels are useful.

### 15.4 Dynamics/Representation Tradeoff

The original dynamics-ranked winners have good `k10/k1` and closure but severe
keypoint collapse. More spatially distributed runs have larger
intermediate-horizon error.

The optimization currently finds a good operator and a good representation in
different regions of the hyperparameter space.

### 15.5 Assisted Versus Emergent Geometry

Inverse and cycle losses encourage well-conditioned, invertible,
rotation-like operators. Their use is scientifically legitimate as an
intervention, but it weakens a claim that the rotation structure arose from
prediction alone.

### 15.6 Inadequate Long-Horizon Gate

`k10/k1` is denominator-sensitive and can reward worse one-step models. It must
be replaced with absolute multi-horizon errors and explicit baseline
comparisons.

### 15.7 No Held-Out Validation

Training, checkpoint selection, and evaluation use the same sequence. Current
metrics establish fitted structure, not generalization.

### 15.8 Single Seed and Single Object

All 324 runs use the hammer and `seed=42`. Hyperparameter comparisons are
descriptive. Seed sensitivity and object dependence are unknown.

### 15.9 One-Pose Duplicate Audit

Observed separation is currently measured at frame 88. A channel pair may be
distinct at that frame and collide elsewhere, or appear duplicated at that
frame but separate elsewhere. Multi-pose aggregation is required.

### 15.10 Pending Baselines and Controls

The following planned comparisons are incomplete:

- Dense-operator sweep.
- Limited-arc action control.
- Multi-seed reruns.
- Multi-object reruns.
- Direct minimum-distance intervention.
- Pairwise-rigidity intervention.
- Proper held-out sequence/phase evaluation.

## 16. Statistical Scope

All current sweep results are descriptive.

### Compositionality Error Bars

The plotted bars are:

```text
mean +/- 1 SEM
SEM = sample std(ddof=1) / sqrt(n)
```

Sample unit:

- A start-frame window from the same sequence.
- `n=177` at `k=1`.
- `n=150` at `k=10`.
- Cyclic full-orbit metrics can use all 180 start phases.

These windows overlap and are strongly correlated. The bars describe
within-sequence phase variability. They are not inferential confidence
intervals and understate uncertainty about rerunning training.

The inferential unit for reproducibility must be an independently trained seed,
with object treated as an additional hierarchical unit when making
multi-object claims.

No population-level hypothesis test is justified by the current sweep.

## 17. Next Experimental Program

### 17.1 Multi-Pose Separation Audit

Before new large training runs:

1. Measure nearest-neighbor separation across multiple evenly spaced poses.
2. Report per-pose and aggregated duplicate fractions.
3. Store the frame set, thresholds, normalization, and aggregation rule in the
   JSON artifact.

Pass criterion:

- No success claim based on a single pose.
- A candidate must satisfy the registered duplicate and coverage thresholds
  across the preselected pose set, not only on average at one favorable frame.

### 17.2 Replace the Composition Gate

Report absolute:

```text
MSE at k = 1, 5, 10, 20, 30, 60
```

Also report:

- identity baseline at matched horizons,
- analytic 6-degree rotation where applicable,
- closed-orbit error,
- area under the full `k=1...60` error curve.

Thresholds must be locked before confirmatory reruns.

### 17.3 Distinctness Intervention

Test a direct minimum-distance or worst-pair penalty rather than relying only
on average dispersion.

The intervention must:

- penalize the closest pair explicitly,
- allow all useful points to remain on the object,
- avoid forcing points into the background merely to increase distance,
- be evaluated using observed locations, not just training loss.

### 17.4 Rigidity Intervention

Test preservation of pairwise distances:

```text
L_rigid =
  mean_(i<j) (
    ||p_i(t)-p_j(t)|| - ||p_i(t+1)-p_j(t+1)||
  )^2
```

Purpose:

- reduce object-relative sliding,
- encourage the keypoint cloud to transform as one rigid structure.

Failure modes:

- collapse also preserves pairwise distances, so dispersion/distinctness must
  remain active,
- static keypoints also satisfy rigidity,
- the loss is well matched to planar roll but not automatically valid for
  projected oblique yaw or pitch.

### 17.5 Clean Versus Assisted Comparison

Rerun matched candidates in two explicitly labeled groups:

```text
clean:
  prediction + entropy + dispersion (+ optional smoothness)

assisted:
  clean objective + inverse and/or cycle
```

The comparison should determine which properties require structural
supervision.

### 17.6 Multi-Seed Confirmation

Rerun Tasks 80, 70, and 65 with at least three independent seeds.

Report across-seed:

- mean and sample standard deviation with `ddof=1`,
- SEM only if clearly labeled,
- individual seed values,
- failure count under the locked semantic criteria.

Three seeds remain a minimal screen, not strong evidence of robustness.
Confidence intervals should only be emphasized after increasing the number of
independent seeds; their method and assumptions must be stated explicitly.

### 17.7 Multi-Object Confirmation

Apply the locked protocol to additional objects with different geometry and
symmetry. Do not tune separate thresholds after seeing each object.

Separate claims:

- same recipe across independently trained objects,
- shared operator across objects,
- transfer to unseen objects.

The current project only supports the first type of experiment, and even that
has not yet been completed for the new roll setup.

### 17.8 Dense-Operator Comparison

Run the matched 324-grid dense sweep or a preregistered reduced comparison.

Question:

> Does the shared-affine constraint improve representation quality, or only
> make the learned transformation easier to interpret?

Compare:

- absolute rollout error,
- localization,
- distinctness,
- sliding,
- ablations,
- operator parameter count.

### 17.9 Limited-Arc Action Control

Use an explicit `-60...+60 deg` arc index and sweep `lambda_act`.

This control should test:

- action direction is decodable on a limited arc,
- the same displacement-only classifier remains near chance on the full orbit,
- closed-orbit metrics are not applied to the arc experiment.

### 17.10 Held-Out Evaluation

Introduce one or more of:

- held-out phase indices,
- held-out contiguous arc,
- separately rendered sequence with changed appearance,
- held-out object.

The split must match the intended claim. A phase holdout tests interpolation;
a new render tests appearance robustness; a new object tests transfer.

## 18. Full-Run Gate for the Next Stage

Before launching a large follow-up sweep, a smoke run must demonstrate:

```text
correct pair index and +6 deg geometry
correct shared-affine code path
correct canonical -theta evaluation
multi-pose separation artifact generated
absolute multi-horizon metrics generated
all metric definitions stored in JSON
no train/validation split ambiguity
```

If any of these fail, the large run should stop.

## 19. Current Conclusions

The strongest defensible conclusions are:

1. A minimal shared 2D affine operator can recover the correct 6-degree
   rotation from keypoints learned jointly from images in this controlled
   hammer sequence.
2. The learned state/operator pair is non-trivial under strong ablations.
3. Most keypoint observations lie on the object.
4. Correct operator geometry, localization, and orbit closure are insufficient
   to establish a useful keypoint representation.
5. Distinctness, dead channels, and object-relative sliding are the current
   bottlenecks.
6. Task 80 is the best screening candidate, but it is assisted and remains
   representationally incomplete.
7. No publication-level generalization or robustness claim is currently
   justified.

## 20. Artifact Map

Primary reports:

```text
/Users/kirubeso.r/Documents/PhD/docs/MOE_2026-06-08.md
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/evaluation_report.md
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/dispersion_aware_report.md
```

Ranking data:

```text
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/all_config_rankings.csv
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/all_config_rankings.json
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/all_config_rankings_dispersion_aware.csv
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/all_config_rankings_dispersion_aware.json
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/dispersion_diagnostics.json
```

Workbooks:

```text
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/Hammer_Full360_Shared_Ranking.xlsx
/Users/kirubeso.r/Documents/PhD/analysis/hammer_full360_evaluation/Hammer_Full360_Shared_Dispersion_Aware_Ranking.xlsx
```

Complete sweep archive:

```text
/Users/kirubeso.r/Documents/PhD/cluster_downloads/hammer_full360_shared_complete
```

Task 80:

```text
/Users/kirubeso.r/Documents/PhD/cluster_downloads/hammer_full360_shared_complete/keypoint_net/runs_hammer_full360_shared/phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_pid3289780
```

Key code:

```text
/Users/kirubeso.r/Documents/PhD/keypoint_net/model.py
/Users/kirubeso.r/Documents/PhD/keypoint_net/dataset.py
/Users/kirubeso.r/Documents/PhD/keypoint_net/train.py
/Users/kirubeso.r/Documents/PhD/keypoint_net/sweep.py
/Users/kirubeso.r/Documents/PhD/keypoint_net/eval_compositionality.py
/Users/kirubeso.r/Documents/PhD/keypoint_net/eval_rollout_viz.py
/Users/kirubeso.r/Documents/PhD/keypoint_net/ablations.py
```

Dataset geometry smoke:

```text
/Users/kirubeso.r/Documents/PhD/tmp/smoke_geometry_20260601/geometry_iou.json
```

## 21. Questions for External Expert Review

1. Is prediction under a shared-affine operator a sufficiently informative
   discovery pressure, or is representational non-identifiability unavoidable
   without an additional principle?
2. Is direct minimum-distance regularization scientifically defensible, or
   does it impose the desired landmark structure too explicitly?
3. Is pairwise rigidity the right intervention for sliding in planar roll, and
   how should it be adapted before moving to projected 3D transformations?
4. What is the strongest held-out split that still tests the intended
   coordinate-emergence claim rather than a different recognition problem?
5. Should the primary evidence be agreement with the analytic 6-degree
   rotation, predictive advantage over baselines, or recovery of group
   structure across multiple transformations?
6. What minimum seed/object replication would make the result scientifically
   persuasive?
