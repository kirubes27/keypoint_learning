# Roll 64-versus-128 head-package training semantic lock v1 (draft)

Date: 2026-07-29

Status: independently reviewed implementation lock; no GPU smoke or full
training authority

Parent decision:

- `docs/decisions/2026-07-26/DECISION_SYNTHESIS_v2_2026-07-26.md`,
  amendment v2.6
- `docs/decisions/2026-07-29/DECISION_SYNTHESIS_v2_8_AMENDMENT_2026-07-29.md`
- `docs/decisions/2026-07-26/REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md`

## 1. Plain-language purpose

This experiment asks one narrow question: if the final keypoint heatmaps are
made finer, do freshly trained hammer models produce keypoints that are more
stable and less duplicated without making the learned roll operator worse?

It does not change or regenerate the images. It does not train yaw, pitch,
scale, or translation. It does not test the descriptor loss yet. It does not
reuse the saved Task 55 or Task 80 weights. Those saved checkpoints remain
replay fixtures only.

The two alternatives are accurately called **head packages**, not a pure
finite-grid intervention:

- `64`: the legacy `/8` encoder feature map followed by the existing `1x1`
  heatmap head;
- `128`: the same `/8` encoder followed by bilinear feature upsampling, a
  `3x3` convolution, batch normalization and ReLU, then the `1x1` heatmap
  head.

The 128 package therefore has a finer spatial grid and additional learned head
parameters. Any result is evidence about the complete 64-versus-128 head
package, not about spatial quantization alone.

For 10 keypoints and 32 base channels, the head-package parameter counts are
1,290 for the 64 path and 74,570 for the 128 path: the 128 path adds 73,280
learned parameters. The unchanged entropy coefficient is also applied to
softmax distributions over 4,096 versus 16,384 cells. Because the implemented
entropy is not divided by `log(HW)`, the same coefficient does not impose an
identical maximum-scale penalty. Both facts are part of the package comparison
and prohibit a grid-resolution-only causal claim.

## 2. Decision lock

**Decision.** Choose one heatmap-head package, `64` or `128`, for the later
descriptor experiment.

**Current claim.** The saved Task 55 and Task 80 fixtures recover the intended
approximately `+6 degree` shared roll operator but retain representation
defects. Under the production v3 evaluator, Task 55 has one flat-dead channel
and two persistent duplicate pairs; Task 80 has one flat-dead channel and one
persistent duplicate pair. Neither is globally collapsed. These are
single-seed historical diagnostics, not fresh-run selection evidence.

**Known evidence.**

- The roll corpus is the existing six-object, 180-frame
  `_tdw_world_z_roll_base_panel_512_v2` corpus.
- The frozen primary transform is cyclic world-Z roll, forward direction,
  stride 3 frames, `+6 degrees`.
- The verified hammer development split has 150 training frames producing 147
  train pairs, 24 validation frames producing 21 validation pairs, and 6 guard
  frames. Training and validation endpoints are disjoint.
- The split manifest and independent report record `gate_pass=true`,
  `structurally_valid=true`, live-corpus validation and byte-identical
  regeneration.
- The planted representation-oracle suite passed. Task 20 v3 reproduced the
  negative-control diagnosis and all 245 historical records. Tasks 55 and 80
  v3 passed their expected non-global-collapse classification and all 245
  historical records.
- The fixed spatial-softmax expectation remains the coordinate readout.

**Critical unknown.** Whether the 128 head package improves fresh-run
representation stability across seeds, rather than merely changing capacity
or optimization, while preserving the learned operator and role-scoped
validation dynamics.

**Main failure modes.**

1. a run changes something besides the declared task recipe and head package;
2. validation aliases training data or the full 180-frame diagnostic selects a
   checkpoint;
3. a finer heatmap looks visually smoother but creates dead, off-object, or
   duplicate channels;
4. one unusually favourable seed determines the choice;
5. a training-success exit code is mistaken for a scientific pass;
6. the 128 package is described as a grid-only causal intervention even though
   it adds learned head layers;
7. the same entropy coefficient is incorrectly described as a
   resolution-independent loss-strength match.

**Constraints.**

- Existing images, masks, metadata and pair indices are read-only.
- Development object only: `engineers_hammer_vray`.
- Fresh initialization for every cell; no weight sharing or warm start.
- No action head or action loss: `lambda_act=0`.
- No raw-RGB, patch, descriptor, mask-localization or separation intervention.
- No full-corpus metric, test object or saved Task 55/80 result may select the
  package.
- Frames, channels, horizons and overlapping rollout starts are correlated
  descriptive units. The three seeds are the replication units. No SEM,
  confidence interval or population-level claim is authorized.
- No overall scalar score may trade operator quality against collapse.

**Next gate.** Implement the fresh-checkpoint evaluator authority in Section
8, Gate 2, while preserving the three immutable fixture paths. The ordering in
Section 8 governs. Only after Gate 2 passes may a committed, clean-source,
CPU-only semantic/training/evaluation smoke prove the exact matrix below can be
constructed, the train and validation endpoints remain disjoint, one optimizer
step changes weights, the selected checkpoint can be reconstructed at both
resolutions, and the production evaluator emits every required validation
axis. Any mismatch blocks the one-job CUDA smoke and the full matrix.

## 3. Frozen development matrix

The full development matrix has exactly 12 independent runs:

`2 task recipes x 2 head packages x 3 seeds`.

- task recipes: `task55_clean`, `task80_assisted`
- head packages: `64`, `128`
- seeds: `42`, `43`, `44`

Every cell uses:

- object: `engineers_hammer_vray`
- indexed mode: `development`
- train artifact:
  `representation_oracle_splits/pairs/roll__world_z__forward__train.json`
- validation artifact:
  `representation_oracle_splits/pairs/roll__world_z__forward__validation.json`
- dataset binding:
  `acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93`
- input resize: `512 x 512`
- center crop: none
- keypoints: 10
- base channels: 32
- temperature: 1.0
- padding: reflect
- operator: shared affine
- epochs: 1000
- batch size: 16
- Adam learning rate: `1e-4`
- weight decay: `1e-5`
- cosine schedule with `T_max=1000`, `eta_min=1e-6`
- dispersion length scale: `sigma=0.1`
- validation evaluation at epoch 1 and every 10 epochs thereafter
- authoritative checkpoint: lowest total validation loss among those recorded
  epochs, with no test loader

The pair artifacts are six-object files. The mandatory `--object` filter and
role lock reduce them to the stated hammer counts of 147 train pairs and 21
validation pairs. `--img_size` defaults to 256 and `--operator_type` defaults
to `dense` in the legacy CLI; the dedicated manifest path must pin them
explicitly to 512 and `shared_affine`.

The task-specific loss weights are:

| recipe | entropy | dispersion | inverse | cycle | smooth | action | localization |
|---|---:|---:|---:|---:|---:|---:|---:|
| `task55_clean` | 0.01 | 0.1 | 0 | 0 | 0 | 0 | 0 |
| `task80_assisted` | 0.01 | 0.1 | 0.5 | 0.5 | 0.001 | 0 | 0 |

`task55_clean` is the mask-free clean headline baseline.
`task80_assisted` is a separately labelled assisted baseline because inverse
and cycle losses are active.

## 4. Run and checkpoint binding

Before an output directory or CUDA device is created, the dedicated run path
must fail closed unless it can prove:

1. the current Git commit is a full committed source state and the worktree has
   no tracked modifications;
2. the exact experiment-manifest file and content hashes;
3. the split-manifest, split-verifier, corpus-inventory and selected pair-file
   hashes;
4. the live dataset basename and binding;
5. the exact object role, recipe, head package, seed and all factors in
   Section 3;
6. no legacy train-as-validation or whole-directory auto-evaluation path is
   reachable;
7. a unique output cell does not already contain an attempted run.

Each run records:

- source commit and relevant source-file hashes;
- experiment-manifest file/content hashes and cell ID;
- train/validation pair-file and dataset-binding hashes;
- full command and resolved arguments;
- Python, PyTorch, torchvision, NumPy and CUDA versions;
- GPU name, driver-visible CUDA version and determinism settings;
- seed, data-loader worker count and checkpoint policy;
- checkpoint epoch, file hash and embedded config;
- Slurm job ID and job-script hash on the cluster;
- completion state and failure reason.

The full matrix must use one reviewed source commit and one reviewed Slurm
script. There is no per-cell editing, resumption with changed arguments,
adaptive epoch extension, seed replacement or failed-cell dropping.

The unrelated user-owned untracked file
`docs/decisions/2026-07-26/representation_oracle_calibration/NUMERIC_CALIBRATION.json`
is outside this programme's source manifest. It must never be read, staged,
edited, moved or deleted by this work. A cluster run uses a clean clone where
that file is absent. Local source-integrity checks require no tracked
modifications and explicitly reject any untracked file that overlaps a bound
source, specification, manifest, runfile or output path.

## 5. Evaluation boundary

Checkpoint selection uses only the 21 hammer validation pairs and the frozen
total validation loss. After the authoritative checkpoint is frozen, a
separate read-only evaluator may:

1. evaluate the 24-frame validation block for scientific selection axes;
2. evaluate all 180 frames for required historical/full-orbit diagnostics.

The full-orbit pass is explicitly labelled mixed train/validation support and
cannot change the checkpoint, package, recipe, duration or seed set.

The role-scoped scientific axes use the same development validation block
whose total loss selects `best_model.pt`. Their absolute values are therefore
selection-optimistic and are not confirmation estimates. Their permitted use
is the matched within-development comparison; later fresh-object fixed-epoch
train/test runs provide independent confirmation.

The production representation evaluator must receive the actual logits and
fixed spatial-softmax coordinates from the selected checkpoint, the masks
processed with the same crop/resize convention, the exact shared affine
operator, and the bound roll geometry. It must report:

1. proper-rotation angle and absolute error from `+6 degrees`;
2. role-scoped validation identity-normalized rollout AUC at `k=1..7`;
3. continuous canonical drift;
4. all-channel and eligible-channel trajectory separation;
5. persistent and recurrent duplicate-pair identities/counts;
6. active/on-object channel identities/counts;
7. flat-dead channels;
8. coordinate- and heatmap-mode switching;
9. the mandatory full-corpus `k=1..59` AUC and `k=60` closure as
   diagnostic-only fields;
10. individual seed records with named descriptive sample units.

Missing logits, masks, split bindings, operator weights, provenance, or any
required metric is a failed cell, not a warning.

## 6. Package decision rule

The comparison is paired by `(recipe, seed)`. No Task 55/Task 80 historical
fixture value enters the decision.

First, a cell is ineligible if any of the following occurs:

- evaluator critical failure;
- wrong roll sign, reflection/improper operator, or unavailable angle;
- structural negative-control collapse;
- fewer than two eligible active/on-object channels;
- void role-scoped validation rollout AUC;
- missing or contradictory provenance.

For each eligible paired comparison, classify the 128 package relative to 64
on the five separate axes:

- operator angle error: lower is better;
- role-scoped validation AUC: lower is better;
- median continuous canonical drift: lower is better;
- persistent duplicate count, then recurrent duplicate count: lexicographically
  lower is better;
- active/on-object eligible count: higher is better.

The 128 package is adopted only when all of these are true:

1. all 12 cells are eligible;
2. in each recipe separately, 128 has lower role-scoped validation AUC in all
   three paired seeds;
3. in each recipe separately, 128 has lower median canonical drift in all
   three paired seeds;
4. in each recipe separately, 128 has no more persistent duplicate pairs and
   no fewer active/on-object channels in at least two of three paired seeds;
5. in each recipe separately, 128 has no worse operator-angle error in at
   least two of three paired seeds;
6. neither recipe has a categorical guardrail failure in any seed.

If exactly two of three seeds support either required 3/3 condition in item 2
or 3, the result is provisional and seeds 45 and 46 run under the identical
frozen matrix for both resolutions of that recipe. After extension, adoption
requires at least four of five paired wins on each extended condition and the
unchanged guardrails. The other recipe is extended too if one common package
is still the decision.

Only the required AUC and drift conditions in items 2 and 3 trigger the
five-seed extension. The categorical and count guardrails in items 4 to 6 do
not create an extension by themselves; they retain 64 when the adoption rule
is not met.

Any other result retains the simpler 64 package. This conservative fallback is
not a claim that 64 is scientifically superior; it means the added 128 head
package did not demonstrate a sufficiently consistent benefit under the frozen
multi-axis rule. Recipe discordance is reported explicitly and also retains
one common 64 package for the later descriptor comparison.

Exact equality is equality. No post-outcome epsilon or practical-equivalence
margin may be added. Because this rule is intentionally descriptive and
conservative, all individual paired values must accompany the decision.

## 7. Grid-jitter interpretation

The estimator oracle already proves the 64 and 128 finite-grid coordinate
paths are numerically correct. Fresh-run canonical drift and heatmap-mode
switching determine whether the finer package changes observed stability.

Do not claim that a lower drift value is entirely caused by smaller cells:
the 128 package adds a learned convolution, batch normalization and ReLU.
The supported statement is only that the complete 128 head package improved,
failed to improve, or gave mixed evidence on the frozen metrics.

## 8. Execution gates

1. **Specification review:** a substantive read-only Fable review of this
   exact draft and the relevant code; unresolved P0/P1 blocks implementation.
2. **Fresh-checkpoint evaluator authority:** extend the secure evaluator
   boundary from the three immutable replay fixtures to hash-bound fresh-run
   manifests and checkpoints without weakening the fixture path. The exact
   candidate requires its own source manifest, mutation tests and independent
   review. Until this passes, no fresh checkpoint is scientifically
   evaluable.
3. **CPU semantic tests:** manifest mutation tests, data-role tests,
   64/128 reconstruction, deterministic dry-run construction, and evaluator
   bundle tests all pass.
4. **CPU training smoke:** each recipe/head combination performs a bounded
   forward/backward/update on disposable data and produces a reconstructable
   checkpoint. At 512 input, both heads must emit the exact 64x64 and 128x128
   heatmap shapes, their encoder state keys must match, and the head parameter
   delta must be exactly 73,280. This is implementation evidence only.
5. **Cluster preflight:** exact reviewed commit and script exist on
   Lichtenberg; dataset and environment hashes match; no training runs are
   active for this matrix.
6. **One-job CUDA smoke:** one sequential job exercises both head packages
   under a short `task55_clean`, seed-42 smoke and proves CUDA, data loading,
   update, checkpoint reconstruction, evaluation and provenance. Its
   scientific output is smoke-only and cannot enter the matrix.
7. **Full matrix:** submit exactly the 12 frozen cells with low concurrency.

Any critical semantic mismatch, non-substantive Fable response, CPU failure,
CUDA absence, dataset/source mismatch, missing evaluation axis, or
untraceable Slurm artifact stops before the next gate.

Here, reconstructable means that the recorded architecture and embedded config
instantiate the intended model and reload the exact saved weights. It does not
mean that a second training run must reproduce checkpoint bytes. The dedicated
path must set and record deterministic-algorithm, cuDNN deterministic and
cuDNN benchmark settings; if an intended CUDA operation cannot satisfy that
policy, the CUDA smoke stops for an explicit amendment rather than silently
relaxing determinism.
