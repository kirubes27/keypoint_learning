# Decision 2.3 diagnostic-head specification v1 — 2026-07-26

Status: frozen for implementation review. Kirubes explicitly authorized the
existing public GitHub repository and branch on 2026-07-26, superseding the
earlier private-remote prerequisite. No cluster launch is allowed until the
implementation, D0 evidence, exact Slurm runfiles, and this amendment pass a
read-only Fable 5 high review.

## Decision lock

**Decision.** Determine whether bypassing the fixed spatial-softmax expectation
restores supervised material-coordinate capability strongly enough to justify
redesigning the coordinate head.

**Current claim.** Gate 0 demoted local/windowed expectation because only
6/150 high-error pairs (4.0%) had the correct dominant mode. A learned
coordinate decoder may still succeed if target information is present in the
raw spatial logits but is lost or poorly conditioned by the current
normalization/readout path.

**Known evidence.**

- Gate 0 is complete and is not rerun by this specification.
- The exact 60/60/60 representative protocol has only one current fixed-grid
  baseline seed (41), so a matched three-seed baseline is required.
- Dense heatmap supervision learned every target on the earlier tiny control,
  while coordinate-through-expectation training showed weak gradients and
  target-dependent failures.
- No Decision 2.3 head or result currently exists.

**Critical unknown.** Does a small, channel-symmetric learned spatial decoder
from raw logits pass when a capacity-matched decoder after softmax and the
fixed expectation do not?

**Main failure mode.** An overpowered or channel-specific direct head can
memorize one correlated orbit and falsely attribute its advantage to the
softmax path.

**Constraints.** One object; the existing full-circle world-Z roll dataset;
no arc60/yaw/pitch data; coordinate MSE only; same encoder, heatmap head, split,
augmentation, optimizer, stopping rule, and checkpoint selector; test remains
unread until all initial arm/seed checkpoints and the finalizer are frozen.

**Next gate.** Unit and semantic tests plus one Slurm smoke covering all three
arms. Train/probe may enumerate the committed split metadata but may open only
train/validation image and mask contents; finalization may open only test image
and mask contents and reconstructs targets from each frozen run config. Any
semantic mismatch, forbidden frame-content read, non-finite value, zero
upstream gradient caused by wiring, provenance mismatch, or inability to
reproduce the fixed baseline stops the full launch.

## Bound data and training recipe

- dataset: `_tdw_world_z_roll_base_panel_512_v2`
- transformation: absolute world-Z in-plane roll, 180 frames, 2 degrees/frame
- object: `engineers_hammer_vray`
- split: train `0,3,...,177`; validation `1,4,...,178`; test
  `2,5,...,179`
- split SHA-256:
  `49f9d2a34c352d3ebb84809ec36e0a46572b0cde6b7a6d357f317dc44e3da486`
- initial optimization seeds: `42, 43, 44`
- input and target construction: exactly the existing representative control
- training augmentation: rotation in `[-5,+5]` degrees and translation in
  `[-8,+8]` pixels
- loss: coordinate MSE only
- optimizer: Adam, learning rate `1e-4`, weight decay `1e-5`, batch size 16
- stopping: minimum 1000 epochs, maximum 3000, evaluate every 25, validation
  plateau patience 400, relative improvement 1%
- selection: minimum validation
  `max(unaugmented, fixed-augmented median-of-channel median error)`
- normalized coordinate range: `[-1,1]`; cell64 error remains Euclidean
  normalized-coordinate error divided by `2/64`

No loss, target, data, encoder, heatmap-head, optimizer, augmentation, budget,
or checkpoint-selection difference is permitted across arms.

## Matched arms

All arms use the existing four-block encoder and shared `1x1` convolution that
produces ten `64x64` raw logit maps.

### Arm A — centered-raw-logit shared linear decoder

For each channel independently, subtract that map's spatial mean, flatten the
4096 values, and apply **one shared** `Linear(4096, 2)` to every channel. The
same weights are reused across all ten channels; there is no cross-channel
mixing.

- added parameters: `2*4096 + 2 = 8,194`
- weight initialization: row 0 is the normalized x grid divided by
  `sqrt(4096)=64`; row 1 is the normalized y grid divided by 64
- bias initialization: zero
- range constraint: none
- training clamp: none

The nonzero spatial initialization preserves upstream gradients on the first
step. Spatial centering removes the additive-logit/heatmap-bias gauge. The
shared decoder preserves channel-permutation symmetry and prevents
channel-specific positional shortcuts.

### Arm B — learned post-softmax shared linear decoder

Apply temperature-1 spatial softmax to each logit map, flatten the probability
map, and apply an identically sized shared `Linear(4096, 2)`.

- added parameters: 8,194
- initial weights: exact normalized x/y coordinate grids
- initial bias: zero

At initialization, Arm B must equal Arm C to absolute tolerance `1e-6`.
Unlike Arm C, its shared spatial decoder is trainable. Arm A versus B tests
whether bypassing softmax conditioning matters; B versus C tests whether a
learned spatial decoder matters.

### Arm C — fixed current spatial expectation

Use the production `spatial_softmax` implementation unchanged, with
temperature 1. It has no added decoder parameters. This matched three-seed
baseline is mandatory because only seed 41 exists under the representative
protocol.

## Evaluation correction for unbounded heads

Arm A and Arm B outputs are not clamped. Evaluation records the fraction of
coordinates inside `[-1,1]^2`. An out-of-range coordinate is always off-mask.
Clipping is permitted only after the in-range flag is computed and only to
make mask-array indexing safe. This corrects the existing evaluator's
otherwise hidden edge-clipping behavior while remaining numerically identical
for in-range baseline outputs.

## Gradient-health audit

Do not reuse `coordinate_path_probe`, whose analytic function reconstructs the
fixed softmax-expectation Jacobian regardless of the active readout.

Use autograd through each arm's actual forward path on the same fixed,
unaugmented validation batch. At initialization, every evaluation epoch, and
the selected checkpoint, record:

- per-channel and pooled `||dL/d(logits)||_2`;
- first encoder-convolution, final encoder-convolution, heatmap-head, and
  learned-decoder parameter-gradient norms;
- exact zero-gradient counts;
- maximum probability and effective support as descriptive heatmap-shape
  context;
- Arm A/B/C coordinate error on that fixed batch.
- Arm A/B learned-decoder weight and bias norms.

Gradient values are continuous descriptive diagnostics. No post-hoc
"dead-gradient" threshold or gradient-ratio gate is introduced.

The audit is observational and must not train on validation data. It uses
`torch.autograd.grad` rather than `loss.backward()`, never populates or changes
parameter `.grad` buffers, runs with BatchNorm in evaluation mode, restores
the model's prior train/eval mode, and leaves parameters, existing parameter
gradients, optimizer state, CPU/CUDA RNG state, and data-loader generator state
bitwise unchanged.

## Frozen-probe scope

After unit tests and before the end-to-end launch, a validation-only
descriptive probe may freeze the seed-41 Gate 0 encoder plus heatmap head and
fit only the shared Arm A and Arm B decoders on train frames. It uses seeds
42–44, never reads test, and cannot block the end-to-end experiment.

- Probe A pass with Probe B failure strengthens evidence that target
  information already exists in raw logits but is poorly conditioned by
  softmax.
- Probe failure is not evidence that an end-to-end direct head cannot learn a
  different representation.

## Gate sequence

### D0 — unit and semantic tests

Required:

1. Arm C reproduces the production spatial expectation exactly.
2. Arm B equals Arm C at initialization to `atol=1e-6`, `rtol=0`.
   This check runs in float64; a separate production-float32 parity check uses
   `atol=1e-5`, `rtol=0`.
3. Arm A's initialized weight rows equal the production x/y grid rows divided
   by 64, including flatten order and y-axis direction.
4. Adding a per-map constant to Arm A logits does not change its output.
5. Permuting heatmap channels permutes Arm A/B outputs identically.
6. Perturbing one channel cannot change another channel's coordinates.
7. Arm A/B decoder parameter counts are exactly 8,194.
8. Actual Arm A/B/C output-to-logit autograd paths are tested; Arm A has
   nonzero logit, heatmap-head, final-encoder, and first-encoder gradients at
   initialization.
9. Out-of-range coordinates count as off-mask, remain unclipped for coordinate
   error, and are reported.
10. Validation augmentation is deterministic.
11. Training/probe modes cannot instantiate or iterate a test loader.
12. Split, dataset basename, object, frame count, axis, seed, and hashes fail
    closed on mismatch.
13. Checkpoint save/restore reproduces the next two optimizer steps.
14. A training step immediately after the validation gradient audit is
    bitwise identical to the same step without the audit, including model,
    optimizer, gradients, RNG, and loader-generator state.

### D1 — implementation smoke

One Slurm job runs Arm A, B, and C sequentially for seed 42. The frozen budget
is exactly two epochs per arm, validation and gradient audit at epochs 0, 1,
and 2, batch size 16, Adam `lr=1e-4`, weight decay `1e-5`, and no early exit.
The Slurm request is exactly one GPU, one task, eight CPUs,
`--mem-per-cpu=5000`, and 30 minutes; no account or partition is specified.
Maximum arm concurrency inside the job is one. It must:

- record the exact Git commit/config hash and CUDA device;
- complete forward, backward, optimizer, validation, checkpoint, restore, and
  gradient-audit paths for every arm;
- produce finite losses/metrics and nonzero expected upstream gradients;
- assert that no test frame was loaded;
- leave the source Gate 0 checkpoint and artifacts unchanged.

This is a wiring/semantic smoke, not a scientific result. The historical
four-frame tiny task may be reported descriptively, but it is not a stop gate
because representative evidence already showed that its catastrophic basins
do not faithfully characterize the 60-frame regime.

Smoke runs use a separate `smoke/` output namespace. Neither training nor
finalization may resolve a smoke checkpoint as a scientific run.

### D2 — matched representative runs

After D0 and D1 pass, train Arm A/B/C for seeds 42/43/44: nine matched runs.
No arm is promoted, dropped, or tuned from interim results.

The frozen D2 Slurm request is an array `0-8%2`: at most two concurrent tasks,
each with one GPU, one task, eight CPUs, `--mem-per-cpu=5000`, and a one-hour
walltime. No account or partition is specified; Lichtenberg's submit plugin
assigns project `10003029`. Every task must run from one clean independent
clone of `agent/preoperator-gates-20260726`, bind the exact expected commit via
the submission environment, and write outside the Git checkout.

### D3 — one frozen test finalization

Freeze all nine selected checkpoints, configs, analysis code, and their hashes
before constructing the test loader. Finalization evaluates all arms and seeds
in one command. It refuses any existing test artifact for the same
arm-and-seed target and never overwrites an aggregate.

Per seed, both unaugmented and fixed-augmented test conditions must satisfy:

- median-of-channel median error `<= 0.50 cell64`;
- pooled p90 error `<= 1.50 cell64`;
- on-mask fraction `>= 0.95`;
- in-range coordinate fraction is reported.

The optimization seed is the replication unit.

- 3/3 passing: arm passes.
- 2/3 passing: provisional only; freeze the code and add seeds 45 and 46
  before any downstream claim.
- 0–1/3 passing: arm fails.

If a provisional arm requires seeds 45/46, neither architecture nor
hyperparameters may change. The two added checkpoints, configs, implementation,
and finalizer are frozen and hashed before their test loader is constructed.
The extension finalizer is authorized only when the immutable initial
aggregate records exactly 2/3 for that arm; it refuses an existing test
artifact for seed 45 or 46, leaves the initial aggregate immutable, and writes
a separately named extension plus combined five-seed report.

The five-seed rule is frozen now:

- both added seeds pass, giving 4/5 total: the provisional arm becomes pass;
- either added seed fails, giving at most 3/5 total: the arm fails.

Every arm/seed checkpoint is evaluated on test exactly once. The initial and
extension reports are never used to tune architecture, hyperparameters,
stopping, or selection.

## Interpretation matrix

- **A passes; B and C fail:** bypassing softmax conditioning is supported as
  the operative head-level intervention. Proceed to a separate, versioned
  spatial-head redesign; direct regression remains diagnostic only.
- **A and B pass; C fails:** a learned spatial decoder is supported, but
  softmax itself is not isolated as the cause.
- **A, B, and C pass:** the representative failure is dominated by
  optimization-seed variability or the fixed seed-41 trajectory; no head
  redesign is justified from this panel.
- **A and B fail; C fails:** these logit-level readout changes are
  insufficient. This does not prove the encoder is incapable. A broader
  compact direct-from-feature observability control requires a new spec and
  Fable review before execution.
- **A fails; B passes:** raw-logit scaling/parameterization is suspect;
  do not claim an upstream representation failure.
- **C passes while either learned arm fails:** the current instrument is
  capable under matched replication; the failed learned arm is not a repair.
- **Any 2/3 arm:** no downstream interpretation until the two-seed extension
  is complete.

Gradient-health evidence controls the strength of the mechanism wording; it
does not override localization outcomes.

## Downstream-benefit boundary

A supervised diagnostic pass is not downstream benefit and cannot establish
representation identifiability. No operator run is authorized by this spec.
Before such a claim, a separate versioned document must freeze a no-coordinate-
label downstream objective, matched arm/budget/seed policy, an improvement
margin, and degeneracy guards for collapse and off-object points.

## Required artifacts

- this specification and its decision-synthesis changelog;
- Fable's raw independent design and blocker review;
- implementation and unit tests;
- exact dataset/split/checkpoint/code hashes;
- hashes of the imported model, data, augmentation, checkpoint, and evaluation
  modules, plus the exact Git commit and clean-worktree assertion;
- D1 smoke report and Slurm job ID;
- per-arm/per-seed config, checkpoint, validation history, gradient audit, and
  parameter-count record;
- frozen finalizer hash and one test-finalization report;
- a plain-language decision record with the one-object/one-orbit limitation.

No error bars are required. If any plot later adds a band, its artifact must
state the statistic, exact computation, seed-level sample unit and `n`, and
that it is descriptive rather than inferential.

## v1.1 execution amendment — 2026-07-26

1. Public-branch publication is authorized by the owner; the obsolete
   private-remote prerequisite is removed.
2. D1 and D2 budgets, concurrency, walltime, CPU, GPU, and memory requests are
   frozen above.
3. "Test untouched" now means test image/mask contents are not opened by
   lock/train/probe, not merely that a test `DataLoader` is absent.
4. Probe A/B is bound to seed-41 checkpoint SHA-256
   `d4777e3abfed3d81698ab07edf0563484b49ed314130aa8e925a0de2e1188c3d`
   and sibling config SHA-256
   `2df6bfd4d0d8487e6f74fb3e63b0eeaa4c2e5604c2ba845c1d7bd56253b9e750`.
5. Test finalization verifies all nine configs, checkpoints, histories,
   gradient audits, imported-source hashes, and the absence of either test
   artifact before constructing test data. Extension seeds 45/46 require the
   immutable initial report to record exactly 2/3 for that arm.

## v1.2 pre-D1 blocker amendment — 2026-07-26

1. Full runtime and GPU identity remain recorded in every config. Cross-job
   matching is required only for the Python/Torch/NumPy/Pillow/CUDA/cuDNN
   software versions and deterministic-algorithm flags; D2 tasks, resumes,
   extensions, and finalization are not required to land on identical GPU
   hardware.
2. "Evaluated exactly once" means one committed, immutable test-result pair
   per checkpoint. If a finalizer is interrupted while a matching claim has
   neither test artifact, the attempt exposed no result and may resume. If
   both immutable artifacts exist, they are hash- and identity-validated and
   committed without reevaluation. Exactly one artifact, an identity
   mismatch, or any inconsistent ledger state fails closed. No architecture,
   hyperparameter, or checkpoint choice may change between attempts.
3. D1 records full validation metrics at epoch 0 as a separate artifact, in
   addition to the epoch-0 gradient audit and epoch-1/2 history.
4. Every D2 and extension array task uses its own clean clone at the bound
   commit and writes outputs outside that clone. A task resumes only from its
   matching immutable config and `last_checkpoint.pt`.
5. The SHA-256 of every D1, D2, finalization, extension, and extension-
   finalization Slurm runfile is bound into the prelaunch lock, every run
   config, the D1 report, finalization plans, and aggregate reports.
