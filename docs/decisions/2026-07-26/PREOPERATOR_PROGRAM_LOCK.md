# Pre-operator programme lock — 2026-07-26

Status: execution authority for branch `agent/preoperator-gates-20260726`.
Owner: Kirubes.

Gate 0 status: **complete; windowing branch closed**. The canonical result is
6/150 (4.0%) correct-dominant-mode high-error pairs versus a frozen 50%
retention threshold. The next scientific branch is the Decision 2.3
diagnostic head, followed by re-planning. Gate 0 must not be rerun or bypassed.

## Decision

Determine whether the coordinate instrument is accurate enough, and whether
each proposed transformation is representable enough, to make later operator
training scientifically interpretable.

No yaw, pitch, scale, or translation operator training is authorized by this
lock. This programme closes the gates that must precede those runs.

## Current claim

The existing 180-frame full-circle world-Z roll experiment shows that a simple
shared affine operator can be recovered even when the learned coordinates are
not all distinct, active, or materially attached. It does not yet establish
that the current coordinate readout can realize requested material points
reliably.

## Canonical current dataset

- dataset: `_tdw_world_z_roll_base_panel_512_v2`
- semantics: absolute TDW world-Z roll about the object centroid
- frames: 180 per object, `0, 2, ..., 358` degrees
- cyclic pair used for the +6-degree control: `pairs_skip3_cyclic.json`
- supervised instrument object: `engineers_hammer_vray`
- split: `split_phase_mod6.json`, 60 train / 60 validation / 60 test frames
- representative checkpoint: seed 41, standard 64x64 heatmap head

The yaw/pitch arc60, scale, and translation datasets are separate future
stimuli. They must never enter Gate 0 or the matched readout gate.

## Must be true

1. Every current-cycle artifact asserts the dataset basename, split hash,
   object, seed, frame count, transformation axis, and cyclic status.
2. Gate 0 replays the frozen seed-41 representative checkpoint without an
   optimizer step or checkpoint mutation.
3. Any trained readout comparison changes only the readout; the CNN, targets,
   split, optimizer, augmentation, budget, and loss remain matched.
4. Validation selects checkpoints. The test split is untouched until all
   preregistered seed runs are frozen.
5. Seed is the optimization-replication unit. Frame/channel summaries are
   descriptive because they belong to one correlated orbit.
6. Full GPU runs occur only after local tests and a one-job Slurm smoke pass.
7. Later yaw/pitch interpretation is conditional on a measured shared-affine
   geometry floor.

## Must not happen

1. Do not use any arc60, scale, or translation frame in the current instrument
   repair cycle.
2. Do not combine a readout change with a loss or operator change.
3. Do not use a model review as evidence; resolve disagreements against code,
   data, geometry, or a bounded test.
4. Do not pool directions, strides, forward/reverse pairs, or transformation
   families under one operator.
5. Do not claim population inference from three seeds or correlated frames.
6. Do not modify the existing dirty local or cluster checkouts.

## Evidence required

- machine-readable provenance manifest;
- Gate 0 lock, per-pair CSV, JSON summary, and human-readable report;
- readout specification, unit tests, smoke artifact, matched seed results, and
  untouched-test decision;
- coincidence runfile and sequential 3a/3b artifacts before any 3c integration;
- geometry and estimator oracle specifications/results before new-dataset
  operator training;
- exact Git commit and Slurm job IDs for every cluster result.

## Gate sequence

1. Completed Gate 0 artifact audit and provenance snapshot.
2. Decision 2.3 diagnostic head, with a frozen specification before execution.
3. Re-plan from the diagnostic-head outcome; the readout branch remains closed
   unless a versioned decision-synthesis changelog explicitly reopens it.
4. Coincidence Gates 3a and 3b; neither silently modifies the baseline.
5. Transformation geometry and estimator oracle package.
6. Dataset manifests and upload verification.
7. Readiness decision for roll control, other objects, translation/scale, and
   finally yaw/pitch.

Failure at a critical gate stops the dependent branch and triggers a written
redirect. Passing file-integrity or runtime checks alone is never sufficient.
