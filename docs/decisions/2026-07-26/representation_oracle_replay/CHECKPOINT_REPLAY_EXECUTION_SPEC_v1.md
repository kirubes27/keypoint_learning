# Saved-checkpoint replay execution specification

Date: 2026-07-28

Status: implementation candidate; real checkpoint loading remains blocked until
the exact candidate commit receives a substantive read-only Fable review and a
separate committed execution authorization binds that review.

## Plain-language purpose

This replay asks the already-trained Task 20, 55, and 80 models to locate their
ten keypoints on the existing 180 hammer roll images. It does not retrain the
models, change their weights, regenerate the dataset, or choose a new training
recipe.

The replay has two jobs:

1. prove that the new evaluator reproduces the old operator and rollout numbers;
2. measure whether the ten keypoint channels are distinct, active, attached to
   the hammer, and stable across the roll.

## Decision

Decide whether the representation evaluator and saved-checkpoint replay are
trustworthy enough to begin the matched 64-versus-128 heatmap-resolution
experiment.

## Current claim

If the replay runtime reconstructs the three historical models exactly, Task 20
should reproduce the preregistered collapsed negative control, while Tasks 55
and 80 should reproduce the preregistered non-collapsed classifications and all
245 definition-identical historical values within the frozen replay tolerance.

## Known evidence

- The hammer roll geometry is bound to an image centre of `(0, 0)` in normalized
  endpoint coordinates, a forward `+6 degree` three-frame transform, and a
  canonical `-theta` unrotation.
- The three checkpoint candidates have passed opaque byte-size and SHA-256
  preflight. That receipt explicitly did not authorize deserialization.
- The external configs, historical output files, dataset inventory, pair index,
  masks, and metadata are registered inputs rather than selection evidence.

## Critical unknown

Whether a reviewed, source-bound, CPU-only runtime can safely reconstruct the
legacy checkpoints and pass the production evaluator without changing any
model state or weakening the evaluator's provenance checks.

## Must be true

- Every committed source and manifest is byte-identical to the exact source
  commit named by the evaluation provenance.
- Each checkpoint, config, and history file matches one registry fixture and
  one run directory.
- Live image, mask, pair-index, and metadata bytes match their committed
  manifests before checkpoint loading.
- The runtime freezes the legacy architecture: RGB input, 512 by 512 images,
  ten keypoints, base width 32, 64 by 64 heatmaps, shared affine operator, no
  action head, and an inverse operator only for Tasks 20 and 80.
- A checkpoint is opened once with no-follow semantics, hashed through that
  same file descriptor, rewound, and loaded from that same descriptor using
  `torch.load(..., map_location="cpu", weights_only=True)`.
- State-dict loading is strict. There is no unsafe fallback, optimizer
  construction, training mode, gradient calculation, or weight update.
- Inference covers exactly frames `0..179` once each, in order, using frozen
  RGB conversion, scaling, and ImageNet normalization with no augmentation.
- The evaluator receives the raw 64 by 64 logits, recomputes the soft-argmax
  coordinates, and verifies them against the runtime coordinates.
- A parameter-and-buffer digest is identical before and after inference.

## Must not happen

- No dataset image, mask, metadata, pair index, checkpoint, config, history, or
  historical output is modified.
- No real checkpoint is opened before the reviewed execution authorization is
  committed.
- No caller-supplied Boolean flag can authorize checkpoint evaluation.
- Task 20 is never treated as a candidate, and Tasks 55/80 are never treated as
  selection evidence.
- Tasks 55/80 are not run if Task 20 fails its collapse-control gate.

## Evidence required

For each replayed task, save a deterministic result that binds the source
commit and every external input hash, records the frozen architecture and
preprocessing, proves state immutability, reports the production evaluator
output, and reports all 245 historical comparisons.

## Next gate and stop conditions

1. Run focused source/provenance/codec tests and a synthetic checkpoint smoke.
   Stop if authorization can be forged, a mutation reaches the loader, strict
   reconstruction fails, or synthetic inference changes state.
2. Commit the candidate and obtain a substantive independent Fable review.
   Stop on any unresolved P0/P1 finding or on a missing/non-substantive review.
3. Commit a separate execution authorization without changing reviewed runtime
   sources.
4. Replay Task 20 only. Continue only if its structural negative-control
   collapse is `true` and all 245 historical comparisons pass.
5. Replay Tasks 55 and 80. The replay gate passes only if both structural
   collapse classifications are `false`, all 245 historical comparisons pass
   for each task, and every safety/provenance check remains true.

Passing this gate makes the matched 64-versus-128 training comparison eligible
to start; it does not establish that either saved model learned scientifically
useful keypoints.
