# Task 55 cross-backend coordinate-tolerance semantic lock

Date: 2026-07-29

## Decision

Decide whether the Task 55 replay may continue after the saved model's
PyTorch coordinates and the independent NumPy evaluator differed by a very
small amount.

## Current claim

The observed stop is a numerical interface defect, not evidence that Task 55
learned bad keypoints: the saved model's coordinates were reproduced exactly
from its logits by a fresh PyTorch calculation, while the mathematically
equivalent NumPy reduction differed by at most
`1.3053417205810547e-05` in normalized coordinates.

## Known evidence

- Task 55 produced 1,800 heatmaps and 3,600 coordinate components.
- Model output versus a fresh PyTorch spatial expectation had maximum absolute
  error `0.0`.
- Model output versus the NumPy evaluator had maximum absolute error
  `1.3053417205810547e-05`; 40 of 3,600 components exceeded the old
  `3.814697265625e-06` numeric tolerance and 7 exceeded `1e-05`.
- The largest difference is approximately `0.000411` of one 64-by-64 heatmap
  pixel. It cannot materially change a keypoint location or any scientific
  classification.
- No Task 55 result was written. Task 80 has not been opened.

## Critical unknown

What implementation-consistency tolerance is wide enough for PyTorch-versus-
NumPy float32 reduction order, but still far too small to hide a materially
different coordinate convention?

## Frozen tolerance semantics

1. The existing `3.814697265625e-06` registry tolerance remains unchanged for
   NumPy-only planted and dataset evaluations.
2. Saved-checkpoint cases, whose supplied coordinates come from PyTorch while
   the independent evaluator uses NumPy, use a separate tolerance of `1e-4`
   normalized coordinate units.
3. `1e-4` is an implementation-consistency tolerance only. At 64-by-64
   resolution it is `0.00315` heatmap pixels: about 7.7 times the observed
   Task 55 maximum, but about 317 times smaller than one grid-cell step.
4. The evaluator still replaces the supplied coordinates with its own
   NumPy-derived coordinates before computing every scientific metric.
5. The checkpoint temperature, endpoint grid, axis order, dtype, resolution,
   and preprocessing remain independently frozen and validated.

## Must be true

- The logits are preserved unchanged and remain the primary heatmap evidence.
- The evaluator continues to recompute its coordinates from logits.
- A cross-backend difference above `1e-4` fails before evaluator execution.
- The bridge does not alter images, masks, splits, checkpoints, model weights,
  heatmaps, operators, historical comparisons, or collapse definitions.
- The bridge performs no training, selection, or hyperparameter tuning.
- Task 55 and Task 80 remain replay-only baselines.

## Must not happen

- Do not change any scientific quality threshold to make Task 55 pass.
- Do not apply the checkpoint-specific tolerance to planted NumPy-only cases.
- Do not describe the numerical bridge as evidence that Task 55 has useful,
  stable, distinct, or attached keypoints.
- Do not run Task 55 or Task 80 officially until the amended runtime has passed
  tests, source binding, and a fresh read-only Fable review.

## Evidence required before replay

- A role-scoping test proves checkpoint cases use `1e-4` while planted cases
  retain the original registry tolerance.
- A discrepancy equal to the observed Task 55 value is below the checkpoint
  tolerance, while a `1e-3` discrepancy remains above it.
- The existing planted oracle suites and checkpoint-blind runtime tests pass.
- The reviewed candidate source hashes are frozen in a new runtime-source
  manifest and independently reviewed by Fable at high effort.

## Next gate

Implement only the checkpoint-scoped tolerance above. Run a v3 Task 20 shadow
replay after review and require its scientific metrics and collapse decision to
match v2 before Tasks 55 and 80. Stop if any existing semantic oracle changes
unexpectedly or if the independent review identifies a P0/P1 defect.
