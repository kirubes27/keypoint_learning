# Diagnostic review point — 2026-07-04

## Current conclusion

The immediate, demonstrated bottleneck is **coordinate localization noise with
structured network residuals**. It is large enough to make the fitted
similarity mechanism unreliable at the current Task-80 noise level. This is
not yet a diagnosis of the upstream cause: input aliasing is implicated, but
the current evidence does not distinguish an architectural localization
ceiling from training/readout effects.

This document is the required review point. Do not begin image-level training
until the next experiment is selected and its thresholds are frozen.

## Evidence that changed the decision

- The width-9 residual filter is not the dominant explanation. Five
  independent, variance-matched filtered-white banks gave affine ordering
  means of 0.880 (K=6; across-bank sample std 0.0267, ddof=1) and 0.789
  (K=10; std 0.0468). Both are much closer to iid-heterogeneous references
  0.907/0.807 than to empirical 0.660/0.373. These are descriptive simulation
  results conditional on one checkpoint/orbit, not population inference.
- At the real empirical noise level, multi-step similarity separation passed
  only 3/5 deterministic optimization seeds, below the frozen 4/5 gate. Both
  failed seeds passed separation, non-collision, loss-decrease, fixed-anchor,
  and finite-value checks; they failed only the radius <=0.8 condition
  (final radii 0.862 and 0.928). The loss can separate duplicates but can push
  them outside the permitted object region.
- With the same learning rate, optimization, seeds, and thresholds at half
  residual amplitude, the test passed 5/5 seeds. This is a toy-level causal
  intervention on noise magnitude: it shows that lowering residual amplitude
  is sufficient to make this mechanism reliable in this simulation. It does
  not prove that a 128-resolution image model will produce that reduction.
- Task-80 median one-step equivariance error is 0.826 cells64. Its sub-pixel
  aliasing gate fires in 10/200 channel/transform cells, concentrated in
  channels 4, 5, and 7. The 128 smoke checkpoint improves median one-step
  error to 0.560 cells64, but its on-mask fraction is worse and the smoke
  training is not a controlled architecture experiment.

## Candidate-cause verdicts

- **Geometry mismatch — EXONERATED for this sequence.** Mask transport mean
  IoU is 0.9866 and even/odd fitted centres agree to 0 px.
- **Architecture localization ceiling — UNRESOLVED.** Higher resolution has a
  better equivariance median, but the existing 64/128 runs differ in training
  outcome and do not isolate architecture.
- **Soft-argmax/multimodality — EXONERATED as the dominant Task-80 cause.**
  Hard readout worsens Task-80 equivariance by 28.9%. It is implicated for the
  128 smoke checkpoint, so it remains a secondary readout issue there.
- **Channel switching — EXONERATED as the dominant cause.** Hungarian
  reassignment does not meet the preregistered improvement criterion.
- **Input aliasing/localization noise — IMPLICATED.** The sub-pixel transform
  gate fires, and halving coordinate residual amplitude changes similarity
  separation from 3/5 to 5/5.
- **Objective/loss conflict — UNRESOLVED at image level.** The coordinate
  mechanism separates points but lacks a sufficient spatial constraint in
  two full-noise seeds. No image-level fitted-loss training has been run.
- **Proxy-signal weakness — UNRESOLVED.** Heatmap spread is not a usable error
  calibrator on these checkpoints, and the oracle matrix has not been run.

## Next gate

The clean tiebreaker is the Days 4-5 supervised coordinate-regression control,
run as a controlled 64-vs-128 architecture/readout test. It must answer:

1. Can each architecture learn stable, on-object target coordinates when the
   objective is unambiguous?
2. Does 128 resolution reduce held-out residual amplitude by enough to cross
   the demonstrated half-noise regime without losing grounding?

This is a multi-run training group estimated near or above one hour, so it
requires an explicit local-versus-cluster decision before launch. If the
supervised control succeeds, design a pruned oracle matrix around proxy
quality and add an explicit spatial barrier to the similarity arm. If it
fails, pause fitted-loss work and fix the localization instrument first.

## Auditable artifacts

- `outputs/geometry_gate.json`
- `outputs/day1_summary.csv`
- `outputs/day2_summary.json` and `outputs/day2_aliasing.csv`
- `outputs/day25_noise_ladder.csv`
- `outputs/day25_filter_artifact_control.json`
- `outputs/day25_similarity_results.csv`
- `outputs/day25_similarity_verdict.json`
