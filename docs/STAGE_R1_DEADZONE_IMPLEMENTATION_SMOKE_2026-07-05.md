# Conditional dead-zone R1 implementation smoke — 2026-07-05

## Scope

Seed 42, fixed four-frame hammer subset, standard 64x64 architecture, 200 CPU
updates. This is an implementation smoke only; it is not an authoritative R1
result.

MPS was requested first but was unavailable in the execution environment. No
optimization began in that attempt. The CPU rerun completed in `43.1` seconds.

## Result

- Median coordinate error decreased from `19.03` to `3.95` cell64.
- Worst-channel error decreased from `25.33` to `16.62` cell64.
- Counterfactual coordinate gradients remained above the frozen floor.
- Heatmaps had not entered the healthy dead zone by update 200; the shape gate
  correctly remained false.
- No NaN, exception, output collision or non-finite gradient occurred.

For context only, the rejected always-active JS run had median error about
`12.69` at update 200 under the same seed/setup. This short comparison is
descriptive and does not establish that the fallback will pass at update 5,000.

## Decision

The implementation is wired correctly and shows no recurrence of the severe
early translation conflict. The synthetic gate, not this smoke, authorizes the
three-seed R1. No scientific claim is made from one seed or 200 updates.

Smoke metrics:
`keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_smoke/tiny_overfit/coordinate_standard64_k10_deadzone_seed42/metrics.json`

No core `model.py` or `train.py` file changed.
