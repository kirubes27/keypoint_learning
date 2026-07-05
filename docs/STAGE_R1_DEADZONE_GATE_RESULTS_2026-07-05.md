# Conditional dead-zone fallback R1 result — 2026-07-05

## Decision

**R1 FAIL: 0/3 joint seed passes. Stop this heatmap-loss instrument design.**

Do not launch R2, tune the dead-zone weight, extend training or design a third
shape loss under the frozen fallback rule. Review a different instrument
architecture with Angela. This result does not reject the operator/keypoint
research hypothesis.

## Evidence

| Seed | Median error cell64 | Worst channel | Failed targets | Shape gate | Gradient gate |
|---:|---:|---:|---|---|---|
| 42 | 0.043 | 4.396 | 1, 3, 6, 9 | fail | fail |
| 43 | 0.439 | 4.110 | 1, 2, 3, 6, 8, 9 | fail | fail |
| 44 | 0.066 | 5.494 | 3, 6, 9 | fail | fail |

Persistent physical targets were `1, 3, 6, 9`; targets `3, 6, 9` failed in all
three seeds. Aggregate PASS required at least 2/3 joint passes and no target
failing in at least 2/3 seeds.

Several channels still collapsed to near-delta heatmaps (maximum probability
`0.9991--0.9998`, effective support approximately `1`), while others remained
too diffuse (maximum support `50.2--78.6`, minimum dominant radius-2 mass
`0.260--0.321`). The affected channels lost usable counterfactual gradients;
minimum per-channel final/initial ratios were `8.76e-5--4.49e-4`.

## Interpretation

The conditional constraint removed the previous global translation conflict:
seeds 42 and 44 achieved good global median localization, and all runs improved
substantially from initialization. However, the CNN learned around the
conditional penalty: some channels remained collapsed or diffuse, and the same
historically persistent targets `3, 6, 9` survived.

The synthetic logit-level repair was therefore necessary but not sufficient at
the shared-CNN parameter level. The existing CNN heatmap plus coordinate-only
soft-argmax design is not a reliable instrument under either of the two frozen
shape repairs tested.

This is a descriptive one-object/four-correlated-frame gate with `n=3`
optimization seeds. Seed is the replication unit. No hypothesis test, error
bar or population inference is claimed.

## Provenance

- Cluster job: `53040568`, commit `0367606`.
- Download:
  `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r1_deadzone_gate_20260705_160834`
- Aggregate JSON:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_gate/R1_DEADZONE_GATE_SUMMARY.json`
- Runs CSV:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_gate/R1_DEADZONE_GATE_RUNS.csv`
- Archive SHA-256 validation: pass.
- All three runs completed 5,000 updates with empty error logs.

No core `model.py` or `train.py` file changed.
