# Stage-A Target-vs-Readout Attribution Lock (2026-07-05)

## Question

Why do the same assigned K=10 targets (especially 3, 6 and 9) fail across
seeds: numerical channel identity, physical target/location, or the
coordinate-MSE -> soft-argmax training signal?

## Frozen design

Existing reference: K=10 coordinate-MSE runs, identity target assignment,
seeds 42/43/44.

Six new runs, all K=10, seeds 42/43/44, the same four frames, standard native
64x64 heatmaps, optimizer and 5,000-update cap:

1. `coordinate_shift1`: coordinate MSE with channel `c` assigned physical
   target `(c+1) mod 10`. Therefore original targets 3, 6 and 9 move to
   numerical channels 2, 5 and 8; numerical channels 3, 6 and 9 receive
   original targets 4, 7 and 0.
2. `heatmap_identity`: original target assignment with Gaussian target-heatmap
   cross-entropy (sigma = 8 input pixels). Evaluation remains the same
   soft-argmax coordinate error and unchanged A0 thresholds.

No K, resolution, backbone, optimizer or threshold changes are allowed.

## Preregistered interpretation

- Failure follows physical targets 3/6/9 to channels 2/5/8 under permutation:
  target/location or its interaction with coordinate optimization.
- Numerical channels 3/6/9 fail on their new physical targets: channel/head
  implementation issue.
- Heatmap supervision passes the full A0 gate in at least 2/3 seeds while the
  coordinate conditions fail: backbone can represent the targets;
  coordinate-only soft-argmax gradients are the bottleneck.
- Heatmap supervision retains the same physical-target failures in at least
  2/3 seeds: investigate target observability/receptive field before any
  fitted-operator training.
- Failures move inconsistently under permutation and heatmap supervision does
  not rescue them: optimization instability remains unresolved.

## Statistical scope

All results are descriptive. The only replication unit is the optimization
seed (`n=3` per condition). The four images are from one object/orbit and are
not independent population samples. No error bars, hypothesis test,
generalization claim or semantic-keypoint claim is permitted.
