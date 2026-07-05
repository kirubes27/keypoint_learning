# Stage R1 shape-gate result — 2026-07-05

## Decision

**R1 FAILS. Do not start R2.**

The preregistered joint gate required at least two of three seeds to pass the
coordinate, heatmap-shape, and counterfactual-gradient criteria, with no
physical target failing in at least two seeds. The observed result was zero of
three joint passes, and every physical target failed in all three seeds.

This is a descriptive one-object, four-correlated-frame experiment with three
optimization seeds. Seed is the replication unit (`n=3`). No population-level
inference or hypothesis test is claimed.

## Semantic lock

For the proposed repair to be supported, all of the following had to be true:

1. fixed target coordinates could be learned to the frozen coordinate gate;
2. every channel retained the frozen healthy heatmap-shape range;
3. every channel retained usable counterfactual coordinate gradients.

Healthy heatmaps alone were not sufficient evidence of success.

## Evidence

| Seed | Joint | Coordinate | Shape | Gradient | Median error (64-cell units) | Worst-channel error |
|---:|---|---|---|---|---:|---:|
| 42 | fail | fail | pass | pass | 2.344 | 6.629 |
| 43 | fail | fail | pass | pass | 2.342 | 7.140 |
| 44 | fail | fail | pass | pass | 3.026 | 8.870 |

Coordinate thresholds were median error at most `0.1` and worst-channel error
at most `0.2` in 64-cell units. All ten physical targets failed in every seed.

The shape repair itself behaved as intended at step 5,000:

- channel-median maximum probabilities were approximately `0.135--0.162`;
- effective supports were approximately `16.2--17.4` cells;
- run-level counterfactual-gradient ratios were `0.600--0.673` of
  initialization, and every per-channel ratio remained above its floor.

The coordinate errors were still decreasing rather than flatlining:

| Seed | Step 100 | Step 1,000 | Step 3,000 | Step 5,000 |
|---:|---:|---:|---:|---:|
| 42 | 11.656 | 10.903 | 5.041 | 2.344 |
| 43 | 13.047 | 10.036 | 5.374 | 2.342 |
| 44 | 11.959 | 10.994 | 5.638 | 3.026 |

These are median coordinate errors in 64-cell units. They show continued
optimization, but they do not satisfy the frozen gate.

For context, the earlier coordinate-only K=10 control at the same 5,000-step
budget reached median errors `0.102`, `0.140`, and `1.048`, with worst-channel
errors `4.396`, `3.179`, and `4.399`. Therefore the prediction-centred JS repair
improved heatmap shape and gradient health but materially slowed coordinate
localization under the frozen budget.

## Interpretation

Supported:

- the prediction-centred JS term can maintain healthy Gaussian-like heatmaps;
- the coordinate readout remains sensitive to counterfactual target shifts;
- the three runs are still making coordinate progress at step 5,000.

Rejected:

- malformed heatmap shape or collapsed logit gradients are the complete cause
  of the coordinate-learning failure;
- the current shape repair solves the supervised coordinate control;
- the current repair is ready for R2 or Stage B.

Still unknown:

- whether a much longer run would eventually pass;
- whether the slowdown is caused by conflict between the coordinate and shape
  objectives, or by attenuation/cancellation after gradients pass from logits
  into the shared CNN parameters.

A longer run would now be a new post-hoc experiment, not a valid reinterpretation
of R1.

## Next discriminating step

Before changing the model or extending training, run one bounded, read-only
gradient-path audit on the saved R1 and matched coordinate-only checkpoints.
On the same frozen four-frame batch and all three seeds, separately measure the
coordinate and shape gradients at:

1. heatmap logits;
2. the final heatmap-head parameters;
3. the shared backbone parameters.

Report gradient norms and coordinate-versus-shape cosine similarity at each
level. This distinguishes a loss-conflict problem from a network-Jacobian or
shared-parameter credit-assignment problem. It requires no training and should
run locally in minutes. Its thresholds and interpretation must be locked before
implementation.

No lambda sweep, longer R1 run, R2 launch, or architectural fallback is
authorized by this result.

## Artifact provenance

- Download root:
  `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r1_shape_gate_20260705_142241`
- Aggregate JSON:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_gate/R1_SHAPE_GATE_SUMMARY.json`
- Per-run table:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_gate/R1_SHAPE_GATE_RUNS.csv`
- Per-seed metrics and models:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_gate/tiny_overfit/`
- Scheduler logs: `slurm_logs/stage_r1_shape_53039735_*.{out,err}`
- Archive checksum: validated by the collector before interpretation.

No core model or training file was changed while producing this result report.
