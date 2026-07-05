# Stage R0 Shape-Constraint Gate — Results (2026-07-05)

## Verdict: PASS

The prediction-centred Gaussian Jensen-Shannon constraint satisfies every
frozen semantic test, its one-shot weight is calibrated and recorded, and no
training-critical core file was changed. Stage R1 tiny supervised coordinate
training may be implemented next.

No model weights were trained in R0.

## Implemented constraint

For each heatmap logit tensor:

1. normalize it spatially to probability `p`;
2. calculate its expected coordinate `mu`;
3. detach `mu` from autograd;
4. render a Gaussian `q` with `sigma = 1.0` heatmap cell at the detached centre;
5. minimize `JS(p || q)`.

The shape constraint receives no ground-truth coordinate, mask, transform or
semantic label.

## Semantic evidence

All 22 relevant diagnostic regression tests pass. The seven R0 meaning tests
show that the constraint:

- has near-zero loss on a matching one-cell Gaussian;
- broadens a delta-like spike in its descent direction;
- concentrates a uniform distribution;
- builds central unimodal mass from two symmetric separated peaks;
- preserves loss under an interior translation;
- sends no gradient through the rendered Gaussian centre;
- remains finite at exact symmetry and extreme collapse.

## Frozen calibration

Fixed probe: seed 42, frames 0/3/6/9, K=10, standard 64x64 architecture.

| Quantity | Value |
|---|---:|
| Coordinate loss | 0.1649168283 |
| Shape loss | 0.6620597243 |
| Coordinate logit-gradient L2 | 0.0009996026 |
| Shape logit-gradient L2 | 0.0008241949 |
| Frozen `lambda_shape` | **1.2128231386** |

Formula:

`lambda_shape = ||dL_coordinate/dlogits|| / max(||dL_shape/dlogits||, 1e-12)`

No result-dependent tuning or candidate sweep occurred.

## Healthy positive-control ranges

Measured on the already successful heatmap-supervised controls, using seeds
42/43/44 and the same four correlated frames:

| Mode | Maximum probability, full range | Effective support, full range |
|---|---:|---:|
| Train | 0.1299--0.1774 | 15.93--20.42 cells |
| Eval | 0.1287--0.1797 | 15.59--20.33 cells |

The frozen R1 bands remain deliberately wider:

- per-channel median maximum probability: `[0.08, 0.30]`;
- per-channel median effective support: `[8, 32]` cells.

These values are descriptive mechanism controls. Frames/channels are not
treated as independent replicates; no inference or error bars are reported.

## Artifacts

- Calibration:
  `/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_shape_r0_20260705/PRELAUNCH_R0_CALIBRATION.json`
- Calibration SHA-256:
  `57f41561e482668212457e6a2c726f7b86eb2c93faca0e725242e7cd502b7129`
- Implementation:
  `keypoint_net/diagnostics/stage_a_shape_constraint.py`
- Calibration runner:
  `keypoint_net/diagnostics/stage_a_shape_r0.py`
- Semantic tests:
  `keypoint_net/diagnostics/test_stage_a_shape_constraint.py`

## Core-file audit

`model.py`, `train.py`, shared losses, datasets and existing checkpoints are
unchanged. R0 code is diagnostic-only and is not yet part of ordinary training.

## Next authorized stage

R1 implementation only: add the frozen constraint and weight to the diagnostic
tiny coordinate control, add the frozen shape/gradient gate metrics, run unit
tests and one short local meaning smoke. Stop and notify the user before the
three-seed training group; use the cluster if its estimated local runtime can
exceed one hour.

