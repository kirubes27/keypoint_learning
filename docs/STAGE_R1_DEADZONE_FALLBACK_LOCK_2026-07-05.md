# Stage R1 conditional dead-zone fallback lock — 2026-07-05

## Scope and terminal rule

This is the single fallback repair authorized after the always-active
prediction-centred JS constraint failed R1. It is implemented only in the
diagnostic control first.

If this candidate fails the synthetic gate or the subsequent three-seed R1
gate, stop modifying heatmap losses. The current CNN-heatmap/coordinate-only
soft-argmax instrument is then classified as instrument-limited for this paper
timeline and reviewed with Angela. This does not reject the operator/keypoint
research hypothesis.

## Required semantics

The fallback must:

1. exert exactly zero loss and gradient on an already healthy heatmap;
2. activate on spike, diffuse and separated-mode failures;
3. become silent after repairing a malformed heatmap;
4. preserve coordinate translation rather than anchor the current mean;
5. use no true coordinate, mask, transform or semantic annotation;
6. remain finite at exact uniformity, equal modes and band boundaries.

It must not use an always-active Gaussian anchored at the current prediction.

## Frozen candidate

For each spatial softmax probability map `p`, calculate:

- maximum probability `m = max(p)`;
- effective support `s = exp(entropy(p))`;
- dominant-neighbourhood mass `d`: probability mass within a two-cell Euclidean
  radius of a detached argmax location.

Use the normalized squared-hinge loss

```text
relu((0.08 - m) / 0.08)^2
+ relu((m - 0.30) / 0.30)^2
+ relu((8 - s) / 8)^2
+ relu((s - 32) / 32)^2
+ relu((0.70 - d) / 0.70)^2
```

averaged over frame/channel units.

The detached argmax makes the dominant-neighbourhood term piecewise
differentiable and gives a deterministic symmetry-breaking direction at an
exact tie. The tie surface is a known discontinuity; exact-tie finiteness and
repair are mandatory tests. No claim of global smoothness is permitted.

## Threshold provenance

The existing successful heatmap-supervised controls were measured in training
mode on the frozen four-frame batch across seeds 42, 43 and 44 (`120`
frame/channel units; correlated descriptive units):

| Statistic | Observed range | Frozen dead zone |
|---|---:|---:|
| maximum probability | `0.1299--0.1774` | `0.08--0.30` |
| effective support | `15.93--20.42` | `8--32` |
| radius-2 dominant mass | `0.7554--0.8839` | `>=0.70` |

Thus every positive-control unit lies strictly inside the dead zone. The first
two ranges were already frozen for R1; only the dominant-mass statistic is new.
These values are descriptive, not inferential. Frames/channels are not treated
as independent samples.

## Weight calibration

On the fixed seed-42 initial CNN and frozen four-frame supervised batch:

```text
lambda_deadzone = ||d L_coordinate / d logits||
                  / max(||d L_deadzone / d logits||, 1e-12)
```

Store both raw norms and the resulting weight before any optimization. There
is no weight sweep.

## Synthetic gate

All conditions must pass with the frozen calibrated weight:

1. **Positive controls:** every one of the 120 successful heatmap-supervised
   units has exactly zero dead-zone loss contribution; aggregate dead-zone
   logit-gradient L2 is at most `1e-10`.
2. **Healthy translations:** interior one-cell Gaussians at multiple locations
   have zero loss/gradient, translation-invariant loss within `1e-8`, and retain
   at least `0.90` coordinate-descent multiplier for half-cell shifts in `+x`,
   `-x`, `+y` and `-y`.
3. **Malformed activation:** spike, diffuse, exact-uniform and equal separated-
   mode prototypes have positive loss and finite nonzero gradients.
4. **Repair then silence:** optimizing each malformed prototype using only the
   dead-zone loss reaches all frozen healthy ranges, then has loss at most
   `1e-10` and gradient L2 at most `1e-8` within 2,000 updates. Report the actual
   update count; prototypes are correlated deterministic cases, not samples.
5. **Active compatibility:** for spike, diffuse and separated-mode prototypes
   with fixed coordinate targets in four directions, the combined coordinate
   plus weighted-dead-zone gradient has coordinate-descent multiplier `>=0.0`.
   Exact uniformity is excluded from this directional criterion because its
   deterministic tie-break necessarily chooses a direction; it remains covered
   by activation, repair and finiteness checks.
6. **Boundary safety:** exact statistic boundaries and equal-mode ties produce
   finite loss and gradients; squared hinges have zero derivative at their
   scalar band boundary.

The audit is valid only if direct combined gradients reconstruct the sum of
separate gradients to relative L2 error `<=1e-5`.

### Recorded validity clarification before verdict

The first gate execution stopped before producing a verdict because the
deliberately extreme repair-spike (`sigma=0.25` cell) had a coordinate-gradient
norm below the frozen numerical floor, so its descent multiplier was undefined.
This is the saturation pathology the repair case is intended to escape, not a
negative multiplier. The extreme spike remains unchanged in activation and
repair tests. Active compatibility uses a still-invalid but movable
`sigma=0.35` spike. All thresholds remain unchanged. Exact scalar boundaries
use `1e-8` numerical comparison tolerance while retaining zero-loss and
zero-gradient requirements.

## After the synthetic gate

Only a complete synthetic pass authorizes the same R1 tiny supervised gate:

- one hammer, frames 0/3/6/9, K=10, standard 64x64;
- coordinate MSE plus frozen conditional constraint;
- seeds 42/43/44, maximum 5,000 updates;
- the existing R1 coordinate, shape and counterfactual-gradient thresholds;
- at least 2/3 joint passes and no persistent physical target failure.

R2 remains blocked until R1 passes. No core `model.py` or `train.py` change is
authorized at this stage.
