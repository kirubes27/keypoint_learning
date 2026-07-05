# Stage-A Soft-Argmax Gradient Audit — Results (2026-07-05)

## Verdict

The preregistered **target-region starvation** hypothesis is **mixed**, not a
clean pass: 60.9% of identity-assignment failed units and 60.0% of shifted-
assignment failed units met the strict spatial signature, below the frozen 75%
threshold but above the 50% rejection boundary.

The broader gradient pathology is nevertheless directly observed in both
assignments. Coordinate-MSE gradients collapse after training while Gaussian
heatmap gradients on the exact same logits remain strong. Failures comprise at
least two subtypes:

1. far wrong peaks with negligible target probability and target gradient;
2. near-target, nearly one-hot peaks whose expectation cannot make the required
   sub-cell correction.

A preregistered read-only temperature tiebreaker restored gradient magnitude
but made coordinate error substantially worse. Therefore a global temperature
increase is **not** a justified repair.

## Local artifacts

All artifacts are stored on the Mac at:

`/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_gradient_audit_20260705`

- `A0_GRADIENT_AUDIT_SUMMARY.json`
- `A0_GRADIENT_AUDIT_UNITS.csv`
- `A0_TEMPERATURE_TIEBREAKER_SUMMARY.json`
- `A0_TEMPERATURE_TIEBREAKER_UNITS.csv`
- `SHA256SUMS.txt`

The analytic coordinate-logit gradient matched PyTorch autograd with maximum
absolute error `3.73e-9`, passing the frozen `1e-6` implementation gate.

## Primary audit

Both losses were differentiated with respect to identical frozen 64x64 logits;
no model weights or source checkpoints were modified.

| Coordinate condition | Failed frame×channel units | Spatial-starvation fraction | Median coordinate-gradient final/initial | Median heatmap-gradient final/initial |
|---|---:|---:|---:|---:|
| Identity assignment | 64 | 0.609 | 0.00836 | 2.68 |
| Shift-1 assignment | 40 | 0.600 | 0.000418 | 3.07 |

Thus the typical coordinate gradient fell to approximately 0.84% of its
initial value in the identity condition and 0.042% in the shifted condition.
The dense heatmap gradient did not vanish; it was larger than at initialization.

Across both coordinate conditions, 75% of failed units met the preregistered
saturated-wrong-peak condition. The heatmap-trained positive control remained
valid: all ten channel medians were <=0.20 cell64.

## Why the strict target-starvation verdict was mixed

For the large failures, the narrow hypothesis is exact. Examples pooled over
the three seeds and four frames:

- shifted channel 2 / physical target 3: median target-region probability
  `7.5e-9`, median error `3.25` cells;
- shifted channel 5 / physical target 6: median target-region probability
  `5.5e-8`, median error `2.90` cells.

However, shifted channel 0 / physical target 1 had median target-region
probability about `0.987`, while still showing median error `0.85` cell and a
coordinate-gradient final/initial ratio of `1.5e-4`. Its probability was at the
target neighborhood but was too concentrated on a discrete cell to produce the
required sub-cell expectation. This fails the narrow “target has no mass” rule
while still exhibiting severe saturation and gradient collapse.

This decomposition is descriptive and was used only to choose the already
documented tiebreaker; it does not retroactively change the primary thresholds.

## Temperature tiebreaker

The same frozen logits were re-read at temperatures 1, 2, 4 and 8, without
weight updates.

At T=2:

- far wrong peaks: gradient norm increased `49.6x` and target probability
  increased `2813x`, satisfying the desaturation criterion, but median error
  became `2.42x` worse;
- near-target peaks: gradient norm increased `3813x`, but median error became
  `7.26x` worse, failing the near-target repair criterion.

Higher temperatures degraded errors further. Global desaturation spreads mass
over unrelated spatial regions and moves the expectation away from the target.
It restores gradient magnitude but does not preserve a useful heatmap shape.

## Scientific conclusion

The evidence now excludes the following primary explanations for A0:

- defective fixed channel indices;
- insufficient 64x64 spatial resolution;
- insufficient receptive field or unrepresentable targets;
- too few training updates;
- keypoint count;
- temperature being simply too low.

The remaining supported problem is that coordinate-only expectation supervision
does not constrain heatmap shape. It permits wrong, overly concentrated or
otherwise malformed distributions; once saturated, their coordinate gradients
can become negligible. Dense Gaussian supervision succeeds because it specifies
the desired spatial distribution directly.

This is still an instrument-level result. It does not prove that every defect
in the original unsupervised keypoints has this cause.

## Statistical scope

The audit is descriptive on one object and four correlated frames. Optimization
seed is the only replication unit (`n=3` per assignment). Frame×channel counts
are mechanism measurements, not independent population samples. No error bars,
hypothesis test, confidence interval or population inference is reported.

## Next decision

Do not launch Stage B and do not use a temperature sweep. The next candidate,
if authorized, should be exactly one unsupervised-compatible spatial-shape
constraint that keeps each heatmap finite-width and unimodal around its own
predicted coordinate without using ground-truth locations. Its semantic lock
must specify how it avoids both one-hot saturation and diffuse/global spreading,
and it must pass the original coordinate A0 gate in at least 2/3 seeds before
any image-level objective comparison resumes.

