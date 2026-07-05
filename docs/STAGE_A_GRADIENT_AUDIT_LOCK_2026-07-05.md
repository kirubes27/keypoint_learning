# Stage-A Soft-Argmax Gradient Audit — Semantic Lock (2026-07-05)

## Question

Why can Gaussian heatmap supervision fit every target while coordinate MSE
through the same soft-argmax readout leaves stable wrong-location failures?

The specific hypothesis is **probability-weighted gradient starvation**: after
a heatmap assigns negligible probability to the target region, coordinate MSE
has negligible logit gradient there and cannot reliably create the missing
target peak, while dense heatmap cross-entropy retains a direct target-region
gradient.

## Must be true before interpreting the result

1. Both losses are differentiated with respect to the exact same heatmap logits.
2. The coordinate-logit gradient implementation agrees with PyTorch autograd on
   a synthetic tensor to maximum absolute error <= `1e-6`.
3. Standard 64x64 architecture, K=10, the same four frames and seeds 42/43/44
   are used. `true_quarter_res` remains false.
4. No model weights are updated by this audit.
5. Failed/pass labels come from the already frozen A0 channel gate, not from a
   new post-hoc threshold.

## Frozen states and comparisons

For every seed, audit:

1. deterministic initialization for identity and shift-1 target assignments;
2. the final coordinate-MSE identity checkpoint;
3. the final coordinate-MSE shift-1 checkpoint;
4. the final heatmap-supervised identity checkpoint as a positive control.

Models are put in training mode and evaluated on the complete fixed four-frame
batch because the question concerns the gradient used by the next training
update, including BatchNorm batch statistics. Saved checkpoints are loaded into
fresh model instances; the source artifacts are never modified.

For each frame x channel unit record:

- coordinate error and argmax-to-target distance in cell64 units;
- maximum spatial probability and normalized entropy;
- probability mass within one heatmap cell of the target and current argmax;
- coordinate-MSE logit-gradient L2 norm;
- heatmap-CE logit-gradient L2 norm on the same logits;
- fraction of absolute gradient mass within one cell of target and argmax for
  each loss;
- final/initial gradient-norm ratios matched by seed, assignment, frame and
  channel.

## Preregistered interpretation

A failed frame x channel unit satisfies the **spatial starvation signature** if:

1. target-neighborhood probability mass is <= `1e-3`; and
2. coordinate-gradient mass fraction near the target is <= `0.1` times the
   heatmap-gradient mass fraction near the target on the same logits.

The mechanism is **supported** if, in both coordinate conditions:

- at least 75% of failed units satisfy the spatial starvation signature;
- median coordinate-gradient final/initial norm ratio among failed units is
  <= `0.10`; and
- median heatmap-gradient final/initial norm ratio on those same logits remains
  >= `0.50`.

It is **not supported** if, in either coordinate condition, fewer than 50% of
failed units satisfy the spatial signature or the median coordinate-gradient
ratio is > `0.50`. Other outcomes are **mixed**.

The narrower **saturated wrong-peak** mechanism is supported only if at least
75% of failed units additionally have maximum probability >= `0.90` or
normalized entropy <= `0.25`. Otherwise the report must call the pattern
target-gradient starvation without claiming one-hot saturation.

The heatmap-trained positive control must have no channel above the original
0.20-cell64 median gate. If it does, stop: the saved-state audit is inconsistent
with the preceding experiment.

## Statistical scope

This is descriptive mechanism tracing on one object. Frames in the four-image
batch and channels within a network are not independent population samples.
Optimization seed is the only replication unit (`n=3` per condition). No
hypothesis test, confidence interval or population claim is authorized.

## Stop rule

Write the audit verdict before designing a repair. If the result is mixed or
contradicts starvation, identify one discriminating measurement; do not begin a
repair sweep. If supported, preregister exactly one unsupervised-compatible
repair and require the original coordinate A0 gate to pass in at least 2/3
seeds before Stage B resumes.

## Mixed-outcome tiebreaker (frozen after primary audit, before temperature evaluation)

The primary audit returned mixed: coordinate-gradient norms collapsed and
heatmap gradients remained strong in both assignments, but only 60--61% of all
frames from failed channels met the narrower target-region signature. Inspection
used only the already-declared audit fields and showed two candidate subtypes:

- far wrong peaks with negligible target probability;
- near-target, nearly one-hot peaks that cannot make a sub-cell correction.

One read-only tiebreaker is authorized. On the same frozen coordinate logits,
recompute soft-argmax and its coordinate gradient at temperatures
`T in {1, 2, 4, 8}`. No parameters are updated. Define near-target units using
the T=1 argmax distance `<= 1.5` cells and far units as `> 1.5` cells.

- **Near-target saturation supported:** for at least one higher temperature,
  the median coordinate error is <=0.5 times T=1 and the median coordinate
  gradient norm is >=10 times T=1.
- **Far wrong-peak desaturation supported:** for at least one higher
  temperature, median target-region probability and coordinate-gradient norm
  are each >=10 times T=1. Error improvement is reported but not required,
  because temperature alone cannot identify the absent target peak.
- If neither condition holds, temperature/desaturation is not a justified
  repair direction. If only one holds, the repair direction is partial and
  must not be sold as a complete explanation.

This is a sensitivity measurement, not a training sweep or repair validation.
