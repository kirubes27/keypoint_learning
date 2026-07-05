# Stage R1 gradient-path audit lock — 2026-07-05

## Question

The R1 prediction-centred shape constraint produced healthy heatmaps and healthy
coordinate gradients with respect to heatmap logits, but coordinate learning
was much slower than the matched coordinate-only controls. This audit asks:

> Does the coordinate-learning signal become materially opposed by the shape
> objective, or materially attenuated when it passes from heatmap logits into
> the final heatmap head or shared CNN backbone?

This is a read-only checkpoint audit. It does not train, update, or select a
model.

## Frozen inputs

- object: `engineers_hammer_vray`;
- frames: the same frozen four-frame tiny-overfit batch;
- K=10, standard 64x64 heatmap architecture;
- seeds 42, 43 and 44;
- repaired checkpoints: authoritative R1 prediction-centred-JS runs;
- controls: matched 5,000-step coordinate-only runs;
- shape weight: the frozen R0 value stored in the R1 checkpoints;
- shape width: one heatmap cell.

Seed is the replication unit (`n=3`). Frames, channels, parameters and tensor
elements are correlated descriptive measurements, not independent samples. No
hypothesis test or population inference is authorized.

## Measurements

For coordinate MSE and the frozen weighted shape loss separately, compute
gradient vectors at:

1. heatmap logits;
2. final heatmap-head parameters;
3. shared encoder/backbone parameters.

For each level record:

- coordinate and weighted-shape gradient L2 norms;
- cosine similarity;
- combined-to-coordinate norm ratio;
- coordinate-descent multiplier
  `dot(g_coordinate, g_coordinate + g_shape) / ||g_coordinate||^2`;
- harmful cancellation fraction
  `max(0, 1 - coordinate_descent_multiplier)`;
- for head and backbone, gradient-transmission gain
  `||g_parameter|| / ||g_logits||` for each loss.

All values must be finite. Separately computed coordinate and shape gradients
must reconstruct the direct combined-loss gradient with relative L2 error at
most `1e-5` at every level.

## Frozen interpretation

### Material loss conflict

Supported only if the harmful cancellation fraction is at least `0.50` at the
head or backbone in at least two of three repaired seeds. This means the shape
term removes at least half of the coordinate objective's local first-order
descent at that parameter level.

Rejected if the fraction is below `0.25` at both head and backbone in every
repaired seed. Values between these boundaries are mixed.

### Material network attenuation

At each parameter level, divide the repaired run's coordinate transmission
gain by the matched coordinate-only seed's gain. Attenuation is supported only
if this ratio is at most `0.25` at the head or backbone in at least two of three
seeds.

It is rejected if the ratio is above `0.50` at both levels in every seed.
Values between these boundaries are mixed.

### Decision

- conflict supported: the present simultaneous shape objective is implicated;
- attenuation supported without conflict: the shared network gradient path is
  implicated;
- both supported: both mechanisms remain active;
- neither supported or either result mixed: audit is inconclusive and no new
  mechanism claim is allowed.

No longer R1 training, lambda sweep, R2 run, Stage-B run, or core-model change
is authorized by this audit itself.

## Required artifacts

- one row per seed, condition and gradient level in CSV;
- aggregate JSON containing thresholds, exact formulas and statistical scope;
- a dated Markdown interpretation referencing the raw artifacts;
- unit tests for vector metrics, threshold classification and gradient
  reconstruction.
