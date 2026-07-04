# Stage-A Keypoint-Count Sweep Lock (2026-07-05)

## Question

Was the K=10 tiny-overfit failure tied to specific output channels, caused by
stochastic soft-argmax saturation, or made worse by increasing the number of
keypoints sharing one CNN?

## Frozen design

- K in {5, 10, 15, 20}; no intermediate K values.
- Seeds {42, 43, 44}.
- Four fixed training frames {0, 3, 6, 9}.
- Standard native 64x64 heatmaps only; true-/4 resolution is not mixed into
  this attribution experiment.
- Same optimizer, targets, 5,000-update cap and unchanged A0 thresholds.
- Every run is retained, including failures.

## Interpretation

- Same channel index fails in all three seeds at a fixed K: stable
  channel/target-ordering issue.
- Median failed-channel fraction rises by at least 0.20 from K=5 to K=20 and
  is nondecreasing on at least two of three adjacent K steps: K-dependent
  shared-capacity/interference pattern.
- Failures occur but indices change across seeds without the K-dependent
  pattern: stochastic saturation.
- K=5 fails in at least two seeds without a >=0.20 rise by K=20: core readout
  issue already present at low K.

These are descriptive pattern labels, not hypothesis tests. The sample unit is
an optimization seed (n=3 per K); all images are from one correlated object
orbit. This experiment does not determine the useful number of semantic
keypoints. It only diagnoses whether K changes the basic coordinate mechanism.

## Resolution decision

One cell64 is approximately eight input pixels. Errors of 2--3 cells are wrong
locations, not 64-to-512 rounding. Native /4 is tested only after this sweep,
and only if failures are predominantly sub-cell rather than wrong-peak errors.
