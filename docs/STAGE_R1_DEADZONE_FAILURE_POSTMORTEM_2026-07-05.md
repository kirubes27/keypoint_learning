# Stage R1 dead-zone failure post-mortem — 2026-07-05

## Corrected conclusion

The 5,000-**update** R1 gate failed in all three optimization seeds. The prior
200-update smoke did not provide positive evidence that the repair worked. It
only proved that the code ran and wrote artifacts; its heatmap-shape gate was
already failing. Describing that smoke as encouraging was an interpretation
error.

The audit found no evidence that the failure was caused by a target-index bug,
an inactive loss, a wrong checkpoint, the optional stride change, or lack of
basic CNN capacity. The evidence instead supports a learning-signal failure:
coordinate-only soft-argmax supervision permits malformed probability maps,
and the attempted post-softmax repair loses corrective gradient in the states
it must repair.

## Semantic lock for this audit

Must be true before blaming the objective:

1. the 5,000-update jobs used the intended standard 64-cell architecture;
2. the coordinate and dead-zone losses entered the same backward pass;
3. channel/target construction and evaluation used the same mapping;
4. the CNN could represent the targets under a stronger supervised control;
5. saved-checkpoint gradients reproduced the suspected failure mechanism.

Must not be claimed:

- a three-seed, one-object diagnostic establishes population generality;
- soft-argmax is universally unusable;
- the operator/keypoint hypothesis has been rejected;
- additional epochs can solve a vanished-gradient state without changing the
  training signal.

## Checks and results

### Code and configuration

- All three cluster jobs completed normally at 5,000 optimizer updates.
- Each run used `coordinate` supervision, `conditional_deadzone`, weight
  `8.723808430294485e-06`, 10 channels and the fixed frames 0, 3, 6 and 9.
- `native_quarter` was absent/false. Therefore the optional third-layer stride
  change was inactive; the legacy 64x64 path was used.
- The training loop computes the coordinate MSE and weighted dead-zone loss,
  calls one backward pass, and steps all extractor parameters.
- The target mapping is identical in training and evaluation. Transported
  targets pass the mask-grounding check.

### Positive controls

Dense Gaussian heatmap supervision on the same targets and architecture passed
all 10 channels in all three seeds within 700--900 updates. This rules out the
strong versions of “the targets are impossible” and “the CNN cannot represent
them.” It does not prove coordinate-only credit assignment is adequate.

The earlier channel/target permutation test also ruled out a simple fixed
numerical-channel or fixed-physical-target indexing defect.

### Saved-checkpoint gradient audit

The audit is reproducible with
`keypoint_net/diagnostics/stage_r1_deadzone_failure_audit.py`. Outputs are in
`keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_failure_audit/`.

- 13 channel/seed units failed the coordinate gate.
- 7/13 failed units had median maximum probability at least 0.99: effectively
  one-cell spikes.
- Other failed units were generally very narrow (effective support commonly
  about 1--2 cells), even when maximum probability was only 0.56--0.71.
- Median weighted-dead-zone/coordinate gradient-norm ratios across the three
  seeds were 0.180 at logits, 0.155 at the heatmap head and 0.104 at the
  backbone.
- Examples at the train-mode optimization path:
  - seed 43, channel 3: max probability 0.9992, coordinate gradient
    `2.69e-6`, repair gradient `1.57e-7`;
  - seed 44, channel 3: max probability 0.9996, coordinate gradient
    `9.60e-7`, repair gradient `2.97e-8`;
  - seed 42, channel 6: max probability 0.559, support 2.01 cells, coordinate
    gradient `1.55e-4`, repair gradient `8.75e-6`.

These measurements are descriptive. The unit of replication is the
optimization seed (`n=3`). The four frames are one correlated cyclic sequence;
no hypothesis test, confidence interval or population inference is reported.

## Why the attempted repair failed

For a spatial-softmax probability `p_i`, grid coordinate `x_i` and expected
coordinate `mu`, the soft-argmax derivative contains the factor
`p_i (x_i - mu)`. When a map saturates on the wrong cell, probabilities away
from that cell approach zero, so the coordinate loss supplies almost no signal
to move probability toward the target.

The repair was also computed from post-softmax quantities (maximum
probability, entropy/effective support and local probability mass). Those
gradients pass through the same softmax Jacobian and therefore attenuate in a
near-delta state. It cannot be assumed that a post-softmax penalty will
unsaturate a saturated softmax.

The frozen weight was calibrated once on the initial, highly diffuse CNN maps
by matching global logit-gradient norms. This had three limitations:

1. the relative gradient scale changed drastically as map shape changed;
2. a global norm did not ensure adequate force for every channel;
3. matching gradients at logits did not guarantee matched gradients through
   the shared head/backbone or matched Adam updates.

The synthetic R0 repair test optimized logits directly using the unweighted
shape loss alone. That proved the formula could reshape isolated prototype
logits. It did not test joint coordinate-plus-shape optimization through a
shared CNN with the calibrated weight. Passing R0 was therefore necessary but
not sufficient, and it was treated as stronger evidence than it was.

## Minor issue that is not causal

Historical diagnostics define one 64-cell unit as `2/64`; the exact spacing of
`torch.linspace(-1, 1, 64)` is `2/63`. This creates about a 1.6% reporting-scale
difference. It cannot explain errors of 1--5 cells or the zero-of-three gate,
but future metrics should use an explicitly named convention and avoid mixing
the two.

## Decision

Per the preregistered terminal rule, stop inventing additional heatmap-shape
losses for this implementation. The next work must compare instrument designs
whose useful gradient does not depend on escaping a saturated spatial softmax.

Before another cluster campaign, require a local smoke gate that jointly proves:

1. all channels fit the four-frame coordinate task in at least 2/3 seeds;
2. useful target-directed parameter gradients remain non-negligible throughout
   training, including deliberately saturated initial states;
3. the test operates through the full CNN, not direct free-logit optimization;
4. success is semantic (correct localized points), not merely valid files or a
   decreasing scalar loss.

This is a failure of the current training instrument, not evidence against the
research hypothesis that learned coordinates may make operators simpler.
