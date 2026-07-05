# Conditional dead-zone fallback R0 result — 2026-07-05

## Decision

**Synthetic R0 PASS.** The three-seed fallback R1 may run. This is not an R1
or image-level scientific result.

The candidate is the final authorized heatmap-loss fallback. If the subsequent
R1 gate fails, stop this instrument design and review alternatives with Angela.

## Frozen candidate

The diagnostic loss is a normalized squared hinge on:

- maximum probability in `[0.08, 0.30]`;
- effective support in `[8, 32]` cells;
- mass within two cells of a detached dominant mode at least `0.70`.

Every term is exactly zero inside its healthy region. The seed-42 fixed-batch
gradient calibration produced:

- coordinate logit-gradient L2: `9.9960e-4`;
- raw dead-zone logit-gradient L2: `114.5833`;
- frozen weight: `8.723808430294485e-06`.

There was no weight sweep.

## Gate evidence

- Successful heatmap controls: all `120` correlated frame/channel units had
  exactly zero loss and zero logit gradient.
- Healthy translated Gaussians: minimum coordinate-descent multiplier `1.0`
  (required `>=0.90`).
- Active malformed cases: minimum multiplier `0.99959` (required `>=0.0`).
- Gradient reconstruction: maximum relative error `0.0` (required `<=1e-5`).
- Spike repaired and became silent in `82` logit updates.
- Diffuse map repaired and became silent in `1,460` updates.
- Exact uniform map repaired and became silent in `36` updates.
- Equal separated modes repaired and became silent in `7` updates.
- All losses and gradients were finite.

The first execution stopped without a verdict because the extreme spike's
coordinate gradient was below the numerical floor, making a descent ratio
undefined. The recorded validity clarification kept that exact spike for repair
and used a still-invalid `sigma=0.35` spike for compatibility. No threshold was
changed.

## Statistical scope

This is a deterministic synthetic mechanism gate. Positive controls contain
one object, four correlated frames and three optimization seeds; seed is the
replication unit for those controls. Prototypes are deterministic cases. No
hypothesis test, error bar or population inference is reported.

## Next

Run the frozen R1 tiny supervised coordinate gate on cluster GPU for seeds
42/43/44, at most 5,000 updates. R2 remains blocked until at least 2/3 seeds
jointly pass with no persistent physical-target failure.

## Artifacts

- Semantic lock: `docs/STAGE_R1_DEADZONE_FALLBACK_LOCK_2026-07-05.md`
- Candidate implementation: `keypoint_net/diagnostics/stage_a_shape_constraint.py`
- Synthetic gate: `keypoint_net/diagnostics/stage_r1_deadzone_r0.py`
- Raw result:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_r0/R0_DEADZONE_FALLBACK_GATE.json`

No core `model.py` or `train.py` file changed.
