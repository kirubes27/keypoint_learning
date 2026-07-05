# Stage R1 gradient-path audit result — 2026-07-05

## Decision

**The simultaneous prediction-centred JS shape loss is materially opposing
coordinate learning. Network-gradient attenuation is not supported.**

This is a descriptive one-object, four-correlated-frame checkpoint audit with
three optimization seeds per condition. Seed is the replication unit (`n=3`).
No hypothesis test or population inference is claimed.

## Meaning in plain language

The CNN-to-softmax coordinate signal is present. It does not disappear when it
passes into the heatmap head or backbone. The problem is that the added shape
loss pushes substantially against the coordinate loss while the coordinate
loss is trying to move a heatmap.

The prediction-centred Gaussian is rendered around the current coordinate with
its centre detached. During one backward pass, coordinate loss tries to deform
the current probability map so its mean moves toward the target, while shape
loss tries to restore a Gaussian around the old current mean. The measured
gradients therefore oppose each other.

This explains the R1 observation: heatmaps remained healthy, but localization
was much slower than the coordinate-only controls.

## Frozen gate results

The audit passed its numerical validity gate. Direct combined-loss gradients
and the sum of separately computed coordinate/shape gradients agreed to a
maximum relative L2 error of `4.55e-12` (required `<=1e-5`). The frozen weights
were evaluated in float64 to keep convolution reduction-order error below this
strict audit tolerance; checkpoint values themselves were not changed.

### Harmful cancellation fraction

This quantity is the fraction of coordinate-loss first-order descent removed
by the weighted shape gradient. The preregistered support threshold was `>=0.50`
in at least two of three repaired seeds at the head or backbone.

| Seed | Heatmap logits | Heatmap head | Backbone |
|---:|---:|---:|---:|
| 42 | 1.092 | 0.898 | 0.947 |
| 43 | 0.946 | 0.952 | 0.828 |
| 44 | 0.927 | 0.901 | 0.317 |

At the heatmap head, the shape term removed approximately `90--95%` of the
coordinate descent in all three seeds. At the backbone it removed `83--95%` in
two seeds. Seed 42's logit-level value above one means the combined local update
was predicted to slightly increase coordinate loss at that checkpoint.

**Loss-conflict verdict: supported** at the head (3/3) and backbone (2/3).

### Coordinate-gradient transmission versus control

Transmission gain is parameter-gradient norm divided by logit-gradient norm.
The table reports repaired gain divided by its matched coordinate-only control.
The preregistered attenuation threshold was `<=0.25` in at least two seeds.

| Seed | Heatmap head | Backbone |
|---:|---:|---:|
| 42 | 0.876 | 11.994 |
| 43 | 1.845 | 24.840 |
| 44 | 0.493 | 8.906 |

No seed met the attenuation threshold at either level. The head comparison is
mixed under the frozen rejection boundary, while backbone transmission was
larger rather than smaller in every repaired checkpoint.

**Network-attenuation verdict: mixed, not supported.**

## What is now established

- This is not evidence of a softmax implementation bug that chooses a different
  point from the CNN's heatmap.
- The coordinate loss produces a usable signal at the heatmap logits.
- The shared CNN path does not materially erase that signal relative to the
  controls under the frozen audit definition.
- The current simultaneous shape loss is the proximate cause of the severe R1
  slowdown and is rejected as the instrument repair.

This does not establish that coordinate-only soft-argmax training is reliable:
its controls still had persistent failed channels and malformed heatmaps. It
establishes why this particular attempted repair did not solve that problem.

## Next step

Do not extend R1 or tune the current shape weight. Replace the always-active
shape penalty with a candidate whose gradient is exactly zero for already
healthy heatmaps and activates only when a map becomes spiked, diffuse or
otherwise invalid. Before image training, require a synthetic translation-
compatibility gate:

1. on healthy one-cell Gaussian maps, at least `0.90` of coordinate first-order
   descent must remain under the combined objective;
2. spike, diffuse and separated-peak failures must still receive corrective
   gradients;
3. location translation must not change the penalty away from boundaries;
4. all symmetry/collapse cases must remain finite;
5. only after these pass may the same three-seed R1 tiny gate be repeated under
   a new experiment ID.

The exact candidate and thresholds must be frozen in a new semantic lock before
implementation. R2 and Stage B remain blocked.

## Artifacts

- Lock: `docs/STAGE_R1_GRADIENT_PATH_AUDIT_LOCK_2026-07-05.md`
- Script: `keypoint_net/diagnostics/stage_r1_gradient_path_audit.py`
- Tests: `keypoint_net/diagnostics/test_stage_r1_gradient_path_audit.py`
- Summary:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_gradient_path_audit/R1_GRADIENT_PATH_SUMMARY.json`
- Row table:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_gradient_path_audit/R1_GRADIENT_PATH_ROWS.csv`

No core model or training file was changed.
