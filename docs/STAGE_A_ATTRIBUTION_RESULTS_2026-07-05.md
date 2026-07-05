# Stage-A Target/Channel/Gradient Attribution — Results (2026-07-05)

## Decision

The standard 64x64 CNN can represent and localize all ten supervised targets.
The failed A0 control is caused by unreliable optimization through the
coordinate-MSE -> soft-argmax path, not by insufficient heatmap resolution, an
intrinsically unobservable hammer location, or fixed numerical channels 3/6/9.

This supports the preregistered **coordinate-only soft-argmax gradient
bottleneck** branch. Stage B remains blocked: its real losses also reach the
heatmaps through predicted coordinates, whereas the successful Gaussian
heatmap target used privileged supervised information unavailable to Stage B.
The heatmap result therefore diagnoses the problem but is not itself a valid
replacement training objective for the unsupervised experiment.

## Artifact verification

- Mac artifact:
  `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_a_attribution_20260705_094134`
- Expected and observed new runs: 6/6.
- Scheduler error logs: 0/6 non-empty (`.err` files are empty; 12 total
  scheduler log files include stdout and stderr).
- Archive SHA-256:
  `2d9ddbdbd0844dff8594041dcf27bd629b17607078c81372c478aaecdf74bad6`.
- Machine-readable results:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_attribution/A0_ATTRIBUTION_SUMMARY.json`
  and `A0_ATTRIBUTION_RUNS.csv` inside the Mac artifact.

## Frozen design

- One hammer object and four fixed frames (0, 3, 6, 9).
- K=10, standard 64x64 path, `true_quarter_res=False`.
- Three optimization seeds per condition: 42, 43 and 44.
- Coordinate conditions: 5,000 updates.
- Heatmap condition: same architecture and unchanged soft-argmax coordinate
  evaluation, trained with Gaussian target-heatmap cross-entropy.
- A0 pass: median error <=0.10 cell64 and every channel <=0.20 cell64.

## Results

| Condition | Passed seeds | Persistent failures across 3 seeds | Interpretation |
|---|---:|---|---|
| Coordinate MSE, identity assignment | 0/3 | numerical channels/targets 3, 6, 9 | Original A0 failure reproduced |
| Coordinate MSE, target shift +1 | 0/3 | channels 0, 2, 5; physical targets 1, 3, 6 | Failure does not stay with channels 3, 6, 9; targets 3 and 6 remain difficult, target 9 is rescued |
| Gaussian heatmap supervision, identity | 3/3 | none | Every target is representable; same soft-argmax evaluation passes |

Heatmap-supervised per-seed outcomes:

| Seed | Pass step | Median error (cell64) | Worst-channel error (cell64) |
|---:|---:|---:|---:|
| 42 | 900 | 0.0662 | 0.1927 |
| 43 | 900 | 0.0566 | 0.1422 |
| 44 | 700 | 0.0899 | 0.1892 |

The target permutation is not a pure “all failures follow targets 3/6/9”
outcome: target 9 becomes learnable and target 1 becomes persistently difficult.
This is an interaction between target/location and coordinate optimization, not
evidence of an immutable bad location. Crucially, original numerical channels
3, 6 and 9 all become valid when assigned different targets, rejecting a fixed
channel/head-index bug.

## Claims supported and rejected

Supported within this diagnostic:

1. Dense spatial supervision can train the unchanged standard-64 architecture
   to sub-cell accuracy on every target in all three seeds.
2. Coordinate-only training enters target-dependent bad optimization basins;
   failures move when targets are reassigned and remain after 5,000 updates.
3. The relevant distinction is the training gradient. The forward soft-argmax
   readout itself is retained during successful heatmap-trained evaluation.

Rejected within this diagnostic:

- fixed defective output channels 3/6/9;
- 64x64 quantization as the cause of 2--5-cell errors;
- the claim that the CNN/receptive field cannot represent the failed targets;
- keypoint count as the primary cause (from the preceding K sweep).

Not established:

- that the same failure explains every defect in the original unsupervised
  Task-80 checkpoint;
- semantic-part discovery, multi-object generalization or population behavior;
- which unsupervised-compatible repair will prevent coordinate-gradient traps.

## Statistical scope

This is a descriptive, one-object/four-correlated-frame optimization diagnostic.
The replication unit is optimization seed, n=3 per condition. All seed outcomes
are shown; no error bars or hypothesis tests are used, and no population-level
inference is claimed.

## Next bounded step

Do not launch Stage B or a resolution experiment. First run a local gradient
audit on the same frozen four-frame problem, comparing coordinate MSE with
heatmap supervision at initialization and after wrong-peak formation. Record,
per channel and seed: target-cell probability, argmax-to-target distance,
coordinate-loss gradient norm, gradient magnitude near the target, and fraction
of gradient mass near the current wrong peak. This directly tests the proposed
probability-weighted gradient-starvation mechanism.

After that audit, preregister exactly one unsupervised-compatible repair to the
coordinate path and require the original coordinate A0 gate to pass in at least
2/3 seeds before Stage B resumes. The heatmap oracle may be retained as a
positive control but may not replace the coordinate gate.

