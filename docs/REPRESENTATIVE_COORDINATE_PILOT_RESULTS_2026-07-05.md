# Representative coordinate-instrument pilot results — 2026-07-05

## Verdict

**Not viable for the three-seed confirmation under the preregistered pilot
criteria.** The test split was not evaluated.

The pilot answers the immediate causal question: the catastrophic
four-frame softmax-saturation trap is **not** the active blocker in the
representative training regime. Nevertheless, the unchanged coordinate
instrument still plateaus above the required localization accuracy.

## Execution record

- Branch: `fitted-operator-diagnostics-20260704`
- Commit: `2690404`
- Cluster job: `53042746`
- Seed: 41
- Instrument: standard 64x64, K=10, plain coordinate MSE
- Shape repair: none
- Split: 60 train / 60 validation / 60 committed test frames
- Stopping: minimum 1,000, maximum 3,000, validation every 25, 1% relative
  improvement patience 400
- Best epoch: 775
- Completed epoch: 1,175
- Stop reason: genuine validation plateau
- Runtime: 384 seconds on one H100

## Best validation checkpoint

| Condition | Median-of-channel medians | P90 | On-mask |
|---|---:|---:|---:|
| Unaugmented | 0.502 cell64 | 2.066 cell64 | 0.987 |
| Fixed augmentation | 0.512 cell64 | 2.068 cell64 | 0.985 |

The median threshold (`<=0.50`) and p90 threshold (`<=1.50`) failed. On-mask
occupancy passed.

Unaugmented per-channel median errors were:

`[0.281, 2.177, 0.345, 1.811, 0.485, 0.518, 0.362, 1.182, 1.212, 0.407]`

Channels 1, 3, 7 and 8 remain the principal errors. This is not the same
failure set as the four-frame gate: channels 6 and 9 now localize adequately.

## Gradient-path result

At the best checkpoint:

- no channel had median maximum probability `>=0.99`;
- no inaccurate channel was saturated;
- no channel's counterfactual-gradient ratio fell below `0.01x` its initial
  value;
- the minimum channel ratio was `0.185x`;
- there were no saturated or collapsed-gradient channels at any inspected
  checkpoint from epoch 200 through the validation plateau.

Therefore data diversity does escape the tiny-regime saturation trap, as the
reviewer proposed. But escaping that trap is insufficient: multiple channels
remain inaccurate despite usable coordinate gradients.

Heatmap shapes remain heterogeneous (some narrow, some extremely diffuse), so
coordinate MSE still does not identify a consistent spatial representation.
The current evidence does not distinguish weak local observability, shared-CNN
interference and underconstrained heatmap shape as causes of the remaining
channel errors. It does rule out vanished coordinate gradients as the primary
explanation for this pilot.

## Decision

Do not launch the three-seed confirmation and do not declare the existing
instrument repaired. The next step is a controlled comparison of alternative
coordinate instruments on this same representative validation protocol. The
four-frame gate remains a diagnostic only and cannot veto future designs.

## Artifact location

The complete Mac archive, checkpoints, validation metrics, probe history and
logs are stored at:

`/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r2_representative_pilot_20260705_210253/`

The archive has a stored SHA-256 checksum. Results are descriptive for one
object and one optimization seed; no error bars, hypothesis test or population
inference is reported.
