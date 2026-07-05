# Stage R1 Diagnostic Integration and Meaning Smoke (2026-07-05)

## Status

Implementation and smoke gate: **PASS**.

Scientific three-seed R1 gate: **not run**.

The frozen Stage-R0 shape constraint is integrated only into the supervised
diagnostic trainer. Default commands retain the exact legacy objective and run
names. `model.py`, `train.py` and shared Phase-A losses remain unchanged.

## Added behavior

When explicitly enabled, the diagnostic objective is:

`L_total = L_coordinate + 1.2128231385721024 * L_shape`

with the frozen one-cell, detached-centre Gaussian JS shape loss.

The R1 joint gate now records:

- original coordinate A0 thresholds;
- per-channel median maximum probability;
- per-channel median effective support;
- a counterfactual coordinate-gradient probe for half-cell moves in `+x`,
  `-x`, `+y` and `-y`;
- final/initial gradient ratios per channel and run;
- explicit joint pass/fail and authoritative/smoke-only scope.

## Verification

- 29 relevant diagnostic tests pass.
- Default no-shape objective is tested as exactly equal to the legacy loss.
- Shape-enabled run names include `shapejs`; old names are unchanged.
- Cluster script syntax and Mac collector syntax pass.
- Aggregation refuses shortened/non-authoritative artifacts.
- Aggregate R1 pass requires 2/3 joint seed passes and no physical target
  failure recurring in at least 2/3 seeds.

## Local smoke

The smoke used seed 42, CPU, 200 updates and the exact frozen R0 shape recipe.
It is explicitly stored as `implementation_smoke_only`; it cannot be aggregated
as an R1 result.

At update 100:

- every channel entered the frozen healthy-shape ranges;
- the counterfactual-gradient gate passed;
- the run median gradient ratio was `0.695`.

At update 200:

- shape gate: PASS;
- counterfactual-gradient gate: PASS;
- run median gradient ratio: `0.670`;
- per-channel maximum probabilities: `0.140--0.242`;
- per-channel effective support: `10.34--15.08` cells;
- coordinate gate: not passed, as expected for a deliberately shortened smoke.

Runtime was 43.23 seconds. Linear extrapolation gives approximately 18 minutes
per 5,000-update CPU seed and approximately 54 minutes for three sequential
seeds. This is close enough to the one-hour boundary that the prepared cluster
array is preferred for the authoritative group.

Smoke artifact:

`/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_smoke_v2_20260705/tiny_overfit/coordinate_standard64_k10_shapejs_seed42/metrics.json`

SHA-256:

`7ebf9c599e1a8078616866439ddf8d3483eeb666da1dc2a7466122b9be6affd6`

## Prepared authoritative run

- Cluster array: `cluster/stage_r1_shape_gate.slurm`.
- Seeds: 42, 43, 44; three concurrent GPU tasks.
- Maximum updates: 5,000.
- Aggregator: `keypoint_net/diagnostics/summarize_stage_r1_shape_gate.py`.
- Mac collector: `cluster/fetch_stage_r1_shape_gate_to_mac.sh`.
- Cluster output:
  `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_gate/`.
- Mac destination:
  `PhD/cluster_downloads/stage_r1_shape_gate_<timestamp>/`.

No cluster job has been submitted. The next action is to push the frozen branch,
submit the three-seed array, and use the collector after completion.

