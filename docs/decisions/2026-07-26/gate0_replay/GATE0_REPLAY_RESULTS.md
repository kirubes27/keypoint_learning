# Gate 0 representative-replay results

## Decision-rule outcome

**Local/windowed readout loses rank.** Of the fixed 150 global-readout
high-error channel-frame pairs, 6 (4.0%) were correct-dominant-mode and
144 (96.0%) were wrong-or-diffuse-mode. The frozen rule requires at least
75/150 (50%) correct-dominant-mode pairs for windowing to keep rank.

Therefore the representative-pilot diagnosis routes to the diagnostic head
(Decision 2.3) and re-planning. This is a ranking result only: per v2.1
amendment 3, it is descriptive evidence from one seed and is not a pass/fail
or population-level claim.

## Frozen scope

- Seed: 41
- Object: `engineers_hammer_vray`
- Checkpoint epoch: 775 (frozen best-validation checkpoint)
- Evaluation data: all 60 unaugmented validation frames; test was untouched
- Channels: 10
- Sample unit: channel-frame pair
- Fixed full denominator: 60 frames x 10 channels = 600 pairs
- High-error denominator: 150 pairs
- Training or weight updates: none
- Replay device: CPU
- Heatmap resolution and temperature: 64 x 64, temperature 1.0
- Statistical scope: one object, one optimization seed, and one correlated
  cyclic validation orbit. All medians, p90 values, counts, and fractions are
  descriptive. No error bars, hypothesis tests, or population inference are
  reported.

## Exact paths used

| Role | Path |
|---|---|
| Frozen checkpoint | `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r2_representative_pilot_20260705_210253/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r2_representative_pilot/runs/coordinate_standard64_k10_seed41/best_model.pt` |
| Frozen run config and frame-0 targets | `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r2_representative_pilot_20260705_210253/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r2_representative_pilot/runs/coordinate_standard64_k10_seed41/config.json` |
| Archived validation metrics used for replay QA | `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r2_representative_pilot_20260705_210253/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r2_representative_pilot/runs/coordinate_standard64_k10_seed41/best_validation_metrics.json` |
| Current model implementation read and loaded | `/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/keypoint_net/model.py` |
| Supervised-target and dataset implementation | `/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/keypoint_net/diagnostics/day45_supervised_control.py` |
| Local representative data root | `/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/_tdw_world_z_roll_base_panel_512_v2` |
| Frozen validation split | `/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/_tdw_world_z_roll_base_panel_512_v2/indices/split_phase_mod6.json` |
| Validation RGB directory | `/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/_tdw_world_z_roll_base_panel_512_v2/train/engineers_hammer_vray/frames/a` |

The exact validation frame IDs were:

`[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178]`.

The corresponding exact RGB paths are the validation RGB directory above
with filenames `img_NNNN.png` for those zero-padded IDs. The complete
expanded list is also stored in
`gate0_replay_metrics.json` at `paths.validation_frame_paths`.

Supervised targets are not a separate label file. The frozen config records
the ten channel-ordered frame-0 pixel targets. The replay transported those
targets over the 180-frame roll with `transported_targets` from the exact
target-code path above, center
`(255.49998435893767, 255.50001568508694)`, and roll sign `+1`, then selected
the 60 frozen validation IDs. Regenerating the frame-0 targets from the local
frame-0 mask reproduced the frozen target array exactly; transported-target
on-mask grounding was 1.0 over all 180 frames.

## Replay script and hashes

- Script: `gate0_replay.py`
- Script SHA-256:
  `2c88604a900274f4ed7a1f41ca44c6f551f3c972124f95515113775b556e5f3b`
- Pair-level/aggregate output: `gate0_replay_metrics.json`
- Metrics JSON SHA-256:
  `7194c8a795b70b10c96b6cb4eada3da56e703eb7bdec60f1140166a6b85ab8c8`
- Checkpoint SHA-256:
  `d4777e3abfed3d81698ab07edf0563484b49ed314130aa8e925a0de2e1188c3d`
- Model-source SHA-256:
  `bc86a495dcb2ec982468fea7a7e2bfad8e0f2cac8f3a1e8c83a14976986fab83`
- Split SHA-256:
  `49f9d2a34c352d3ebb84809ec36e0a46572b0cde6b7a6d357f317dc44e3da486`

Execution command:

```sh
PYTHONDONTWRITEBYTECODE=1 python gate0_replay.py
```

## Frozen definitions and implementation

- Global readout: the current `model.py` softmax over each flattened heatmap,
  followed by expectation on the normalized `[-1, 1]` grid.
- Local/windowed readout: first row-major hard argmax; square radius
  `r in {2, 4, 8}`; edge clipping with no padding; temperature unchanged;
  softmax and renormalization only over the retained window; expectation on
  the retained grid.
- Cell argmax: normalized coordinate of the hard-argmax cell. It is reported
  only as the mode-correctness probe, not as a replayable coarse+offset head.
- Coordinate error: Euclidean normalized-coordinate error divided by `2/64`,
  matching the representative pilot's `CELL64_NORM`.
- High-error: global-readout coordinate error strictly above its 75th
  percentile. The observed threshold was **1.277964 cell64** (NumPy linear
  quantile); there were no threshold ties affecting the 150-pair denominator.
- Target cell: nearest cell on the 64 x 64 expectation grid. No target lay
  exactly on a half-cell quantization tie.
- Correct-dominant-mode: Euclidean distance between integer argmax and target
  cell indices at most 1 cell.
- Wrong-or-diffuse-mode: the logical complement of correct-dominant-mode
  within the fixed global high-error denominator. The amendments freeze no
  separate quantitative diffuse cutoff, so the replay does not invent one.

## Fixed-denominator strata

| Stratum | Count / 150 high-error pairs | Fraction of high-error denominator | Count / 600 all pairs |
|---|---:|---:|---:|
| Correct-dominant-mode high-error | 6 / 150 | 4.0% | 6 / 600 (1.0%) |
| Wrong-or-diffuse-mode high-error | 144 / 150 | 96.0% | 144 / 600 (24.0%) |
| Total fixed high-error denominator | 150 / 150 | 100.0% | 150 / 600 (25.0%) |

## Per-variant errors

Every value is in cell64 units. The high-error columns use the same 150-pair
mask defined once from the global readout; the mask is not recomputed for each
variant.

| Variant | All 600 median | All 600 p90 | Fixed high-error 150 median | Fixed high-error 150 p90 |
|---|---:|---:|---:|---:|
| Current global expectation | 0.6543 | 2.0654 | 1.9310 | 3.1211 |
| Local/windowed r=2 | 3.2261 | 12.9470 | 2.7728 | 8.2424 |
| Local/windowed r=4 (primary) | 2.8123 | 12.0862 | 2.7540 | 8.0804 |
| Local/windowed r=8 | 2.5150 | 10.4198 | 2.7371 | 7.2002 |
| Cell argmax probe only | 3.3399 | 13.2697 | 2.7350 | 8.0213 |

## Per-stratum errors by readout

The stratum membership remains fixed from the global high-error distribution
and the cell-argmax mode probe.

| Variant | Correct-dominant n=6 median | Correct-dominant n=6 p90 | Wrong-or-diffuse n=144 median | Wrong-or-diffuse n=144 p90 |
|---|---:|---:|---:|---:|
| Current global expectation | 1.3496 | 1.5182 | 1.9710 | 3.1526 |
| Local/windowed r=2 | 1.3469 | 1.5063 | 2.8603 | 8.5848 |
| Local/windowed r=4 (primary) | 1.3478 | 1.5143 | 2.8258 | 8.4491 |
| Local/windowed r=8 | 1.3668 | 2.0163 | 2.8274 | 7.3084 |
| Cell argmax probe only | 1.3276 | 1.3664 | 2.8014 | 8.5511 |

The local readouts do not materially improve the six correct-dominant
high-error pairs and substantially worsen the 144 wrong-or-diffuse pairs.
Across all 600 pairs, every tested window radius is also worse than the
current global expectation. These error comparisons are descriptive support
for the frozen ranking decision; they are not an additional gate.

## Verification

- The local split hash exactly matched the hash frozen in the checkpoint.
- All 180 RGB frames and masks loaded; the 60 replay IDs matched the checkpoint
  and split file exactly.
- The current global calculation matched the extractor forward result exactly
  (maximum absolute coordinate difference 0).
- The archived CUDA global baseline was reproduced on CPU within the
  prereplay 0.01-cell QA tolerance: median difference 0.003439 cell64, p90
  difference 0.000131 cell64, and maximum per-channel median difference
  0.001769 cell64.
- Explicit sliced-window calculations agreed with the vectorized
  implementation to maximum absolute coordinate differences of
  `2.38e-7`, `2.98e-7`, and `5.96e-7` for radii 2, 4, and 8.
- Model parameters were bitwise unchanged in memory, and the checkpoint hash
  was identical before and after replay.
- A second complete execution produced the same metrics JSON hash.
- The pair-level audit contains exactly 600 unique `(frame, channel)` records;
  the two high-error strata partition all 150 high-error records.

The amendments do not state whether "within 1 cell" uses Euclidean or
coordinate-wise/Chebyshev cell distance. The primary result above uses
Euclidean distance, consistent with the pilot's coordinate-error norm. As a
non-decision sensitivity check, the more inclusive nearest-cell Chebyshev
interpretation gives 23/150 (15.3%) correct-dominant high-error pairs, still
far below the 75/150 rank-retention threshold. Continuous-target Euclidean and
floor-cell Euclidean variants give 1/150 and 13/150, respectively. Thus this
unfrozen geometric detail cannot change the Gate 0 branch on these data.

## Closing interpretation

Gate 0 does not support the hypothesized representative failure mechanism
that local windowing is designed to repair. The dominant heatmap cell is
usually not target-local among the global-readout high-error pairs, and
windowing around that cell entrenches the wrong mode. Under the frozen v2.1
rule, local/windowed readout therefore loses rank and the next branch is the
Decision 2.3 diagnostic head, followed by re-planning based on that diagnostic
result.
