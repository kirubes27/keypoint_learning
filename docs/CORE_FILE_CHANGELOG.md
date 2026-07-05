# Core File Change Log

Purpose: permanently record every modification to training-critical files on
`fitted-operator-diagnostics-20260704`. Update this file in the same commit as
any future core-file change.

Core files include model architecture, losses, training, datasets and target
generation. Each entry must state the old behavior, new behavior, default
behavior, checkpoint/config implications and commit.

## 2026-07-04 — configurable heatmap head

- **Files:** `keypoint_net/model.py`, `keypoint_net/train.py`
- **Commit:** `d3bc471`
- **Before:** `KeypointExtractor` always produced native 64x64 heatmaps from
  `/8` encoder features for 512x512 inputs. `train.py` had no heatmap-resolution
  argument.
- **After:** added `heatmap_res` with choices 64 or 128. At 128, `/8` features
  are bilinearly upsampled and passed through a new 3x3 convolution before the
  heatmap head. `train.py` exposes `--heatmap_res` and stores it in config.
- **Default behavior:** `heatmap_res=64`; unchanged from Phase A.
- **Compatibility:** old commands and old checkpoints retain the 64x64
  architecture. A 128-head checkpoint has additional parameters and must be
  reconstructed with `heatmap_res=128`.
- **Training risk:** using `--heatmap_res 128` is an architecture change; runs
  must not be compared to 64-head runs without labeling it.

## 2026-07-04 — native `/4` diagnostic architecture

- **File:** `keypoint_net/model.py`
- **Commit:** `d0f66f0`
- **Before:** the only 128x128 option upsampled already-computed `/8` features.
- **After:** added `true_quarter_res`. When true, encoder layer 3 uses stride 1,
  producing native `/4` features and a 128x128 heatmap without the upsampling
  block. It requires `heatmap_res=128`.
- **Default behavior:** `true_quarter_res=False`; unchanged from Phase A and
  unchanged for every normal `train.py` command.
- **Compatibility:** this flag is currently used only by
  `diagnostics/stage_a_supervised_control.py`; it is not exposed by `train.py`.
  Therefore ordinary new training runs cannot enable it accidentally.
- **Training risk:** checkpoints produced with native `/4` are not architecture-
  compatible with standard 64 or upsampled-128 checkpoints.

## 2026-07-05 — variable-K diagnostic support

- **Training-critical core files:** none.
- **Commit:** `a47b60b`
- **Diagnostics changed:** Stage A now accepts `--num-keypoints`; the K sweep
  uses K in {5,10,15,20}. `day45_supervised_control.py` cycles visualization
  colors when K > 10.
- **Default training behavior:** unchanged. `model.py` and `train.py` were not
  modified in this commit.

## 2026-07-05 — target/readout attribution diagnostic

- **Training-critical core files:** none.
- **Diagnostics changed:** Stage A accepts a recorded target-channel cyclic
  shift and either coordinate-MSE or Gaussian target-heatmap supervision. A
  six-task cluster array and attribution summarizer were added.
- **Default behavior:** remains coordinate supervision, identity assignment
  (`target_shift=0`) and K=10. Normal `train.py`, `model.py`, losses and Phase-A
  training behavior are unchanged.
- **Purpose:** distinguish physical target difficulty, numerical channel
  identity and coordinate-only soft-argmax gradient failure.

## Required future entry format

1. Date and purpose.
2. Exact files and commit.
3. Before -> after behavior.
4. Whether defaults changed.
5. Checkpoint/config compatibility.
6. Consequence for old and new training runs.
