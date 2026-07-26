# Gate 0 representative-replay lock — 2026-07-26

Status: **historical pre-execution draft; not execution authority**.

Gate 0 was completed before this branch was prepared. The canonical executed
protocol and result are the byte-for-byte snapshot under
`gate0_replay/` beside this file. That result observed 6/150 (4.0%)
correct-dominant-mode high-error pairs, so local/windowed readout is demoted
under the frozen 50% rule. Do not rerun Gate 0 from this draft. Any extension
requires a changelog line in `DECISION_SYNTHESIS_v2_2026-07-26.md`.

## Question

Do the high-error predictions from the frozen representative roll checkpoint
usually have the correct dominant heatmap mode, such that a local/windowed
expectation is mechanistically matched to the observed failure?

## Bound inputs

- local dataset:
  `/Users/kirubeso.r/Documents/PhD/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2`
- dataset basename: `_tdw_world_z_roll_base_panel_512_v2`
- split file: `indices/split_phase_mod6.json`
- split SHA-256:
  `49f9d2a34c352d3ebb84809ec36e0a46572b0cde6b7a6d357f317dc44e3da486`
- object: `engineers_hammer_vray`
- checkpoint seed: 41
- checkpoint SHA-256:
  `d4777e3abfed3d81698ab07edf0563484b49ed314130aa8e925a0de2e1188c3d`
- checkpoint architecture: standard 64x64, K=10
- primary cohort: the 60 unaugmented validation frames
- sample unit: validation frame x supervised channel, `n = 60 x 10 = 600`
- fixed-augmentation cohort: sensitivity analysis only
- device: CPU is authoritative; no optimizer is constructed

The script must fail before inference if the dataset basename, 180-frame
partition, split hash, checkpoint hash, object, seed, architecture, or K differs.

## Readouts replayed from the same frozen logits

1. current global spatial softmax expectation;
2. heatmap argmax cell, as a mode-correctness probe only;
3. argmax-centred square-window expectations with integer radii
   `r = {1, 2, 4, 8}` cells.

For a windowed expectation, logits outside the clipped square window are
excluded, the remaining logits are softmax-normalized at temperature 1, and
coordinates use the production `torch.linspace(-1, 1, H/W)` grid.

Ties use PyTorch's deterministic flattened `argmax` ordering. Windows clip at
boundaries and are never padded or wrapped. Gate 0 is a replay, so no
straight-through or other surrogate gradient is involved.

## Frozen definitions

- Existing metric convention:
  coordinate error in "64-grid cells" is Euclidean normalized-coordinate error
  divided by `2/64`.
- Heatmap target-cell coordinate:
  `((target_norm + 1) / 2) * (size - 1)`, kept continuous.
- Correct dominant mode:
  Euclidean distance between the integer argmax cell centre and the continuous
  target-cell coordinate is at most 1.0 cell.
- High error:
  global-readout error strictly above the empirical 75th percentile over the
  600 primary pairs. The threshold, count, and tie count must be reported.
- Primary Gate 0 fraction:
  correct-dominant high-error pairs divided by all high-error pairs.
- Concentration descriptors:
  max softmax probability, effective support `1 / sum(p^2)`, and probability
  mass within radius 2 of the dominant mode. These remain continuous; no
  unsupported "diffuse" threshold controls the decision.

## Decision

- fraction at least 0.50: local/windowed readout keeps first rank and authorizes
  writing `READOUT_SPEC_v1.md`;
- fraction below 0.50: local/windowed readout loses first rank; do not retrain
  it as the primary repair. Route to the diagnostic-head/observability branch
  and write a new lock.

Gate 0 is one checkpoint, one object, and one correlated orbit. It ranks a
mechanism; it is not a capability pass and supports no population inference.

## Required outputs

- `GATE0_REPLAY_RESULTS.json`
- `GATE0_REPLAY_PAIRS.csv`
- `GATE0_REPLAY_RESULTS.md`
- a top-error visual panel showing image, target, global prediction, dominant
  cell, and each windowed prediction

Every output must echo the resolved dataset path, split path/hash, checkpoint
path/hash, object, seed, frame IDs, Git commit, runtime environment, sample
unit, and descriptive-only scope.
