# OCR-ZNCC Stage 0 semantic and decision lock

Date frozen: 2026-08-09, before opening any OCR-ZNCC direction result.

## Decision

Decide whether a minimal paired-RGB correspondence anchor is semantically
capable of pointing frozen shared-affine keypoints toward the known material
correction in Task 55 and Task 80, and, only if both recipe gates pass, choose
one non-arbitrary auxiliary coefficient without training.

## Current claim and critical unknown

The shared affine operator appears to predict approximate roll motion, but the
exact learned descriptor-attachment objective did not identify material
transport: canonical material drift worsened in all six paired runs and its
local coordinate-descent audit passed zero of twelve checkpoints.

The critical unknown is whether Operator-Centred Reciprocal ZNCC Transport
(OCR-ZNCC) can use only paired RGB evidence inside the detached operator basin
to find a correction with usable coverage and the correct material direction.

## Must be true

- A fixed 7x7 RGB patch is sampled at the detached source keypoint on the
  retained 64x64 grid.
- A fixed radius-eight-cell target search is centred on the detached shared
  operator prediction.
- Matching uses whole-RGB ZNCC after per-channel spatial centring.
- The target match is accepted only when patch evidence, peak score, peak
  margin, non-boundary, and reciprocal checks pass.
- Source coordinate, operator centre, RGB evidence, correspondence target,
  confidence, and all acceptance decisions are detached.
- Only the target-frame keypoint/heatmap path receives direct auxiliary
  gradient.
- The SmoothL1 residual is normalized by the eight-cell search radius and the
  batch loss by accepted match count.
- A zero accepted-match batch returns finite differentiable zero.
- Other keypoint channels never act as positives or negatives.
- The complete 147-pair hammer training split is used for each frozen Task-55
  and Task-80 control seed 42, 43, and 44.
- The material oracle is used only for evaluation.
- No optimizer or scheduler step occurs.

## Must not happen

- No GPU job, 1,000-epoch run, 5,000-epoch run, or training run starts.
- No Task-80 training, held-out object, translation, scale, yaw, or pitch data
  is opened.
- No mask, known transform, renderer material ID, reconstruction objective,
  optical-flow teacher, tracker, pretrained feature teacher, or learned
  descriptor field is consumed by OCR-ZNCC.
- No auxiliary gradient reaches the shared operator, inverse operator, action
  head, or source keypoint path.
- No raw lambda sweep or componentwise base-loss gradient decomposition occurs.
- Main and all existing worktrees remain unchanged.
- The protected untracked `NUMERIC_CALIBRATION.json` is not read or touched.

## Frozen OCR-ZNCC engineering configuration

- Retained grid: 64x64, endpoint-aligned xy coordinates in [-1, 1].
- Patch: 7x7 RGB cells.
- Target and reciprocal search radius: 8 cells.
- Minimum patch RMS after per-channel spatial centring: 0.02.
- Minimum best ZNCC: 0.80.
- Minimum best-minus-second-best ZNCC margin: 0.03.
- Reciprocal return tolerance: 1.0 grid cell.
- A match on the search-window boundary abstains.

These are preregistered engineering thresholds for this bounded diagnostic.
They are not established scientific constants and will not be changed after
viewing the result.

## Frozen material-direction gate

For each recipe and seed, evaluate only channels that the bound primary
evaluator classified as motion-active and on-object. A row is usable only when
its match is accepted and both material-correction and OCR-ZNCC descent vectors
have nonzero finite norm.

A seed passes only when all conditions hold:

- usable-direction coverage is at least 50% of eligible pair-channel rows;
- median material-direction cosine is strictly positive;
- median one-heatmap-cell material-error change is strictly negative; and
- the number of originally eligible channels with at least 75% target-mask
  occupancy is non-worse after the one-cell intervention.

A recipe passes when at least two of its three seeds pass. The 50% coverage
floor, 75% on-object classification floor, and two-of-three aggregation are
preregistered engineering choices, not established evidence.

## Calibration gate and meaning

Only if both Task-55 and Task-80 recipe gates pass, run one combined six-cell
calibration at initializations 42, 43, and 44 over the complete training split.
For each cell, compare:

- the global L2 norm of the full-split OCR-ZNCC gradient; and
- the global L2 norm of the full recipe base-loss gradient

over all trainable extractor encoder and heatmap-head parameters. The base loss
is not decomposed. The comparison measures relative update-signal size, not
gradient agreement or a training outcome.

Choose one shared coefficient:

`min(0.5, 0.10 / median(six auxiliary-to-base gradient ratios))`.

The target is an approximately 10% median contribution. The value 0.5 is only
a coefficient safety cap, never a second arm. Any missing, zero, or non-finite
required gradient, zero accepted calibration coverage, or provenance mismatch
invalidates calibration.

## Frozen checkpoint and selector provenance

The six frozen controls are the completed `task55_clean__r64__seed{42,43,44}`
and `task80_assisted__r64__seed{42,43,44}` cells under the downloaded fresh
primary matrix. Their receipts bind:

- source branch `agent/representation-oracles-20260726`;
- source commit `196245fdc8b6d65a5348d1addebcfc3c58ddb3d6`;
- the exact 147-pair training split and dataset binding; and
- historical checkpoint selection by minimum total validation loss.

Stage 0 does not rewrite or reselect those frozen controls. For every future
auxiliary experiment, checkpoint selection is frozen to minimum base validation
loss, because minimum total validation loss would allow the intervention to
select itself.

## Evidence required

- Exact experiment worktree branch, commit, source-file hashes, and clean
  tracked status.
- Receipt/config/history/checkpoint hashes and strict checkpoint load for all
  six controls.
- Exact split and dataset binding, object, recipe, seed, and evaluator-eligible
  channel provenance.
- Raw pair-channel correspondence rows and per-seed summaries.
- Per-recipe pass decisions under the frozen rule.
- If authorized by both gates, per-cell full-split gradient norms, Task-55 and
  Task-80 comparative summaries, and the one shared coefficient.
- One independent Fable High review of the actual code and raw Stage-0 evidence,
  followed by claim-by-claim verification against primary artifacts.

## Stop boundary

Report and stop after Stage 0. A pass is evidence that the local auxiliary has
a usable material-correction direction under these frozen roll controls; it is
not evidence that training will improve drift. A failure blocks training and
does not authorize threshold changes, a larger search, another patch scale, a
teacher, another transform, or another object.
