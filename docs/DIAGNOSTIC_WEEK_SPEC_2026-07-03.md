# Diagnostic Week — Implementation Specification (v1.2, 2026-07-03)

v1.2 implementation-lock corrections: distinct active channels are counted
as connected components of the persistent duplicate graph; heatmap statistics
use the configured temperature; the noise audit includes cross-channel
dependence; similarity optimization fixes healthy anchors and evaluates on a
held-out noise bank; supervised controls use spatial augmentations; cluster
work requires the Days 1-2 review point and explicit user authorization.

v1.1 corrections (external review): rotation-centre estimation replaces the
invalid silhouette-centroid gate; baseline arm pinned to the EXACT smoke
config (operator_type=dense — verified in runs_res_smoke config.json, NOT
shared_affine); noise ladder gains remove-one-from-full ablations; similarity
separation criterion completed (boundedness, all-pairs distance, healthy-point
bias, loss decrease) + lr stability pre-check; aliasing probe multi-frame with
control reported as a measurement floor (never subtracted); activity metric
fixed-point exemption; numerical oracle-success thresholds calibrated from
control runs and frozen pre-matrix; loss-weight gradient-scale calibration;
staged execution order + runtime/cluster policy.

Audience: a coding agent with NO access to prior conversations. Everything needed is in this file. Execute top to bottom. The goal is DIAGNOSIS, not fixing: by Jul 10 we must be able to name the dominant cause of bad keypoints among {geometry mismatch, architecture localization limit, soft-argmax/multimodality, channel switching, input aliasing, objective/loss conflict, proxy-signal weakness} with evidence.

## 0. Context (read once)

Project: unsupervised keypoints via transformation prediction. A trained CNN
(`KeypointExtractor` in `keypoint_net/model.py`) maps a 512x512 image to 10
heatmaps -> soft-argmax -> 10 (x,y) coords in [-1,1]. Dataset: 180 frames of
a hammer rotating 2 deg/frame in the image plane (world-Z roll), cyclic
(frame 179 -> 0). Known problems measured so far: per-channel localization
jitter 0.66-1.56 heatmap cells (mean ~0.97), one dead channel (kp8 in the
Task-80 checkpoint), heavy duplication in fresh 300-epoch runs, and a
coordinate-level result (Block 0 / E7b, in `keypoint_net/block0/`) showing
that fitted-operator consistency losses drown in this noise for the affine
family. This week determines WHERE the noise and duplication come from.

## 1. Environment facts (verified 2026-07-03)

- Python: `/opt/anaconda3/envs/phd/bin/python` (torch 2.10, PIL, numpy,
  scipy 1.15.3). NO pytest (write assert-based runners executed via
  `python file.py`). NO matplotlib in this env (use PIL for images, CSV for
  numbers). MPS is available for training; `train.py` auto-selects device.
- Repo: `/Users/kirubeso.r/Documents/PhD/keypoint_net/` (NOT a git repo).
- Put ALL new code in `keypoint_net/diagnostics/`, outputs in
  `keypoint_net/diagnostics/outputs/`, training runs in
  `keypoint_net/diagnostics/runs_oracle/`.
- `keypoint_net/block0/` is a frozen gate artifact: IMPORT from it freely,
  NEVER modify it. Reuse: `block0/compare_res_smoke.py::load_run,
  trajectories` (checkpoint loading), `block0/block0.py::fit_similarity`
  (regularized similarity solver), `block0/extract_empirical_jitter.py`
  (high-pass jitter pipeline), `outputs/empirical_jitter_residuals.npz`
  (real residuals (180,10,2), normalized coords).
- Dataset root (parent dir has a TRAILING SPACE — always quote):
  `/Users/kirubeso.r/Documents/PhD/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2`
  Hammer object dir: `train/engineers_hammer_vray/` containing
  `frames/a/img_%04d.png` (180, 512x512 RGB) and
  `masks/a/mask_%04d.png` (180, 512x512 uint8, binary {0,255}, object
  ~7.9% of pixels). `id_passes/a/` also exists. `meta.jsonl` in the object
  dir has per-frame angles.
- Checkpoints to diagnose (all seed 42, hammer):
  - `task80` (1000 epochs, full losses):
    `/Users/kirubeso.r/Documents/PhD/cluster_downloads/hammer_full360_shared_complete/keypoint_net/runs_hammer_full360_shared/phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_pid3289780/`
  - `smoke64`, `smoke128` (300 epochs, disp=0.1 ent=0.01 only): the two dirs
    under `keypoint_net/runs_res_smoke/` — identify by `config.json` key
    `heatmap_res` (64 vs 128); do not hardcode dir names.
  - Loading: `config.json` is authoritative (the checkpoint-embedded config
    dict LACKS `heatmap_res`); build `KeypointExtractor(in_channels=3,
    num_keypoints, base_channels, temperature, padding_mode, heatmap_res)`
    from config.json, load `model_state_dict` entries with prefix
    `extractor.` stripped, `strict=True`, call `.eval()`.
- Preprocessing (must match training): PIL RGB, resize (512,512), /255,
  normalize mean [0.485,0.456,0.406] std [0.229,0.224,0.225].
- Coordinate convention (from `model.py::spatial_softmax`): outputs (x,y)
  in [-1,1]; x indexes WIDTH (columns), y indexes HEIGHT (rows).
  Pixel mapping: col = (x+1)/2*(W-1), row = (y+1)/2*(H-1).
  One 64-res heatmap cell: CELL64 = 2/64 = 0.03125 normalized = 8 px.
- Fixed seeds everywhere; every script prints a summary table AND writes
  CSV; every claim about keypoint placement must be backed by a saved
  overlay PNG (PIL: colored circles + channel index on the frame).

## 2. Standing rules

1. Report PER CHANNEL, never only model averages.
2. Preregister pass/fail thresholds in each script's docstring BEFORE
   running (they are given below — copy them verbatim).
3. Oracle/privileged signals (true masks, true rotation) are DIAGNOSTIC
   ONLY and must be labeled `oracle_` in code, CSV columns and filenames.
4. No new losses enter any "science" training. Days 6-7 arms are diagnosis.
5. If a precondition assert fails, STOP and report; do not work around.

## 3. Step 0 — shared utilities: `diagnostics/dxutils.py` (build first)

Functions (with tests in `diagnostics/test_dxutils.py`, assert-runner):

1. `load_run(run_dir) -> (extractor, cfg)` — as described in section 1.
2. `trajectories(extractor) -> np.ndarray (180,10,2)` — batched forward
   over the 180 frames (batch 12, `torch.no_grad`), float32.
3. `load_masks() -> np.ndarray bool (180,512,512)`;
   assert 180 files, assert binary.
4. **Rotation model with built-in sanity gate** (CRITICAL — every
   equivariance number below depends on it):
   - NOTE: the silhouette centroid of an ASYMMETRIC object legitimately
     traces a circle around the true rotation centre while the object
     rotates — centroid motion is NOT an error signal. Do not gate on it.
   - Centre estimation: initialize `c0 = mean_t(mask centroid_t)` (over the
     FULL 360-deg cyclic orbit the mean of a circle of points is its
     centre, so this is a valid initializer here) and sanity-compare to the
     image centre (255.5, 255.5). Then REFINE: optimize c (and pick sign s
     in {+1,-1}) to maximize mask-transport IoU over hops
     k in {1,3,15,45,90}. This prevents a one-pixel centre error from
     looking harmless at 2 degrees while accumulating over the orbit. Use a
     coarse 5-px grid search around c0 followed by 1-px local refinement.
     The reported G0 IoU remains the full t->t+1 score over t in {0..29}.
   - `transport(p_norm, k_frames)` = rotate points by chosen sign * 2k deg
     about the refined c (convert to px, rotate, convert back).
   - **GATE G0 (preregistered, tests stability of the ESTIMATED rotation
     centre, not the silhouette centroid): (i) mean transport-IoU at the
     refined c >= 0.95; (ii) c estimated on even frame-pairs vs odd
     frame-pairs agrees within 3 px.** If violated: print `GEOMETRY FLAG`,
     still continue Day 1 but mark every equivariance metric
     `geometry_suspect`, and prioritize Day 3.
5. `overlay(frame_idx, coords_px, path)` — PIL render, 10 distinct colors.
6. Metric helpers: `to_px`, `cell64`, Hungarian matching wrapper around
   `scipy.optimize.linear_sum_assignment`.

Tests: transport of the mask by 90 frames (180 deg) should approximately
match mask_{t+90}; transport is its own inverse composed with -k; loader
reproduces the known result `jitter median ~0.89-0.97 cells` on task80 via
the block0 high-pass pipeline (import it).

## 4. Day 1 — per-channel evaluation suite: `diagnostics/day1_channel_suite.py`

For EACH of {task80, smoke64, smoke128}, one CSV row per channel
(`outputs/day1_channels_<model>.csv`) with:

- `on_mask_frac`: fraction of 180 frames whose keypoint pixel lies in the
  TRUE per-frame mask; `on_mask_frac_dilated` with 8-px dilation
  (max-pool). (This replaces the useless sequence-union motion mask.)
- `eq_err_1`, `eq_err_3`: median over t of |p_{t+1} - transport(p_t, 1)|
  and |p_{t+3} - transport(p_t, 3)| in cells64; also `eq_err_1_p90`.
- `nn_dist_median`: median over frames of distance to nearest other
  channel (cells64); `dup_partner`: channel index that is the nearest
  neighbour in >50% of frames, else -1; `dup_flag`: nn_dist_median < 1.
- `activity_ratio`: median |p_{t+1}-p_t| / median |transport(p_t,1)-p_t|;
  `static_flag`: activity_ratio < 0.2 **AND expected motion is meaningful:
  median |transport(p_t,1)-p_t| > 0.5 cells64** (fixed-point exemption — a
  legitimate keypoint near the rotation centre barely moves and must NOT be
  flagged static; both numerator and denominator approach zero there).
- Heatmap stats (median over frames, computed from the raw heatmaps the
  extractor returns using `softmax(logits / configured_temperature)`):
  `hm_entropy` (nats over that probability map), `hm_std_px`
  (spatial std of softmax), `hm_maxprob`, `hm_peak_ratio` (max softmax /
  max outside a 15-px radius of the argmax), `hm_mass_in_mask` (softmax
  mass inside the true mask).
- `dead_flag`: median entropy > log(H*W)-0.5 AND median max probability <
  2/(H*W). This distinguishes a near-uniform heatmap parked at soft-argmax
  centre from a legitimate sharply localized fixed point.
- Per-axis noise: `sig_x_cells`, `sig_y_cells` from the block0 high-pass
  pipeline (feeds Day 2.5 anisotropy).

Model-level summary (`outputs/day1_summary.csv`): means/medians of the
above + `coverage`: fraction of mask pixels within 24 px of any keypoint
(median over frames) + `n_distinct_active_channels`. Compute the latter as
the number of connected components in the persistent duplicate graph after
removing static OR dead channels: vertices are informative non-static
channels; add edge (i,j)
when their median pairwise distance over frames is < 1 cell64. Do not
subtract per-channel flags because duplicate pairs and overlapping
static/duplicate flags would be counted twice.

**Calibration analysis (prerequisite for ANY future uncertainty
weighting):** (a) Spearman correlation across the 10 channels between
`hm_std_px` and `eq_err_1` — NOTE: n=10 is weak evidence on its own;
(b) frame-level calibration within channels using TEMPORALLY BLOCKED
summaries: split the 180 frames into 12 contiguous blocks of 15, compute
per-block median spread and per-block median error, correlate across
blocks within each channel (blocks, not frames, because residuals are
temporally structured). Preregistered read: both (a) and (b) rho >= 0.6 ->
heatmap spread is a usable error proxy; either < 0.3 -> it is NOT (any
"calibrated weighting" plan dies here).

Evidence PNGs: overlays for frames {0,45,90,135} per model; a 2x5 grid of
per-channel heatmaps (grayscale + peak marker) at frame 45 per model.

## 5. Day 2 — hidden failure modes (no training): `diagnostics/day2_failure_modes.py`

All on the three existing checkpoints.

1. **Channel switching:** recompute `eq_err_1` after Hungarian-matching the
   set {transport(p_t)} to the set {p_{t+1}} per frame pair.
   `switch_gain = (eq_preserved - eq_matched) / eq_preserved`.
   CONFOUND CONTROL: duplicated channels make matching look better for
   free (any permutation among coincident points is cost-neutral). So
   ALSO report: (i) the fraction of frame pairs where the optimal
   assignment is the identity permutation; (ii) the frequency of each
   non-identity permutation; (iii) switch_gain computed only over
   channels NOT flagged `dup_flag` in Day 1.
   Preregistered read: switch_gain > 0.3 AMONG NON-DUPLICATE channels ->
   a large part of "jitter" is identity switching (an optimization/
   assignment problem, not a localization problem).
2. **Hard vs soft argmax:** recompute trajectories with hard argmax (peak
   cell centre). Report eq_err_1 and on_mask_frac for both readouts.
   Preregistered read: "hard clearly better on both" means >=20% lower
   median eq_err_1 AND >=0.05 absolute increase in on_mask_frac; this
   implicates multimodal heatmaps making the soft mean invalid.
3. **Aliasing probe:** 12 evenly spaced frames {0,15,30,...,165} (a single
   frame would give a content-specific answer); shifts {±0.25,±0.5,±1,±2}
   px on each axis and rotations {±0.25°,±0.5°} about the estimated centre,
   applied with bilinear `grid_sample` at 512; run the extractor; map
   output coords back through the exact inverse transform; residual vs the
   unperturbed output, PER-CHANNEL DISTRIBUTIONS (median + p90 over the 12
   frames), in cells64. CONTROL: transform-then-inverse-transform the image
   (net zero motion) — report the control residual SEPARATELY as the
   interpolation measurement floor; do NOT numerically subtract it (the
   model is nonlinear; subtraction is invalid).
   Preregistered read: sub-pixel-shift residual > 0.5 cells AND >2x the
   matched transform-inverse control median -> strided-conv aliasing
   implicated.
4. **Diagnostic-only levers:** temperature sweep T in {0.25,0.5,1,2}
   (divide heatmap logits by T at inference) and test-time averaging over 8
   jittered inputs (±0.5 px): report eq_err_1 deltas. These are
   measurements, NOT solutions.

## 6. Day 2.5 — coordinate-level (parallel, imports block0): `diagnostics/day25_noise_ladder.py`

1. **Noise-decomposition ladder** (isolates WHICH property of real
   residuals kills affine subset ordering — currently unproven):
   conditions (a) iid homogeneous Gaussian sigma=0.97 cells;
   (b) iid heterogeneous: per-channel sigmas from Day-1 `sig_*_cells`
   (isotropic per channel); (c) + x/y anisotropy (per-axis sigmas);
   (d) + AR(1) temporal correlation (fit rho per channel from residuals);
   (e) full bootstrap of real 3-frame residual windows.
   Test: duplicate-vs-healthy batch-mean ordering exactly as
   `block0/e7b_confirmation.py` (B=16, N=150, affine K in {6,10}, seed
   base 20260704).
   Properties can INTERACT, so "first failing cumulative condition"
   is not sufficient for attribution. Run BOTH directions:
   - cumulative additions (the ladder above), AND
   - remove-one-property-from-full ablations: full empirical residuals
     with (i) channel variances equalized (rescale each channel to the
     global mean sigma), (ii) temporal order shuffled (destroys temporal
     structure), (iii) axes isotropized (rotate/rescale each channel's
     residuals to equal x/y variance), (iv) tails Gaussianized (rank-map
     each channel's marginals to a Gaussian of matched sigma).
   Include cross-channel dependence explicitly: a cumulative multivariate
   Gaussian condition with measured channel covariance, plus a
   remove-from-full condition that independently permutes channel identities
   across bootstrap samples while preserving each channel's marginal and
   within-channel temporal windows.
   Attribution requires AGREEMENT of both directions: a property is named
   the killer only if adding it breaks ordering AND removing it from the
   full distribution restores ordering (>= 95%).
2. **Multi-step similarity separation** (the missing validation for the
   similarity rung): toy anchors K=6 (3 distinct + 3 duplicates at 0.5
   cell), similarity-family subset loss (mean over exhaustive 2-subsets,
   fit hop1 evaluate complement hop2), noise RESAMPLED from the empirical
   bootstrap at every step; 5 seeds; also at half noise.
   **LR stability pre-check first:** lr 0.02 is ~0.64 cells/step under
   Adam normalization — large relative to the geometry. Run 200-step
   smokes at lr {0.02, 0.005, 0.001}; pick the largest lr with no
   oscillation/divergence, freeze it, THEN commit to 2000 steps.
   Keep the 3 healthy anchors fixed and optimize only the 3 near-duplicate
   anchors; otherwise global similarity-gauge drift confounds the healthy
   displacement and boundedness checks. Evaluate initial and final loss on
   the same fixed held-out bank of 100 empirical-noise batches, independent
   of noise resampled during optimization.
   **Preregistered success (ALL required, >= 4/5 seeds):**
   (i) min pairwise distance among former duplicates > 2 cells64 at the
   final step; (ii) min distance to EVERY other anchor > 1 cell64 (no
   separation-by-collision); (iii) all coordinates remain within the
   object region (radius <= 0.8); (iv) healthy-anchor displacement from
   init < 1 cell64 (no bias injected into good points); (v) final
   fitted loss < initial loss; (vi) no NaN/Inf at any step.
   This, not the E7b per-batch gradient sign, decides whether the
   similarity mechanism separates duplicates under realistic noise.

3. **Mandatory filter-artifact control before the full similarity run.**
   The frozen Block-0 residual bank uses a width-9 centred moving-average
   high-pass. That filter itself gives white noise negative lag-1
   correlation, so the empirical temporal result is not yet interpretable.
   Generate five independent 180-frame white-noise banks at the empirical
   per-channel/per-axis standard deviations, apply the exact frozen
   high-pass, and rescale each filtered channel/axis to its empirical
   post-filter standard deviation (the rescaling isolates dependence from
   noise magnitude). For each bank run the preregistered empirical-bootstrap
   affine ordering cells at K in {6,10}, B=16, N=150.

   The decision is frozen before execution. Compare the across-bank mean at
   each K with the already measured empirical result (0.660, 0.373) and iid
   heterogeneous result (0.907, 0.807). If both K values are closer in
   absolute distance to empirical, label the current temporal hostility
   **filter-artifact-like** and rebuild the operational bank by cubic
   per-channel/per-axis detrending of the ground-truth-derotated trajectory.
   If both are closer to iid heterogeneous, label it
   **network-residual-like** and retain the current bank. A split decision is
   **unresolved** and conservatively triggers the parametric rebuild. Report
   individual-bank values, across-bank mean and sample standard deviation
   (ddof=1), and pooled Wilson intervals as descriptive simulation results;
   do not treat them as checkpoint/object population inference. The full
   similarity run may start only after the selected bank and its lag-1
   statistics are recorded.

## 7. Day 3 — geometry/family verification: `diagnostics/day3_geometry_check.py`

1. Read `meta.jsonl`; assert the roll angle steps are 2.000 deg and note
   the axis field.
2. Full mask-transport curve: IoU(transport(mask_t,1), mask_{t+1}) for all
   180 t (nearest-neighbour warp). Report median/min.
3. If median IoU < 0.95 under pure rotation: fit similarity then affine to
   mask-boundary correspondences (nearest-point iteration, 20 rounds) per
   frame pair and report which family reaches IoU >= 0.95. CAVEAT
   (mandatory in the report): silhouette nearest-neighbour ICP is NOT true
   correspondence (boundary aperture problem; a family can fit silhouettes
   spuriously). A higher-family ICP fit is DESCRIPTIVE evidence only —
   never conclude "affine works" from silhouette ICP alone; if rotation
   fails, ESCALATE with the numbers.
4. Photometric spot check: 200 farthest-point-sampled mask pixels at frame
   0, transported through all frames; median RGB error at transported
   locations vs a random-point baseline. DESCRIPTIVE ONLY: fixed lighting
   interacting with rotating surface normals changes appearance even for
   perfect correspondence; do not gate on this number.
**Preregistered read:** rotation valid -> proceed; rotation fails ->
escalate immediately with the IoU curve and the descriptive family fits
(the fitted-operator premise is in doubt on this data); do NOT silently
substitute a higher family.

## 8. Days 4-5 — supervised localization control: `diagnostics/day45_supervised_control.py`

Question: can THIS architecture localize easy, known points when told
exactly what to predict? (Separates architecture ceiling from
objective-induced noise.)

- Targets: M=10 farthest-point-sampled mask-interior points at frame 0,
  transported to all frames with the verified transform.
- Split: even frames train, odd frames eval. During both training and
  evaluation apply deterministic-seeded random digital translations and
  small rotations to RGB, mask and targets together. Report unaugmented and
  augmented held-out results separately. This prevents success by memorizing
  the deterministic angle-to-coordinate orbit rather than localizing targets.
- Arm A: standard `KeypointExtractor` (heatmap_res 64), end-to-end
  soft-argmax + MSE on coordinates. 500 epochs, Adam 1e-4, batch 16,
  seeds 42 and 43.
- Arm B: same but supervised at the heatmap: per-channel cross-entropy to
  a Gaussian target map (sigma 8 px) + soft-argmax readout at eval.
- Arm C (run ONLY if A and B land between 0.4 and 0.8 cells): encoder with
  layer-3 stride set to 1 (true /4 feature resolution — the resolution
  test the earlier upsampled-head smoke did NOT perform).
- Metrics: eval-frame median localization error per channel (cells64) +
  jitter of predictions via the block0 high-pass pipeline.
**Preregistered read: <= 0.4 cells median -> architecture capable, the
unsupervised objective causes the noise; >= 0.8 cells -> architecture/
readout is the bottleneck (antialiasing, head, argmax work comes BEFORE
any fitted-loss work); in between -> mixed, Arm C decides.**

## 9. Days 6-7 — oracle intervention matrix: `diagnostics/train_oracle.py` + `run_day67.sh`

Write a SELF-CONTAINED training loop (do not patch `train.py`) that
imports `model.py` components. Data: triplets (t, t+3, t+6) mod 180 — the
constant 2-deg roll makes both hops identical 6-deg transforms, so the
fit-hop1/predict-hop2 structure is valid on this data as-is.

Arms (each = baseline plus the listed additions), 3 seeds (42,43,44),
300 epochs, hammer only:

**Pre-run calibration (mandatory, before any arm trains):**
- **Loss-weight scale calibration:** on a fixed probe batch at
  initialization, measure the gradient norm of EVERY loss term w.r.t. the
  shared extractor parameters; set weights so all terms land within one
  order of magnitude of L_pred's gradient norm; FREEZE the weights and
  record them in the run configs. (The nominal 0.5/1.0 weights below are
  placeholders until calibrated — a 100x gradient imbalance would make an
  arm fail for scale reasons, not mechanism reasons.)
- **Held-out split:** use the existing index
  `indices/split_phase_mod6.json` (keys train/val/test; inspect its exact
  schema before wiring). Train on its train split; evaluate ALL Day-1
  metrics on the held-out phase frames. Training-set improvement alone is
  only an optimization diagnostic.
- **Benchmark first:** time 5 epochs of arm 4 before launching — the 45
  subset fits per triplet may be much slower than the ~25-30 min baseline;
  vectorize the 2-point similarity fits over subsets (closed-form complex
  ops, batchable). See the runtime policy in section 9b.

1. `baseline`: the EXACT smoke configuration, verified from
   `runs_res_smoke/*/config.json`: **operator_type = "dense"** (NOT
   shared_affine — the smoke runs used train.py's legacy default), pair
   loss L_pred + disp 0.1 + ent 0.01, img 512, K=10, lr 1e-4, batch 16.
   Copy the config file values; do not re-derive from memory.
2. `+oracle_mask`: heatmap-mass grounding with the TRUE per-frame mask:
   L_fg = -log(sum_u mask_t(u) * softmax(H_i)(u)), weight 0.5.
3. `+oracle_rot`: L_rot = mean_i |p_{i,t+3} - transport(p_{i,t}, 3)|^2,
   weight 1.0. (Imposes the answer — diagnostic ceiling only.)
4. `+sim_subset`: similarity-family subset-consistency on triplets: port
   `fit_similarity` from block0 (regularized denominator, lambda 1e-4);
   exhaustive 2-subsets of K=10 (45 subsets); fit on hop (t -> t+3),
   evaluate mean squared displacement on the 8 complement channels at hop
   (t+3 -> t+6); mean aggregation; weight 1.0. NO oracle input.
5. `all`: arms 2+3+4 together.

Instrumentation: every 25 epochs, on a fixed probe batch, per-loss
gradient norms w.r.t. shared extractor parameters + pairwise cosine
similarities between loss gradients (`outputs/day67_grad_conflict.csv`).

After training: automatically run the Day-1 suite on all 15 checkpoints
(on the HELD-OUT split) -> `outputs/day67_matrix.csv` (arm x seed x the
five property groups) + overlays for every run. Add: (a) channel-shuffle
probe on baseline checkpoints (evaluate L_pred with channels randomly
permuted at eval; if prediction barely degrades, the operator is ignoring
keypoint identity); (b) **leave-one-channel-out ablation** (evaluate
L_pred with each channel removed in turn; permutation-sensitivity alone
does not prove every channel contributes — LOO measures per-channel
marginal contribution directly).

**Numerical success thresholds (structure fixed now; exact values
CALIBRATED from the Days 4-5 supervised control and the oracle arms'
initialization behavior, then FROZEN before the matrix trains — never set
after seeing matrix results):**
- median on_mask_frac >= 0.95, no channel < 0.80;
- eq_err_3 <= max(1.0 cell64, 2x the supervised-control eval error);
- >= 8 distinct active channels (not dup_flag, not static_flag);
- coverage >= baseline coverage + 20% (relative);
- all of the above in 3/3 seeds, on the held-out split.

**Preregistered interpretation matrix:**
- Full oracle (arm 5) produces distinct, on-object, tracking keypoints ->
  proxy signals are the bottleneck; engineering path = better proxies.
- Arm 5 fails while Days 4-5 supervised control succeeded -> objective/
  optimization conflict; inspect gradient-cosine log for cancellation.
- Days 4-5 control failed -> architecture bottleneck; stop all fitted-loss
  work until the instrument is fixed.
- Arm 4 recognizes duplicates (loss higher) but does not separate them ->
  optimization hardening needed (consistent with Day 2.5 outcome).
- Channel-shuffle probe shows insensitivity -> operator ignores identity;
  rethink the readout, not the losses.

## 9b. Execution order and runtime/cluster policy

**Staged execution — do NOT run this document top-to-bottom blindly:**
1. Build utilities + geometry gate (Step 0); review G0 output.
2. Run Days 1-2 on the three existing checkpoints (no training).
3. **REVIEW POINT: report the Days 1-2 diagnosis to the user before ANY
   training starts.** The diagnosis may prune or reshape everything below.
4. Run Day 2.5 coordinate-level tests (noise ladder + similarity
   multi-step) — benchmark each before full execution.
5. Run Days 4-5 supervised control ONLY if the architecture/readout
   question is still unresolved after steps 2-4.
6. Design the final oracle matrix FROM the diagnosis (arms may be dropped
   or replaced); calibrate weights and thresholds; then run.

**Runtime policy:** no run or run-group expected to exceed ONE HOUR starts
without informing the user first. Expected costs: Step 0 + Days 1-2:
< 1 h each (local). Noise ladder: benchmark, likely < 1 h. Similarity
multi-step: benchmark before committing. Supervised control: ~45-60 min
PER RUN, >= 4 runs -> raise the local-vs-cluster question with the user.
Oracle matrix: >= 8 h even before subset-fitting overhead -> CLUSTER, and
benchmark arm 4's per-epoch cost first (45 subset fits/triplet; vectorize
before benchmarking).

## 10. Wrap-up deliverable

`diagnostics/DIAGNOSIS.md`, generated at the end: for each candidate cause
{geometry, architecture ceiling, soft-argmax/multimodality, channel
switching, aliasing, loss conflict, proxy weakness} one verdict line
{IMPLICATED / EXONERATED / UNRESOLVED} with the specific numbers and the
PNG/CSV evidence path. Plus the calendar gates:

- **End of week 1 (Jul 10):** the dominant bottleneck is named in
  DIAGNOSIS.md with evidence, or the honest statement that two candidates
  remain tied (name the tiebreaker experiment).
- **End of week 2 (Jul 17):** a positive 3-seed one-object image result
  under the chosen intervention exists, or the fitted-operator programme
  is paused and reframed with the supervisor.

## 11. Explicitly out of scope this week

New losses in science training; WLS on minimal subsets (a no-op — minimal
fits are exactly determined); any new renderer or dataset; cluster jobs before
the Days 1-2 review point or without explicit user authorization;
Transporter; the affine mechanism-hardening list (overdetermined fits,
IRLS, calibrated weighting) — that list activates only AFTER Day 2.5 and
Day 1 calibration results say it can work.
