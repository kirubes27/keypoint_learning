# Project History: Phase-A Keypoint Learning

**Last Updated**: 2026-03-06
**Authoring note**: This file is the canonical experiment log for Phase A. Metrics are copied from saved JSON artifacts in `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/`.

---

## 1) Project Goal

Show that keypoints (coordinate charts) can emerge from raw images because they make rigid object transformations:

- linear (single global operator in keypoint space),
- predictable (low one-step keypoint prediction error),
- compositional (multi-step rollout remains stable),

without reconstruction and without supervised keypoint labels.

---

## 2) Semantic Lock (Pass/Fail Criteria)

Before claiming progress, these must be checked:

1. **Must be true**: learned operator outperforms identity (`MSE_baseline < MSE_identity`).
2. **Must be true**: action direction is decodable from keypoint displacement (`act_acc >> 50%` for binary yaw+/yaw-).
3. **Must be true**: k-step error does not explode over the tested range.
4. **Must be true**: keypoint visualizations are interpretable (on-object and temporally coherent).
5. **Must not happen**: static-keypoint + identity shortcut.

Current status is mixed: criteria 1-3 pass on recent runs; criterion 4 is improved but not fully solved.

---

## 3) Fixed Experimental Setup

- Dataset: TDW rigid objects, 61 frames per sequence, yaw from -60 deg to +60 deg in 2 deg increments.
- Training stride: `frame_skip=3` (effective 6 deg operator step).
- Core model: CNN heatmaps -> soft-argmax keypoints `p_t in R^(2N)` -> linear operator `p_hat = W p_t + b`.
- Action head: linear classifier on `delta_k = p_{t+1} - p_t` for yaw direction.

Loss form:

`L = L_pred + lambda_smooth*L_smooth + lambda_disp*L_disp + lambda_ent*L_ent + lambda_act*L_act`

With forward masking:

- `L_pred` and `L_smooth` use forward pairs only (`action_label=0`).
- `L_act` uses both forward and backward pairs.

---

## 4) Experiment Log

## Run 0 (Baseline Degenerate) - `phase_a_scissors_20260206_032138`

Config:

- Object: `scissors`
- `N=10`, `lambda_disp=0.1`, `lambda_smooth=0.0`, `lambda_ent=0.0`, `lambda_act=0.0`
- `frame_skip=3`, `epochs=200`

Result:

- Identity beat baseline (degenerate shortcut).
- Keypoints mostly static/off-object.

---

## Run 1 (Baseline Degenerate) - `phase_a_engineers_hammer_vray_20260206_090739`

Config:

- Object: `engineers_hammer_vray`
- Same losses as Run 0 (`L_pred + 0.1*L_disp`)

Result:

- Identity beat baseline harder than Run 0.
- Confirms objective loophole is not object-specific.

---

## Run 2 (Fewer Keypoints, Still Degenerate) - `phase_a_engineers_hammer_vray_20260206_100910`

Config:

- `N=5`, `lambda_ent=0.0`, `lambda_act=0.0`

Result:

- Slightly improved geometry, but identity still beat baseline.
- Changing `N` alone did not fix degeneracy.

---

## Run 3 (Introduced L_act; Degeneracy Broken at 200 Epochs) - `phase_a_engineers_hammer_vray_20260213_040355`

Config:

- `N=10`
- `lambda_disp=0.1`, `lambda_smooth=0.0`, `lambda_ent=0.0`, `lambda_act=0.1`
- `num_action_classes=2`, backward pairs enabled
- `frame_skip=3`, `epochs=200`

Key metrics:

- Ablations (`/ablations/ablation_results.json`):
  - Baseline: `5.224735e-04`
  - Identity: `4.365153e-03`
  - Identity/Baseline: `8.36x` (baseline better)
- Compositionality (`/compositionality/compositionality_results.json`):
  - k1 mean: `5.224735e-04`
  - k10 mean: `4.902089e-04`
  - k10/k1: `0.94x`
- History (`/history.json`):
  - Best/Final epoch: `200`
  - `val_pred=2.698915e-04`
  - `val_act=0.6435`
  - `val_act_acc=0.71875`

Interpretation:

- L_act + forward masking successfully broke the identity shortcut.
- Visualization quality still mixed because `lambda_ent=0.0`.

---

## Run 4 (L_act, No Entropy, 1000 Epochs) - `phase_a_engineers_hammer_vray_20260213_074841`

Config:

- `N=10`
- `lambda_disp=0.1`, `lambda_smooth=0.0`, `lambda_ent=0.0`, `lambda_act=0.1`
- `frame_skip=3`, `epochs=1000`

Key metrics:

- Ablations:
  - Baseline: `5.003866e-04`
  - Identity: `5.710376e-03`
  - Identity/Baseline: `11.41x`
- Compositionality:
  - k1 mean: `5.003866e-04`
  - k10 mean: `1.398469e-03`
  - k10/k1: `2.79x`
- History:
  - Best epoch: `1000`
  - `val_pred=2.721575e-04`
  - `val_act=0.2549`
  - `val_act_acc=1.0`

Interpretation:

- Strongest anti-degeneracy signal (very large identity gap, perfect action accuracy).
- But keypoint localization/heatmaps still had qualitative issues (shortcut-like patterns).

---

## Run 5 (N=5 + Entropy Refinement) - `phase_a_engineers_hammer_vray_20260213_092501_032144_seed42_pid66077`

Config:

- `N=5`
- `lambda_disp=0.1`, `lambda_smooth=0.0`, `lambda_ent=0.01`, `lambda_act=0.1`
- `frame_skip=3`, `epochs=1000`

Final best-checkpoint metrics:

- Ablations:
  - Baseline: `1.922084e-03`
  - Identity: `3.019527e-03`
  - Identity/Baseline: `1.57x`
- Compositionality:
  - k1 mean: `1.922084e-03`
  - k10 mean: `3.024386e-03`
  - k10/k1: `1.57x`
- History:
  - Best epoch (by val loss): `980`
  - `val_pred=1.013954e-03`
  - `val_act=0.6641`
  - `val_act_acc=0.578125`

Midpoint (epoch-500 checkpoint):

- Ablations e500:
  - Baseline: `2.293726e-03`
  - Identity: `2.956006e-03`
  - Identity/Baseline: `1.29x`
- Compositionality e500:
  - k10/k1: `1.20x`

Interpretation:

- Entropy + fewer keypoints improved localization aesthetics.
- But action-direction signal weakened (accuracy near chance), reducing causal-strength claim.

---

## Run 6 (N=10 + Entropy; Current Best Balance) - `phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542`

Config:

- `N=10`
- `lambda_disp=0.1`, `lambda_smooth=0.0`, `lambda_ent=0.01`, `lambda_act=0.1`
- `frame_skip=3`, `epochs=1000`

Final best-checkpoint metrics:

- Ablations (`/ablations/ablation_results.json`):
  - Baseline: `1.786238e-03`
  - Identity: `5.744913e-03`
  - Identity/Baseline: `3.22x`
- Compositionality (`/compositionality/compositionality_results.json`):
  - k1 mean: `1.786238e-03`
  - k10 mean: `3.707023e-03`
  - k10/k1: `2.08x`
- History (`/history.json`):
  - Best epoch (by val loss): `920`
  - `val_pred=9.539396e-04`
  - `val_act=0.5663`
  - `val_act_acc=0.9765625`
  - Best observed `val_act_acc=0.9921875` at epoch `680`

Midpoint (epoch-500 checkpoint):

- Ablations e500:
  - Baseline: `2.464405e-03`
  - Identity: `6.826671e-03`
  - Identity/Baseline: `2.77x`
- Compositionality e500:
  - k1 mean: `2.464405e-03`
  - k10 mean: `4.888977e-03`
  - k10/k1: `1.98x`

Interpretation:

- Better tradeoff than Run 5: strong direction signal and meaningful identity gap while keeping improved localization pressure.
- Still not perfect visually: residual banding/shortcut patterns remain in some heatmaps.

---

## 5) New Diagnostics Added After These Runs

## A) Compositionality error-bar clarification (`compositionality_v2`)

File: `/Users/kirubeso.r/Documents/PhD/keypoint_net/eval_compositionality.py`

What changed:

- Error bars now explicitly defined as:
  - mean +/- 1 sample std across valid start frames `t` (with `ddof=1`)
- JSON now stores:
  - `error_bar_definition`
  - `sem` (in addition to `std`)

Artifact:

- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/compositionality_v2/compositionality_results.json`

Important statistics caveat:

- These bars are descriptive within-sequence variability, not a population confidence interval (start frames are correlated).

## B) Operator visualization and metrics (`visualizations_v2`)

File: `/Users/kirubeso.r/Documents/PhD/keypoint_net/visualize.py`

New outputs:

- `operator_summary.png`
- `operator_metrics.json`

For Run 6 best checkpoint:

- `||W||_F = 2.7752`
- `||W-I||_F = 4.9344`
- Off-diagonal block mixing ratio: `0.8788`
- `||b|| = 0.6029`
- `max |eig(W)| = 1.0700`

Interpretation:

- Operator is useful for prediction and not identity.
- Operator is strongly mixed across keypoints (not yet physically interpretable as near block-diagonal rigid rotation in the current learned basis).

---

## 6) Cross-Run Summary (Hammer, L_act enabled)

| Run | N | lambda_ent | Epochs | Baseline MSE | Identity MSE | Identity/Baseline | Best val_act_acc | k10/k1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260213_040355` | 10 | 0.00 | 200 | 5.2247e-04 | 4.3652e-03 | 8.36x | 0.7188 | 0.94x |
| `20260213_074841` | 10 | 0.00 | 1000 | 5.0039e-04 | 5.7104e-03 | 11.41x | 1.0000 | 2.79x |
| `20260213_092501...` | 5 | 0.01 | 1000 | 1.9221e-03 | 3.0195e-03 | 1.57x | 0.5781 | 1.57x |
| `20260213_102905...` | 10 | 0.01 | 1000 | 1.7862e-03 | 5.7449e-03 | 3.22x | 0.9922 | 2.08x |

Main takeaway:

- `N=10, lambda_ent=0.01, lambda_act=0.1` currently gives the best balance between direction decodability and non-degenerate operator performance.
- `N=5, lambda_ent=0.01` gives cleaner localization but weakens action signal substantially.

---

## 7) Code Changes to Date (Confirmed)

- `/Users/kirubeso.r/Documents/PhD/keypoint_net/model.py`
  - Added `ActionClassifier` and `L_act`.
  - Added forward-mask logic for `L_pred` and `L_smooth`.
  - Added `act_acc` reporting.
  - Added configurable `padding_mode` in keypoint CNN conv layers.
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/dataset.py`
  - Added backward pairs and `action_label`.
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/train.py`
  - Added `lambda_act`, `num_action_classes`, auto-eval, action metrics in logs/history.
  - Added `--padding_mode` and persisted `padding_mode` in checkpoint config.
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/ablations.py`
  - Checkpoint compatibility and seeded perturbations.
  - Added `--padding_mode_override` for legacy checkpoints missing padding metadata.
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/eval_compositionality.py`
  - Checkpoint compatibility.
  - Explicit error-bar definition, sample std (`ddof=1`), and SEM output.
  - Added `--padding_mode_override` for legacy checkpoints missing padding metadata.
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/eval_rollout_viz.py`
  - Added `--padding_mode_override` for legacy checkpoints missing padding metadata.
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/visualize.py`
  - Checkpoint compatibility.
  - Added operator diagnostics plots and JSON metrics.
  - Added `--padding_mode_override` for legacy checkpoints missing padding metadata.

---

## 8) Current Status vs Success Criteria

1. Visualization makes sense: **partial**
2. Stable/smooth keypoints: **partial to good** (improved, still some artifacts)
3. Linear map predicts next keypoints well: **met**
4. Compositionality non-explosive over tested range: **met**
5. Removing linear predictor breaks performance (identity/random worse): **met**
6. Randomizing keypoints breaks performance: **met**

---

## 9) Immediate Next Steps (Priority Ordered)

1. Add a quantitative localization metric (for example, percent of keypoints inside object mask per frame) so criterion #1 is numerical, not subjective.
2. Run a small `lambda_smooth` sweep (for example `0.001`, `0.01`) to reduce swaps/jitter without collapsing action signal.
3. Replicate Run 6 settings on at least 2 additional objects (scissors, mug or pepper) and report cross-object mean/std for:
   - identity/baseline ratio
   - best val action accuracy
   - k10/k1
   - localization metric
4. Test mild operator-structure regularization (small penalty on off-diagonal 2x2 block energy) and check whether operator interpretability improves without hurting predictive metrics.

---

## 10) Artifact Map

Latest balanced run:

- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/config.json`
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/history.json`
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/ablations/ablation_results.json`
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/compositionality/compositionality_results.json`
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/compositionality_v2/compositionality_results.json`
- `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_102905_855088_seed42_pid68542/visualizations_v2/operator_metrics.json`

---

## 11) Full Sweep Completion (2026-02-27 to 2026-02-28)

Objective:

- Move from anecdotal manual runs to systematic parameter evidence for the hammer setting.

Sweep setup:

- Script: `keypoint_net/sweep.py`
- Grid:
  - `lambda_ent in {0.0, 0.005, 0.01, 0.05, 0.1}`
  - `lambda_act in {0.0, 0.1, 0.5, 1.0}`
  - `lambda_smooth in {0.0, 0.001, 0.01}`
- Fixed:
  - `lambda_disp=0.1`, `sigma=0.1`, `frame_skip=3`, `num_keypoints=10`
  - `epochs=1000`, `seed=42`, `batch_size=16`, `lr=1e-4`
- Total configs: `5 * 4 * 3 = 60`

Sweep artifacts:

- `keypoint_net/runs/sweep/sweep_results.json` (60 entries)
- `keypoint_net/runs/sweep/sweep_summary.csv` (60 rows)
- `keypoint_net/runs/sweep/sweep_summary.png`
- `keypoint_net/runs/sweep/promoted_configs.json`

Initial promotion gates:

- Mandatory:
  - `identity_ratio > 2.0`
  - `on_foreground_pct > 0.5`
- Soft:
  - `val_act_acc > 0.7`
  - `active_kp_frac > 0.3`
  - `k10_k1_ratio < 5.0`
- Outcome: `6` full promotions, `0` soft.

Critical correction after manual visual audit:

- The run `phase_a_engineers_hammer_vray_20260213_074841` passed mean foreground but fails in tail frames (clear off-object drift in sequence visualization).
- Mean-only localization screening is insufficient.

Stricter localization-tail post-filter used on the 6 promoted runs:

- Keep only runs with:
  - `min(localization_per_frame) >= 0.3`
  - `mean(last_3 localization_per_frame) >= 0.45`
- Result: `4` robust candidates remain.

Robust 4-run shortlist (post-filter):

1. `keypoint_net/runs/phase_a_engineers_hammer_vray_20260227_201406_119939_seed42_pid6175`
   - `lambda_ent=0.05`, `lambda_act=0.5`, `lambda_smooth=0.0`
   - `identity_ratio=2.2073`, `on_foreground_pct=0.6590`, `val_act_acc=0.9609`, `k10/k1=2.0713`
2. `keypoint_net/runs/phase_a_engineers_hammer_vray_20260227_203530_910039_seed42_pid6665`
   - `lambda_ent=0.05`, `lambda_act=0.5`, `lambda_smooth=0.001`
   - `identity_ratio=2.1733`, `on_foreground_pct=0.6246`, `val_act_acc=0.9219`, `k10/k1=1.9556`
3. `keypoint_net/runs/phase_a_engineers_hammer_vray_20260227_205643_885233_seed42_pid7119`
   - `lambda_ent=0.05`, `lambda_act=0.5`, `lambda_smooth=0.01`
   - `identity_ratio=2.1364`, `on_foreground_pct=0.6557`, `val_act_acc=0.9063`, `k10/k1=2.2817`
4. `keypoint_net/runs/phase_a_engineers_hammer_vray_20260228_031715_807134_seed42_pid16696`
   - `lambda_ent=0.05`, `lambda_act=1.0`, `lambda_smooth=0.001`
   - `identity_ratio=2.5402`, `on_foreground_pct=0.6508`, `val_act_acc=0.9375`, `k10/k1=2.7647`

Sweep-level interpretation:

- Positive:
  - Degeneracy remains broken in many configs (`identity_ratio > 2` in 39/60).
  - Action direction decodability is generally strong when `lambda_act > 0`.
- Negative / limiting:
  - Localization remains the weakest criterion and is sensitive to how it is measured.
  - Mean-only on-foreground gate can produce false positives versus visual semantics.
- Net:
  - The sweep is useful, but does not yet justify a final claim that localization is solved.

---

## 12) Updated Next Steps (Post-Sweep)

1. Use stricter localization-tail criteria (not mean-only) for any future promotion.
2. Run multi-seed replication on the 4-run shortlist before final config selection.
3. Validate shortlist behavior on at least 2 additional objects.
4. Introduce anti-shortcut data augmentation (translation/crop jitter and mild appearance jitter) before any new large sweep.

---

## 13) Reproducibility Correction: Padding-Mode Replay Mismatch (2026-03-06)

Issue:

- Legacy checkpoints from early runs do not store extractor `padding_mode` in config.
- Re-evaluating those checkpoints under newer code can silently change padding behavior and produce inconsistent visualizations/metrics.

Validated evidence (same checkpoint, same sample, same frame pair `t=29 -> t1=32`):

- Run `phase_a_engineers_hammer_vray_20260213_074841`:
  - replay with `reflect`: prediction error `0.9484005`
  - replay with `zeros`: prediction error `0.05124839`
- Run `phase_a_engineers_hammer_vray_20260213_040355`:
  - replay with `reflect`: prediction error `0.6459559`
  - replay with `zeros`: prediction error `0.03731566`

Interpretation:

- This is a replay-compatibility problem for legacy checkpoints, not new evidence that sweep conclusions changed.
- For recent sweep shortlist runs, the measured reflect-vs-zeros difference at the tested sample was negligible, supporting that shortlist conclusions are not driven by this replay issue.

Action taken:

1. New training runs now save `padding_mode` in checkpoint config.
2. Evaluation scripts now fail fast on legacy checkpoints unless `--padding_mode_override` is provided.
3. Legacy-compatible refresh output for `074841` was generated at:
   - `/Users/kirubeso.r/Documents/PhD/keypoint_net/runs/phase_a_engineers_hammer_vray_20260213_074841/visualizations_zeros_override`

Reporting rule going forward:

- For any checkpoint without `padding_mode` in `config.json`, explicitly pass `--padding_mode_override` in visualization/ablation/compositionality/rollout commands and record that override in notes.
