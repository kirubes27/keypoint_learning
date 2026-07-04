# Phase A Roll — First Results & Open Questions (for feedback)

Date: 2026-06-04
Continues from: `docs/PHASE_A_ROLL_TECHNICAL_NOTES.md` (ends 2026-06-01 at the *planned* 432-config sweep, before any cluster run).

This note covers everything **after** that: the first real cluster run, what we
saw, and the design questions we'd like feedback on. Nothing here changes the
dataset; it's all training/loss/eval interpretation.

---

## 1. What we ran

First full run on the cluster — single object (engineers' hammer), to act as a
**semantic gate** before launching the big sweep.

- Operator: `shared_affine` (one shared 2×2 `A` + bias, applied to every keypoint — the "strong hypothesis" model).
- Data: full 360° World-Z roll, `pairs_skip3_cyclic.json` (6° step, closed orbit).
- Config: `lambda_act=1.0, lambda_ent=0.01, lambda_disp=0.1, lambda_inv=0.1, lambda_cycle=0.1, lambda_smooth=0, lambda_loc=0`, 1000 epochs, `img_size=512`.

(Note: this was *not* the clean-emergence config — `inv`/`cycle` were on, and `ent` was low. See §4.)

## 2. Results

| Metric | Value | Read |
|---|---|---|
| **Shared operator `A` vs `R(6°)`** | angle **5.90°**, ‖A−R(6°)‖=**0.014**, det=0.98, orth-err=0.028 | ✅ **emerged as a near-exact 6° rotation** |
| on_object_pct (TDW masks) | **0.736**, border occ. 0 | ✅ on the object, no border shortcut |
| active_kp_frac | **0.80** | ✅ keypoints participate |
| identity_ratio (operator vs W=I) | **1.63** | ⚠️ below our old `>2.0` gate (see §5) |
| val_act_acc (+6° vs −6°) | **0.538** | ❗ ≈ chance — see §3 (it's not a bug) |
| canonical drift (−θ, RMS) | **0.158** | ⚠️ moderate sliding (not tightly part-locked) |
| closed orbit `T⁶⁰` (full loop) | MSE **0.024** | ✅ closes; interpret jointly (see §5) |
| inverse / cycle losses | `l_inv≈l_pred≈0.0027`, `l_cycle≈1e-5` | ✅ inverse works; cycle near-perfect (near-trivial for a rotation) |

**Headline positive:** the single shared 2×2 operator converged to an almost-exact
6° rotation. That's the meeting's #1 question (does a *simple shared* law emerge,
not a dense mixing matrix) answered positively for this controlled setting.

**Visual check** (`figures_roll_update/hammer_keypoints_sequence.png`): the keypoints
sit on the hammer and co-rotate with it across the full sweep — a head-cluster stays
on the head, handle points on the handle. Much better than the old off-object/banding
failures, though individual points still slide somewhat (consistent with the 0.158 drift).

## 3. The main finding: the current delta-linear action loss is not a valid full-orbit diagnostic

`val_act_acc` sat at **chance (0.54)**. We initially suspected a code bug; it is
better explained as a geometry/diagnostic mismatch:

- A backward pair's keypoint motion `Δk` is exactly the negative of the forward pair's.
- The current action head is **linear on displacement only**: `logits = Linear(p_t1 - p_t)`.
- On a **full 360°** closed trajectory, forward displacements sweep around the full
  tangent field. The reversed samples are `-Δk`. For a closed periodic trajectory,
  the average forward displacement is zero, so there is no single linear projection
  that can stay positive for every forward step and negative for every backward step.
- For object-attached circular motion the problem is even more concrete: a `+6°` step
  at one phase can look like a `-6°` step at the opposite phase if the classifier sees
  only `Δk` and not the starting phase.
- On a **limited arc** (the old ±60° yaw), the object never reaches φ+180°, so the two
  stay locally separable, which is why action classification was useful in the earlier
  short-arc setup.

Figures:
- `figures_roll_update/why_action_loss_fails_on_full_circle.png` — scatter of the actual
  `Δk=(Δx,Δy)` the classifier sees: full circle = one overlapping ring (50%); ±60° arc =
  two separated arcs split by a line (~100%).
- `figures_roll_update/action_direction_roll_vs_yaw.png` — the same point as direction dials.

**Implication:** `val_act_acc>0.7` should not be a promotion gate for the full-360
roll experiment when using the current `delta_linear` action head. The closed-orbit
structure that makes the compositionality test (`T⁶⁰≈I`) compelling is the same
structure that makes displacement-only direction classification ill-posed.

## 4. What this means for the "clean" run

The run above had `inv/cycle` on and low `ent`. So:
- The beautiful `A≈R(6°)` is **partly assisted** — `inv/cycle` reward invertibility,
  which nudges `A` toward orthogonal (a rotation). Their success here is also near-trivial
  (`cycle≈0` follows automatically once `A` is a rotation).
- The jitter symptoms (chance action acc, 0.158 drift) likely trace to low `ent=0.01`.

So the **headline emergence run still needs to be done**: `shared_affine`,
`lambda_act=0`, `inv=cycle=loc=smooth=0`, `ent≈0.05` — to test whether `A≈R(6°)` and
on-object keypoints emerge from **prediction + the shared constraint alone**.

## 5. Proposed design decisions (where we'd like feedback)

**(a) Two-experiment split.** Keep the full 360° dataset (for `T⁶⁰≈I`, closed-orbit
compositionality, group structure). Run two protocols from the *same* data (index
subsets only, no regeneration):
- **`full360_main`** — `lambda_act=0`, minimal priors. Tests the strong claim: *do
  coordinates emerge because they make the transformation predictable/compositional?*
- **`local_arc_action`** — a signed ±60° arc subset, `lambda_act` swept. Action
  direction is decodable here (control), but **closed-orbit metrics don't apply**.
  This arc run also serves as the clean *control* that demonstrates the §3 finding
  on the real model (≈chance on the full circle, ~90% on the arc).

Current leaning after discussion: do **not** make a roll-cross action loss primary.
It would hard-code centered angular motion and therefore weaken the main emergence
claim. If retained at all, it should be a clearly labeled ablation/sanity check, not
the headline objective.

**(b) Gates for roll.** Drop `val_act_acc` (unpassable). Recalibrate `identity_ratio`
(our 1.63 was a *good* run — `>2.0` was calibrated for the harder oblique-yaw regime).
Gate `closed_orbit` only *jointly* (a static/identity solution closes the orbit
trivially), alongside on_object + active_kp + the operator angle.

**(c) Anti-static for the clean run.** With `lambda_act=0`, nothing explicitly breaks
the static-keypoint degeneracy. Plan: **test first** (the earlier run did *not* collapse
— uniform background makes object-tracking the natural solution). If it collapses, use a
generic, weak **`lambda_motion`** ("keypoints must move"), *not* a centered-rotation
prior. Caveat we noted: `lambda_motion` mildly **penalizes legitimately-stable points
near the rotation center**, so it's a fallback, not a default.

One safer form is a **quota** rather than forcing every keypoint to move:
`motion_i = ||p_i(t+1)-p_i(t)||`, `active_i = sigmoid((motion_i-m0)/tau)`,
`L_motion = max(0, q - mean_i active_i)^2`. This asks for enough active keypoints
while allowing 1-2 central/stable points.

**(d) Drift / "same-part" stability — Angela's relative-distance idea.** The 0.158
canonical drift = keypoints sliding within the object. The natural fix is a
**pairwise-distance (rigidity) loss** `L_rigid = Σ(‖p_i−p_j‖_t − ‖p_i−p_j‖_{t+1})²`,
which forces the keypoint cloud to move as one rigid body (a sliding point changes its
distances to the others). Well-matched to roll (a rotation preserves distances). Caveats:
needs dispersion on (else collapse satisfies it trivially), doesn't break static, and is
roll-specific (projected oblique yaw/pitch foreshortens). We'd run it as a **tagged
intervention** (headline keeps it off; drift stays a *measured* quantity).

Implementation update after discussion (2026-06-05): we decided to keep the
full hammer sweep broad because cluster compute is available, but make the
protocol explicit so the scientific claims do not get mixed:

1. **`full360_main`:** full cyclic roll, `lambda_act=0`, `lambda_loc=0`;
   sweep `lambda_disp × lambda_ent × lambda_inv × lambda_cycle × lambda_smooth`.
   This gives `324` configs per operator. Run it once with `shared_affine` and
   once with `dense` for the `648`-run hammer operator comparison. `val_act_acc`
   is reported only, not a gate.
2. **`local_arc_action`:** future ±60° arc control; sweep `lambda_act` only in
   this non-closed setting, and do not apply closed-orbit gates. The code now
   refuses to run this protocol unless an explicit arc pair index is provided.
3. **Updated diagnostics:** in addition to `active_kp_frac`, evaluation now
   reports an exhaustive motion/object partition:
   `active_on_object_frac`, `active_off_object_frac`, `static_on_object_frac`,
   `dead_off_object_frac`, and `motion_object_partition_sum`. This prevents
   active background points or static on-object points from being hidden inside
   a single average.

## 6. Specific questions for you

1. Is the **two-experiment split** (full 360° = compositionality with `lambda_act=0`;
   ±60° arc = action-decodability control) the right framing, or do you want action
   supervision retained on the full orbit (which would require the hand-crafted
   centered-rotation "roll-cross" loss — a stronger prior)?
2. For the **clean emergence run**, are you comfortable with **no action loss** and
   relying on prediction + shared-affine (+ a weak motion fallback only if it collapses)?
3. The **`A≈R(6°)`** result — is matching the analytic 6° rotation the right primary
   evidence for "a simple shared operator emerged," or do you want an additional
   comparison (e.g. Kabsch-optimal rotation vs learned `A`)?
4. **Drift:** add the relative-distance rigidity loss as an intervention now, or keep
   drift purely as a measured diagnostic for the first clean result?
