# Plan v4.2 — Fitted-Operator Programme (2026-07-03)

Supersedes v4.1 (same day). Status: spec-complete, pending the Block-0 gate.
**Rule: do not expand the implementation before the algebraic gate. After the gate, freeze the tested implementation while keeping scientific interpretation conditional on the empirical result.**

**v4.2 deltas over v4.1:** fixed-coordinate-in-motion-support degeneracy + activity-constraint testing; K > |S| complement requirement; regularized similarity/rotation solvers; P1 solver-relative wording; P3 per-family scope; corrected affine/3D framing; matched-motion causal control; mask-baseline supervision tiers; differentiable-geometric-estimation literature refresh (DSAC, Kabsch layers, BPnP).

**Block-0 pre-implementation amendments (2026-07-03, incorporated below):** (1) lone-tracker rescue tests ((K−1) static + 1 tracking, and the reverse — decides whether the all-static optimum is attracting or escapable); (2) complement size treated as a variance parameter (affine K=4, |Sᶜ|=1 = near-floor diagnostic only; K∈{6,10} primary; report |Sᶜ| with loss variance); (3) activity is a quality SCORE with ordering A_static < A_jitter < A_tracking — if minimized as a loss, transform so the optimization direction matches; (4) activity remains conditional on passing all five Block-0 conditions; (5) solver-safe degeneracy handling everywhere incl. coincident two-point similarity subsets, with conditioning + fallback-activation logging; (6) non-empty complement: rotation K≥2, similarity K≥3, affine K≥4; (7) no formal non-inferiority claims at n=3 — paired seed-level effects, raw points, directional replication in ≥2/3 seeds; (8) terminology: "self-supervised motion-grounded" / "no-oracle-mask", never "mask-free"; (9) dataset lock note recorded for Week 2 (document only): no existing generator assumed; triplet frames generated INDEPENDENTLY from the same high-res source — x₀=D(W(x_HR,I)), x₁=D(W(x_HR,G)), x₂=D(W(x_HR,G²)); never generate x₂ by warping x₁.

**Working thesis:** *Transformation prediction remains non-identifiable under constant and redundant representations; operator-family complexity and minimal-subset consistency increase landmark informativeness, but explicit grounding and symmetry breaking remain necessary.*

**Narrative (state bluntly in the intro):** the representation is the ONLY learned object. G is fitted per-triplet by a closed-form solver; operator recovery is a solver output. Bridge arm (learned shared W on rung ①) reconnects Phase A.

---

## 1. Theory section (impossibility propositions + mechanism)

- **P1 (grounding necessary; solver-relative wording):** *under the specified identity-centred, regularized minimum-norm solver*, a constant representation selects the identity update and attains zero representation-prediction loss — under ANY transformation diversity. (Not "every fitted operator is identity": infinitely many G fix a collapsed point; the solver selects identity.)
- **P1b (fixed-coordinate-in-motion-support):** a constant image coordinate inside the motion support sees changing pixels every frame → satisfies motion-mask grounding while having a static trajectory → identity fit → zero loss. Under full-360 roll the swept disk is permanently in the motion mask, so this is the default degenerate solution, not a corner case. Four distinct properties, never conflated: (i) on-foreground; (ii) changing appearance at the coordinate; (iii) non-static coordinate trajectory; (iv) motion consistent with G.
- **Activity constraint (candidate, decided in Block 0):** must satisfy the ordering **static < random jitter < correct tracking** — never reward raw displacement (jitter games it). Two structural facts any candidate must respect: (a) genuinely fixed points of G (e.g., the rotation centre) are legitimately static → activity must be G-aware, comparing observed to G-expected displacement at that location; (b) G-relative activity computed from the fitted Ĝ inherits the collapse circularity (all-static → Ĝ=I → zero expected motion, self-consistent) → an image-referenced signal is required to break the loop. Block 0 tests candidates with oracle stubs (true flow / appearance-change flags); the image-space estimator is chosen only after the ordering is established. **A is a quality score (higher = better); if it enters training as a minimized loss it must be sign-flipped/transformed — never accidentally optimize the score in the wrong direction.** No activity loss enters training unless a candidate passes ALL of: static coordinates fail; random jitter does not count as correct activity; correct tracking passes; legitimate fixed points of G (rotation centre) are not penalized; the all-static Ĝ=I circularity is exposed rather than silently accepted.
- **P2 (symmetry-breaking necessary):** at exact coordinate duplication, a permutation-symmetric loss gives identical coordinate-gradients; separation only via parametrization asymmetry (distinct heatmap weights → unstable equilibrium, not deterministic escape). Method prevents NEAR-duplicates; exact case documented honestly.
- **P3 (exposed-geometry bound, per-family):** subset consistency cannot exceed the geometry the representation exposes *relative to the family's DOF*. For the affine family, collinear-but-tracking configurations get ~zero loss (ridge fills the transverse nullspace with identity). For the similarity family, two distinct points fully determine the transform — collinearity is NOT a similarity degeneracy. One proposition per family; no universal statement across the ladder.
- **M (mechanism):** family-matched identifiability floors (1/1/2/3) create diversity pressure; minimal-subset consistency extends it beyond the floor toward all channels.

Draft P1–P3+M during weeks 3–4 (while cluster runs), NOT in week 7.

---

## 2. Core specs

### 2.1 Subset-consistency loss
- Fit G on minimal subset S (|S| = family floor) from hop x₀→x₁; evaluate displacement error on the **complement** at hop x₁→x₂.
- **Complement must be non-empty: K > |S|.** Affine subset experiments need **K ≥ 4** (K=3 reserved for plain-fit floor diagnostics only); similarity needs K ≥ 3; rotation K ≥ 2.
- **Complement size is a variance parameter:** affine K=4 (|Sᶜ|=1) is a near-floor diagnostic, NOT the primary stable configuration; primary complement-consistency tests at K∈{6,10}; report complement size alongside loss variance.
- Aggregate: **mean over exhaustive subsets** (min/RANSAC aggregation makes duplicates invisible — never primary). Softmin = one ablation arm.
- Engine (state in methods): ridge G = I + Δ ⇒ rank-deficient subsets shrink toward identity ⇒ complement error ≈ full motion magnitude ⇒ clean separating gradient.

### 2.2 Solvers (family-matched, closed-form, differentiable, ALL regularized at their degeneracies)
| Rung | Family | Solver | Floor |
|---|---|---|---|
| ① | rotation, known centre | 1-DOF Procrustes: atan2(Σcross, Σdot), **norm-floor on denominator** | 1 non-central pt |
| ② | + isotropic scale | complex regression through origin, **denominator Σ|p̃|² + λ** | 1 |
| ③ | + translation (similarity) | centroid-centred complex LS, **denominator + λ** (coincident 2-pt subsets → zero variance otherwise) | 2 distinct pts |
| ④ | full affine | ridge affine | 3 non-collinear |

- Rung ④ samples **G = R(θ₁)·diag(sx,sy)·R(θ₂)** (SVD form; R(θ)diag alone is 3-DOF, misses shear).
- Ridge: solve (XᵀX+λI)Δ = Xᵀ(Y−X) via `torch.linalg.solve`/Cholesky, never explicit inverse; centre by centroids; translation unregularized; λ=1e-4 + sensitivity sweep; float64 in toy.
- **Fallback behavior at degeneracy = shrink to identity, uniformly across families.** Finite-output and finite-gradient tests + conditioning logging mandatory in Block 0 (duplicates are the pathology under study — the solver must not NaN on them).
- Transform-error metric: mean displacement over a fixed evaluation grid, never raw matrix-parameter error.
- Disclosure: solvers are family-matched (method NOT group-agnostic); known centre on ①②, estimated translation on ③④.

### 2.3 Grounding
- Heatmap-mass form: L_fg = −log(Σ_u M(u)·π_i(u)) per channel (coordinate-level grounding gameable by diffuse heatmaps).
- **Necessary but NOT sufficient** (P1b: fixed coordinates in the motion support pass it). Activity handled separately per §1.
- Dual condition: self-sup motion mask (blurred frame diff) + TDW oracle mask; motion-mask precision/recall vs TDW reported. Headline condition is called **"self-supervised motion-grounded" (no-oracle-mask)** — never "mask-free" (frame differencing is itself a mask-like motion cue); oracle arms labeled in the figures themselves.
- Grounding ON in every arm.

### 2.4 Dataset
- Triplets **(x, Gx, G²x)** — one G per triplet, applied twice. The loss never sees G (eval-only metadata).
- **Composited exact 2D warps PRIMARY** for rungs ②–④; rung ① dual-arm (TDW physical roll + composited rotation; cross-check = pipeline validation + resampling-shortcut detector). Hygiene: render high-res → warp → downsample; identical resampling across ALL rungs; object stays in frame; ≥3 background/lighting variants per rung.
- **Generation-independence lock (Week-2 build item; no existing generator is assumed — the original was deleted):** a standalone compositing generator produces every triplet frame independently from the same high-res source: x₀ = D(W(x_HR, I)), x₁ = D(W(x_HR, G)), x₂ = D(W(x_HR, G²)). **Never generate x₂ by warping x₁** (compounds resampling error and breaks the exact-G² guarantee).
- **Matched-motion sampling (primary causal control):** tune per-family magnitude ranges so **median object-normalized pixel displacement matches across rungs**; report median, p90 displacement, visible-mask-area change (boundary displacement if cheap). Primary evidence = within-family plain-vs-subset interaction + matched-motion across-family comparison. The unmatched across-family ladder is supporting evidence only (otherwise "higher estimator floor" is confounded with "more image motion").
- Splits in the generator: held-out transformation parameters + start poses; object lists for per-object and joint training. `val = train` is dead.
- **Corrected framing:** full affine is a controlled image-space deformation family that *locally approximates* some projected rigid-motion effects (planar-patch / first-order); it is **not** generally equivalent to weak-perspective rigid 3D motion of a nonplanar object. No global Tomasi–Kanade claim.

---

## 3. Week-by-week

### Week 1 (Jul 3–10) — Block-0 algebraic gate + parallel housekeeping
**Compute note: Block 0 needs NO GPU/cluster/training — coordinate-only, CPU, minutes on the Mac.**

**Track A (days 1–3, decision Mon Jul 7):**
1. Implement the 4 regularized solvers (§2.2) + subset loss (§2.1) + heatmap-mass grounding stub + candidate activity measures.
2. Test worlds: {healthy; exact 3+7 duplicates; near-duplicates ε∈{1e-3,1e-2}; full collapse; **static-off-object**; **static-on-object / fixed-coordinate-with-changing-pixels** (oracle appearance-change flag); collinear-tracking; collinear-nontracking; **coincident 2-pt similarity subsets** (degenerate denominator); **lone-tracker A**: (K−1) static + 1 tracking; **lone-tracker B**: (K−1) tracking + 1 static} × K grid respecting K>|S| (affine complement at K∈{6,10} primary, K=4 near-floor diagnostic; K=3 plain-fit only) × matched solvers (+1 documented mismatch) × {shared, independent} parametrization.
3. Jitter arms σ ∈ {0, 0.5, 1, 2} heatmap cells (minimal fits amplify noise ~σ/r).
4. **Activity-candidate ordering test:** each candidate must produce static < jitter < correct-tracking, with the rotation-centre point NOT penalized (G-aware) and the circularity documented (G-relative measures fail the all-static world without an image-referenced signal).
5. ε-escape: ≥10 inits, spatial rank must increase; gradient-norm-vs-ε logging (measured scaling, not assumed); clipping/λ from measurement. Finite-gradient checks at ALL degeneracies including coincident similarity pairs. **Lone-tracker rescue optimization:** from both lone-tracker inits, optimize the toy representation and log whether the fitted operator moves toward G_true or identity, and whether the minority trajectory is recruited or suppressed — decides whether the all-static optimum is escapable via symmetry-broken initialization or an attracting basin requiring an activity constraint. **Timebox: mechanical solver/subset gate first; activity-candidate exploration briefly afterward — do not let it broaden into a new research project.**
6. Aggregation: mean (primary) vs softmin vs min (documented failure).
7. **GATE (Jul 7), family-specific:** ordering holds where theory predicts (duplicates/collapse/static-off-object > healthy under subset+grounding); collinear-tracking ≈ 0 (documented P3, affine only); fixed-coordinate world separated ONLY by a passing activity candidate (this decides whether an activity term ships); plain-fit floors = 1/1/2/3; ε-escape works; **ordering survives σ = 1 cell**. FAIL → impossibility-paper pivot (honest: workshop-to-borderline).

**Track B (parallel):** 3-seed audit (Tasks 55/80/20, seeds 43/44, 6 runs); two Pareto fixes in `run_reanalysis.py` ~L917; conditioning battery on existing checkpoints → F2 (SV spectrum, %full-rank subsets, minimal-subset error, leverage). Writing starts now: verify MSP/NFT objectives from the papers; **begin the literature refresh** — the method now sits in differentiable closed-form geometric estimation; audit DSAC (differentiable RANSAC), Kabsch/Procrustes layers, BPnP, declarative layers, correspondence/geometric-consistency self-supervision, robust subset fitting, BEFORE novelty claims.

### Week 2 (Jul 10–17) — Dataset + training branch
8. `DATASET_SEMANTIC_LOCK.md` first (triplets, compositing hygiene, matched-motion sampling, corrected affine framing, splits, nuisance variants).
9. Compositing pipeline (rungs ②–④) + rung-① dual arm; pilot rung ④ on hammer ~300 triplets → whole-mask warp-residual verification → full generation with matched-motion calibration.
10. Code branch: `FittedOperator` ported from Block 0 (its grid = permanent regression suite); `TripletDataset`; L_fg; activity term ONLY if a candidate passed the Block-0 ordering test; inv/cycle disabled; proper val loader; fitted-G rollout + transform-recovery eval.
11. Tiered regression gate: (i) oracle coords = pass/fail; (ii) rendered-marker/mask points = geometry pass/fail; (iii) Task 55/80 keypoints = diagnostic only.
12. Transporter adaptation starts. Sync to git tree.
**GATE (Jul 17):** pilot verified + tiers (i)/(ii) pass → cluster allowed.

### Week 3 (Jul 18–24) — Screening, seed 42 (~20 runs)
13. 4 rungs × {plain-fit, subset} × {disp+ent on, off}; + K=4 rung ④; + oracle-vs-motion grounding rung ④; optional GT-operator arm (labeled).
14. **Pre-registered predictions:** plain-fit arms: spatial rank / minimal-subset recovery increase with family complexity, no change ①→② (floor-equal control), no equality-with-floor claims. Subset arms: anti-duplication on ③④, none on floor-1 rungs (interaction control). Secondary: radial migration (mean radial distance within object support, subset vs plain, matched seeds). Matched-motion comparisons primary.
15. Draft propositions P1–P3+M during runs.

### Week 4 (Jul 25–31) — Confirmation + targeted ablation
16. Promoted contrasts at seeds 43/44 (~16 runs); report paired seed-level effects with raw data points and directional replication in ≥2/3 seeds — **no formal non-inferiority claims at n=3**.
17. Targeted ablation at winning rung: all-on / −dispersion / −entropy / −both / subset-off. Expected: subset replaces coordinate repulsion; heatmap concentration remains necessary.
18. Held-out transformation parameters + start poses.
**GATE (Jul 31):** mechanism trend on ≥2/3 seeds → proceed; else fallback (taxonomy + conditioning + propositions), honestly assessed.

### Week 5 (Aug 1–7) — Generalization as three separate claims
19. Core: within-object holdouts + per-object replication ≥4 objects, frozen hyperparameters.
20. Joint-train unseen-object: planned and reported either way, NOT gate-coupled.
21. Seed-/object-level error bars only.

### Week 6 (Aug 8–14) — Baselines + spectrum figure
22. Transporter, neutral protocol.
23. Generic-20D latent: matched trained probe; direct-estimation vs learned-probing reported as different protocols; never judged on on-object/duplicate metrics.
24. **Mask-baseline supervision tiers (never presented as equivalent; tier labeled in the figure):**
    - T1 current-frame-mask geometric detectors: random mask points / farthest-point sampling / silhouette-extremity points;
    - T2 canonical-mask + GT-G propagated optimal layout (oracle upper bound);
    - T3 full-triplet optimization (most privileged).
    Purpose: does the system learn appearance-attached repeatable landmarks, or approximate an optimal geometric sensor arrangement?
25. F7: latent-operator eigenspectrum vs keypoint operator — rung ① or per-magnitude-bin only (shared W undefined on mixed-G data).
26. Bridge arm: learned shared W on rung ① (2 runs). Downstream: transform estimation vs GT + tracking repeatability. Slot Attention only with slack.

### Week 7 (Aug 15–24) — Figures + assembly
27. Frozen Aug 24: F1 degeneracy taxonomy · F2 conditioning battery · F3 complexity trend + subset interaction (matched-motion) · F4 targeted ablation · F5 baselines incl. mask tiers · F6 transform recovery + holdouts · F7 spectra.
28. Theory = assembly of drafted propositions. Refresh `docs/positioning_refs.csv`: MSP, Wang 2026, Umeyama, RANSAC, OptNet/declarative, Nibali DSNT, **DSAC, Kabsch layers, BPnP**.

### Week 8 (Aug 25–31) — Full draft
29. Draft Aug 31; Angela reads Sep 1 week.
30. Venue: ICLR 2027 dates UNKNOWN (iclr.cc: "West Coast NA, TBA"; "Brazil / Sep 19–24" trackers are year-shifted ICLR-2026 junk — do not cite). Late-Sep expected by cadence only; weekly iclr.cc + arXiv scoop-watch. NeurIPS 2027 fallback.

---

## 4. Run budget
Audit 6 + screening ~22 + confirmation ~16 + ablation ~5 + multi-object ~12 + baselines/bridge/spectrum/mask-tiers ~12 ≈ **73 runs, all figure-mapped. No sweeps.**

## 5. Standing rules
- No cluster run without verified data + passing tiered regression + a named figure.
- The loss never sees G. Every privileged arm labeled in the figure itself.
- No activity loss in training unless it passed the Block-0 ordering test (static < jitter < tracking).
- "Good enough" = pre-registered thresholds, ≥2/3 seeds, per claim type. Operator-on-roll: done.
- 128-res + contrastive-descriptor: secondary.
- Honest odds ~35–45% at ICLR conditional on clean confirmation. Top risk: toy confirms but encoder doesn't transmit the gradient (uninsurable → fallback).

## 6. Literature map
As v4.1 (Transporter wk2/6; MSP/NFT wk1/7; HAE, Ruiz-Morales, Wang, Minderer wk7; Suwajanakorn/B-KinD/Lorenz wk1–3; Thewlis/Jakab/Zhang, Higgins, Cohen&Welling/Worrall wk7; Fang appendix; Yu ToI framing) **plus the differentiable-geometric-estimation lineage: DSAC, Kabsch/Procrustes layers, BPnP, OptNet/declarative networks, Nibali DSNT, Umeyama, RANSAC.** Related-work audit refresh REQUIRED before novelty claims (starts wk1, finalized wk7) — the old survey was organized around a learned shared operator, which is no longer the central method.

## 7. Immediate next action
**Block 0, Track A (steps 1–7). No GPU, no cluster, no training — one CPU file, decision July 7.**
